from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models.unet import _get_sfs_idxs

# export
class LateralUpsampleMerge(nn.Module):

    def __init__(self, ch, ch_lat, hook):
        super().__init__()
        self.hook = hook
        self.conv_lat = conv2d(ch_lat, ch, ks=1, bias=True)

    def forward(self, x):
        return self.conv_lat(self.hook.stored) + F.interpolate(x, scale_factor=2)


class RetinaNet(nn.Module):
    "Implements RetinaNet from https://arxiv.org/abs/1708.02002"

    def __init__(self, encoder: nn.Module, n_classes, final_bias:float=0.,  n_conv:float=4,
                 chs=256, n_anchors=9, flatten=True, sizes=None):
        super().__init__()
        self.n_classes, self.flatten = n_classes, flatten
        imsize = (256, 256)
        self.sizes = sizes
        sfs_szs, x, hooks = self._model_sizes(encoder, size=imsize)
        sfs_idxs = _get_sfs_idxs(sfs_szs)
        self.encoder = encoder
        self.c5top5 = conv2d(sfs_szs[-1][1], chs, ks=1, bias=True)
        self.c5top6 = conv2d(sfs_szs[-1][1], chs, stride=2, bias=True)
        self.p6top7 = nn.Sequential(nn.ReLU(), conv2d(chs, chs, stride=2, bias=True))
        self.merges = nn.ModuleList([LateralUpsampleMerge(chs, szs[1], hook)
                                     for szs, hook in zip(sfs_szs[-2:-4:-1], hooks[-2:-4:-1])])
        self.smoothers = nn.ModuleList([conv2d(chs, chs, 3, bias=True) for _ in range(3)])
        self.classifier = self._head_subnet(n_classes, n_anchors, final_bias, chs=chs, n_conv=n_conv)
        self.box_regressor = self._head_subnet(4, n_anchors, 0., chs=chs, n_conv=n_conv)

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256):
        layers = [self._conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)

    def _apply_transpose(self, func, p_states, n_classes):
        if not self.flatten:
            sizes = [[p.size(0), p.size(2), p.size(3)] for p in p_states]
            return [func(p).permute(0, 2, 3, 1).view(*sz, -1, n_classes) for p, sz in zip(p_states, sizes)]
        else:
            return torch.cat(
                [func(p).permute(0, 2, 3, 1).contiguous().view(p.size(0), -1, n_classes) for p in p_states], 1)

    def _model_sizes(self, m: nn.Module, size:tuple=(256,256), full:bool=True) -> Tuple[Sizes,Tensor,Hooks]:
        "Passes a dummy input through the model to get the various sizes"
        hooks = hook_outputs(m)
        ch_in = in_channels(m)
        x = torch.zeros(1,ch_in,*size)
        x = m.eval()(x)
        res = [o.stored.shape for o in hooks]
        if not full: hooks.remove()
        return res,x,hooks if full else res

    def _conv2d_relu(self, ni:int, nf:int, ks:int=3, stride:int=1,
                    padding:int=None, bn:bool=False, bias=True) -> nn.Sequential:
        "Create a `conv2d` layer with `nn.ReLU` activation and optional(`bn`) `nn.BatchNorm2d`"
        layers = [conv2d(ni, nf, ks=ks, stride=stride, padding=padding, bias=bias), nn.ReLU()]
        if bn: layers.append(nn.BatchNorm2d(nf))
        return nn.Sequential(*layers)

    def forward(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        for merge in self.merges:
            p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        if self.sizes is not None:
            p_states = [p_state for p_state in p_states if p_state.size()[-1] in self.sizes]
        return [self._apply_transpose(self.classifier, p_states, self.n_classes),
                self._apply_transpose(self.box_regressor, p_states, 4),
                [[p.size(2), p.size(3)] for p in p_states]]
