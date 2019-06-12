from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models.unet import _get_sfs_idxs

from models.RetinaNet import RetinaNet



class RetinaNetRPN(RetinaNet):

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
                [[p.size(2), p.size(3)] for p in p_states], p_states]
