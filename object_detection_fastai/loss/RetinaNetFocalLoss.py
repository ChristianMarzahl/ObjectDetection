from fastai import *
from fastai.vision import *
from torch.autograd import Variable

from object_detection_fastai.helper.object_detection_helper import *

class RetinaNetFocalLoss(nn.Module):

    def __init__(self, anchors: Collection[float], gamma: float = 2., alpha: float = 0.25, pad_idx: int = 0,
                 reg_loss: LossFunction = F.smooth_l1_loss):
        super().__init__()
        self.gamma, self.alpha, self.pad_idx, self.reg_loss = gamma, alpha, pad_idx, reg_loss
        self.anchors = anchors
        self.metric_names = ['BBloss', 'focal_loss']

    def _unpad(self, bbox_tgt, clas_tgt):
        i = torch.min(torch.nonzero(clas_tgt - self.pad_idx)) if sum(clas_tgt)>0 else 0
        return tlbr2cthw(bbox_tgt[i:]), clas_tgt[i:] - 1 + self.pad_idx

    def _focal_loss(self, clas_pred, clas_tgt):
        encoded_tgt = encode_class(clas_tgt, clas_pred.size(1))
        ps = torch.sigmoid(clas_pred)
        weights = Variable(encoded_tgt * (1 - ps) + (1 - encoded_tgt) * ps)
        alphas = (1 - encoded_tgt) * self.alpha + encoded_tgt * (1 - self.alpha)
        weights.pow_(self.gamma).mul_(alphas)
        clas_loss = F.binary_cross_entropy_with_logits(clas_pred, encoded_tgt, weights, reduction='sum')
        return clas_loss

    def _one_loss(self, clas_pred, bbox_pred, clas_tgt, bbox_tgt):
        bbox_tgt, clas_tgt = self._unpad(bbox_tgt, clas_tgt)
        matches = match_anchors(self.anchors, bbox_tgt)
        bbox_mask = matches >= 0
        if bbox_mask.sum() != 0:
            bbox_pred = bbox_pred[bbox_mask]
            bbox_tgt = bbox_tgt[matches[bbox_mask]]
            bb_loss = self.reg_loss(bbox_pred, bbox_to_activ(bbox_tgt, self.anchors[bbox_mask]))
        else:
            bb_loss = 0.
        matches.add_(1)
        clas_tgt = clas_tgt + 1
        clas_mask = matches >= 0
        clas_pred = clas_pred[clas_mask]
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]
        return bb_loss, self._focal_loss(clas_pred, clas_tgt) / torch.clamp(bbox_mask.sum(), min=1.)

    def forward(self, output, bbox_tgts, clas_tgts):
        clas_preds, bbox_preds, sizes = output
        if bbox_tgts.device != self.anchors.device:
            self.anchors = self.anchors.to(clas_preds.device)

        bb_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        focal_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        for cp, bp, ct, bt in zip(clas_preds, bbox_preds, clas_tgts, bbox_tgts):
            bb, focal = self._one_loss(cp, bp, ct, bt)

            bb_loss += bb
            focal_loss += focal

        self.metrics = dict(zip(self.metric_names, [bb_loss / clas_tgts.size(0), focal_loss / clas_tgts.size(0)]))
        return (bb_loss+focal_loss) / clas_tgts.size(0)

class SigmaL1SmoothLoss(nn.Module):

    def forward(self, output, target):
        reg_diff = torch.abs(target - output)
        reg_loss = torch.where(torch.le(reg_diff, 1/9), 4.5 * torch.pow(reg_diff, 2), reg_diff - 1/18)
        return reg_loss.mean()
