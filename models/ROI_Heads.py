from fastai import *
from fastai.vision import *

import torch
from torch.nn import functional as F
from torch import nn

from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops.roi_align import RoIAlign

class RoIHeadsFast(nn.Module):


    def __init__(self, rpn, n_classes, anchors):
        super().__init__()

        self.n_classes = n_classes
        self.rpn = rpn
        self.anchors = anchors

        self.classifier = self._head_subnet(n_classes, 2)
        self.roi_align = RoIAlign((7, 7), spatial_scale=1.0, sampling_ratio=-1)


    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256):
        layers = [self._conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)

    def _conv2d_relu(self, ni:int, nf:int, ks:int=3, stride:int=1,
                    padding:int=None, bn:bool=False, bias=True) -> nn.Sequential:
        "Create a `conv2d` layer with `nn.ReLU` activation and optional(`bn`) `nn.BatchNorm2d`"
        layers = [conv2d(ni, nf, ks=ks, stride=stride, padding=padding, bias=bias), nn.ReLU()]
        if bn: layers.append(nn.BatchNorm2d(nf))
        return nn.Sequential(*layers)

    def forward(self, x):

        clas_preds, bbox_preds, sizes, features = self.rpn(x)

        #roi_features_dict =
        for feature_map in features:
            roi_features = self.roi_align(feature_map, bbox_preds)





