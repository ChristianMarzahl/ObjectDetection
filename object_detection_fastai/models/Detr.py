from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from util.misc import NestedTensor, get_world_size, is_dist_avail_and_initialized

from models.transformer import Transformer
from models.backbone import Joiner
from models.detr import DETR
from models.position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine



class Backbone(nn.Module):
    def __init__(self, body, return_interm_layers=False):
        super().__init__()

        if return_interm_layers:
            return_layers = {"4": 0, "5": 1, "6": 2, "7": 3}
        else:
            return_layers = {"7": 0}
        self.body = IntermediateLayerGetter(body, return_layers=return_layers)

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class DETRFastAi(DETR):
    def __init__(self, body, num_classes=90, num_queries=100, aux_loss=True, num_channels=512, hidden_dim=64, dropout=.1, nheads=8,   
                    dim_feedforward=256, enc_layers=2, dec_layers=2, 
                        pre_norm=False, return_intermediate_dec=True, position_embedding=None):

        backbone = Backbone(body=body)
        N_steps = hidden_dim // 2

        position_embedding = position_embedding if position_embedding is not None else PositionEmbeddingSine(N_steps, normalize=True)
        model = Joiner(backbone, position_embedding)
        model.num_channels = num_channels

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=True,
        )

        super().__init__(model, transformer, num_classes=num_classes, num_queries=num_queries, aux_loss=aux_loss)

class DETRFastAiSimple(nn.Module):

    def __init__(self, body, num_classes, num_queries=100, num_channels=512, hidden_dim=256, nheads=8,
                 num_layers=6):
        super().__init__()

        self.backbone = body

        # create conversion layer
        self.conv = nn.Conv2d(num_channels, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_layers, num_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):

        x = self.backbone(inputs)

        # convert from num_channels to hidden_dim feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}