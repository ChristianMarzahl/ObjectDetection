import numpy as np
import torch
from util.misc import NestedTensor, get_world_size, is_dist_avail_and_initialized
from models.detr import SetCriterion

class SetCriterionFastAi(SetCriterion):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, metric_names=None):
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses)

        if metric_names is not None:
            self.metric_names = metric_names
        else:
            self.metric_names = ['loss_ce', 'class_error', 'loss_bbox', 'loss_giou', 'cardinality_error', 'loss_ce_0', 'loss_bbox_0', 'loss_giou_0', 'cardinality_error_0']


    def forward(self, outputs, bbox_tgts, clas_tgts):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device=next(iter(outputs.values())).device

        targets = []
        # for each image
        for bbox_gt, class_gt in zip(bbox_tgts, clas_tgts):
            # extract non zero boxes and labels
            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0] - 1
            # change gt from y,x,y2,x2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]
            bbox_gt = bbox_gt[:, [1,0,3,2]]
            # change gt from x,y,w,h -> cxcywh
            bbox_gt[:, :2] = bbox_gt[:, :2] + 0.5 * bbox_gt[:, 2:]

            # scale form input(-1, 1) to expected (0, 1)
            bbox_gt[:, 2:] = bbox_gt[:, 2:] / 2.
            bbox_gt[:, :2] = (bbox_gt[:, :2] + 1) / 2.

            targets.append({
                "boxes": bbox_gt.to(device),
                "labels": class_gt.to(device),
            })


        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.metrics = {}
        for name in losses:
            if name in self.metric_names:
                self.metrics[name] = losses[name] if name not in self.weight_dict else losses[name] * self.weight_dict[name]

        losses = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
        return losses