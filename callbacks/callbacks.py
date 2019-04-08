from fastai.callbacks import *

from helper.object_detection_helper import *

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *

class BBLossMetrics(LearnerCallback):
    "Add `loss_func.metrics` to metrics named by `loss_func.metric_names`"
    _order = -20 #Needs to run before the recorder

    def on_train_begin(self, **kwargs):
        "Add the metrics names to the `Recorder`."
        self.names = ifnone(self.learn.loss_func.metric_names, [])
        if not self.names: warn('LossMetrics requested by no loss_func.metric_names provided')
        self.learn.recorder.add_metric_names(self.names)

    def on_epoch_begin(self, **kwargs):
        "Initialize the metrics for this epoch."
        self.metrics = {name:0. for name in self.names}
        self.nums = 0

    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        if train: return
        bs = last_target[0].size(0)
        for name in self.names:
            self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if not self.nums: return
        metrics = [self.metrics[name]/self.nums for name in self.names]
        return {'last_metrics': last_metrics + metrics}


class BBMetrics(LearnerCallback):
    "Add `loss_func.metrics` to metrics named by `loss_func.metric_names`"
    _order = -20 #Needs to run before the recorder

    def on_train_begin(self, **kwargs):
        "Add the metrics names to the `Recorder`."
        self.names = ifnone(self.learn.loss_func.metric_names, [])
        if not self.names: warn('LossMetrics requested by no loss_func.metric_names provided')

        for m in self.learn.metrics:
            if hasattr(m, 'metric_names'):
                for name in m.metric_names:
                    if name not in self.learn.loss_func.metric_names:
                        self.names.append(name)
        if not self.names: warn('Metrics names requested by no metrics.metric_names provided')
        self.learn.recorder.add_metric_names(self.names)

    def on_epoch_begin(self, **kwargs):
        "Initialize the metrics for this epoch."
        self.metrics = {name:0. for name in self.names}
        self.nums = 0

    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        if train: return
        bs = last_target[0].size(0)
        for name in self.names:
            if name in self.learn.loss_func.metrics:
                self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if not self.nums: return
        metrics = [self.metrics[name]/self.nums for name in self.names if name in self.learn.loss_func.metrics]

        for name in self.names:
            for metric in self.learn.metrics:
                if hasattr(metric, 'metric_names') and name in metric.metrics.keys():
                    metrics.append(metric.metrics[name])

        return {'last_metrics': last_metrics + metrics}


class PascalVOCMetric(Callback):

    def __init__(self, anchors, size, metric_names: list, detect_thresh: float=0.3, nms_thresh: float=0.3
                 , images_per_batch: int=-1):
        self.ap = 'AP'
        self.anchors = anchors
        self.size = size
        self.detect_thresh = detect_thresh
        self.nms_thresh = nms_thresh

        self.images_per_batch = images_per_batch
        self.metric_names_original = metric_names
        self.metric_names = ["{}-{}".format(self.ap, i) for i in metric_names]

        self.evaluator = Evaluator()
        self.boundingBoxes = BoundingBoxes()

    def on_epoch_begin(self, **kwargs):
        self.boundingBoxes.removeAllBoundingBoxes()
        self.imageCounter = 0


    def on_batch_end(self, last_output, last_target, **kwargs):
        bbox_gt_batch, class_gt_batch = last_target
        class_pred_batch, bbox_pred_batch = last_output[:2]

        self.images_per_batch = self.images_per_batch if self.images_per_batch > 0 else class_pred_batch.shape[0]
        for bbox_gt, class_gt, clas_pred, bbox_pred in \
                list(zip(bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch))[: self.images_per_batch]:

            bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
            if bbox_pred is None:# or len(preds) > 3 * len(bbox_gt):
                continue

            #image = np.zeros((512, 512, 3), np.uint8)

            # if the number is to hight evaluation is very slow
            total_nms_examples = len(class_gt) * 3
            bbox_pred = bbox_pred[:total_nms_examples]
            scores = scores[:total_nms_examples]
            preds = preds[:total_nms_examples]
            to_keep = nms(bbox_pred, scores, self.nms_thresh)
            bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()

            t_sz = torch.Tensor([(self.size, self.size)])[None].cpu()
            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0]
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))
            bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
            # change from center to top left
            bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2

            class_gt = to_np(class_gt) - 1
            preds = to_np(preds)
            scores = to_np(scores)

            for box, cla in zip(bbox_gt, class_gt):
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                               w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(self.size,self.size))

                self.boundingBoxes.addBoundingBox(temp)

            # to reduce math complexity take maximal three times the number of gt boxes
            num_boxes = len(bbox_gt) * 3
            for box, cla, scor in list(zip(bbox_pred, preds, scores))[:num_boxes]:
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                                   w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute, classConfidence=scor,
                                   bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(self.size, self.size))

                self.boundingBoxes.addBoundingBox(temp)

            #image = self.boundingBoxes.drawAllBoundingBoxes(image, str(self.imageCounter))
            self.imageCounter += 1


    def on_epoch_end(self, last_metrics, **kwargs):
        if self.boundingBoxes.count() > 0:
            self.metrics = {}
            metricsPerClass = self.evaluator.GetPascalVOCMetrics(self.boundingBoxes, IOUThreshold=0.3)
            self.metric = max(sum([mc[self.ap] for mc in metricsPerClass]) / len(metricsPerClass), 0)

            for mc in metricsPerClass:
                self.metrics['{}-{}'.format(self.ap, mc['class'])] = max(mc[self.ap], 0)

            return {'last_metrics': last_metrics + [self.metric]}
        else:
            self.metrics = dict(zip(self.metric_names, [0 for i in range(len(self.metric_names))]))
            return {'last_metrics': last_metrics + [0]}