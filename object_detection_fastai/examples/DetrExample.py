# add detr
import sys
sys.path.append("/root/ProgProjekte/detr")

import warnings
warnings.filterwarnings('ignore')


from fastai import *
from fastai.vision import *

from object_detection_fastai.callbacks.callbacks import BBMetrics, PascalVOCMetric, PascalVOCMetricDETR
from pathlib import Path

from object_detection_fastai.loss.DetrCriterion import SetCriterionFastAi 
from object_detection_fastai.models.Detr import DETRFastAi

from models.matcher import HungarianMatcher


aux_loss = False
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2
eos_coef = 0.1
num_classes = 6



body = create_body(models.resnet18, True, -2)
model = DETRFastAi(body, num_classes=num_classes, aux_loss=aux_loss)


losses = ['labels', 'boxes', 'cardinality']
weight_dict = {'loss_ce': 1, 'loss_bbox': set_cost_bbox, 'loss_giou': set_cost_giou}
matcher = HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)

crit = SetCriterionFastAi(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=eos_coef, losses=losses)


size = 128
coco = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco/'train.json') #'annotations/train_sample.json'
img2bbox = dict(zip(images, lbl_bbox))
get_y_func = lambda o:img2bbox[o.name]

data = (ObjectItemList.from_folder(coco)
            #Where are the images? -> in coco
            .random_split_by_pct()
            #How to split in train/valid? -> randomly with the default 20% in valid
            .label_from_func(get_y_func)
            #How to find the labels? -> use get_y_func
            .transform(get_transforms(), tfm_y=True, size=size)
            #Data augmentation? -> Standard transforms with tfm_y=True
            .databunch(bs=16, collate_fn=bb_pad_collate)) #, num_workers=1
            #Finally we convert to a DataBunch and we use bb_pad_collate


learn = Learner(data, model, loss_func=crit, callback_fns=[ShowGraph, BBMetrics]) #,metrics=[voc]

learn.fit_one_cycle(3, 1e-3)

print()