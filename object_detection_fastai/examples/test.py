from pathlib import Path
from multiprocessing import *
freeze_support()

from fastai import *
from fastai.vision import *

from object_detection_fastai.helper.object_detection_helper import *
from object_detection_fastai.loss.RetinaNetFocalLoss import RetinaNetFocalLoss
from object_detection_fastai.models.RetinaNet import RetinaNet
from object_detection_fastai.callbacks.callbacks import BBLossMetrics, BBMetrics, PascalVOCMetric


if __name__ == "__main__":


    dots = Path('D:/Datasets/DotsSmall')

    images, lbl_bbox = get_annotations(dots/'train.json')
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o:img2bbox[o.name]

    data = (ObjectItemList.from_folder(dots)
            #Where are the images? -> in coco
            .random_split_by_pct()
            #How to split in train/valid? -> randomly with the default 20% in valid
            .label_from_func(get_y_func)
            #How to find the labels? -> use get_y_func
            .transform(get_transforms(), tfm_y=True, size=256)
            #Data augmentation? -> Standard transforms with tfm_y=True
            .databunch(bs=16, collate_fn=bb_pad_collate, num_workers=0)) #, num_workers=1
            #Finally we convert to a DataBunch and we use bb_pad_collate

    data.show_batch(rows=2, ds_type=DatasetType.Valid, figsize=(6,6))

    ratios = [1]
    scales = [0.1]
    sizes = [(16, 16)]# [(32, 32),(16, 16), (8, 8), (4, 4), (2, 2)]
    anchors = create_anchors(sizes, ratios, scales)

    n_classes = 2
    crit = RetinaNetFocalLoss(anchors)
    encoder = create_body(models.resnet18, True, -2)
    model = RetinaNet(encoder, n_classes=data.train_ds.c, n_anchors=1, sizes=[16], chs=128, final_bias=-4., n_conv=3)

    voc = PascalVOCMetric(anchors, 256, [str(i) for i in data.train_ds.y.classes[1:]])
    learn = Learner(data, model, loss_func=crit, callback_fns=[BBMetrics, ShowGraph],  metrics=[voc])

    learn.split([model.encoder[6], model.c5top5])
    learn.freeze_to(-2)

    learn.fit_one_cycle(1, 1e-4)


