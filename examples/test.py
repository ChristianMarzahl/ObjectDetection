from pathlib import Path
from multiprocessing import *
freeze_support()

from fastai import *
from fastai.vision import *

from helper.object_detection_helper import *
from loss.RetinaNetFocalLoss import RetinaNetFocalLoss
from models.RetinaNet import RetinaNet

if __name__ == "__main__":


    dots = Path('D:/Datasets/DotsSmall')

    images, lbl_bbox = get_annotations(dots/'train.json')
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o:img2bbox[o.name]

    def bb_pad_collate(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
        "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
        samples = [s for s in samples if s[1].data[0].shape[0] > 0] # check that labels are available

        max_len = max([len(s[1].data[1]) for s in samples])
        bboxes = torch.zeros(len(samples), max_len, 4)
        labels = torch.zeros(len(samples), max_len).long() + pad_idx
        imgs = []
        for i,s in enumerate(samples):
            imgs.append(s[0].data[None])
            bbs, lbls = s[1].data
            bboxes[i,-len(lbls):] = bbs
            labels[i,-len(lbls):] = lbls
        return torch.cat(imgs,0), (bboxes,labels)

    data = (ObjectItemList.from_folder(dots)
            #Where are the images? -> in coco
            .random_split_by_pct()
            #How to split in train/valid? -> randomly with the default 20% in valid
            .label_from_func(get_y_func)
            #How to find the labels? -> use get_y_func
            .transform(get_transforms(), tfm_y=True, size=256)
            #Data augmentation? -> Standard transforms with tfm_y=True
            .databunch(bs=16, collate_fn=bb_pad_collate)) #, num_workers=1
            #Finally we convert to a DataBunch and we use bb_pad_collate

    data.show_batch(rows=2, ds_type=DatasetType.Valid, figsize=(6,6))

    n_classes = 2
    encoder = create_body(models.resnet18, True, -2)
    model = RetinaNet(encoder, n_classes, n_anchors=1, sizes=[16], chs=64)

    x = torch.randn(2,3,256,256)
    output = model(x)

    print([y.size() for y in output[:2]], output[2])

    ratios = [1]
    scales = [0.1]
    sizes = [(16, 16)]# [(32, 32),(16, 16), (8, 8), (4, 4), (2, 2)]
    anchors = create_anchors(sizes, ratios, scales)
    print(anchors.size())

    crit = RetinaNetFocalLoss(scales=scales, ratios=ratios)
    learn = Learner(data, model, loss_func=crit)

    learn.split([model.encoder[6], model.c5top5])
    learn.freeze()

    learn.fit_one_cycle(1, 1e-3)
    print()