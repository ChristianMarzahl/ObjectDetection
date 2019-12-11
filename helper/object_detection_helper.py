import numpy as np
from fastai import *
from fastai.vision import *


def create_anchors(sizes, ratios, scales, flatten=True):
    "Create anchor of `sizes`, `ratios` and `scales`."
    aspects = [[[s*math.sqrt(r), s*math.sqrt(1/r)] for s in scales] for r in ratios]
    aspects = torch.tensor(aspects).view(-1,2)
    anchors = []
    for h,w in sizes:
        #4 here to have the anchors overlap.
        sized_aspects = 4 * (aspects * torch.tensor([2/h,2/w])).unsqueeze(0)
        base_grid = create_grid((h,w)).unsqueeze(1)
        n,a = base_grid.size(0),aspects.size(0)
        ancs = torch.cat([base_grid.expand(n,a,2), sized_aspects.expand(n,a,2)], 2)
        anchors.append(ancs.view(h,w,a,4))
    return torch.cat([anc.view(-1,4) for anc in anchors],0) if flatten else anchors


def create_grid(size):
    "Create a grid of a given `size`."
    H, W = size if is_tuple(size) else (size,size)
    grid = FloatTensor(H, W, 2)
    linear_points = torch.linspace(-1+1/W, 1-1/W, W) if W > 1 else tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(-1+1/H, 1-1/H, H) if H > 1 else tensor([0.])
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1,2)


def tlbr2cthw(boxes):
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[:,:2] + boxes[:,2:])/2
    sizes = boxes[:,2:] - boxes[:,:2]
    return torch.cat([center, sizes], 1)


def encode_class(idxs, n_classes):
    target = idxs.new_zeros(len(idxs), n_classes).float()
    mask = idxs != 0
    i1s = LongTensor(list(range(len(idxs))))
    target[i1s[mask],idxs[mask]-1] = 1
    return target


def show_anchors(ancs, size):
    _,ax = plt.subplots(1,1, figsize=(5,5))
    ax.set_xticks(np.linspace(-1,1, size[1]+1))
    ax.set_yticks(np.linspace(-1,1, size[0]+1))
    ax.grid()
    ax.scatter(ancs[:,1], ancs[:,0]) #y is first
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim(-1,1)
    ax.set_ylim(1,-1) #-1 is top, 1 is bottom
    for i, (x, y) in enumerate(zip(ancs[:, 1], ancs[:, 0])):
        ax.annotate(i, xy = (x,y))


def show_boxes(boxes):
    "Show the `boxes` (size by 4)"
    _, ax = plt.subplots(1,1, figsize=(5,5))
    ax.set_xlim(-1,1)
    ax.set_ylim(1,-1)
    for i, bbox in enumerate(boxes):
        bb = bbox.numpy()
        rect = [bb[1]-bb[3]/2, bb[0]-bb[2]/2, bb[3], bb[2]]
        draw_rect(ax, rect)


def activ_to_bbox(acts, anchors, flatten=True):
    "Extrapolate bounding boxes on anchors from the model activations."
    if flatten:
        acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
        centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
        sizes = anchors[...,2:] * torch.exp(acts[...,2:])
        return torch.cat([centers, sizes], -1)
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res


def cthw2tlbr(boxes):
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[:,:2] - boxes[:,2:]/2
    bot_right = boxes[:,:2] + boxes[:,2:]/2
    return torch.cat([top_left, bot_right], 1)


def intersection(anchors, targets):
    "Compute the sizes of the intersections of `anchors` by `targets`."
    ancs, tgts = cthw2tlbr(anchors), cthw2tlbr(targets)
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(a,t,4), tgts.unsqueeze(0).expand(a,t,4)
    top_left_i = torch.max(ancs[...,:2], tgts[...,:2])
    bot_right_i = torch.min(ancs[...,2:], tgts[...,2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[...,0] * sizes[...,1]


def IoU_values(anchors, targets):
    "Compute the IoU values of `anchors` by `targets`."
    inter = intersection(anchors, targets)
    anc_sz, tgt_sz = anchors[:,2] * anchors[:,3], targets[:,2] * targets[:,3]
    union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter
    return inter/(union+1e-8)


def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    "Match `anchors` to targets. -1 is match to background, -2 is ignore."
    ious = IoU_values(anchors, targets)
    matches = anchors.new(anchors.size(0)).zero_().long() - 2

    if ious.shape[1] > 0:
        vals,idxs = torch.max(ious,1)
        matches[vals < bkg_thr] = -1
        matches[vals > match_thr] = idxs[vals > match_thr]
    #Overwrite matches with each target getting the anchor that has the max IoU.
    #vals,idxs = torch.max(ious,0)
    #If idxs contains repetition, this doesn't bug and only the last is considered.
    #matches[idxs] = targets.new_tensor(list(range(targets.size(0)))).long()
    return matches

def bbox_to_activ(bboxes, anchors, flatten=True):
    "Return the target of the model on `anchors` for the `bboxes`."
    if flatten:
        t_centers = (bboxes[...,:2] - anchors[...,:2]) / anchors[...,2:]
        t_sizes = torch.log(bboxes[...,2:] / anchors[...,2:] + 1e-8)
        return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res


def _draw_outline(o:Patch, lw:int):
    "Outline bounding box onto image `Patch`."
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax:plt.Axes, b:Collection[int], color:str='white', text=None, text_size=14):
    "Draw bounding box on `ax`."
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    _draw_outline(patch, 4)
    if text is not None:
        patch = ax.text(*b[:2], text, verticalalignment='top', color=color, fontsize=text_size, weight='bold')
        _draw_outline(patch,1)


def nms(boxes, scores, thresh=0.5):
    idx_sort = scores.argsort(descending=True)
    boxes, scores = boxes[idx_sort], scores[idx_sort]
    to_keep, indexes = [], torch.LongTensor(range_of(scores))
    while len(scores) > 0:
        #pdb.set_trace()
        to_keep.append(idx_sort[indexes[0]])
        iou_vals = IoU_values(boxes, boxes[:1]).squeeze()
        mask_keep = iou_vals <= thresh
        if len(mask_keep.nonzero()) == 0: break
        idx_first = mask_keep.nonzero().min().item()
        boxes, scores, indexes = boxes[mask_keep], scores[mask_keep], indexes[mask_keep]
    return LongTensor(to_keep)


def show_preds(img, bbox_pred, preds, scores, classes, figsize=(5,5)):

    _, ax = plt.subplots(1, 1, figsize=figsize)
    for bbox, c, scr in zip(bbox_pred, preds, scores):
        img.show(ax=ax)
        txt = str(c.item()) if classes is None else classes[c.item()+1]
        draw_rect(ax, [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr:.2f}')


def show_results_side_by_side(learn: Learner, anchors, detect_thresh:float=0.2, nms_thresh: float=0.3,  image_count: int=5):

    with torch.no_grad():
        img_batch, target_batch = learn.data.one_batch(DatasetType.Valid, False, False, False)

        prediction_batch = learn.model(img_batch[:image_count])
        class_pred_batch, bbox_pred_batch = prediction_batch[:2]

        bbox_gt_batch, class_gt_batch = target_batch[0][:image_count], target_batch[1][:image_count]

        for img, bbox_gt, class_gt, clas_pred, bbox_pred in list(
                zip(img_batch, bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch)):
            if hasattr(learn.data, 'stats'):
                img = Image(learn.data.denorm(img))
            else:
                img = Image(img)

            bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, anchors, detect_thresh)
            if bbox_pred is not None:
                to_keep = nms(bbox_pred, scores, nms_thresh)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()

            t_sz = torch.Tensor([*img.size])[None].cpu()
            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0] - 1
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))
            if bbox_pred is not None:
                bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
                # change from center to top left
                bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2

            show_results(img, bbox_pred, preds, scores, learn.data.train_ds.classes[1:] , bbox_gt, class_gt, (15, 15), titleA="GT", titleB="Prediction")


def show_results(img, bbox_pred, preds, scores, classes, bbox_gt, preds_gt, figsize=(5,5)
                 , titleA: str="", titleB: str=""):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].set_title(titleA)
    ax[1].set_title(titleB)
    # show prediction
    img.show(ax=ax[1])
    if bbox_pred is not None:
        for bbox, c, scr in zip(bbox_pred, preds, scores):
            txt = str(c.item()) if classes is None else classes[c.item()]
            draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr:.2f}')

    # show gt
    img.show(ax=ax[0])
    for bbox, c in zip(bbox_gt, preds_gt):
        txt = str(c.item()) if classes is None else classes[c.item()]
        draw_rect(ax[0], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt}')

def process_output(clas_pred, bbox_pred, anchors, detect_thresh=0.25):
    bbox_pred = activ_to_bbox(bbox_pred, anchors.to(clas_pred.device))
    clas_pred = torch.sigmoid(clas_pred)

    detect_mask = clas_pred.max(1)[0] > detect_thresh
    if np.array(detect_mask.cpu()).max() == 0:
        return None, None, None

    bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
    bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))
    scores, preds = clas_pred.max(1)
    return bbox_pred, scores, preds


def rescale_boxes(bboxes, t_sz: Tensor):

    bboxes[:, 2:] = bboxes[:, 2:] * t_sz / 2
    bboxes[:, :2] = (bboxes[:, :2] + 1) * t_sz / 2

    return bboxes

def show_anchors_on_images(data, anchors, figsize=(15,15)):
    all_boxes = []
    all_labels = []
    x, y = data.one_batch(DatasetType.Train, True, True)
    for image, bboxes, labels in zip(x, y[0], y[1]):
        image = Image(image.float().clamp(min=0, max=1))

        # 0=not found; 1=found; found 2=anchor
        processed_boxes = []
        processed_labels = []
        for gt_box in tlbr2cthw(bboxes[labels > 0]):
            matches = match_anchors(anchors, gt_box[None, :])
            bbox_mask = matches >= 0
            if bbox_mask.sum() != 0:
                bbox_tgt = anchors[bbox_mask]

                processed_boxes.append(to_np(gt_box))
                processed_labels.append(2)
                for bb in bbox_tgt:
                    processed_boxes.append(to_np(bb))
                    processed_labels.append(3)
            else:
                processed_boxes.append(to_np(gt_box))
                processed_labels.append(0)
                val, idx = torch.max(IoU_values(anchors, gt_box[None, :]), 0)
                best_fitting_anchor = anchors[idx][0]
                processed_boxes.append(to_np(best_fitting_anchor))
                processed_labels.append(1)

        all_boxes.extend(processed_boxes)
        all_labels.extend(processed_labels)

        processed_boxes = np.array(processed_boxes)
        processed_labels = np.array(processed_labels)

        _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ax[0].set_title("Matched Anchors")
        ax[1].set_title("No match")

        if sum(processed_labels == 2) > 0:
            imageBB = ImageBBox.create(*image.size, cthw2tlbr(tensor(processed_boxes[processed_labels > 1])),
                                           labels=processed_labels[processed_labels > 1],
                                           classes=["", "", "Match", "Anchor"], scale=False)

            image.show(ax=ax[0], y=imageBB)
        else:
            image.show(ax=ax[0])

        if sum(processed_labels == 0) > 0:
            imageBBNoMatch = ImageBBox.create(*image.size, cthw2tlbr(tensor(processed_boxes[processed_labels <= 1])),
                                                  labels=processed_labels[processed_labels <= 1],
                                                  classes=["No Match", "Anchor"], scale=False)
            image.show(ax=ax[1], y=imageBBNoMatch)
        else:
            image.show(ax=ax[1])


    return np.array(all_boxes), np.array(all_labels)
