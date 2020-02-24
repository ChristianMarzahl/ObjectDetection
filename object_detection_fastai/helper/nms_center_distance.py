import numpy as np
from sklearn.neighbors import KDTree


def non_max_suppression_by_distance(boxes, scores, radius: float = 25, return_ids = False):

    center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
    center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

    X = np.dstack((center_x, center_y))[0]
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]

    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        id = sorted_ids[0]
        ids_to_keep.append(id)
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[id]).nonzero()[0])

    return boxes[ids_to_keep] if return_ids == False else ids_to_keep
