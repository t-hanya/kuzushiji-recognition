"""
Postprocessing module.
"""


from typing import List
from typing import Tuple
from typing import Union

import chainer.functions as F
from chainer.backends import cuda
import numpy as np


NdArray = Union[np.ndarray, cuda.ndarray]


def heatmap_to_labeled_bboxes(heatmap: NdArray,
                              score_threshold: float = 0.5,
                             ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert heatmap tensor to labeled bounding boxes."""
    xp = cuda.get_array_module(heatmap)
    N = len(heatmap)

    values, indices = F.max_pooling_2d(
        heatmap[:, 0:-4, :, :], ksize=3, pad=1, stride=1, return_indices=True)

    batch_indices, labels, i_indices, j_indices = xp.where(
        xp.logical_and(values.array >= score_threshold, indices == 4))

    w = heatmap[batch_indices, -4, i_indices, j_indices]
    h = heatmap[batch_indices, -3, i_indices, j_indices]
    dx = heatmap[batch_indices, -2, i_indices, j_indices]
    dy = heatmap[batch_indices, -1, i_indices, j_indices]
    scores = heatmap[batch_indices, labels, i_indices, j_indices]

    x1 = j_indices + dx - w / 2.
    y1 = i_indices + dy - h / 2.
    x2 = j_indices + dx + w / 2.
    y2 = i_indices + dy + h / 2.

    bboxes = xp.transpose(xp.stack([x1, y1, x2, y2]))

    bboxes_list = []
    labels_list = []
    scores_list = []
    for i in range(N):
        batch_mask = batch_indices == i
        bboxes_list.append(bboxes[batch_mask])
        labels_list.append(labels[batch_mask])
        scores_list.append(scores[batch_mask])

    return bboxes_list, labels_list, scores_list