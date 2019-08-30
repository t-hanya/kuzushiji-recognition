"""
Loss module
"""


from typing import Union
from typing import Tuple

from chainer import Variable
import chainer.functions as F
from chainer.backends import cuda
import numpy as np


NdArray = Union[np.ndarray, cuda.ndarray]


def calc_loss(heatmap: Variable,
              gt_heatmap: NdArray,
              gt_labels: NdArray,
              gt_indices: NdArray,
              alpha: int = 2,
              beta: int = 4,
              w_size: float = 0.4,  # use 1/4 scaled size.
              w_offset: float = 1) -> Tuple[Variable, Variable, Variable]:
    """Calculate loss."""

    xp = cuda.get_array_module(gt_heatmap)
    N = max(len(gt_labels), 1)

    i_ids = gt_indices[:, 0]
    j_ids = gt_indices[:, 1]
    flags = xp.zeros_like(gt_heatmap[0:-4], dtype=np.int32)
    flags[gt_labels, i_ids, j_ids] = 1

    y_raw = heatmap[0:-4]
    y = F.sigmoid(y_raw)
    y_gt = gt_heatmap[0:-4]

    cross_entropy = F.sigmoid_cross_entropy(y_raw, flags, reduce='no')

    pos_focal_weight = (1 - y) ** alpha
    neg_focal_weight = ((1 - y_gt) ** beta) * (y ** alpha)
    focal_weight = flags * pos_focal_weight + (1 - flags) * neg_focal_weight

    score_loss = F.sum(focal_weight * cross_entropy) / N

    if len(gt_labels):
        size_loss = F.mean_absolute_error(heatmap[-4:-2, i_ids, j_ids],
                                          gt_heatmap[-4:-2, i_ids, j_ids])
        offset_loss = F.mean_absolute_error(heatmap[-2:, i_ids, j_ids],
                                            gt_heatmap[-2:, i_ids, j_ids])
    else:
        size_loss = Variable(xp.array(0., dtype=xp.float32))
        offset_loss = Variable(xp.array(0., dtype=xp.float32))

    return score_loss, w_size * size_loss, w_offset * offset_loss
