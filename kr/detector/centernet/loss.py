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
    eps = 1e-20

    i_ids = gt_indices[:, 0]
    j_ids = gt_indices[:, 1]
    mask = xp.zeros_like(gt_heatmap[0:-4])
    mask[gt_labels, i_ids, j_ids] = 1

    # center score loss
    y = heatmap[0:-4]
    y_gt = gt_heatmap[0:-4]
    if len(gt_labels):
        pos_loss = -((1 - y) ** alpha * F.log(y + eps))
        neg_loss = -((1 - y_gt) ** beta * (y ** alpha) * F.log(1 - y + eps))
        score_loss = F.mean((mask * pos_loss + (1 - mask)  * neg_loss))
    else:
        neg_loss = -(1 - y_gt) ** beta * (y ** alpha) * F.log(1 - y + eps)
        score_loss = F.mean(neg_loss)

    if len(gt_labels):
        # size loss
        size_loss = F.mean_absolute_error(heatmap[-4:-2, i_ids, j_ids],
                                          gt_heatmap[-4:-2, i_ids, j_ids])
        # offset loss
        offset_loss = F.mean_absolute_error(heatmap[-2:, i_ids, j_ids],
                                            gt_heatmap[-2:, i_ids, j_ids])
    else:
        size_loss = Variable(xp.array(0., dtype=xp.float32))
        offset_loss = Variable(xp.array(0., dtype=xp.float32))

    return score_loss, w_size * size_loss, w_offset * offset_loss
