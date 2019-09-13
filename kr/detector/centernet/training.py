"""
Training module.
"""


from typing import List
from typing import Union

import chainer
import chainer.functions as F
from chainer.backends import cuda
import numpy as np

from .loss import calc_loss


NdArray = Union[np.ndarray, cuda.ndarray]


class TrainingModel(chainer.Chain):
    """Training model."""

    def __init__(self, model: chainer.Chain) -> None:
        super().__init__()
        with self.init_scope():
            self.model = model

    def __call__(self,
                 images: NdArray,
                 gt_heatmap_list: List[NdArray],
                 gt_labels_list: List[NdArray],
                 gt_indices_list: List[NdArray]
                ) -> chainer.Chain:

        heatmaps = self.model(images)

        score_loss = 0.
        size_loss = 0.
        offset_loss = 0.
        for heatmap, gt_heatmap, gt_labels, gt_indices in zip(
            heatmaps, gt_heatmap_list, gt_labels_list, gt_indices_list):

            ret = calc_loss(heatmap, gt_heatmap, gt_labels, gt_indices)
            score_loss += ret[0]
            size_loss += ret[1]
            offset_loss += ret[2]

        # normalize scale
        batchsize = len(heatmap)
        score_loss /= batchsize
        size_loss /= batchsize
        offset_loss /= batchsize

        loss = score_loss + size_loss + offset_loss

        chainer.report({'loss': loss,
                        'score_loss': score_loss,
                        'size_loss': size_loss,
                        'offset_loss': offset_loss}, self)

        return loss
