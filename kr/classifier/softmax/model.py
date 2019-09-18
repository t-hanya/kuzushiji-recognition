"""
Base model for classification.
"""


from typing import Tuple

import chainer
import chainer.functions as F
from chainer.backends import cuda
import numpy as np
from PIL import Image


class SoftmaxClassifierBase(chainer.Chain):
    """Base class of softmax classifier."""

    input_size = (112, 112)

    def classify(self,
                 image: Image.Image,
                 bboxes: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:

        if not len(bboxes):
            return (np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32))

        # crop character regions
        # - with small padding around character bbox
        # - keep original aspect ratio
        crop_bboxes = _calc_padded_bboxes(bboxes)
        images = [image.crop(bbox) for bbox in crop_bboxes]

        # convert to NCHW tensor format
        images = [img.resize(self.input_size, resample=Image.BILINEAR)
                  for img in images]
        img_arr = np.array([np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
                            for img in images])

        # inference
        if self.xp != np:
            img_arr = cuda.to_gpu(img_arr)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            probs = F.softmax(self(img_arr)).array

        # get labels and scores
        labels = self.xp.argmax(probs, axis=1)
        scores = self.xp.max(probs, axis=1)

        if self.xp != np:
            labels = cuda.to_cpu(labels)
            scores = cuda.to_cpu(scores)

        return labels, scores


def _calc_padded_bboxes(bboxes: np.ndarray,
                        pad_scale: float = 0.1) -> np.ndarray:

    centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
    longer_sides = (bboxes[:, 2:4] - bboxes[:, 0:2]).max(axis=1)
    padded_size = longer_sides * (1 + pad_scale * 2)

    padded_bboxes = np.empty_like(bboxes)
    padded_bboxes[:, 0:2] = np.floor(centers - padded_size[:, None] / 2.)
    padded_bboxes[:, 2:4] = np.ceil(centers + padded_size[:, None] / 2.)

    return padded_bboxes