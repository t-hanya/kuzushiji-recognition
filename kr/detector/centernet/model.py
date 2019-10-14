"""
Model module.
"""


from typing import Tuple

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainercv.utils import non_maximum_suppression
import numpy as np
from PIL import Image

from .postprocess import heatmap_to_labeled_bboxes


def _sigmoid(x):
    xp = cuda.get_array_module(x)
    return 1. / (1. + xp.exp(-x))


class UnetCenterNet(chainer.Chain):
    """U-Net like encoder-decoder model."""

    stride: int = 4
    image_min_side: int = 832

    def __init__(self,
                 n_fg_class: int = 1,
                 score_threshold: float = 0.5
                ) -> None:
        super().__init__()
        out_ch = n_fg_class + 4
        self.score_threshold = score_threshold

        with self.init_scope():
            # change stride size from 1 to 2 to reduce output size
            self.c0 = L.Convolution2D(None, 32, 3, 2, 1)

            self.c1 = L.Convolution2D(32, 64, 4, 2, 1)
            self.c2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.c3 = L.Convolution2D(64, 128, 4, 2, 1)
            self.c4 = L.Convolution2D(128, 128, 3, 1, 1)
            self.c5 = L.Convolution2D(128, 256, 4, 2, 1)
            self.c6 = L.Convolution2D(256, 256, 3, 1, 1)
            self.c7 = L.Convolution2D(256, 512, 4, 2, 1)
            self.c8 = L.Convolution2D(512, 512, 3, 1, 1)

            self.dc8 = L.Deconvolution2D(1024, 512, 4, 2, 1)
            self.dc7 = L.Convolution2D(512, 256, 3, 1, 1)
            self.dc6 = L.Deconvolution2D(512, 256, 4, 2, 1)
            self.dc5 = L.Convolution2D(256, 128, 3, 1, 1)
            self.dc4 = L.Deconvolution2D(256, 128, 4, 2, 1)
            self.dc3 = L.Convolution2D(128, 64, 3, 1, 1)
            self.dc2 = L.Deconvolution2D(128, out_ch, 3, 1, 1)

            self.bnc0 = L.BatchNormalization(32)
            self.bnc1 = L.BatchNormalization(64)
            self.bnc2 = L.BatchNormalization(64)
            self.bnc3 = L.BatchNormalization(128)
            self.bnc4 = L.BatchNormalization(128)
            self.bnc5 = L.BatchNormalization(256)
            self.bnc6 = L.BatchNormalization(256)
            self.bnc7 = L.BatchNormalization(512)
            self.bnc8 = L.BatchNormalization(512)

            self.bnd8 = L.BatchNormalization(512)
            self.bnd7 = L.BatchNormalization(256)
            self.bnd6 = L.BatchNormalization(256)
            self.bnd5 = L.BatchNormalization(128)
            self.bnd4 = L.BatchNormalization(128)
            self.bnd3 = L.BatchNormalization(64)

    def forward(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        d2 = self.dc2(F.concat([e2, d3]))

        # force 0-1 value range to scores and offsets
        C = d2.shape[1]
        scores, sizes, offsets = F.split_axis(d2, (C - 4, C - 2), axis=1)
        offsets = F.sigmoid(offsets)

        return F.concat([scores, sizes, offsets])

    def detect(self,
               image: Image.Image,
               nms_iou_threshold: float = 0.5
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect characters from the image."""
        img_w, img_h = image.size

        if img_w < img_h:
            w = self.image_min_side
            h = img_h * self.image_min_side / img_w
            h = 32 * int(round(h / 32))
        else:
            h = self.image_min_side
            w = img_w * self.image_min_side / img_h
            w = 32 * int(round(w / 32))

        image = image.resize((w, h), resample=Image.BILINEAR)

        img = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        imgs = img.reshape(1, *img.shape)

        if self.xp != np:
            imgs = cuda.to_gpu(imgs)

        imgs = (imgs - 127.5) / 128.0
        with chainer.using_config('train', False), chainer.no_backprop_mode():

            heatmap = self(imgs)
            heatmap = heatmap.array
            heatmap[:-4] = _sigmoid(heatmap[:-4])

        bboxes, _, scores = heatmap_to_labeled_bboxes(heatmap,
                                                      self.score_threshold)
        bboxes, scores = bboxes[0], scores[0]

        hm_h, hm_w = heatmap.shape[2:4]

        bboxes[:, 0::2] *= img_w / hm_w
        bboxes[:, 1::2] *= img_h / hm_h

        keep = non_maximum_suppression(bboxes, nms_iou_threshold, score=scores)
        bboxes = bboxes[keep]
        scores = scores[keep]

        if self.xp != np:
            bboxes = cuda.to_cpu(bboxes)
            scores = cuda.to_cpu(scores)

        return bboxes, scores
