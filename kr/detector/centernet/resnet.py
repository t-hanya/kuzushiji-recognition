"""
Resnet-Unet
"""


from typing import Tuple

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
from chainer.backends import cuda
from chainercv.utils import non_maximum_suppression
from chainercv.links import SEBlock
import numpy as np
from PIL import Image

from .postprocess import heatmap_to_labeled_bboxes


class Block(chainer.Chain):
    """Basic block"""

    def __init__(self,
                 out_ch: int,
                 stride: int = 1) -> None:
        super().__init__()
        kw = {'initialW': normal.HeNormal(scale=1.0)}
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_ch, ksize=3, stride=stride,
                                         pad=1, nobias=True, **kw)
            self.bn1 = L.BatchNormalization(out_ch)
            self.conv2 = L.Convolution2D(out_ch, out_ch, ksize=3,
                                         pad=1, nobias=True, **kw)
            self.bn2 = L.BatchNormalization(out_ch)
            self.seblock = SEBlock(out_ch)
            if stride != 1:
                self.conv_skip = L.Convolution2D(None, out_ch, ksize=1,
                                                 stride=stride,
                                                 nobias=True, **kw)
                self.bn_skip = L.BatchNormalization(out_ch)

        self.stride = stride

    def forward(self, x: chainer.Variable) -> chainer.Variable:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.seblock(h)
        if self.stride != 1:
            x = self.bn_skip(self.conv_skip(x))

        return F.relu(h + x)


class SEResnet34(chainer.Chain):
    """SEResnet34 backbone CNN."""

    def __init__(self) -> None:
        super().__init__()

        kw = {'initialW': normal.HeNormal(scale=1.0)}
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=7,
                                         stride=2, pad=3, **kw)
            self.bn1 = L.BatchNormalization(64)
            self.block2_1 = Block(64, stride=2)
            self.block2_2 = Block(64)
            self.block2_3 = Block(64)

            self.block3_1 = Block(128, stride=2)
            self.block3_2 = Block(128)
            self.block3_3 = Block(128)
            self.block3_4 = Block(128)

            self.block4_1 = Block(256, stride=2)
            self.block4_2 = Block(256)
            self.block4_3 = Block(256)
            self.block4_4 = Block(256)
            self.block4_5 = Block(256)
            self.block4_6 = Block(256)

            self.block5_1 = Block(512, stride=2)
            self.block5_2 = Block(512)
            self.block5_3 = Block(512)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))

        h = self.block2_1(h)
        h = self.block2_2(h)
        e4 = self.block2_3(h)

        h = self.block3_1(e4)
        h = self.block3_2(h)
        h = self.block3_3(h)
        e8 = self.block3_4(h)

        h = self.block4_1(e8)
        h = self.block4_2(h)
        h = self.block4_3(h)
        h = self.block4_4(h)
        h = self.block4_5(h)
        e16 = self.block4_6(h)

        h = self.block5_1(e16)
        h = self.block5_2(h)
        e32 = self.block5_3(h)
        return e4, e8, e16, e32


class DeconvBlock(chainer.Chain):
    """Deconvolution block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Deconvolution2D(in_ch, in_ch // 2,
                                           ksize=4, stride=2, pad=1)
            self.bn1 = L.BatchNormalization(in_ch // 2)
            self.conv2 = L.Convolution2D(in_ch // 2, out_ch,
                                         ksize=3, stride=1, pad=1)
            self.bn2 = L.BatchNormalization(out_ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        return h


def _sigmoid(x):
    xp = cuda.get_array_module(x)
    return 1. / (1. + xp.exp(-x))


class SERes34UnetCenterNet(chainer.Chain):
    """CenterNet with SEResnet34-Unet backbone."""

    stride: int = 4
    image_min_side: int = 832

    def __init__(self, n_fg_class: int = 1) -> None:
        super().__init__()
        out_ch = n_fg_class + 4

        with self.init_scope():
            self.enc = SEResnet34()
            self.dc1 = DeconvBlock(512, 256)
            self.dc2 = DeconvBlock(512, 128)
            self.dc3 = DeconvBlock(256, 64)
            self.dc4 = L.Convolution2D(128, out_ch, ksize=3, stride=1, pad=1)

    def forward(self, x):
        e4, e8, e16, e32 = self.enc(x)
        h = self.dc1(e32)
        h = self.dc2(F.concat([e16, h]))
        h = self.dc3(F.concat([e8, h]))
        h = self.dc4(F.concat([e4, h]))

        # force 0-1 value range to scores and offsets
        C = h.shape[1]
        scores, sizes, offsets = F.split_axis(h, (C - 4, C - 2), axis=1)
        offsets = F.sigmoid(offsets)

        return F.concat([scores, sizes, offsets])

    def detect(self,
               image: Image.Image,
               score_threshold: float = 0.5,
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
                                                      score_threshold)
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
