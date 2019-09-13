"""
MobileNetV3
"""


from typing import Tuple

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
import numpy as np
from PIL import Image


def relu6(x):
    """ReLU 6 activation function."""
    return F.clipped_relu(x, 6.)


def hard_sigmoid(x):
    """Hard version of sigmoid function."""
    return relu6(x + 3.) / 6.


def hard_swish(x):
    """Hard version of swish function."""
    return x * relu6(x + 3.) / 6.


class ConvBnActiv(chainer.Chain):
    """Conv-BN-Activation block."""

    def __init__(self, in_ch, out_ch, ksize, stride=1, activ=F.relu):
        assert ksize in (1, 3)
        pad = (ksize - 1) // 2
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_ch, out_ch, ksize,
                                        stride=stride, pad=pad)
            self.bn = L.BatchNormalization(out_ch)
        self.activ = activ

    def forward(self, x):
        h = self.activ(self.bn(self.conv(x)))
        return h


class SEModule(chainer.Chain):
    """Squeeze-and-Excitation module."""

    def __init__(self, ch):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(ch, ch // 4)
            self.fc2 = L.Linear(ch // 4, ch)

    def forward(self, x):
        N, C, H, W = x.shape
        h = F.average_pooling_2d(x, (H, W)).reshape(N, C)
        h = F.relu(self.fc1(h))
        h = hard_sigmoid(self.fc2(h))
        h = F.transpose(F.broadcast_to(h, (H, W, N, C)), (2, 3, 0, 1))
        return x * h


class Bneck(chainer.Chain):
    """Bottleneck module."""

    def __init__(self, in_ch, exp_ch, out_ch, ksize,
                 stride=1, use_se=False, activ=F.relu):
        assert ksize in (3, 5)
        pad = (ksize - 1) // 2
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_ch, exp_ch, 1, nobias=True)
            self.bn1 = L.BatchNormalization(exp_ch)
            self.conv2 = L.Convolution2D(exp_ch, exp_ch, ksize, stride, pad,
                                         groups=exp_ch, nobias=True)
            self.bn2 = L.BatchNormalization(exp_ch)
            self.conv3 = L.Convolution2D(exp_ch, out_ch, 1, nobias=True)
            self.bn3 = L.BatchNormalization(out_ch)
            self.se = SEModule(exp_ch) if use_se else None

        self.activ = activ
        self.skip = in_ch == out_ch and stride == 1

    def forward(self, x):
        h = self.activ(self.bn1(self.conv1(x)))
        h = self.activ(self.bn2(self.conv2(h)))
        if self.se:
            h = self.se(h)
        h = self.activ(self.bn3(self.conv3(h)))
        if self.skip:
            h += x
        return h


class MobileNetV3(chainer.Chain):
    """MobileNetV3-Large model."""

    input_size = (112, 112)

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        hs = {'activ': hard_swish}
        with self.init_scope():
            self.conv1 = ConvBnActiv(3, 16, ksize=3, **hs, stride=2)
            self.bneck2 = Bneck(16, 16, 16, ksize=3)
            self.bneck3 = Bneck(16, 64, 24, ksize=3, stride=2)
            self.bneck4 = Bneck(24, 72, 24, ksize=3)
            self.bneck5 = Bneck(24, 72, 40, ksize=5, use_se=True, stride=2)
            self.bneck6 = Bneck(40, 120, 40, ksize=5, use_se=True)
            self.bneck7 = Bneck(40, 120, 40, ksize=5, use_se=True)
            self.bneck8 = Bneck(40, 240, 80, ksize=3, **hs, stride=2)
            self.bneck9 = Bneck(80, 200, 80, ksize=3, **hs)
            self.bneck10 = Bneck(80, 184, 80, ksize=3, **hs)
            self.bneck11 = Bneck(80, 184, 80, ksize=3, **hs)
            self.bneck12 = Bneck(80, 480, 112, ksize=3, use_se=True, **hs)
            self.bneck13 = Bneck(112, 672, 112, ksize=3, use_se=True, **hs)
            self.bneck14 = Bneck(112, 672, 160, ksize=5, use_se=True, **hs, stride=2)
            self.bneck15 = Bneck(160, 960, 160, ksize=5, use_se=True, **hs)
            self.bneck16 = Bneck(160, 960, 160, ksize=5, use_se=True, **hs)
            self.conv17 = ConvBnActiv(160, 960, ksize=1, **hs)
            self.fc18 = L.Linear(960, 1280)
            self.fc19 = L.Linear(1280, out_ch)

    def forward(self, x) -> chainer.Chain:
        h = self.conv1(x)
        for i in range(2, 16 + 1):
            h = getattr(self, f'bneck{i}')(h)
        h = self.conv17(h)
        h = F.average_pooling_2d(h, h.shape[2:4])
        h = hard_swish(self.fc18(h.reshape(h.shape[0], -1)))
        h = F.dropout(h, ratio=0.2)
        h = self.fc19(h)
        return h

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
