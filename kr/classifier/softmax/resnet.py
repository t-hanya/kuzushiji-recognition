"""
Resnet
"""


import chainer
import chainer.functions as F
import chainer.links as L

from .model import SoftmaxClassifierBase


class Block(chainer.Chain):
    """Basic block"""

    def __init__(self,
                 out_ch: int,
                 stride: int = 1) -> None:
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_ch, ksize=3, stride=stride,
                                         pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(out_ch)
            self.conv2 = L.Convolution2D(out_ch, out_ch, ksize=3,
                                         pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(out_ch)
            if stride != 1:
                self.conv_skip = L.Convolution2D(None, out_ch, ksize=1,
                                                 stride=stride,
                                                 nobias=True)
                self.bn_skip = L.BatchNormalization(out_ch)

        self.stride = stride

    def forward(self, x: chainer.Variable) -> chainer.Variable:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.stride != 1:
            x = self.bn_skip(self.conv_skip(x))

        return F.relu(h + x)


class Resnet18(SoftmaxClassifierBase):
    """Resnet18 based CNN."""
    input_size = (64, 64)

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=5, stride=2, pad=2)
            self.bn1 = L.BatchNormalization(64)
            self.block2 = Block(64, stride=2)
            self.block3 = Block(64)
            self.block4 = Block(128, stride=2)
            self.block5 = Block(128)
            self.block6 = Block(256, stride=2)
            self.block7 = Block(256)
            self.block8 = Block(512, stride=2)
            self.block9 = Block(512)
            self.fc10 = L.Linear(512, out_ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = self.block9(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.fc10(h.reshape(len(h), -1))
        return h
