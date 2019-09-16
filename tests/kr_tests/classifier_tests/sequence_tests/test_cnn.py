"""
Unittest for kr.classifier.sequence.cnn
"""


import numpy as np

from kr.classifier.sequence.cnn import Resnet18


class TestResnet18:

    def test(self):
        model = Resnet18(100)
        imgs = np.random.uniform(-1, 1, (5, 3, 64, 64)).astype(np.float32)
        ret = model(imgs)
        assert ret.shape == (5, 100)
