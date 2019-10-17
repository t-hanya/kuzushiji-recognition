"""
Unittest for kr.classification.softmax.model module.
"""


import chainer.functions as F
import numpy as np
from PIL import Image

from kr.classifier.softmax.model import SoftmaxClassifierBase


class DummyModel(SoftmaxClassifierBase):
    input_size = (32, 32)

    def forward(self, x):
        h = F.average_pooling_2d(x, x.shape[2:4]).reshape(len(x), -1)
        return h


class TestSoftmaxClassifierBase:

    def test_classify(self):
        model = DummyModel()
        img = Image.new('RGB', (500, 300))
        bboxes = np.array([[10, 10, 20, 20],
                           [30, 30, 50, 50]])

        labels, scores = model.classify(img, bboxes)

        assert labels.shape == (2,)
        assert scores.shape == (2,)
