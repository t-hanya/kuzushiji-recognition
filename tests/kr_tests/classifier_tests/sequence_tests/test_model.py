"""
Unittest for kr.classifier.sequence.model
"""


import numpy as np

from kr.classifier.sequence.model import SequenceClassifier


class TestSequenceClassifier:

    def test(self):
        model = SequenceClassifier()
        images = [np.random.uniform(-1, 1, (5, 3, 64, 64)).astype(np.float32),
                  np.random.uniform(-1, 1, (3, 3, 64, 64)).astype(np.float32)]
        ret = model(images)
        assert type(ret) == list
        assert ret[0].shape == (5, 4787)
        assert ret[1].shape == (3, 4787)
