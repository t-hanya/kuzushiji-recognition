"""
Unittest for kr.detector.extensions module.
"""


import chainer
from chainer.iterators import SerialIterator
import numpy as np
from PIL import Image

from kr.detector.extensions import DetectionMapEvaluator


class _Model(chainer.Link):
    def detect(self, image):
        bboxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        return bboxes, scores


def _dataset():
    data = {
        'image': Image.new('RGB', (500, 300)),
        'bboxes': np.array([[10, 10, 20, 20]])
    }
    return [data, data, data, data]


class TestDetectionMapEvaluator:

    def test(self):
        model = _Model()
        dataset = _dataset()
        iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)
        evaluator = DetectionMapEvaluator(iterator, model)
        evaluator()
