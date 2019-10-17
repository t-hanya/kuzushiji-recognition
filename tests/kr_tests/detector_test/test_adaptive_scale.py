"""
Unittest for adaptive_scale module.
"""


import numpy as np
from PIL import Image

from kr.detector.adaptive_scale import ResizedImageWrapper
from kr.detector.adaptive_scale import AdaptiveScaleWrapper


class Detector:
    image_min_side = 832

    def detect(self, image):
        bboxes = np.array([[10, 10, 20, 20],
                           [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.7, 0.8], dtype=np.float32)
        return bboxes, scores


class TestResizedImageWrapper:

    def test(self):
        detector = ResizedImageWrapper(Detector())
        image = Image.new('RGB', (500, 300))
        bboxes, scores = detector.detect(image)

        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        assert scores.ndim == 1


class TestAdaptiveScaleWrapper:

    def test(self):
        detector = AdaptiveScaleWrapper(Detector())
        image = Image.new('RGB', (500, 300))
        bboxes, scores = detector.detect(image)

        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        assert scores.ndim == 1
