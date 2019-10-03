"""
Unittest for kr.detector.bbox_voting module.
"""


import numpy as np
from PIL import Image

from kr.detector.bbox_voting import BboxVoting


class DetectorA:
    def detect(self, image):
        bboxes = np.array([[10, 10, 20, 20],
                           [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.7, 0.8], dtype=np.float32)
        return bboxes, scores


class DetectorB:
    def detect(self, image):
        bboxes = np.array([[21, 21, 30, 30],
                           [30, 30, 40, 40]], dtype=np.float32)
        scores = np.array([0.8, 0.9], dtype=np.float32)
        return bboxes, scores


class TestBboxVoting:
    def test(self):
        image = Image.new('RGB', (300, 300))
        expected_bboxes = np.array([[10, 10, 20, 20],
                                    [20.5, 20.5, 30, 30],
                                    [30, 30, 40, 40]], dtype=np.float32)
        expected_scores = np.array([0.7, 0.8, 0.9], dtype=np.float32)

        detector = BboxVoting([
            DetectorA(),
            DetectorB()
        ])
        bboxes, scores = detector.detect(image)
        order = np.argsort(bboxes[:, 0])  # sort by x1 value
        bboxes = bboxes[order]
        scores = scores[order]

        assert np.isclose(bboxes, expected_bboxes).all()
        assert np.isclose(scores, expected_scores).all()

    def test_min_votes(self):
        image = Image.new('RGB', (300, 300))
        expected_bboxes = np.array([[20.5, 20.5, 30, 30]], dtype=np.float32)
        expected_scores = np.array([0.8], dtype=np.float32)

        detector = BboxVoting([
            DetectorA(),
            DetectorB()
        ], min_votes=2)
        bboxes, scores = detector.detect(image)

        assert np.isclose(bboxes, expected_bboxes).all()
        assert np.isclose(scores, expected_scores).all()
