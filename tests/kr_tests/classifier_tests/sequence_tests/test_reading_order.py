"""
Unittest for reading_order module.
"""


import numpy as np

from kr.classifier.sequence.reading_order import predict_reading_order


class TestPredictReadingOrder:

    def test_empty(self):
        bboxes = np.empty((0, 4), dtype=np.float32)
        sequences = predict_reading_order(bboxes)
        assert sequences == []

    def test_normal(self):
        bboxes = np.array([[0, 0, 10, 10],
                           [0, 10, 10, 20],
                           [0, 20, 10, 30]])
        sequences = predict_reading_order(bboxes)
        assert sequences == [[0, 1, 2]]
