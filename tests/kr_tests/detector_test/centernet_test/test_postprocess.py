"""
Unittest for postprocess module.
"""


import numpy as np

from kr.detector.centernet.postprocess import heatmap_to_labeled_bboxes


class TestHeatmapToLabeledBboxes:

    def test_no_objects(self):
        heatmap = np.zeros((1, 7, 80, 80), dtype=np.float32)
        bboxes, labels, scores = heatmap_to_labeled_bboxes(heatmap)
        assert type(bboxes) == list
        assert type(labels) == list
        assert type(scores) == list
        assert bboxes[0].shape == (0, 4)
        assert labels[0].shape == (0,)
        assert scores[0].shape == (0,)

    def test_normal_case(self):
        heatmap = np.zeros((1, 7, 80, 80), dtype=np.float32)

        # object1: label=0, w=10, h=20, offset_x=0, offset_y=0
        heatmap[0, 0, 5, 5] = 0.9
        heatmap[0, 3, 5, 5] = 10
        heatmap[0, 4, 5, 5] = 20
        heatmap[0, 5, 5, 5] = 0
        heatmap[0, 6, 5, 5] = 0

        # object2: label=2, w=20, h=10, offset_x=0.5, offset_y=0.5
        heatmap[0, 2, 10, 20] = 1.0
        heatmap[0, 3, 10, 20] = 20
        heatmap[0, 4, 10, 20] = 10
        heatmap[0, 5, 10, 20] = 0.5
        heatmap[0, 6, 10, 20] = 0.5

        bboxes, labels, scores = heatmap_to_labeled_bboxes(heatmap)
        assert type(bboxes) == list
        assert type(labels) == list
        assert bboxes[0].shape == (2, 4)
        assert labels[0].shape == (2,)

        assert np.isclose(bboxes[0], np.array([[5 - 5, 5 - 10,
                                                5 + 5, 5 + 10],
                                               [20.5 - 10, 10.5 - 5,
                                                20.5 + 10, 10.5 + 5]],
                                              dtype=np.float32)).all()
        assert (labels[0] == np.array([0, 2], dtype=np.int32)).all()
        assert (scores[0] == np.array([0.9, 1.0], dtype=np.float32)).all()
