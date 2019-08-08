"""
Unittest for heatmap module.
"""


import numpy as np

from kr.detector.centernet.heatmap import generate_heatmap


class TestGenerateHeatmap:

    def test_no_object(self):
        bboxes = np.empty((0, 4), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int32)
        heatmap, center_indices = generate_heatmap(
            bboxes, labels, num_classes=3, heatmap_size=(200, 100))
        assert heatmap.shape == (7, 100, 200)  # 7 = 3 + 2 + 2
        assert np.isclose(heatmap[0:3], 0).all()
        assert len(center_indices) == 0

    def test_with_objects(self):
        bboxes = np.array([[10, 10, 15, 15],
                           [20, 40, 30, 50]], dtype=np.float)
        labels = np.array([1, 2], dtype=np.int)
        heatmap, center_indices = generate_heatmap(
            bboxes, labels, num_classes=3, heatmap_size=(200, 100))

        assert heatmap.shape == (7, 100, 200)
        assert center_indices.shape == (2, 2)

        # center heatmap
        assert np.isclose(heatmap[0], 0).all()  # heatmap for class 0
        i_ids = center_indices[:, 0]
        j_ids = center_indices[:, 1]
        assert (heatmap[labels, i_ids, j_ids] > 0.7).all()

        # bbox size
        assert np.isclose(heatmap[-4:-2, i_ids, j_ids], np.array([[5, 10],
                                                                  [5, 10]])).all()

        # center offset
        assert np.isclose(heatmap[-2:, i_ids, j_ids], np.array([[0.5, 0],
                                                                [0.5, 0]])).all()

        # center_indices
        assert (i_ids == np.array([12, 45])).all()
        assert (j_ids == np.array([12, 25])).all()
