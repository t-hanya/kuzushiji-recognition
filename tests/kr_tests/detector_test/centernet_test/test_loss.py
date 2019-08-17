"""
Unittest for loss module.
"""


import numpy as np

from kr.detector.centernet.loss import calc_loss


class TestCalcLoss:

    def test_with_objects(self):
        heatmap = np.random.uniform(0, 1, (7, 100, 200))
        gt_heatmap = np.random.uniform(0, 1, (7, 100, 200))
        gt_labels = np.array([1, 2])
        gt_indices = np.array([[10, 20], [30, 40]])

        (score_loss,
         size_loss,
         offset_loss) = calc_loss(heatmap, gt_heatmap, gt_labels, gt_indices)

        score_loss.backward()  # test: do not raise an error
        size_loss.backward()  # test: do not raise an error
        offset_loss.backward()  # test: do not raise an error

    def test_no_object(self):
        heatmap = np.random.uniform(0, 1, (7, 100, 200))
        gt_heatmap = np.zeros((7, 100, 200))
        gt_labels = np.empty((0,), dtype=np.int)
        gt_indices = np.empty((0, 2), dtype=np.int)

        (score_loss,
         size_loss,
         offset_loss) = calc_loss(heatmap, gt_heatmap, gt_labels, gt_indices)

        score_loss.backward()  # test: do not raise an error
        size_loss.backward()  # test: do not raise an error
        offset_loss.backward()  # test: do not raise an error
