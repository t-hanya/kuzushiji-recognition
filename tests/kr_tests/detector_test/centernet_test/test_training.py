"""
Unittest for training module.
"""


import chainer
import chainer.links as L

import numpy as np

from kr.detector.centernet.training import TrainingModel


class SimpleModel(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(3, 7, ksize=3, stride=2, pad=1)
            self.l2 = L.Convolution2D(7, 7, ksize=3, stride=2, pad=1)

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return h


class TestTrainingModel:

    def test(self):
        image = np.random.uniform(-1, 1, (2, 3, 400, 800)).astype(np.float32)
        gt_heatmap = np.random.uniform(0, 1, (7, 100, 200)).astype(np.float32)
        gt_labels = np.array([1, 2], dtype=np.int32)
        gt_indices = np.array([[10, 20], [30, 40]], dtype=np.int32)

        model = SimpleModel()
        train_model = TrainingModel(model)

        loss = train_model(image,
                           [gt_heatmap, gt_heatmap],
                           [gt_labels, gt_labels],
                           [gt_indices, gt_indices])
        model.cleargrads()
        loss.backward()
