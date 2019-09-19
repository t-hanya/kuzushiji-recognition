"""
Unittest for kr.datasets.sampling module.
"""


from chainer.dataset import DatasetMixin
import numpy as np

from kr.datasets import RandomSampler
from kr.datasets import OverSampler


class TestRandomSampler:

    def test(self):
        dataset = RandomSampler([1, 2, 3, 4, 5], virtual_size=10)
        assert len(dataset) == 10
        for _ in range(10):
            assert 1 <= dataset[0] <= 5


class DummyDataset(DatasetMixin):

    def __init__(self):
        self.all_labels = np.array([0, 0, 0, 0, 0, 1])

    def __len__(self):
        return len(self.all_labels)

    def get_example(self, i):
        return self.all_labels[i]


class TestOverSampler:

    def test(self):
        dataset = OverSampler(DummyDataset(), min_samples=5)
        expected = np.array([0.5, 0.5], dtype=np.float32)
        assert np.isclose(dataset.class_probs, expected).all()
        for _ in range(10):
            assert dataset[0] in (0, 1)
