"""
Utility dataset wrapper to sample data from original dataset.
"""


import random

from chainer.dataset import DatasetMixin


class RandomSampler(DatasetMixin):
    """Dataset wrapper to sample data."""

    def __init__(self, dataset, virtual_size=10000):
        self.dataset = dataset
        self.virtual_size = virtual_size

    def __len__(self):
        return self.virtual_size

    def get_example(self, i):
        return random.choice(self.dataset)

