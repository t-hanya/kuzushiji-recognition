"""
Utility dataset wrapper to sample data from original dataset.
"""


from collections import defaultdict
import random

from chainer.dataset import DatasetMixin
import numpy as np


class RandomSampler(DatasetMixin):
    """Dataset wrapper to sample data."""

    def __init__(self, dataset, virtual_size=10000):
        self.dataset = dataset
        self.virtual_size = virtual_size

    def __len__(self):
        return self.virtual_size

    def get_example(self, i):
        return random.choice(self.dataset)


class OverSampler(DatasetMixin):
    """Dataset wrapper to enagle oversampling."""

    def __init__(self, dataset, min_samples=5, virtual_size=10000):
        assert hasattr(dataset, 'all_labels')
        self.dataset = dataset
        self.virtual_size = virtual_size

        samples_per_class = defaultdict(int)
        self.class_sample_mapping = defaultdict(list)
        for i, label in enumerate(dataset.all_labels):
            samples_per_class[label] += 1
            self.class_sample_mapping[label].append(i)

        self.labels = sorted(samples_per_class.keys())
        num_samples = np.array(
            [samples_per_class[k] for k in self.labels], dtype=np.float32)
        num_samples_mod = np.maximum(num_samples, min_samples)
        self.class_probs = num_samples_mod / num_samples_mod.sum()

    def __len__(self):
        return self.virtual_size

    def get_example(self, _):
        label = random.choices(self.labels, weights=self.class_probs)[0]
        index = random.choice(self.class_sample_mapping[label])
        return self.dataset[index]
