"""
Cutmix target generator.
"""


import random
from typing import Sequence
from typing import Tuple

from chainer.dataset import DatasetMixin
import numpy as np


def _sample_bbox(w: int, h: int, ratio: float) -> Tuple[int, int, int, int]:

    raw_x = np.random.uniform(0, w)
    raw_y = np.random.uniform(0, h)
    raw_w = w * np.sqrt(1 - ratio)
    raw_h = h * np.sqrt(1 - ratio)

    x1 = int(np.round(np.maximum(raw_x - raw_w / 2, 0)))
    y1 = int(np.round(np.maximum(raw_y - raw_h / 2, 0)))
    x2 = int(np.round(np.minimum(raw_x + raw_w / 2, w)))
    y2 = int(np.round(np.minimum(raw_y + raw_h / 2, h)))

    return x1, y1, x2, y2


class CutmixSoftLabelDataset(DatasetMixin):
    """CutMix soft label dataset."""

    def __init__(self, dataset: Sequence, n_class: int, prob: float = 1.0) -> None:
        self.dataset = dataset
        self.n_class = n_class
        self.prob = prob

    def __len__(self) -> int:
        return len(self.dataset)

    def get_example(self, i) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() <= self.prob:
            img1, label1 = self.dataset[i]
            img2, label2 = random.choice(self.dataset)

            h, w = img1.shape[1:]
            raw_ratio = np.random.uniform(0, 1)
            x1, y1, x2, y2 = _sample_bbox(w, h, raw_ratio)

            img = img1.copy()
            img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
            ratio = 1 - (x2 - x1) * (y2 - y1) / (w * h)
            label = np.zeros(self.n_class, dtype=np.float32)
            label[label1] += ratio
            label[label2] += 1 - ratio

            return img, label

        else:
            img1, label1 = self.dataset[i]
            label = np.zeros(self.n_class, dtype=np.float32)
            label[label1] = 1
            return img1, label
