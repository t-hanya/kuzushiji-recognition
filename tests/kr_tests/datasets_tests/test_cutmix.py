"""
Unittest for cutmix module.
"""


import numpy as np

from kr.datasets import CutmixSoftLabelDataset


class TestCutmixSoftlabelDataset:

    def test(self):
        dataset = [(np.zeros((3, 64, 64), dtype=np.float32),
                    np.array(3, dtype=np.int32))]
        cutmix_dataset = CutmixSoftLabelDataset(dataset, n_class=10)
        img, label = cutmix_dataset[0]

        assert len(cutmix_dataset) == len(dataset)
        assert img.shape == (3, 64, 64)
        assert img.dtype == np.float32
        assert np.isclose(img, 0).all()

        assert label.shape == (10,)
        assert np.isclose(
            label,
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])).all()
        assert label.dtype == np.float32
