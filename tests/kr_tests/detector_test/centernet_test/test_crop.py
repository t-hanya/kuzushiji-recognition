"""
Unittest for crop module.
"""


import numpy as np
from PIL import Image

from kr.detector.centernet.crop import RandomCropAndResize
from kr.detector.centernet.crop import CenterCropAndResize


class TestRandomCropAndResize:

    def test(self):
        image = Image.new('RGB', (400, 300))
        bboxes = np.array([
            [10, 10, 20, 20],
            [20, 20, 30, 30],
            [30, 30, 40, 40]
        ])
        unicodes = ['A', 'B', 'C']

        crop_func = RandomCropAndResize((0.4, 0.6), (200, 100))
        ret = crop_func(image, bboxes, unicodes)

        assert isinstance(ret[0], Image.Image)
        assert ret[0].size == (200, 100)

        assert type(ret[1]) == np.ndarray
        assert type(ret[2]) == list
        assert len(ret[1]) == len(ret[2])


class TestCenterCropAndResize:

    def test(self):
        image = Image.new('RGB', (400, 300))
        bboxes = np.array([
            [10, 10, 20, 20],  # out of crop bounds
            [20, 20, 30, 30],  # out of crop bounds
            [200, 200, 210, 210]
        ])
        unicodes = ['A', 'B', 'C']

        crop_func = CenterCropAndResize(0.5, (200, 100))
        ret = crop_func(image, bboxes, unicodes)

        assert isinstance(ret[0], Image.Image)
        assert ret[0].size == (200, 100)

        assert type(ret[1]) == np.ndarray
        assert type(ret[2]) == list
        assert len(ret[1]) == 1
        assert ret[2] == ['C']
