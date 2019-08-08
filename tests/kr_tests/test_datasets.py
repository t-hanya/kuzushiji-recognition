"""
Unittest for dataset loader.
"""


import numpy as np
from PIL import Image

from kr.datasets import KuzushijiRecognitionDataset


class TestKuzushijiRecognitionDataset:

    def test(self):
        dataset = KuzushijiRecognitionDataset()
        assert len(dataset) == 3881

        data = dataset[0]
        assert type(data) == dict
        assert isinstance(data['image'], Image.Image)
        assert type(data['bboxes']) == np.ndarray
        assert type(data['unicodes']) == list
        assert type(data['unicodes'][0]) == str
