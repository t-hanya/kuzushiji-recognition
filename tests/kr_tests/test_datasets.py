"""
Unittest for dataset loader.
"""


import numpy as np
from PIL import Image

from kr.datasets import KuzushijiRecognitionDataset
from kr.datasets import KuzushijiUnicodeMapping
from kr.datasets import KuzushijiCharCropDataset


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


class TestKuzushijiUnicodeMapping:

    def test(self):
        mapping = KuzushijiUnicodeMapping()

        assert mapping.unicode_to_char('U+0031') == '1'

        assert mapping.index_to_unicode(1) == 'U+0032'
        assert mapping.unicode_to_index('U+0032') == 1


class TestKuzushijiCharCropDataset:

    def test(self):
        dataset = KuzushijiCharCropDataset()

        data = dataset[0]
        assert isinstance(data['image'], Image.Image)
        assert type(data['label']) == int
        assert len(dataset) == 683464
