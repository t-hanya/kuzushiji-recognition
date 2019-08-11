"""
Unittest for dataset loader.
"""


import numpy as np
from PIL import Image

import pytest

from kr.datasets import KuzushijiRecognitionDataset
from kr.datasets import KuzushijiUnicodeMapping
from kr.datasets import KuzushijiCharCropDataset


class TestKuzushijiRecognitionDataset:

    @pytest.mark.parametrize('split, expected_size', [
        (None, 3881),  # default -> trainval
        ('trainval', 3881),
        ('train', 3686),
        ('val', 195),
    ])
    def test(self, split, expected_size):
        dataset = KuzushijiRecognitionDataset(split=split)
        assert len(dataset) == expected_size

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
