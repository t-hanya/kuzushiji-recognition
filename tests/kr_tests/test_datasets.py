"""
Unittest for dataset loader.
"""


import numpy as np
from PIL import Image

import pytest

from kr.datasets import KuzushijiRecognitionDataset
from kr.datasets import KuzushijiUnicodeMapping
from kr.datasets import KuzushijiCharCropDataset
from kr.datasets import KuzushijiTestImages


class TestKuzushijiRecognitionDataset:

    @pytest.mark.parametrize('split, cv_index, expected_size', [
        (None, None, 3881),  # default -> trainval
        ('trainval', None, 3881),
        ('train', 0, None),
        ('val', 0, None),
        ('train', 1, None),
        ('val', 1, None),
    ])
    def test(self, split, cv_index, expected_size):
        dataset = KuzushijiRecognitionDataset(split=split, cv_index=cv_index)
        if expected_size is not None:
            assert len(dataset) == expected_size

        data = dataset[0]
        assert type(data) == dict
        assert isinstance(data['image'], Image.Image)
        assert type(data['bboxes']) == np.ndarray
        assert type(data['unicodes']) == list
        if data['unicodes']:
            assert type(data['unicodes'][0]) == str


class TestKuzushijiUnicodeMapping:

    def test(self):
        mapping = KuzushijiUnicodeMapping()

        assert mapping.unicode_to_char('U+0031') == '1'

        assert mapping.index_to_unicode(1) == 'U+0032'
        assert mapping.unicode_to_index('U+0032') == 1


class TestKuzushijiCharCropDataset:

    @pytest.mark.parametrize('split, expected_size', [
        (None, 683464),  # default -> trainval
        ('trainval', 683464),
        ('train', 648774),
        ('val', 34690),
    ])
    def test(self, split, expected_size):
        dataset = KuzushijiCharCropDataset(split=split)

        data = dataset[0]
        assert isinstance(data['image'], Image.Image)
        assert type(data['label']) == int
        assert len(dataset) == expected_size


class TestKuzushijiTestImages:

    def test(self):
        dataset = KuzushijiTestImages()
        assert len(dataset) == 4150

        data = dataset[0]
        assert isinstance(data['image'], Image.Image)
        assert data['image_id'] == 'test_00145af3'
