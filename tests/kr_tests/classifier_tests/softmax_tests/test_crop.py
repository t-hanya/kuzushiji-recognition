"""
Unittest for crop module.
"""


from PIL import Image

from kr.classifier.softmax.crop import CenterCropAndResize
from kr.classifier.softmax.crop import RandomCropAndResize


class TestCenterCropAndResize:

    def test(self):
        image = Image.new('RGB', (300, 200))
        crop_func = CenterCropAndResize(scale=0.8, size=(48, 32))
        ret = crop_func(image)
        assert isinstance(ret, Image.Image)
        assert ret.size == (48, 32)


class TestRandomCropAndResize:

    def test(self):
        image = Image.new('RGB', (300, 200))
        crop_func = RandomCropAndResize(scale_range=(0.6, 0.9), size=(48, 32))
        ret = crop_func(image)
        assert isinstance(ret, Image.Image)
        assert ret.size == (48, 32)
