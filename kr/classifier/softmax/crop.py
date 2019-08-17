"""
Crop function.
"""


from typing import Tuple

import numpy as np
from PIL import Image


class CenterCropAndResize:
    """Center crop and resize."""

    def __init__(self,
                 scale: float = 1.2 / 1.4,
                 size: Tuple[int, int] = (112, 112)
                ) -> None:
        self.scale = scale
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size
        crop_w = int(img_w * self.scale)
        crop_h = int(img_h * self.scale)
        x = (img_w - crop_w) // 2
        y = (img_h - crop_h) // 2
        cropped = image.crop((x, y, x + crop_w, y + crop_h))
        resized = cropped.resize(self.size, resample=Image.BILINEAR)
        return resized


class RandomCropAndResize:
    """Random crop and resize."""

    def __init__(self,
                 scale_range: Tuple[float, float] = (1.1 / 1.4, 1.3 / 1.4),
                 size: Tuple[int, int] = (112, 112)
                ) -> None:

        assert 0.5 < scale_range[0] < scale_range[1] <= 1

        self.scale_range = scale_range
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size
        crop_w = int(img_w * np.random.uniform(*self.scale_range))
        crop_h = int(img_h * np.random.uniform(*self.scale_range))
        x = (img_w - crop_w) // 2
        y = (img_h - crop_h) // 2
        cropped = image.crop((x, y, x + crop_w, y + crop_h))
        resized = cropped.resize(self.size, resample=Image.BILINEAR)
        return resized
