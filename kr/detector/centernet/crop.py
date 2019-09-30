"""
Crop image
"""

from typing import List
from typing import Tuple

import numpy as np
from PIL import Image


class RandomCropAndResize:
    """Random crop and resize."""

    def __init__(self,
                 scale_range: Tuple[int, int] = (0.4, 0.6),
                 size: Tuple[int, int] = (416, 416)) -> None:

        assert 0 < scale_range[0] < scale_range[1] <= 1

        self.scale_range = scale_range
        self.size = size

    def __call__(self,
                 image: Image.Image,
                 bboxes: np.ndarray,
                 unicodes: List[str],
                ) -> Tuple[Image.Image, np.ndarray, List[str]]:

        org_w, org_h = image.size

        crop_size = int(np.random.uniform(*self.scale_range) * min(org_w, org_h))
        crop_x = np.random.randint(0, org_w - crop_size + 1)
        crop_y = np.random.randint(0, org_h - crop_size + 1)
        cropped = image.crop((crop_x, crop_y,
                              crop_x + crop_size, crop_y + crop_size))

        bboxes = bboxes.copy()
        bboxes[:, 0::2] -= crop_x
        bboxes[:, 1::2] -= crop_y
        centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.

        mask = np.logical_and.reduce((
            0 < centers[:, 0],
            0 < centers[:, 1],
            centers[:, 0] < crop_size,
            centers[:, 1] < crop_size
        ))
        bboxes = bboxes[mask]
        unicodes = [u for u, m in zip(unicodes, mask) if m]

        # resize
        resized = cropped.resize(self.size, resample=Image.BILINEAR)
        bboxes[:, 0::2] = bboxes[:, 0::2] * (self.size[0] / crop_size)
        bboxes[:, 1::2] = bboxes[:, 1::2] * (self.size[1] / crop_size)

        return resized, bboxes, unicodes


class CenterCropAndResize:
    """Center crop and resize."""

    def __init__(self,
                 scale: float = 0.5,
                 size: Tuple[int, int] = (416, 416)) -> None:

        assert 0 < scale < 1
        self.scale = scale
        self.size = size

    def __call__(self,
                 image: Image.Image,
                 bboxes: np.ndarray,
                 unicodes: List[str],
                ) -> Tuple[Image.Image, np.ndarray, List[str]]:

        org_w, org_h = image.size

        # crop
        crop_size = int(self.scale * min(org_w, org_h))
        crop_x = (org_w - crop_size) // 2
        crop_y = (org_h - crop_size) // 2
        cropped = image.crop((crop_x, crop_y,
                              crop_x + crop_size, crop_y + crop_size))

        bboxes = bboxes.copy()
        bboxes[:, 0::2] -= crop_x
        bboxes[:, 1::2] -= crop_y
        centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.

        mask = np.logical_and.reduce((
            0 < centers[:, 0],
            0 < centers[:, 1],
            centers[:, 0] < crop_size,
            centers[:, 1] < crop_size
        ))
        bboxes = bboxes[mask]
        unicodes = [u for u, m in zip(unicodes, mask) if m]

        # resize
        resized = cropped.resize(self.size, resample=Image.BILINEAR)
        bboxes[:, 0::2] = bboxes[:, 0::2] * (self.size[0] / crop_size)
        bboxes[:, 1::2] = bboxes[:, 1::2] * (self.size[1] / crop_size)

        return resized, bboxes, unicodes
