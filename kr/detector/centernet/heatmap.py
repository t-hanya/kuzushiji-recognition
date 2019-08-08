"""
Generate training target heatmap.
"""


from typing import Tuple
import numpy as np


def gaussian_radius(bbox_size: Tuple[int, int], min_overlap=0.7):
    """Compute gaussian radius."""
    width, height = bbox_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return min(r1, r2, r3)


def generate_heatmap(bboxes: np.ndarray,
                     labels: np.ndarray,
                     num_classes: int,
                     heatmap_size: Tuple[int, int]  # (w, h)
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a training target heatmap from object data."""
    w, h = heatmap_size

    # prepare mesh center points
    x_arr = np.arange(w) + 0.5
    y_arr = np.arange(h) + 0.5
    xy_mesh = np.stack(np.meshgrid(x_arr, y_arr))  # [2, h, w]

    # initialize heatmap and object center index list
    heatmap = np.zeros((num_classes + 4, h, w), dtype=np.float32)
    center_indices = []

    for bbox, label in zip(bboxes, labels):

        # center coordinate
        center = (bbox[0:2] + bbox[2:4]) / 2.
        j, i = np.floor(center).astype(np.int)
        center_indices.append((i, j))

        # center heatmap
        size = bbox[2:4] - bbox[0:2]
        sigma = gaussian_radius(size)
        dist_squared = np.sum((xy_mesh - center[:, None, None]) ** 2, axis=0)
        gauss = np.exp(-1 * dist_squared / (2 * sigma ** 2))
        heatmap[label, :, :] = np.maximum(heatmap[label, :, :], gauss)

        # size
        heatmap[-4:-2, i, j] = size

        # offset
        heatmap[-2:, i, j] = center - np.floor(center)

    if len(center_indices):
        center_indices = np.array(center_indices, dtype=np.int32)
    else:
        center_indices = np.empty((0, 2), dtype=np.int32)

    return heatmap, center_indices
