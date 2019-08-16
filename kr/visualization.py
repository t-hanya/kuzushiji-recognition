"""
Visualization
"""


from typing import List
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image

from kr.datasets import KuzushijiUnicodeMapping


def visualize_labeled_image(image: Image.Image,
                            bboxes: np.ndarray,
                            unicodes: List[str],
                            fig_size: int = 15,
                            display: bool = True,
                            save_path: Optional[str] = None
                           ) -> None:
    """Visualize annotation data of Kuzushiji Recognition."""
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    mapping = KuzushijiUnicodeMapping()

    ax.imshow(image)
    for bbox, unicode in zip(bboxes, unicodes):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        char = mapping.unicode_to_char(unicode)
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fc='none', ec='blue', linewidth=1))
        ax.text(x2, (y1 + y2) / 2, char, color='red', va='center', fontsize=14)
    if display:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_matching_result(image: Image.Image,
                              tp_list: List[dict],
                              fp_list: List[dict],
                              fn_list: List[dict],
                              fig_size: int = 15,
                              display: bool = True,
                              save_path: Optional[str] = None
                             ) -> None:
    """Visualize matching result."""
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    mapping = KuzushijiUnicodeMapping()
    ax.imshow(image)

    for tp in tp_list:
        x1, y1, x2, y2 = [tp['gt_bbox'][k] for k in ('x1', 'y1', 'x2', 'y2')]
        w, h = x2 - x1, y2 - y1
        char = mapping.unicode_to_char(tp['gt_unicode'])
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fc='none', ec='blue', linewidth=1))
        ax.text(x2, (y1 + y2) / 2, char, color='blue', va='center', fontsize=14)

    for fp in fp_list:
        x, y = [fp['pred_point'][k] for k in ('x', 'y')]
        char = mapping.unicode_to_char(fp['pred_unicode'])
        ax.add_patch(plt.Circle((x, y), radius=10, fc='white', ec='none'))
        ax.text(x - 20, y, char, color='white', ha='right', va='center', fontsize=14)

    for fn in fn_list:
        x1, y1, x2, y2 = [fn['gt_bbox'][k] for k in ('x1', 'y1', 'x2', 'y2')]
        w, h = x2 - x1, y2 - y1
        char = mapping.unicode_to_char(fn['gt_unicode'])
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fc='none', ec='red', linewidth=1))
        ax.text(x2, (y1 + y2) / 2, char, color='red', va='center', fontsize=14)

    if display:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
