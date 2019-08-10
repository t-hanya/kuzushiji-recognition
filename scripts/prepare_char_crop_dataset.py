"""
Prepare cropped character image dataset.
"""


import json
from pathlib import Path
import sys

import numpy as np

PRJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PRJ_ROOT))

from kr.datasets import KuzushijiRecognitionDataset


OUTPUT_ROOT = PRJ_ROOT / 'data'/ 'kuzushiji-recognition-char-crop'
PADDING_SCALE = 0.2


def calc_padded_bboxes(bboxes: np.ndarray) -> np.ndarray:
    centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
    longer_sides = (bboxes[:, 2:4] - bboxes[:, 0:2]).max(axis=1)
    padded_size = longer_sides * (1 + PADDING_SCALE * 2)

    padded_bboxes = np.empty_like(bboxes)
    padded_bboxes[:, 0:2] = np.floor(centers - padded_size[:, None] / 2.)
    padded_bboxes[:, 2:4] = np.ceil(centers + padded_size[:, None] / 2.)

    return padded_bboxes


def main() -> None:

    assert not OUTPUT_ROOT.exists(), 'Already converted.'
    OUTPUT_ROOT.mkdir()

    dataset = KuzushijiRecognitionDataset()

    idx = 0
    annotations = []
    for i, data in enumerate(dataset):
        print('progress: {}/{}'.format(i + 1, len(dataset)))

        image = data['image']
        bboxes = data['bboxes']
        unicodes = data['unicodes']
        padded_bboxes = calc_padded_bboxes(bboxes)

        for bbox, padded_bbox, unicode_ in zip(bboxes, padded_bboxes, unicodes):

            # define output directory and file name
            dirname = str(idx // 10000 * 10000).zfill(8)
            output_dir = OUTPUT_ROOT / 'images' / dirname
            output_dir.mkdir(parents=True, exist_ok=True)

            # save cropped image
            fname = str(idx).zfill(8) + '.png'
            cropped = image.crop(padded_bbox)
            cropped.save(output_dir / fname)

            # save annotation
            modified = bbox - np.tile(padded_bbox[0:2], 2)
            annotations.append({
                'image_path': str((output_dir / fname).relative_to(OUTPUT_ROOT)),
                'original_bbox': bbox.tolist(),
                'bbox': modified.tolist(),
                'unicode': unicode_
            })

            idx += 1

    annotation_path = OUTPUT_ROOT / 'annotations.json'
    with annotation_path.open('w') as f:
        json.dump({'annotations': annotations}, f, indent=2)


if __name__ == '__main__':
    main()
