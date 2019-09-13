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


OUTPUT_ROOT = PRJ_ROOT / 'data'/ 'kuzushiji-recognition-gsplit'
PADDING_SCALE = 0.2
N_CV_SPLITS = 4


def calc_padded_bboxes(bboxes: np.ndarray) -> np.ndarray:
    centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
    longer_sides = (bboxes[:, 2:4] - bboxes[:, 0:2]).max(axis=1)
    padded_size = longer_sides * (1 + PADDING_SCALE * 2)

    padded_bboxes = np.empty_like(bboxes)
    padded_bboxes[:, 0:2] = np.floor(centers - padded_size[:, None] / 2.)
    padded_bboxes[:, 2:4] = np.ceil(centers + padded_size[:, None] / 2.)

    return padded_bboxes


def extract_annotations(dataset: KuzushijiRecognitionDataset,
                        image_path_mapping: dict
                       ) -> list:
    annotations = []
    for data in dataset:
        image_id = data['image_id']
        bboxes = data['bboxes']
        unicodes = data['unicodes']
        padded_bboxes = calc_padded_bboxes(bboxes)

        for bbox, padded_bbox, unicode in zip(bboxes, padded_bboxes, unicodes):

            key = (image_id, *bbox.tolist())
            cropped_image_path = image_path_mapping[key]

            modified = bbox - np.tile(padded_bbox[0:2], 2)
            annotations.append({
                'image_path': str(cropped_image_path),
                'original_bbox': bbox.tolist(),
                'bbox': modified.tolist(),
                'unicode': unicode
            })
    return annotations


def main() -> None:

    assert OUTPUT_ROOT.exists(), "Run 'prepare_train_val_split.py first.'"

    idx = 0
    image_path_mapping = {}  # (image_id, x1, y1, x2, y2) -> Path

    # crop all characters
    dataset = KuzushijiRecognitionDataset('trainval')
    for i, data in enumerate(dataset):
        print('[{}/{}] {}'.format(i + 1, len(dataset), data['image_id']))

        image_id = data['image_id']
        image = data['image']
        bboxes = data['bboxes']
        unicodes = data['unicodes']
        padded_bboxes = calc_padded_bboxes(bboxes)

        for bbox, padded_bbox, unicode in zip(bboxes, padded_bboxes, unicodes):

            # define output directory and file name
            dirname = str(idx // 10000 * 10000).zfill(8)
            output_dir = OUTPUT_ROOT / 'char_images' / dirname
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = str(idx).zfill(8) + '.png'
            fpath = output_dir / fname

            # save cropped image
            cropped = image.crop(padded_bbox)
            cropped.save(fpath)

            # keep cropped image file path
            key = (image_id, *bbox.tolist())
            image_path_mapping[key] = fpath.relative_to(OUTPUT_ROOT)

            idx += 1

    # extract and save annotation data
    annotation_path = OUTPUT_ROOT / 'char_images_trainval.json'
    with annotation_path.open('w') as f:
        annotations = extract_annotations(dataset, image_path_mapping)
        json.dump({'annotations': annotations}, f, indent=2)

    # extract and save annotation data for each cross validation index and split
    for cv_index in range(N_CV_SPLITS):
        for split in ('train', 'val'):
            annotation_path = OUTPUT_ROOT / f'char_images_{split}-{cv_index}.json'
            with annotation_path.open('w') as f:
                annotations = extract_annotations(
                    KuzushijiRecognitionDataset(split=split, cv_index=cv_index),
                    image_path_mapping
                )
                json.dump({'annotations': annotations}, f, indent=2)


if __name__ == '__main__':
    main()

