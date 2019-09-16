"""
Prepare character sequence dataset.
"""


import json
from pathlib import Path
import pickle
import sys

import numpy as np

PRJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PRJ_ROOT))

from kr.datasets import KuzushijiRecognitionDataset
from kr.classifier.sequence.reading_order import predict_reading_order


OUTPUT_DIR = PRJ_ROOT / 'data'/ 'kuzushiji-recognition-seq'


def _calc_padded_bboxes(bboxes: np.ndarray,
                        pad_scale: float = 0.2) -> np.ndarray:
    """Calculate padded bounding boxes."""

    centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
    longer_sides = (bboxes[:, 2:4] - bboxes[:, 0:2]).max(axis=1)
    padded_size = longer_sides * (1 + pad_scale * 2)

    padded_bboxes = np.empty_like(bboxes)
    padded_bboxes[:, 0:2] = np.floor(centers - padded_size[:, None] / 2.)
    padded_bboxes[:, 2:4] = np.ceil(centers + padded_size[:, None] / 2.)

    return padded_bboxes


def main():
    sequences_dir = OUTPUT_DIR / 'sequences'
    sequences_dir.mkdir(parents=True, exist_ok=True)

    dataset = KuzushijiRecognitionDataset()
    image_id_to_sequences = {}  # str -> list

    for di, data in enumerate(dataset):
        if di % 100 == 0:
            print('[{}/{}] {}'.format(di + 1, len(dataset), data['image_id']))

        sequences = predict_reading_order(data['bboxes'])
        image_id_to_sequences[data['image_id']] = []

        for si, seq in enumerate(sequences):
            bboxes = np.array([data['bboxes'][i] for i in seq])
            padded_bboxes = _calc_padded_bboxes(bboxes)
            images = [data['image'].crop(bb) for bb in padded_bboxes]
            unicodes = [data['unicodes'][i] for i in seq]

            fn = '{}_{}.pkl'.format(data['image_id'], si)
            fp = sequences_dir / fn
            with fp.open('wb') as f:
                pickle.dump({'images': images, 'unicodes': unicodes}, f)

            image_id_to_sequences[data['image_id']].append(
                str(fp.relative_to(OUTPUT_DIR)))

    splits = [('trainval', None)]
    for split in ('train', 'val'):
        for cv_index in range(4):
            splits.append((split, cv_index))

    for split, cv_index in splits:
        dataset = KuzushijiRecognitionDataset(split=split, cv_index=cv_index)
        annotations = []
        for data in dataset:
            for path in image_id_to_sequences[data['image_id']]:
                annotations.append({'image_id': data['image_id'],
                                    'data_path': path})

        if cv_index is None:
            fn = '{}.json'.format(split)
        else:
            fn = '{}-{}.json'.format(split, cv_index)
        annt_path = OUTPUT_DIR / fn
        with annt_path.open('w') as f:
            json.dump({'annotations': annotations}, f)


if __name__ == '__main__':
    main()
