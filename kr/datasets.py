"""
Kuzushiji dataset
"""


from pathlib import Path

from chainer.dataset import DatasetMixin
import numpy as np
import pandas as pd
from PIL import Image


_prj_root = Path(__file__).resolve().parent.parent
_dataset_dir = _prj_root / 'data' / 'kuzushiji-recognition'


class KuzushijiRecognitionDataset(DatasetMixin):
    """Kaggle Kuzushiji Recognition training dataset."""

    def __init__(self) -> None:
        assert (_dataset_dir.exists(),
                ('Download Kaggle Kuzushiji Recognition dataset '
                 'and move files to <prj>/data/kuzushiji-recognition/'))

        self.table = pd.read_csv(_dataset_dir / 'train.csv')
        self.image_dir = _dataset_dir / 'train_images'

    def __len__(self) -> int:
        return len(self.table)

    def get_example(self, i: int) -> dict:
        row = self.table.iloc[i]

        image = Image.open(self.image_dir / (row.image_id + '.jpg'))

        try:
            labels = row.labels.split()
            unicodes = labels[0::5]
            x = [int(v) for v in labels[1::5]]
            y = [int(v) for v in labels[2::5]]
            w = [int(v) for v in labels[3::5]]
            h = [int(v) for v in labels[4::5]]
            bboxes = np.transpose(np.array([x, y, w, h]))
            bboxes[:, 2:4] += bboxes[:, 0:2]  # (x1, y1, x2, y2)
        except AttributeError:
            unicodes = []
            bboxes = np.empty((0, 4), dtype=np.int)

        return {'image': image, 'bboxes': bboxes, 'unicodes': unicodes}
