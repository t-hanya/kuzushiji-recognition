"""
Prepare train / validation split.

The first block of image file name separated by '-' or '_' is defined as
book title here, and group split with book title is used to provide
cross validation set.
"""


from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold


PRJ_ROOT = Path(__file__).resolve().parent.parent
TRAINVAL_CSV = PRJ_ROOT / 'data' / 'kuzushiji-recognition' / 'train.csv'
TRAINVAL_IMAGE_DIR = PRJ_ROOT / 'data' / 'kuzushiji-recognition' / 'train_images'
OUTPUT_ROOT = PRJ_ROOT / 'data' / 'kuzushiji-recognition-gsplit'

N_SPLIT = 4


def extract_book_title(image_path: Path) -> str:
    """Extract book title from image file path."""
    return image_path.parts[-1].split('_')[0].split('-')[0]


if __name__ == '__main__':

    OUTPUT_ROOT.mkdir(exist_ok=True)

    image_paths = sorted(TRAINVAL_IMAGE_DIR.iterdir())
    book_titles = sorted(set([extract_book_title(p) for p in image_paths]))
    group_labels = [book_titles.index(extract_book_title(p))
                    for p in image_paths]

    table = pd.read_csv(str(TRAINVAL_CSV))

    np.random.seed(0)
    cv = GroupKFold(n_splits=4)
    splits = cv.split(image_paths, groups=group_labels)

    for i, (train_indices, val_indices) in enumerate(splits):
        table_train = table.iloc[train_indices]
        table_val = table.iloc[val_indices]

        table_train.to_csv(str(OUTPUT_ROOT / f'train-{i}.csv'), index=False)
        table_val.to_csv(str(OUTPUT_ROOT / f'val-{i}.csv'), index=False)
