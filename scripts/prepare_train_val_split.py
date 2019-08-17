"""
Prepare train / validation split.
"""


from pathlib import Path

import pandas as pd
import numpy as np


PRJ_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PRJ_ROOT / 'data' / 'kuzushiji-recognition' / 'train.csv'
OUTPUT_ROOT = PRJ_ROOT / 'data' / 'kuzushiji-recognition-converted'

TRAIN_RATIO = 0.95


if __name__ == '__main__':

    OUTPUT_ROOT.mkdir(exist_ok=True)

    table = pd.read_csv(str(TRAIN_CSV))
    n_samples = len(table)
    n_train = int(n_samples * TRAIN_RATIO)

    np.random.seed(0)
    random_order = np.random.permutation(n_samples)
    train_indices = random_order[:n_train]
    val_indices = random_order[n_train:]

    table_train = table.iloc[train_indices]
    table_val = table.iloc[val_indices]

    table_train.to_csv(str(OUTPUT_ROOT / 'train.csv'), index=False)
    table_val.to_csv(str(OUTPUT_ROOT / 'val.csv'), index=False)
