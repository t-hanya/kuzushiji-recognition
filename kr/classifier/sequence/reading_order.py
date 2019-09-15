"""
Reading order prediction.
"""


from typing import List
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _distance_matrix(bboxes: np.ndarray) -> np.ndarray:
    """Distance matrix between characters."""
    centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
    sizes = np.mean(bboxes[:, 2:4] - bboxes[:, 0:2], axis=1)
    size_mat = np.sqrt(sizes[:, None] * sizes[None, :])
    vec = (centers[None, :, :] - centers[:, None, :]) / size_mat[:, :, None]

    # euclidean distance between character centers
    center_dist = np.sqrt(np.sum(vec ** 2, axis=2))

    # cosine value with y-axis vector
    target = np.array([0, 1])
    norm = np.maximum(np.linalg.norm(vec, axis=2), 1)  # to avoid zero division
    angle_dist = 1 - np.sum(vec * target[None, None, :], axis=2) / norm

    distance = center_dist + 1.5 * angle_dist + np.eye(len(bboxes)) * 10000
    return distance


def _build_sequence_from_prev_list(idx: int,
                                   prev_indices: np.ndarray
                                  ) -> List[int]:
    """Build sequence from previous indices list."""
    seq = [idx]
    i = idx
    while True:
        if prev_indices[i] < 0:
            break
        seq.insert(0, int(prev_indices[i]))
        i = prev_indices[i]
    return seq


def _build_sequence_from_next_list(idx: int,
                                   next_indices: np.ndarray
                                  ) -> List[int]:
    """Build sequence from next indices list."""
    seq = [idx]
    i = idx
    while True:
        if next_indices[i] < 0:
            break
        seq.append(int(next_indices[i]))
        i = next_indices[i]
    return seq


def predict_reading_order(bboxes: np.ndarray,
                          distance_threshold: float = 2.) -> List[List[int]]:
    """Predict reading order."""
    # distance matrix
    dist_mat = _distance_matrix(bboxes)

    # check order
    centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
    top_right_score = centers[:, 0] / 10 - centers[:, 1]
    check_order = np.argsort(top_right_score)[::-1]

    next_indices = -np.ones(len(bboxes), dtype=np.int)
    prev_indices = -np.ones(len(bboxes), dtype=np.int)

    # determine reading order based on distance matrix
    # in greedy manner
    for i in check_order:
        if next_indices[i] >= 0:
            continue
        idx_prev = i
        seq_prev = _build_sequence_from_prev_list(idx_prev, prev_indices)

        while True:
            j = np.argmin(dist_mat[idx_prev])
            if dist_mat[idx_prev][j] > distance_threshold:
                break
            seq_next = _build_sequence_from_next_list(j, next_indices)
            if set(seq_prev).intersection(set(seq_next)):
                dist_mat[idx_prev, j] = 1e10
                continue

            idx_next = j
            dist_mat[idx_prev, :] = 1e10
            dist_mat[:, idx_next] = 1e10
            next_indices[idx_prev] = idx_next
            prev_indices[idx_next] = idx_prev

            idx_prev = idx_next
            seq_prev = _build_sequence_from_prev_list(idx_prev, prev_indices)

    # parse next_indices and prev_indices into sequence list
    used = np.zeros(len(bboxes), dtype=np.bool)
    sequences = []
    for i in range(len(bboxes)):
        if used[i]:
            continue
        seq = (_build_sequence_from_prev_list(i, prev_indices)[:-1] +
               _build_sequence_from_next_list(i, next_indices))
        for j in seq:
            used[j] = True
        sequences.append(seq)

    return sequences


def visualize_sequences(data: dict,
                        sequences: List[List[int]],
                        fig_size: int = 10,
                        display: bool = True,
                        save_path: Optional[str] = None
                       ) -> None:
    """Visualize sequences."""
    bboxes = data['bboxes']
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(data['image'])
    cmap = plt.cm.get_cmap('tab20')
    for si, sequence in enumerate(sequences):
        for i, j in zip(sequence[:-1], sequence[1:]):
            p1 = (bboxes[i][0:2] + bboxes[i][2:4]) / 2.
            p2 = (bboxes[j][0:2] + bboxes[j][2:4]) / 2.
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'ro-')
        for i in sequence:
            x1, y1, x2, y2 = bboxes[i]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fc=cmap(si), ec=cmap(si), alpha=0.5))
    if display:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()

