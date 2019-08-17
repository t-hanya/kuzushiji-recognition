"""
Evaluation module.
"""

from typing import List
from typing import Tuple

import numpy as np


def compare_with_ground_truth(pred_labels: List[dict],
                              gt_labels: List[dict]
                             ) -> Tuple[List[dict], List[dict], List[dict]]:
    """Compare prediction with ground truth data.

    input example::

        pred_labels = [
            {'unicode': 'A', 'point': {'x': 15, 'y': 15}},
            {'unicode': 'C', 'point': {'x': 55, 'y': 55}},
        ]
        gt_labels = [
            {'unicode': 'A',
             'bbox': {'x1': 10, 'y1': 10, 'x2': 20, 'y2': 20}},
            {'unicode': 'B',
             'bbox': {'x1': 50, 'y1': 50, 'x2': 60, 'y2': 60}},
        ]
    """
    tp_list = []
    fp_list = []
    fn_list = []

    pred_used_masks = np.zeros(len(pred_labels), dtype=np.bool)
    gt_used_masks = np.zeros(len(gt_labels), dtype=np.bool)

    if pred_labels and gt_labels:
        pred_x = np.array([pred['point']['x'] for pred in pred_labels])
        pred_y = np.array([pred['point']['y'] for pred in pred_labels])
        pred_unicodes = np.array([pred['unicode'] for pred in pred_labels])

        for i, gt in enumerate(gt_labels):
            match = np.logical_and.reduce((
                gt['bbox']['x1'] < pred_x,
                gt['bbox']['y1'] < pred_y,
                pred_x < gt['bbox']['x2'],
                pred_y < gt['bbox']['y2'],
                pred_unicodes == gt['unicode'],
                np.logical_not(pred_used_masks)
            ))
            if match.any():
                j = np.argmax(match)
                gt_used_masks[i] = True
                pred_used_masks[j] = True
                tp_list.append(
                    {'gt_unicode': gt_labels[i]['unicode'],
                     'gt_bbox': gt_labels[i]['bbox'],
                     'pred_unicode': pred_labels[j]['unicode'],
                     'pred_point': pred_labels[j]['point']})

    for gt_label, used in zip(gt_labels, gt_used_masks):
        if not used:
            fn_list.append(
                {'gt_unicode': gt_label['unicode'],
                 'gt_bbox': gt_label['bbox']})

    for pred_label, used in zip(pred_labels, pred_used_masks):
        if not used:
            fp_list.append(
                {'pred_unicode': pred_label['unicode'],
                 'pred_point': pred_label['point']})

    return tp_list, fp_list, fn_list


def calc_f1_score(n_tp: int, n_fp: int, n_fn: int) -> float:
    """Calculate F1 score."""

    if (n_tp + n_fp) == 0 or (n_tp + n_fn) == 0:
        return 0

    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)

    if precision > 0 or recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


def evaluate_kuzushiji_recognition(all_pred_labels: List[List[dict]],
                                   all_gt_labels: List[List[dict]]
                                  ) -> Tuple[dict, list, list, list]:
    """Evaluate prediction result of kuzushiji recognition task.

    Args:
        all_pred_labels (list of list): Prediction labels for all samples.
        all_gt_labels (list of list): Ground truth labels for all samples.

    Returns:
        dict: Evaluation metrics.
        dict: Dictionary holding TP, FP, FN lists.
    """
    all_true_positives = []
    all_false_positives = []
    all_false_negatives = []

    for pred_labels, gt_labels in zip(all_pred_labels, all_gt_labels):
        tp_list, fp_list ,fn_list = compare_with_ground_truth(
            pred_labels, gt_labels)

        all_true_positives.append(tp_list)
        all_false_positives.append(fp_list)
        all_false_negatives.append(fn_list)

    n_tp = sum([len(tp_list) for tp_list in all_true_positives])
    n_fp = sum([len(fp_list) for fp_list in all_false_positives])
    n_fn = sum([len(fn_list) for fn_list in all_false_negatives])

    f1_score = calc_f1_score(n_tp, n_fp, n_fn)

    metrics = {
        'f1_score': f1_score
    }

    matching_results = {
        "true_positives": all_true_positives,
        "false_positives": all_false_positives,
        "false_negatives": all_false_negatives
    }
    return metrics, matching_results
