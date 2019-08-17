"""
Unittest for evaluation module.
"""


import math

import pytest

from kr.evaluation import compare_with_ground_truth
from kr.evaluation import calc_f1_score
from kr.evaluation import evaluate_kuzushiji_recognition


class TestCompareWithGroundTruth:

    def test(self):
        pred_labels = [
            {'unicode': 'A', 'point': {'x': 15, 'y': 35}},
            {'unicode': 'C', 'point': {'x': 55, 'y': 55}},
        ]
        gt_labels = [
            {'unicode': 'A',
             'bbox': {'x1': 10, 'y1': 30, 'x2': 20, 'y2': 40}},
            {'unicode': 'B',
             'bbox': {'x1': 50, 'y1': 50, 'x2': 60, 'y2': 60}},
        ]

        tp_list, fp_list, fn_list = compare_with_ground_truth(
            pred_labels, gt_labels)

        assert len(tp_list) == 1
        assert tp_list[0]['gt_unicode'] == 'A'
        assert tp_list[0]['gt_bbox'] == gt_labels[0]['bbox']
        assert tp_list[0]['pred_unicode'] == 'A'
        assert tp_list[0]['pred_point'] == pred_labels[0]['point']

        assert len(fp_list) == 1
        assert fp_list[0]['pred_unicode'] == 'C'
        assert fp_list[0]['pred_point'] == pred_labels[1]['point']

        assert len(fn_list) == 1
        assert fn_list[0]['gt_unicode'] == 'B'
        assert fn_list[0]['gt_bbox'] == gt_labels[1]['bbox']


class TestCalcF1Score:

    @pytest.mark.parametrize('n_tp, n_fp, n_fn, expected', [
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (1, 1, 1, 0.5),
    ])
    def test(self, n_tp, n_fp, n_fn, expected):
        ret = calc_f1_score(n_tp, n_fp, n_fn)
        assert math.isclose(ret, expected)


class TestEvaluateKuzushijiRecognition:

    def test(self):
        all_pred_labels = [
            [{'unicode': 'A', 'point': {'x': 15, 'y': 35}},
             {'unicode': 'C', 'point': {'x': 55, 'y': 55}}]
        ]
        all_gt_labels = [
            [{'unicode': 'A',
              'bbox': {'x1': 10, 'y1': 30, 'x2': 20, 'y2': 40}},
             {'unicode': 'B',
              'bbox': {'x1': 50, 'y1': 50, 'x2': 60, 'y2': 60}}]
        ]
        metrics, matching_results = evaluate_kuzushiji_recognition(
            all_pred_labels, all_gt_labels)

        assert math.isclose(metrics['f1_score'], 0.5)
        assert len(matching_results["true_positives"][0]) == 1
        assert len(matching_results["false_positives"][0]) == 1
        assert len(matching_results["false_negatives"][0]) == 1
