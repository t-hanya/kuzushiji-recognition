"""
Trainer extensions for detector training.
"""


import numpy as np

import chainer
from chainer import reporter
from chainer.training.extensions import Evaluator

from chainercv.evaluations import eval_detection_voc


def _apply_iterator(iterator, target):

    pred_bboxes, pred_labels, pred_scores = [], [], []
    gt_bboxes, gt_labels = [], []

    for batch in iterator:
        for data in batch:

            bboxes, scores = target.detect(data['image'])
            labels = np.zeros(len(bboxes), dtype=np.int32)
            pred_bboxes.append(bboxes)
            pred_labels.append(labels)
            pred_scores.append(scores)

            gt_bboxes.append(data['bboxes'])
            gt_labels.append(np.zeros(len(data['bboxes']), dtype=np.int32))

    return pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels


class DetectionMapEvaluator(Evaluator):
    """Detection mean average-precision evaluator."""

    trigger = 1, 'epoch'
    default_name = 'eval'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target):
        super().__init__(iterator, target)

    def evaluate(self):
        target = self._targets['main']
        iterator = self._iterators['main']
        iterator.reset()

        ret = _apply_iterator(iterator, target)
        result = eval_detection_voc(*ret)

        report = {'map': result['map']}

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
