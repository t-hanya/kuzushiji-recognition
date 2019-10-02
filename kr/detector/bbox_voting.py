"""
Bounding box voting for test time augmentation.

ref: https://arxiv.org/abs/1505.01749
"""


import numpy as np
from PIL import Image
from chainercv.utils import non_maximum_suppression


def _area(bboxes):
    """Calculate areas of bounding boxs."""
    size = bboxes[:, 2:4] - bboxes[:, 0:2]
    area = size[:, 0] * size[:, 1]
    return area


def calc_iou_mat(bboxes1, bboxes2):
    """Calculate IOU matrix from two bbox array."""
    area1 = _area(bboxes1)
    area2 = _area(bboxes2)

    xx1 = np.maximum(bboxes1[:, None, 0], bboxes2[None, :, 0])
    yy1 = np.maximum(bboxes1[:, None, 1], bboxes2[None, :, 1])
    xx2 = np.minimum(bboxes1[:, None, 2], bboxes2[None, :, 2])
    yy2 = np.minimum(bboxes1[:, None, 3], bboxes2[None, :, 3])

    w = np.maximum(xx2 - xx1, 0)
    h = np.maximum(yy2 - yy1, 0)
    overlap = w * h

    iou = overlap / (area1[:, None] + area2[None, :] - overlap)

    return iou


class BboxVoting:
    """Detector wrapper to implement bounding box voting."""

    def __init__(self, detectors):
        self.detectors = detectors

    def detect(self, image: Image.Image):

        # get all prediction results
        pred_bboxes_set = []
        pred_scores_set = []

        for detector in self.detectors:
            bboxes, scores = detector.detect(image)
            pred_bboxes_set.append(bboxes)
            pred_scores_set.append(scores)

        all_bboxes = np.concatenate(pred_bboxes_set)
        all_scores = np.concatenate(pred_scores_set)

        # apply NMS to obtain base bounding boxes for refinement
        keep = non_maximum_suppression(all_bboxes,
                                       thresh=0.5,
                                       score=all_scores)
        base_bboxes = all_bboxes[keep]
        base_scores = all_scores[keep]

        # get matched bboxes
        iou_mat = calc_iou_mat(base_bboxes, all_bboxes)
        match_mat = iou_mat >= 0.5

        # refine bboxes by bbox voting
        refined_bboxes = np.empty_like(base_bboxes)
        refined_scores = np.empty_like(base_scores)

        for i in range(len(base_bboxes)):
            match = match_mat[i]
            scores = all_scores[match]
            bboxes = all_bboxes[match]

            refined_bboxes[i] = np.sum(scores[:, None] * bboxes, axis=0) / scores.sum()
            refined_scores[i] = np.average(scores)

        return refined_bboxes, refined_scores
