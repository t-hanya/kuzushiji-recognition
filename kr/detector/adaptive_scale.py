"""
Detector wrapper to implement adaptive & multi scale inference.
"""


import numpy as np

from .bbox_voting import BboxVoting


class ResizedImageWrapper:
    """Detector wrapper to change 'image_min_side' value."""

    def __init__(self, detector, scale=1.0):
        assert hasattr(detector, 'image_min_side')
        self.detector = detector

        image_min_side = scale * detector.image_min_side
        self.image_min_side = 32 * int(round(image_min_side // 32))

    def detect(self, image):
        # change value
        org_value = self.detector.image_min_side
        self.detector.image_min_side = self.image_min_side

        # detection
        bboxes, scores = self.detector.detect(image)

        # restore value
        self.detector.image_min_side = org_value
        return bboxes, scores


class AdaptiveScaleWrapper:
    """Adaptive scale detection wrapper."""

    # calculated value from training set.
    # 1. resize all trainig images so that each shorter side is 832px.
    # 2. calculate bbox size (=(w+h)/2) for all bboxes.
    # 3. take median of all bbox sizes to determine this ``target_size``.
    target_size = 35.
    image_min_side = 832

    def __init__(self, detector, scales=(0.8, 1, 1.25)):
        assert detector.image_min_side == self.image_min_side

        self.detector = detector
        self.scales = scales

    def detect(self, image):

        # predict bboxes on original image
        bboxes, scores = self.detector.detect(image)
        if len(bboxes) == 0:
            return bboxes, scores

        # calculate ideal scale
        size = (bboxes[:, 2:4] - bboxes[:, 0:2]).mean()
        size *= (self.image_min_side / min(*image.size))
        scale = self.target_size / size

        detectors = [ResizedImageWrapper(self.detector, scale * s)
                     for s in self.scales]
        voting_wrapper = BboxVoting(detectors)
        bboxes, scores = voting_wrapper.detect(image)

        return bboxes, scores
