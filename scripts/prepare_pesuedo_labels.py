"""
Create pesuedo labels.
"""


import argparse
import json
from pathlib import Path

import chainer
import numpy as np

from kr.datasets import KuzushijiTestImages
from kr.datasets import KuzushijiUnicodeMapping
from kr.detector.centernet.resnet import Res18UnetCenterNet
from kr.detector.adaptive_scale import AdaptiveScaleWrapper
from kr.classifier.softmax.mobilenetv3 import MobileNetV3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('detector_path', type=str)
    parser.add_argument('classifier_path', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', type=str,
                        default='data/kuzushiji-recognition-pesuedo')
    args = parser.parse_args()
    return args


def load_model(detector_path, classifier_path, gpu, num_classes):
    """Load model."""
    detector = Res18UnetCenterNet()
    chainer.serializers.load_npz(
        detector_path,
        detector
    )
    classifier = MobileNetV3(out_ch=num_classes)
    chainer.serializers.load_npz(
        classifier_path,
        classifier
    )
    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        detector.to_gpu()
        classifier.to_gpu()

    detector = AdaptiveScaleWrapper(detector)
    return detector, classifier


def calc_padded_bboxes(bboxes: np.ndarray, pad_ratio=0.2) -> np.ndarray:
    """Calculate expanded bboxes size.

    - keep aspect ratio
    - padding
    """
    centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
    longer_sides = (bboxes[:, 2:4] - bboxes[:, 0:2]).max(axis=1)
    padded_size = longer_sides * (1 + pad_ratio * 2)

    padded_bboxes = np.empty_like(bboxes)
    padded_bboxes[:, 0:2] = np.floor(centers - padded_size[:, None] / 2.)
    padded_bboxes[:, 2:4] = np.ceil(centers + padded_size[:, None] / 2.)

    return padded_bboxes


def main():
    """Main procedure."""
    args = parse_args()

    dataset = KuzushijiTestImages()
    mapping = KuzushijiUnicodeMapping()

    detector, classifier = load_model(
        args.detector_path, args.classifier_path, args.gpu,
        num_classes=len(mapping))

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)

    idx = 0
    pesuedo_labels = []
    for i, data in enumerate(dataset):
        print('[{}/{}] {}'.format(i + 1, len(dataset), data['image_id']))
        image = data['image']

        # prediction
        bboxes, _ = detector.detect(image)
        labels, scores = classifier.classify(image, bboxes)
        unicodes = [mapping.index_to_unicode(l) for l in labels]

        # crop position
        padded_bboxes = calc_padded_bboxes(bboxes)

        for bbox, padded_bbox, unicode, score in zip(bboxes, padded_bboxes,
                                                     unicodes, scores):
            # ignore low confident predictions
            if score < 0.9:
                continue

            # define output directory and file name
            dirname = str(idx // 10000 * 10000).zfill(8)
            output_dir = output_root / 'char_images' / dirname
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = str(idx).zfill(8) + '.png'
            fpath = output_dir / fname

            # save cropped image
            cropped = image.crop(padded_bbox)
            cropped.save(fpath)

            # keep cropped image file path
            modified = bbox - np.tile(padded_bbox[0:2], 2)
            pesuedo_labels.append(
                {'image_path': str(fpath.relative_to(output_root)),
                 'original_bbox': bbox.tolist(),
                 'bbox': modified.tolist(),
                 'unicode': unicode})

            idx += 1

    out = {
        'detector': args.detector_path,
        'classifier': args.classifier_path,
        'pesuedo_labels': pesuedo_labels,
    }
    with (output_root / 'pesuedo_labels.json').open('w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
