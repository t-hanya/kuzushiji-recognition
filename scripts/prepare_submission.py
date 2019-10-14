"""
Prepare submission data using trained model.
"""


import argparse
import datetime
import json
from pathlib import Path

import chainer

from kr.datasets import KuzushijiTestImages
from kr.datasets import KuzushijiUnicodeMapping
from kr.detector.centernet.resnet import Res18UnetCenterNet
from kr.classifier.softmax.mobilenetv3 import MobileNetV3
from kr.detector.adaptive_scale import AdaptiveScaleWrapper


_project_root = Path(__file__).resolve().parent.parent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('detector_path', type=str)
    parser.add_argument('classifier_path', type=str)
    parser.add_argument('--bbox-score-threshold', type=float, default=0.5)
    parser.add_argument('--class-score-threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


def load_model(detector_path, classifier_path, gpu, num_classes,
               bbox_score_threshold):
    """Load model."""
    detector = Res18UnetCenterNet(
        score_threshold=bbox_score_threshold)

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


def main():
    """Main procedure."""
    args = parse_args()

    # prepare output path
    submission_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    submission_dir = _project_root / 'submissions' / submission_id
    submission_dir.mkdir(parents=True)
    csv_path = submission_dir / 'submission.csv'

    # load data
    dataset = KuzushijiTestImages()
    mapping = KuzushijiUnicodeMapping()

    # load model
    detector, classifier = load_model(
        args.detector_path, args.classifier_path, args.gpu,
        num_classes=len(mapping),
        bbox_score_threshold=args.bbox_score_threshold)

    with csv_path.open('w') as f:
        f.write('image_id,labels\n')

        for i, data in enumerate(dataset):
            print('[{}/{}] {}'.format(i + 1, len(dataset), data['image_id']))

            bboxes, _ = detector.detect(data['image'])
            unicode_indices, scores = classifier.classify(data['image'], bboxes)
            centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.
            label_items = []
            for unicode_index, center, score in zip(unicode_indices, centers, scores):
                if score < args.class_score_threshold:
                    continue
                unicode = mapping.index_to_unicode(unicode_index)
                x = int(round(center[0]))
                y = int(round(center[1]))
                label_items += [unicode, str(x), str(y)]
            labels = ' '.join(label_items)

            row = '{},{}\n'.format(data['image_id'], labels)
            f.write(row)

    cond_path = submission_dir / 'cond.json'
    with cond_path.open('w') as f:
        json.dump({
            'submission_id': submission_id,
            'detector_path': args.detector_path,
            'classifier_path': args.classifier_path,
            'bbox_score_threshold': args.bbox_score_threshold,
            'class_score_threshold': args.class_score_threshold
        }, f, indent=2)


if __name__ == '__main__':
    main()
