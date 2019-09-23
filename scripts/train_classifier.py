"""
Training script of kuzushiji character classification model.
"""


import argparse
import json
from pathlib import Path

import albumentations as alb
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extension
from chainer.training import extensions
from chainer.training import triggers
from chainer.datasets import split_dataset_random
from chainer.datasets import TransformDataset
from PIL import Image
import numpy as np


from kr.classifier.softmax.mobilenetv3 import MobileNetV3
from kr.classifier.softmax.resnet import Resnet18
from kr.classifier.softmax.resnet import Resnet34
from kr.classifier.softmax.crop import CenterCropAndResize
from kr.classifier.softmax.crop import RandomCropAndResize
from kr.datasets import KuzushijiCharCropDataset
from kr.datasets import KuzushijiUnicodeMapping
from kr.datasets import RandomSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Resume from the specified snapshot')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--batchsize', '-b', type=int, default=512,
                        help='Validation minibatch size')
    parser.add_argument('--lr', '-l', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--weight-decay', '-w', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'mobilenetv3'],
                        default='resnet18', help='Backbone CNN model.')
    parser.add_argument('--full-data', '-F', action='store_true', default=False,
                        help='Flag to use all training dataset.')
    args = parser.parse_args()
    return args


class Preprocess:

    def __init__(self, image_size=(112, 112), augmentation=False):
        self.image_size = image_size
        if augmentation:
            self.crop_func = RandomCropAndResize(size=image_size)
            w, h = image_size
            self.aug_func = alb.Compose([
                alb.RGBShift(),
                alb.RandomBrightnessContrast(),
                alb.OneOf([
                    alb.Rotate(limit=5),
                    alb.GridDistortion(distort_limit=0.2),
                ]),
                alb.OneOf([
                    alb.GaussNoise(),
                    alb.IAAAdditiveGaussianNoise()
                ]),
                alb.CoarseDropout(
                    max_holes=1, max_height=h // 2, max_width=w // 2,
                    min_height=h // 4, min_width=w // 4, fill_value=128)
            ])
        else:
            self.crop_func = CenterCropAndResize(size=image_size)
            self.aug_func = None

    def __call__(self, data):
        image = data['image']
        label = data['label']

        image = self.crop_func(image)
        if self.aug_func:
            image = np.asarray(image)
            image = self.aug_func(image=image)['image']
            image = image.astype(np.float32).transpose(2, 0, 1)
        else:
            image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)

        image = (image - 127.5) / 128.
        label = np.array(label, dtype=np.int32)
        return image, label


def prepare_dataset(image_size=(64, 64), full_data=False):

    train_split = 'trainval' if full_data else 'train'
    train = TransformDataset(
        RandomSampler(
            KuzushijiCharCropDataset(split=train_split),
            virtual_size=10000),
        Preprocess(image_size=image_size, augmentation=True))

    val = TransformDataset(
        split_dataset_random(
            KuzushijiCharCropDataset(split='val'),
            first_size=5000, seed=0)[0],
        Preprocess(image_size=image_size, augmentation=False))

    return train, val


class LearningRateDrop(extension.Extension):

    def __init__(self, drop_ratio, attr='lr', optimizer=None):
        self._drop_ratio = drop_ratio
        self._attr = attr
        self._optimizer = optimizer

    def __call__(self, trainer):
        opt = self._optimizer or trainer.updater.get_optimizer('main')

        lr = getattr(opt, self._attr)
        lr *= self._drop_ratio
        setattr(opt, self._attr, lr)


def dump_args(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / 'args.json'
    with dump_path.open('w') as f:
        json.dump(vars(args), f, indent=2)


def main():
    args = parse_args()
    dump_args(args)

    # setup model
    n_classes = len(KuzushijiUnicodeMapping())
    if args.model == 'resnet18':
        model = Resnet18(n_classes)
    elif args.model == 'resnet34':
        model = Resnet34(n_classes)
    elif args.model == 'mobilenetv3':
        model = MobileNetV3(n_classes)
    train_model = L.Classifier(model)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        train_model.to_gpu()

    # setup dataset
    train, val = prepare_dataset(image_size=model.input_size,
                                 full_data=args.full_data)
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize,
                                                      repeat=False,
                                                      shuffle=False)

    # setup optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=args.lr, momentum=0.9)
    optimizer.setup(train_model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(val_iter, train_model,
                                        device=args.gpu))
    trainer.extend(extensions.snapshot(), trigger=(100, 'epoch'))
    trainer.extend(extensions.snapshot_object(
                   model, 'model_{.updater.epoch}.npz'), trigger=(100, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # learning rate scheduling
    lr_drop_epochs = [int(args.epoch * 0.5),
                      int(args.epoch * 0.75)]
    lr_drop_trigger = triggers.ManualScheduleTrigger(lr_drop_epochs, 'epoch')
    trainer.extend(LearningRateDrop(0.1), trigger=lr_drop_trigger)
    trainer.extend(extensions.observe_lr())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # start training
    trainer.run()


if __name__ == '__main__':
    main()
