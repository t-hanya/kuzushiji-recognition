"""
Training script of kuzushiji character classification model.
"""


import argparse
import json
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import training
from chainer.training import extension
from chainer.training import extensions
from chainer.training import triggers
from chainer.datasets import split_dataset
from chainer.datasets import TransformDataset
from PIL import Image
import numpy as np


from kr.classifier.softmax.model import MobileNetV3
from kr.classifier.softmax.crop import CenterCropAndResize
from kr.classifier.softmax.crop import RandomCropAndResize
from kr.datasets import KuzushijiCharCropDataset
from kr.datasets import KuzushijiUnicodeMapping
from kr.datasets import RandomSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Resume from the specified snapshot')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--batchsize', '-b', type=int, default=192,
                        help='Validation minibatch size')
    parser.add_argument('--lr', '-l', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--weight-decay', '-w', type=float, default=1e-5,
                        help='Weight decay')
    args = parser.parse_args()
    return args


class ClassBalancedTrainingModel(chainer.Chain):

    def __init__(self, model, num_samples, beta=0.9) -> None:
        super().__init__()
        with self.init_scope():
            self.model = model

        self.beta = beta  # class balance parameter
        self.num_samples = num_samples  # 1d array holding
                                        # number of samples for each class
        self._class_weight = None

    @property
    def class_weight(self):
        if self._class_weight is None:
            weight = (1 - self.beta) / (1 - self.beta ** self.num_samples)
            weight = weight.astype(np.float32)
            if self.xp != np:
                weight = cuda.to_gpu(weight)
            self._class_weight = weight
        return self._class_weight

    def __call__(self, x, labels):
        y = self.model(x)

        # class balanced softmax loss
        # ref: https://arxiv.org/abs/1901.05555
        loss = F.softmax_cross_entropy(y, labels, class_weight=self.class_weight)
        acc = F.accuracy(y, labels)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss


class Preprocess:

    def __init__(self, image_size=(112, 112), augmentation=False):
        self.image_size = image_size
        if augmentation:
            self.crop_func = RandomCropAndResize(size=image_size)
        else:
            self.crop_func = CenterCropAndResize(size=image_size)

    def __call__(self, data):
        image = data['image']
        label = data['label']

        image = self.crop_func(image)
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        label = np.array(label, dtype=np.int32)
        return image, label


def prepare_dataset():

    train_raw = KuzushijiCharCropDataset(split='train')
    train = TransformDataset(
        RandomSampler(
            train_raw,
            virtual_size=10000),
        Preprocess(augmentation=True))

    val = TransformDataset(
        split_dataset(
            KuzushijiCharCropDataset(split='val'),
            split_at=5000)[0],
        Preprocess(augmentation=False))

    return train, val, train_raw.num_samples


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

    train, val, num_samples = prepare_dataset()
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize,
                                                      repeat=False,
                                                      shuffle=False)

    # setup model
    n_classes = len(KuzushijiUnicodeMapping())
    model = MobileNetV3(n_classes)
    train_model = ClassBalancedTrainingModel(model, num_samples)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=args.lr, momentum=0.9)
    optimizer.setup(train_model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(val_iter, train_model,
                                        device=args.gpu))
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
                   model, 'model_{.updater.epoch}.npz'), trigger=(10, 'epoch'))
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
