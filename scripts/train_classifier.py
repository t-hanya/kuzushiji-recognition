"""
Training script of DenseNet on CIFAR-10 dataset.
"""


import argparse

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


from kr.classifier.softmax.model import MobileNetV3
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
    parser.add_argument('--batchsize', '-b', type=int, default=96,
                        help='Validation minibatch size')
    args = parser.parse_args()
    return args


class Preprocess:

    def __init__(self, image_size=(112, 112)):
        self.image_size = image_size

    def __call__(self, data):
        image = data['image']
        label = data['label']

        image = image.resize(self.image_size, resample=Image.BILINEAR)
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        label = np.array(label, dtype=np.int32)
        return image, label


def prepare_dataset():
    dataset = KuzushijiCharCropDataset()
    n_train = int(len(dataset) * 0.99)
    train, val = split_dataset_random(dataset, n_train, seed=0)
    train = RandomSampler(train, virtual_size=10000)
    train = TransformDataset(train, Preprocess())
    val = TransformDataset(val, Preprocess())
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


def main():
    args = parse_args()

    train, val = prepare_dataset()
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize,
                                                      repeat=False,
                                                      shuffle=False)

    # setup model
    n_classes = len(KuzushijiUnicodeMapping())
    model = MobileNetV3(n_classes)
    train_model = L.Classifier(model)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=0.1, momentum=0.9)
    optimizer.setup(train_model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

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
