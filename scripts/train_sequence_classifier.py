"""
Training script of kuzushiji character classification model.
"""


import argparse
import json
from pathlib import Path
from typing import Tuple
from typing import List


import chainer
import chainer.links as L
from chainer.backends import cuda
from chainer import training
from chainer.training import extension
from chainer.training import extensions
from chainer.training import triggers
from chainer.datasets import TransformDataset
from chainer.datasets import split_dataset_random
from PIL import Image
import numpy as np


from kr.datasets import KuzushijiUnicodeMapping
from kr.datasets import KuzushijiSequenceDataset
from kr.datasets import RandomSampler
from kr.classifier.softmax.crop import RandomCropAndResize
from kr.classifier.softmax.crop import CenterCropAndResize
from kr.classifier.sequence.model import SequenceClassifier
from kr.classifier.sequence.training import TrainModel
from kr.classifier.sequence.sampler import KuzushijiMaskedSequenceGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Resume from the specified snapshot')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Validation minibatch size')
    parser.add_argument('--lr', '-l', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--weight-decay', '-w', type=float, default=1e-5,
                        help='Weight decay')
    args = parser.parse_args()
    return args


class Augmentation:

    def __init__(self, image_size=(64, 64)):
        self.crop_func = RandomCropAndResize(size=image_size)
        self.mapping = KuzushijiUnicodeMapping()

    def __call__(self, seq: list) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           List[np.ndarray],
                                           np.ndarray]:
        images = [self.crop_func(data['image']) for data in seq]
        images = np.stack([np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
                           for img in images])
        labels = np.array([self.mapping.unicode_to_index(data['unicode'])
                           for data in seq], dtype=np.int32)
        none = np.empty((0, 3, 64, 64), dtype=np.float32)
        candidates = []
        mask_positions = []
        for i, data in enumerate(seq):
            if data['candidates'] is None:
                candidates.append(none)
            else:
                tmp = [self.crop_func(img) for img in data['candidates']]
                candidates.append(
                    np.stack([np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
                              for img in tmp]))
                mask_positions.append(i)
        mask_positions = np.array(mask_positions, dtype=np.int32)
        return images, labels, candidates, mask_positions


class Preprocess:

    def __init__(self, image_size=(64, 64)):
        self.crop_func = CenterCropAndResize(size=image_size)
        self.mapping = KuzushijiUnicodeMapping()

    def __call__(self, seq: dict) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           List[np.ndarray],
                                           np.ndarray]:
        images = [self.crop_func(img) for img in seq['images']]
        images = np.stack([np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
                           for img in images])
        labels = np.array([self.mapping.unicode_to_index(uni)
                           for uni in seq['unicodes']], dtype=np.int32)
        none = np.empty((0, 3, 64, 64), dtype=np.float32)
        candidates = [none for _ in range(len(images))]
        mask_positions = np.empty((0,), dtype=np.int32)
        return images, labels, candidates, mask_positions


def prepare_dataset():
    train_raw = KuzushijiMaskedSequenceGenerator(
        KuzushijiSequenceDataset(split='train'))
    val_raw = KuzushijiSequenceDataset(split='val')

    train = TransformDataset(
        RandomSampler(train_raw, 10000),
        Augmentation())

    val = TransformDataset(
        split_dataset_random(val_raw, 1000, seed=0)[0],
        Preprocess())

    return train, val


def converter(batch, gpu_id=-1):
    if gpu_id >= 0:
        to_device = lambda x: cuda.to_gpu(x)
    else:
        to_device = lambda x: x

    imgs = [to_device(s[0]) for s in batch]
    labels = [to_device(s[1]) for s in batch]
    candidates = [[to_device(c) for c in s[2]] for s in batch]
    mask_positions = [to_device(s[3]) for s in batch]

    return imgs, labels, candidates, mask_positions


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

    train, val = prepare_dataset()
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize,
                                                        shared_mem=4000000)
    val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize,
                                                      repeat=False,
                                                      shuffle=False,
                                                      shared_mem=4000000)

    # setup model
    n_classes = len(KuzushijiUnicodeMapping())
    model = SequenceClassifier()
    train_model = TrainModel(model)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=args.lr, momentum=0.9)
    optimizer.setup(train_model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(val_iter, train_model,
                                        device=args.gpu,
                                        converter=converter))
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
                   model, 'model_{.updater.epoch}.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/cls_acc', 'validation/main/cls_acc']))
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

