"""
Training script of kuzushiji charactor detection model.
"""


from typing import Tuple
import argparse
import json
from pathlib import Path

import chainer
from chainer.backends import cuda
from chainer import training
from chainer.training import extension
from chainer.training import extensions
from chainer.training import triggers
from chainer.datasets import split_dataset_random
from chainer.datasets import TransformDataset
from PIL import Image
import numpy as np

from kr.detector.centernet.model import UnetCenterNet
from kr.detector.centernet.training import TrainingModel
from kr.detector.centernet.heatmap import generate_heatmap
from kr.detector.centernet.crop import RandomCropAndResize
from kr.detector.centernet.crop import CenterCropAndResize
from kr.datasets import KuzushijiRecognitionDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Validation minibatch size')
    parser.add_argument('--resume', '-r', default='',
                            help='Initialize the trainer from given file')
    args = parser.parse_args()
    return args


class Preprocessor:

    def __init__(self,
                 scale_range: float = (0.35, 0.65),
                 input_size: int = (416, 416),
                 augmentation: bool = False) -> None:

        if augmentation:
            self.crop_func = RandomCropAndResize(scale_range, input_size)
        else:
            scale = (scale_range[0] + scale_range[1]) / 2.
            self.crop_func = CenterCropAndResize(scale, input_size)

        self.heatmap_stride = 4
        self.heatmap_size = (input_size[0] // self.heatmap_stride,
                             input_size[1] // self.heatmap_stride)

    def __call__(self, data: dict) -> Tuple[np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray]:

        # crop
        image, bboxes, unicodes = self.crop_func(
            data['image'], data['bboxes'], data['unicodes'])

        # prepare image
        image = np.asarray(image, dtype=np.float32)
        image = image.transpose(2, 0, 1)

        # prepare training target
        labels = np.zeros(len(bboxes), dtype=np.int32)
        heatmap, indices = generate_heatmap(
            bboxes / self.heatmap_stride,
            labels,
            num_classes=1,
            heatmap_size=self.heatmap_size)

        return image, heatmap, labels, indices


def prepare_dataset():

    train = TransformDataset(
        KuzushijiRecognitionDataset('train'),
        Preprocessor(augmentation=True))

    val = TransformDataset(
        KuzushijiRecognitionDataset('val'),
        Preprocessor(augmentation=False))

    return train, val


def converter(batch, gpu_id=-1):
    if gpu_id >= 0:
        to_device = lambda x: cuda.to_gpu(x)
    else:
        to_device = lambda x: x

    imgs = to_device(np.stack([s[0] for s in batch]))
    heatmaps = to_device(np.stack([s[1] for s in batch]))
    labels = [to_device(s[2]) for s in batch]
    indices = [to_device(s[3]) for s in batch]
    return imgs, heatmaps, labels, indices


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

    # prepare dataset
    train, val = prepare_dataset()
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize,
                                                        shared_mem=4000000)
    val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize,
                                                      repeat=False,
                                                      shuffle=False,
                                                      shared_mem=4000000)

    # setup model
    model = UnetCenterNet()
    training_model = TrainingModel(model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        training_model.to_gpu()

    # setup optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=1e-3)
    optimizer.setup(training_model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
    optimizer.add_hook(chainer.optimizer.GradientClipping(100.))

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer,
                                       device=args.gpu,
                                       converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                               out=args.out)

    # set trainer extensions
    trainer.extend(extensions.Evaluator(val_iter, training_model,
                                        device=args.gpu,
                                        converter=converter))
    trainer.extend(extensions.snapshot_object(
                   model, 'model_{.updater.epoch}.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
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
