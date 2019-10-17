=====================
Kuzushiji Recognition
=====================

`Kaggle Kuzushiji Recognition <https://www.kaggle.com/c/kuzushiji-recognition>`_: Code for 8th place solution.

The kuzushiji recognition pipeline is consists of two models: `CenterNet <https://arxiv.org/abs/1904.07850>`_ character detection model and `MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ per-character classification model.


.. contents::


Setup
=====

Language environment
--------------------

Python version:

* 3.7.3

Libraries:

* chainer (6.2.0)
* chainercv (0.13.1)
* cupy-cuda92 (6.2.0)
* albumentations (0.3.1)
* opencv-python (4.1.0.25)
* Pillow (6.1.0)
* pandas (0.25.0)
* numpy (1.17.0)
* matplotlib (3.1.1)
* japanize-matplotlib (1.0.4)

For unittest:

* pytest==4.4.1

Download dataset
----------------

Please download and unzip the competition dataset to ``<repo root>/data/kuzushiji-recognition``.

The expected directory structure is as follows::

   kuzushiji-recognition/
       data/
           kuzushiji-recognition/
               train.csv
               train_images
               test_images
               unicode_translation.csv
               sample_submission.csv



Training procedure
==================

Please follow the steps below to train kuzushiji recognition models.

1. Set environment variable::

   cd <path to this repo>
   export PYTHONPATH=`pwd`

2. Split all annotated samples written in ``train.csv`` into train and validation split::

   python scripts/prepare_train_val_split.py

3. Prepare per-character cropped image set for character classifier training::

   python scripts/prepare_char_crop_dataset.py

4. Train character detection model::

   python scripts/train_detector.py --gpu 0 --out ./results/detector --full-data

5. Train character classification model::

   python scripts/train_classifier.py --gpu 0 --out ./results/classifier --full-data

6. Prepare pseudo label using trained detector and classifier::

   python scripts/prepare_pseudo_labels.py --gpu 0 \
       ./result/detector/model_700.npz \
       ./result/classifier/model_900.npz \
       --out data/kuzushiji-recognition-pesuedo

7. Finetune classifier using pseudo label and original training data::

   python scripts/finetune_classifier.py --gpu 0 \
       --pseudo-labels-dir  data/kuzushiji-recognition-pesuedo \
       --out ./results/classifier-finetune \
       ./result/classifier/model_900.npz


Prepare submission
==================

To generate a CSV for submission, please execute the following commands.::

   python scripts/prepare_submission.py --gpu 0 \
       ./result/detector/model_700.npz \
       ./results/classifier-finetune/model_100.npz


License
=======

Released under the MIT license.

