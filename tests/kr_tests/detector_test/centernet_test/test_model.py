"""
Unittest for model module.
"""


from chainer import Variable
import numpy as np
from PIL import Image

from kr.detector.centernet.model import UnetCenterNet


class TestUnet:

    def test_call(self):
        img = np.random.uniform(-1, 1, (1, 3, 64, 64)).astype(np.float32)
        model = UnetCenterNet(n_fg_class=4)
        ret = model(img)
        assert isinstance(ret, Variable)
        assert ret.shape == (1, 8, 16, 16)

    def test_detect(self):
        img = Image.new('RGB', (640, 480))
        model = UnetCenterNet(n_fg_class=4)
        bboxes, scores = model.detect(img)

        assert type(bboxes) == np.ndarray
        assert type(scores) == np.ndarray

        assert bboxes.ndim == 2 and bboxes.shape[1] == 4
        assert scores.ndim == 1
