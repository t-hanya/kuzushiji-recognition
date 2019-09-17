"""
Sequence classification model.
"""


from typing import List

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

from kr.datasets import KuzushijiUnicodeMapping
from kr.classifier.sequence.cnn import Resnet18


class SequenceClassifier(chainer.Chain):
    """Sequence classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.unicode_mapping = KuzushijiUnicodeMapping()
        with self.init_scope():
            self.cnn = Resnet18(512)
            self.lstm = L.NStepBiLSTM(
                n_layers=2, in_size=512, out_size=512, dropout=0.2)
            self.fc = L.Linear(512 * 2, len(self.unicode_mapping))

    def forward(self,
                images: List[chainer.Variable],
                return_embeddings: bool = False
               ) -> List[chainer.Variable]:
        """Forward computation of sequence classifier."""
        N = len(images)
        batch_indices = np.concatenate([i * np.ones(len(img), dtype=np.int32)
                                        for i, img in enumerate(images)])

        # CNN
        h = F.relu(self.cnn(F.concat(images, axis=0)))
        h = [h[batch_indices == i] for i in range(N)]

        # Bi-LSTM
        _, _, embs = self.lstm(None, None, h)

        # FC
        h = self.fc(F.relu(F.concat(embs, axis=0)))
        ret = [h[batch_indices == i] for i in range(N)]
        if return_embeddings:
            return ret, embs
        else:
            return ret
