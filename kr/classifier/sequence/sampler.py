"""
Self-supervised target generator.
"""


from chainer.dataset import DatasetMixin
import numpy as np
from PIL import Image

from kr.datasets import KuzushijiRecognitionDataset
from kr.datasets import KuzushijiSequenceDataset


class KuzushijiMaskedSequenceGenerator(DatasetMixin):
    """kuzushiji masked sequence generator."""

    def __init__(self,
                 dataset: KuzushijiSequenceDataset,
                 max_length: int = 10
                ) -> None:
        self.dataset = dataset
        self.max_length = max_length
        self.mask_prob = 0.15

    def __len__(self) -> int:
        return len(self.dataset)

    def get_example(self, index) -> list:
        page_index = self.dataset.page_indices[index]

        same_page_indices = np.where(
            self.dataset.page_indices == page_index)[0]

        # handle the case only 1 sequence in the page
        if len(same_page_indices) == 1:
            sequence = self.dataset[index]
            ret = [{'image': img, 'unicode': uni, 'candidates': None}
                   for img, uni in zip(sequence['images'], sequence['unicodes'])]
            return ret

        other_sequence_indices = same_page_indices[same_page_indices != index]
        other_sequence_indices = np.random.choice(
            other_sequence_indices,
            size=min(4, len(other_sequence_indices)),
            replace=False)

        other_chars = []
        for i in range(len(other_sequence_indices)):
            seq = self.dataset[i]
            for img, uni in zip(seq['images'], seq['unicodes']):
                other_chars.append((img, uni))

        def _random_pop(l):
            i = np.random.choice(np.arange(len(l)))
            return l.pop(i)

        ret = []
        sequence = self.dataset[index]
        if len(sequence) > self.max_length:
            i = np.random.randint(0, len(sequence) - self.max_length + 1)
            sequence = sequence[i: i + self.max_length]

        for img, uni in zip(sequence['images'], sequence['unicodes']):
            if np.random.rand() <= self.mask_prob and len(other_chars) >= 4:
                # sample image
                v = np.random.rand()
                if v < 0.2:
                    src_img = img
                elif v < 0.4:
                    d = _random_pop(other_chars)
                    src_img = d[0]
                else:
                    src_img = Image.new('RGB', (80, 80), (128, 128, 128))

                # sample candidates
                candidates = [img] + [_random_pop(other_chars)[0] for _ in range(3)]
                ret.append({'image': src_img, 'candidates': candidates, 'unicode': uni})
            else:
                ret.append({'image': img, 'candidates': None, 'unicode': uni})
        return ret
