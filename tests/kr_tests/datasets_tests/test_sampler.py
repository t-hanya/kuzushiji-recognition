"""
Unittest for datasets.sampler module.
"""


from kr.datasets import RandomSampler


class TestRandomSampler:

    def test(self):
        dataset = list(range(10))

        sampler = RandomSampler(dataset, virtual_size=20)
        assert len(sampler) == 20
        assert 0 <= sampler[0] <= 9
