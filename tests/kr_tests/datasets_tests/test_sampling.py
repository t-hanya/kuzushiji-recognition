"""
Unittest for kr.datasets.sampling module.
"""


from kr.datasets import RandomSampler


class TestRandomSampler:

    def test(self):
        dataset = RandomSampler([1, 2, 3, 4, 5], virtual_size=10)
        assert len(dataset) == 10
        for _ in range(10):
            assert 1 <= dataset[0] <= 5
