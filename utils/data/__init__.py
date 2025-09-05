from torch.utils.data import Dataset
import numpy as np
import random


class Merged_Dataset(Dataset):
    def __init__(self, datasets, probs=None):
        if not probs is None:
            probs = np.array(probs) / sum(probs)
            probs = probs.tolist()
        self.datasets = map(iter, datasets)
        self.probs = probs
        self._prob_intervals = np.cumsum([0] + probs)

    def __len__(self):
        len_s = map(len, self.datasets)
        return sum(len_s)

    def __getitem__(self, index):
        if not self.probs is None:
            random_v = random.random()
            # todo: impl with bisearch
            for interval_start in range(self.probs):
                prob_start = self._prob_intervals[interval_start]
                prob_end = self._prob_intervals[interval_start + 1]
                if prob_start <= random_v <= prob_end:
                    interval_index = interval_start
                break
            dataset = self.datasets[interval_index]
        else:
            dataset = random.choice(self.datasets)
        return next(dataset)


def merge_datasets(*datasets, probs=None):
    pass
