from torch.utils.data import Dataset
import numpy as np
import random
from loguru import logger
from multiprocessing import Value, RLock


class Merged_Dataset(Dataset):
    def __init__(self, *datasets, probs=None, shuffle=True):
        if probs is not None:
            probs = np.array(probs) / sum(probs)
            probs = probs.tolist()
        else:
            if shuffle:
                probs = list(map(len, datasets))
                probs = np.array(probs) / sum(probs)
                probs = probs.tolist()
        self.probs = probs

        self.datasets_original = datasets
        self.datasets = map(iter, datasets)
        # self.probs = probs
        self._prob_intervals = np.cumsum([0] + probs)
        self.processed_value = Value('i', 0)
        self.lock = RLock()

        self.cumulative_sizes = self.get_cumulative_sizes()

    def get_cumulative_sizes(self):
        sizes = [len(d) for d in self.datasets]
        cumulative_sizes = np.cumsum(sizes)
        return cumulative_sizes

    def __len__(self):
        len_s = map(len, self.datasets)
        return sum(len_s)

    def reset(self):
        for dataset in self.datasets:
            if hasattr(dataset, 'reset'):
                dataset.reset()

    def __getitem__(self, index):
        if self.probs is not None:
            if self.processed_value.value == self.__len__():
                raise StopIteration

            random_v = random.random()
            # todo: impl with bisearch
            for interval_start in range(self.probs):
                prob_start = self._prob_intervals[interval_start]
                prob_end = self._prob_intervals[interval_start + 1]
                if prob_start <= random_v <= prob_end:
                    dataset_index = interval_start
                break
            dataset = self.datasets[dataset_index]
            try:
                sample = next(dataset)
            except Exception as e:
                logger.warning(
                    'LOCAL_RANK: {}, an error in dataset sampling: {}, the {}th dataset will be reset and reused'.format(
                        os.getenv('LOCAL_RANK', -1),
                        e,
                        dataset_index
                    ))
                dataset_orig = self.datasets_original[dataset_index]
                if hasattr(dataset_orig, 'reset'):
                    dataset_orig.reset()
                self.datasets[dataset_index] = iter(dataset_orig)
                dataset = self.datasets[dataset_index]
                sample = next(dataset)
            with self.lock:
                self.processed_value.value += 1

        else:
            dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            sample = self.datasets_original[dataset_idx][sample_idx]
        return sample


def merge_datasets(*datasets, probs=None, shuffle=True):
    if len(datasets) == 1:
        return datasets[0]
    else:
        return Merged_Dataset(*datasets, probs, shuffle)
