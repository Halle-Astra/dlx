import os
import glob
import json
import numpy as np
import torch
from dlx.tokenizer.tiktoken import Tokenizer


class WuDao:
    def __init__(self, root, tokenizer, dtype=torch.long, device='cpu'):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device

    def open_file_func(self, file):
        with open(file) as f:
            contents = json.load(f)
        return contents

    def collate_fn(self, batch):
        b_lengths = [len(i) for i in batch]
        min_b_length = min(b_lengths)
        max_b_length = max(b_lengths)
        test_max_length = 150#2048
        if max_b_length > test_max_length:  # 2048:
            batch = [i[:test_max_length] for i in batch]
            max_b_length = test_max_length

        bs = len(batch)
        input_ndarray = np.ones((bs, max_b_length)) * self.tokenizer.pad_id
        for i in range(bs):
            input_ndarray[i, :b_lengths[i]] = batch[i]

        input_x = torch.tensor(input_ndarray, dtype=self.dtype)  # .to(self.device)
        input_y = input_x[:, 1:]
        input_y = input_y.flatten()

        return [input_x, input_y, {'start_pos': 0}]


if __name__ == '__main__':
    from dlx.utils.data.nlp.file_segments_dataloader import FileSegmentsDataloader

    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    wudao_dataset = WuDao(root, Tokenizer())

    dataloader = FileSegmentsDataloader(wudao_dataset)
