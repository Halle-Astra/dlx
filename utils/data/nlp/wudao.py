import os
import glob
import json
import numpy as np
import torch
from dlx.tokenizer.tiktoken import Tokenizer
from torch.utils.data import Dataset
from multiprocessing import Value, RLock, Manager
import time
import random
from loguru import logger
import torch.distributed as dist


# For file segments dataloader
class WuDao:
    def __init__(self, root, tokenizer, dtype=torch.long, max_tokens=150, device='cpu'):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.device = device

    def open_file_func(self, file):
        with open(file) as f:
            contents = json.load(f)
        return contents

    def __len__(self):
        return 59132213

    def collate_fn(self, batch):
        b_lengths = [len(i) for i in batch]
        min_b_length = min(b_lengths)
        max_b_length = max(b_lengths)
        # test_max_length = 150
        if max_b_length > self.max_tokens:  # 2048:
            batch = [i[:self.max_tokens] for i in batch]
            max_b_length = self.max_tokens

        bs = len(batch)
        input_ndarray = np.ones((bs, max_b_length)) * self.tokenizer.pad_id
        for i in range(bs):
            input_ndarray[i, :b_lengths[i]] = batch[i]

        input_x = torch.tensor(input_ndarray, dtype=self.dtype)  # .to(self.device)
        input_y = input_x[:, 1:]
        input_y = input_y  # .flatten()

        return [
            input_x, input_y,
            {
                'start_pos': 0,
                'tokens_num': sum([len(i) for i in batch])
            }
        ]


# for torch.utils.data.DataLoader
class WuDao_Dataset(Dataset):
    def __init__(self, root_or_files, tokenizer, max_length=2048, shuffle=False, random_seed=40, samples_num=0):
        if isinstance(root_or_files, list):
            self.files = root_or_files
        else:
            self.files = glob.glob(os.path.join(root_or_files, '*'))

        self.shuffle = shuffle
        if self.shuffle:
            random.seed(random_seed)
            random.shuffle(self.files)
        self.current_file_index = 0
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.process_count = Value('i', 0)
        self.lock = RLock()
        self.manager = Manager()
        self.file_lens = self.manager.list([0])
        self.data = self.manager.list()
        self.load_file(self.current_file_index)

        if samples_num>0:
            self.samples_num=samples_num
        else:
            samples_num = 0
            for file in self.files:
                with open(file) as f:
                    data = json.load(f)
                    samples_num += len(data)
                f.close()
            self.samples_num = samples_num
        logger.info('local rank: {}, samples_num: {}'.format(
            os.getenv('LOCAL_RANK', -1), self.samples_num
        ))

    def load_file(self, file_index):
        file_path = self.files[file_index]
        logger.debug('local rank: {}, load file: {}'.format(os.getenv('LOCAL_RANK', -1), file_path))
        with open(file_path) as f:
            data = json.load(f)
            if self.shuffle: random.shuffle(data)
            for _ in range(len(self.data)):
                self.data.pop()
            for item in data:
                self.data.append(item)
            self.file_lens.append(len(self.data))
            logger.debug('local rank: {}, loaded, samples num: {}'.format(
                os.getenv('LOCAL_RANK', -1), len(self.data)
            ))

    def __len__(self):
        return self.samples_num#59132213

    def __getitem__(self, index):
        logger.debug('local rank: {}, index: {}'.format(
            os.getenv('LOCAL_RANK', -1), index
        ))
        if index >= self.__len__():
            raise StopIteration
        if not dist.is_initialized():
            stop_border = sum(self.file_lens)
            process_border = stop_border
            # condition = index == sum(self.file_lens)
        else:
            border_temp = sum(self.file_lens)
            range_temp = range(dist.get_rank(), border_temp, dist.get_world_size())
            stop_border = range_temp[-1] + dist.get_world_size()  # kernel line for stopping is adding world size
            process_border = len(range_temp)
        if index > 0 and index == stop_border:
            # while self.process_count.value < sum(self.file_lens):
            while self.process_count.value < process_border:
                logger.debug('local rank: {}, process count: {}, process_border: {}, index: {}, stop_border: {}, '
                             'waiting to change file...'.format(
                    os.getenv('LOCAL_RANK', -1),
                    self.process_count.value, process_border, index, stop_border
                ))
                time.sleep(3)
            self.current_file_index += 1
            logger.debug('local rank: {} start to change file')
            self.load_file(self.current_file_index)
            logger.debug(
                'local rank: {} {} {}'.format(os.getenv('LOCAL_RANK', -1), self.current_file_index, len(self.data)))
        original_index = index
        while True:
            index = original_index - sum(self.file_lens[:-1])
            logger.debug('local rank: {}, original index: {}, index: {}'.format(
                os.getenv('LOCAL_RANK', -1),
                original_index, index
            ))
            try:
                sample = self.data[index]
                break
            except IndexError as e:
                time.sleep(0.5)
        title = sample['title']
        content = sample['content']
        logger.debug('local rank: {}, title: {}'.format(
            os.getenv('LOCAL_RANK', -1), title[:15]
        ))
        text = '\n'.join([title, content])
        token_ids = self.tokenizer.encode(text, bos=True, eos=True)
        self.lock.acquire()
        self.process_count.value += 1
        self.lock.release()
        return token_ids


if __name__ == '__main__':
    from dlx.utils.data.nlp.file_segments_dataloader import FileSegmentsDataloader

    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    wudao_dataset = WuDao(root, Tokenizer())

    dataloader = FileSegmentsDataloader(wudao_dataset)
