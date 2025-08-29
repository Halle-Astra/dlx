import os
import glob
import json
import numpy as np
import torch
import tqdm
from datasets import load_dataset
from multiprocessing import Value, RLock, Manager
import time
import random
from loguru import logger
import torch.distributed as dist
from datasets.formatting.formatting import LazyRow


class HF_Dataset:
    def __init__(self, root, tokenizer,
                 max_length=2048,
                 shuffle=False,
                 random_seed=40,
                 samples_num=0,
                 # drop_short=False,
                 split='train',
                 extract_text_func=None,
                 enable_dsmap=False
                 ):
        """
        Provide an init func of Huggingface Dataset subject to users needn't write dataset.map logics again.
        It will use 80% of cpu cores to process dataset mapping functions.
        :param root:
        :param tokenizer:
        :param max_length:
        :param shuffle:
        :param random_seed:
        :param samples_num:
        :param split:
        """
        self.root = root
        self.tokenizer = tokenizer
        self._n_cpu4map = int(0.8*os.cpu_count())
        self.dataset = load_dataset(root, split=split, trust_remote_code=True)
        if shuffle:
            random.seed(random_seed)
            self.dataset = self.dataset.shuffle(seed=random_seed)
            # dataset = dataset.flatten_indices()

        self.samples_num = len(self.dataset)
        logger.info('local rank: {}, samples_num: {}, dataset root: {}'.format(
            os.getenv('LOCAL_RANK', -1), samples_num, root
        ))

        self.enable_dsmap = enable_dsmap
        self.extract_text_func = extract_text_func or self.extract_text_func
        if self.enable_dsmap:
            self.dataset = self.dataset.map(self.extract_text_func, num_proc=self._n_cpu4map)
            self.dataset = self.dataset.map(self.tokenize, num_proc=self._n_cpu4map)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.enable_dsmap:
            token_ids = sample['token_ids']
        else:
            sample = self.extract_text_func(sample)
            token_ids = self.tokenize(sample['input'])
        return token_ids

    def tokenize(self, sample_or_text):
        if isinstance(sample_or_text, dict):
            text = sample_or_text['input']
        elif isinstance(sample_or_text, str):
            text = sample_or_text
        else:
            assert False, 'Type of sample gonna be tokenized is wrong.'
        token_ids = self.tokenizer.encode(text, bos=True, eos=True)
        if self.enable_dsmap:
            return {'token_ids': token_ids}
        else:
            return token_ids


    def extract_text_func(self, sample):
        def get_random_sep():
            seps = [
                '\n', '  ', '\n\n',
                '\n{}\n'.format(
                    random.choice('=-+_@#$%') * random.randint(3, 40)
                )
            ]
            sep = random.choice(seps)
            return sep

        inputs = []
        if isinstance(sample, LazyRow) or isinstance(sample, dict):
            if isinstance(sample, LazyRow) and \
                    hasattr(sample, 'features') and \
                    len(sample.features) == 1:
                key = list(dict(sample.features).keys())[0]
            else:
                key = list(sample.keys())[0]

            value = sample[key]
            assert isinstance(value, str), f'sample of {self.root} have no string.'
            return {'input': value}

        else:
            if 'prompt' in sample:
                inputs.append(sample['prompt'])
            if 'input' in sample:
                inputs.append(sample['input'])
            if len(inputs) == 0:
                logger.error('dataset root {} samples have no prompt and input keys.\nsample keys: {}'.format(
                    self.root, list(sample.keys())
                ))
            elif len(inputs) == 1:
                text = inputs[0]
            else:
                text = get_random_sep().join(inputs)
            return {'input': text}


if __name__ == "__main__":
    from dlx.tokenizer.tiktoken import Tokenizer

    root = '/workspace/downloads/Skylion007/openwebtext'
    tokenizer = Tokenizer()
    dataset = HF_Dataset(root, tokenizer, shuffle=True)
    for example in dataset:
        print(example)
