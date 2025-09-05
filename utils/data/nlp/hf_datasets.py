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


class HF_Dataset:
    def __init__(self, root, tokenizer,
                 max_length=2048,
                 shuffle=False,
                 random_seed=40,
                 samples_num=0,
                 # drop_short=False,
                 split='train',
                 extract_text_func=None,
                 ):
        """
        Provide an init func of Huggingface Dataset subject to users needn't write dataset.map logics again.
        :param root:
        :param tokenizer:
        :param max_length:
        :param shuffle:
        :param random_seed:
        :param samples_num:
        :param split:
        """
        self.root = root

        dataset = load_dataset(root, split=split, trust_remote_code=True)
        self.shuffle = shuffle
        if self.shuffle:
            random.seed(random_seed)
            dataset = dataset.shuffle(seed=random_seed)
            # dataset = dataset.flatten_indices()
        self.samples_num = len(dataset)
        logger.info('local rank: {}, samples_num: {}, dataset root: {}'.format(
            os.getenv('LOCAL_RANK', -1), self.samples_num, root
        ))

        # dataset = dataset.map(extract_text_func or self.extract_text_func, batched=True)
        return dataset

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
        if list(sample.keys()) == ['text']:
            return sample

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
            return {'text': text}


if __name__ == "__main__":
    from dlx.tokenizer.tiktoken import Tokenizer

    root = '/workspace/downloads/Skylion007/openwebtext'
    tokenizer = Tokenizer()
    dataset = HF_Dataset(root, tokenizer, shuffle=True)
    for example in dataset:
        print(example)
