import glob
from torch import nn
from dlx.train.llm.trainer import AutoRegressiveTrainer
import os
from dlx.models.llm.llama3 import Transformer, ModelArgs
import torch
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from dlx.utils.data.nlp.wudao import WuDao
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
from dlx.tokenizer.tiktoken import Tokenizer
from dlx.utils.data.nlp.file_segments_dataloader import FileSegmentsDataloader
from dlx.utils.stat import stat_parameters_num
from loguru import logger
import sys
from torch.utils.tensorboard import SummaryWriter
from dlx.utils.data.nlp.wudao import WuDao_Dataset
from dlx.utils.data.collate_funcs import create_collate_fn_normal_batch
from torch.utils.data import DistributedSampler
import random



# logger.add('log.txt')
# torch.autograd.set_detect_anomaly(True)


def get_args():
    args_helper = ArgumentParser()
    args_helper.add_argument('--tensorboard_dir', type=str,
                             default='', help='Defaultly, tensorboard is disabled.')
    args_helper.add_argument('--model_parallel_size', type=int, default=1)
    args_helper.add_argument('--save_folder', type=str,
                             default='models_pretrain_test')
    args_helper.add_argument('--accumulate', type=int, default=1)
    args_helper.add_argument('--eval_log_iters', type=int, default=200000)
    args_helper.add_argument('--resume', action='store_true')
    args_helper.add_argument('--max_length', default=2048, type=int)
    # args_helper.add_argument('--world_size', default=1, type=int)
    args_helper.add_argument('--batch_size', default=32, type=int)
    # args_helper.add_argument('--local_rank', default=-1, type=int)
    args_helper.add_argument('--debug', action='store_true')
    args = args_helper.parse_args()
    return args


def init_parallel(model_parallel_size=None):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

if __name__ == '__main__':
    # set up the arguments
    args = get_args()

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    model_parallel_size = world_size if args.model_parallel_size is None else args.model_parallel_size
    init_parallel(model_parallel_size) if world_size > 1 else ...

    margs = {
        "dim": 512,
        "n_layers": 8,
        "n_heads": 4,
        "n_kv_heads": 2,
        "vocab_size": 128256,
        "multiple_of": 1024,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "max_seq_len": args.max_length,
        "mode": "train"
    }
    margs = ModelArgs(**margs)
    model = Transformer(margs)
    stat_parameters_num(model)

    # others
    tokenizer = Tokenizer()
    optimizer = Adam(model.parameters(), lr=1e-5)
    summary_writer = SummaryWriter(args.tensorboard_dir) if args.tensorboard_dir else None

    # dataloader
    wudao_root = '/dataset/fd5061f6/chinese_data/WuDao'
    fs = glob.glob(wudao_root+'/*')
    random.seed(0)
    random.shuffle(fs)
    val_files = fs[:1]
    train_files = fs[1:2]
    train_dataset = WuDao_Dataset(train_files, tokenizer, args.max_length)
    val_dataset = WuDao_Dataset(val_files, tokenizer, args.max_length)

    train_sampler = DistributedSampler(train_dataset, shuffle=False) if dist.is_initialized() else None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, num_workers=12,
        collate_fn=create_collate_fn_normal_batch(tokenizer, args.max_length),
        sampler=train_sampler
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, num_workers=12,
        collate_fn=create_collate_fn_normal_batch(tokenizer, args.max_length),
        sampler=val_sampler
    )

    trainer = AutoRegressiveTrainer(
        model, train_dataloader,
        optimizer=optimizer,
        world_size=world_size,
        tokenizer=tokenizer,
        model_is_kv_cache_enabled=False,
        ids_dtype=torch.long, parallel='ddp',
        grad_clip=None, device='cuda',
        amp=True, profile_dir=None,
        profile_steps=None,
        vocab_size=margs.vocab_size,
        summary_writer=summary_writer,
        resume=args.resume,
        save_iters=80000,
        save_folder=args.save_folder,
        accumulate_iters=args.accumulate,
        eval_dataloader=val_dataloader,
        eval_log_iters=args.eval_log_iters
    )
    trainer.start()
