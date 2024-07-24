from torch import nn
from dlx.train.llm.trainer import AutoRegressiveTrainer
import os
from dlx.models.llm.llama3 import Transformer, ModelArgs
import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from dlx.data.datasets.nlp.wudao import WuDao
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
from dlx.tokenizer.tiktoken import Tokenizer


args = {
    "dim": 4096,
    "n_layers": 3,
    "n_heads": 2,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "multiple_of": 1024,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0
}
margs = ModelArgs(**args)

if __name__ == '__main__':
    # ddp setting

    # dataloader
    wudao_root = '/dataset/fd5061f6/chinese_data/WuDao'
    train_dataloader = WuDao(wudao_root)

    # model
    ckpt_path = '~/.cache/dlx/Meta-Llama-3-8B-Instruct/consolidated_instruct.00.pth'
    weights = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(weights, dict):
        te_weight = {'tok_embeddings.weight': weights}
    model = Transformer(margs)
    model.load_state_dict(weights)
    model = DDP(model)

    # others
    optimizer = Adam()
    loss_func = CrossEntropyLoss()
    tokenizer = Tokenizer()

    trainer = AutoRegressiveTrainer(
        model, train_dataloader, loss_func, optimizer,
        2, tokenizer, True, torch.float32
    )
    trainer.start()
