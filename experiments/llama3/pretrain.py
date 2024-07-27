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
from dlx.utils.data.nlp.wudao_draft import WuDao
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
from dlx.tokenizer.tiktoken import Tokenizer

#torch.autograd.set_detect_anomaly(True)

args = {
    "dim": 4,
    "n_layers": 3,
    "n_heads": 2,
    "n_kv_heads": 2,
    "vocab_size": 128256,
    "multiple_of": 1024,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "max_seq_len": 2048
}
margs = ModelArgs(**args)

if __name__ == '__main__':
    # ddp setting
    model_parallel_size = 1
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)
    torch.cuda.set_device(dist.get_rank())
    print(f'当前的rank为{dist.get_rank()}')
    print(f'当前的world_size为{dist.get_world_size()}')


    # dataloader
    wudao_root = '/dataset/fd5061f6/chinese_data/WuDao'
    train_dataloader = WuDao(wudao_root, num_worker=1, batch_size=32, max_seq_len=args.max_seq_len)

    # model
    ckpt_path = '/root/.cache/dlx/Meta-Llama-3-8B-Instruct/consolidated_instruct.00.pth'
    # weights = torch.load(ckpt_path, map_location="cpu")
    # if not isinstance(weights, dict):
    #     te_weight = {'tok_embeddings.weight': weights}
    model = Transformer(margs).cuda()
    # model.load_state_dict(weights)
    model = DDP(model, broadcast_buffers=False)

    # others
    optimizer = Adam(model.parameters(), lr=1e-5)
    loss_func = CrossEntropyLoss()
    tokenizer = Tokenizer()

    trainer = AutoRegressiveTrainer(
        model, train_dataloader, loss_func, optimizer,
        2, tokenizer, True, dtype=torch.long, parallel='ddp',
        grad_clip=None, device='cuda'
    )
    trainer.start()
