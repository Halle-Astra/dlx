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
import mlflow

# logger.remove()
# logger.add(sys.stderr, level='INFO')

# torch.autograd.set_detect_anomaly(True)

args = {
    "dim": 512,
    "n_layers": 8,
    "n_heads": 1,
    "n_kv_heads": 1,
    "vocab_size": 128256,
    "multiple_of": 1024,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "max_seq_len": 150,
    "mode": "train"
}
margs = ModelArgs(**args)


def get_args():
    args_helper = ArgumentParser()
    args_helper.add_argument('--tensorboard_dir', type=str,
                             default='', help='Defaultly, tensorboard is disabled.')
    args_helper.add_argument('--model_parallel_size', type=int, default=1)
    args_helper.add_argument('--save_folder', type=str,
                             default='models_pretrain')
    args_helper.add_argument('--resume_folder', type=str, default='')
    args = args_helper.parse_args()
    return args


if __name__ == '__main__':
    # set up the arguments
    args = get_args()

    # mlflow setting
    mlflow.set_tracking_uri(uri="http://ai-universe.cn:7516")
    cli = mlflow.MlflowClient()
    run = cli.create_run('0')
    mlflow.set_experiment("Llama3 pretraining")
    some_trainer_arguments = dict(
        model_is_kv_cache_enabled=False,
        ids_dtype=torch.long, parallel='ddp',
        grad_clip=None, device='cuda',
        amp=True, profile_dir=None,
        profile_steps=None,
        vocab_size=margs.vocab_size,
        world_size=1,

    )
    dataset_args = dict(
        num_worker=24, batch_size=32,
    )
    lr=5e-5
    cli.log_params(dict(
        model_args=margs,
        trainer_args=some_trainer_arguments,
        dataset_args=dataset_args,
        script_args=dict(args),
        lr=lr
    ))

    # tokenizer
    tokenizer = Tokenizer()

    # dataloader
    wudao_root = '/dataset/fd5061f6/chinese_data/WuDao'
    train_dataset = WuDao(wudao_root, tokenizer)
    train_dataloader = FileSegmentsDataloader(train_dataset, **dataset_args)
    # train_dataloader = ((torch.randint(100, (48, 150,), dtype=torch.long, device='cuda'),
    #                     torch.randint(100, (48, 150), dtype=torch.long, device='cuda')[:,1:].flatten(),
    #                      dict(start_pos=0))
    #                     for i in range(100000000))

    # model
    # ckpt_path = '/root/.cache/dlx/Meta-Llama-3-8B-Instruct/consolidated_instruct.00.pth'
    # weights = torch.load(ckpt_path, map_location="cpu")
    # if not isinstance(weights, dict):
    #     te_weight = {'tok_embeddings.weight': weights}
    model = Transformer(margs).cuda()
    # model.load_state_dict(weights)
    # model = DDP(model, broadcast_buffers=False)
    stat_parameters_num(model)

    # others
    optimizer = Adam(model.parameters(), lr=lr)
    tokenizer = Tokenizer()

    trainer = AutoRegressiveTrainer(
        model, train_dataloader,
        optimizer=optimizer,
        tokenizer=tokenizer,cli=cli,
        **some_trainer_arguments
    )
    trainer.start()
