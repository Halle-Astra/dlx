from torch import nn
from dlx.train.llama3.trainer import Llama3Trainer
import os
from dlx.models.llm.llama3 import Transformer, ModelArgs
import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
model_parallel_size = 1
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
if not model_parallel_is_initialized():
    if model_parallel_size is None:
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(model_parallel_size)

args = {
   "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "multiple_of": 1024,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0
}
args = ModelArgs(**args)

ckpt_path = 'Meta-Llama-3-8B-Instruct/consolidated_instruct.00.pth'
weights = torch.load(ckpt_path, map_location="cpu")
model = Transformer(args)
model.load_state_dict(weights)
print(weights['tok_embeddings'])
print(model)