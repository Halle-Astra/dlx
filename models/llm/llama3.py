# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
from dlx.utils.time import timer
from loguru import logger


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    mode: str = 'infer'


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class TempLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        remove_keys = ['input_is_parallel', 'gather_output', 'init_method']
        for k in remove_keys:
            if k in kwargs:
                del kwargs[k]

        super(TempLinear, self).__init__(*args, **kwargs)
        # self.forward = super().forward


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size() if torch.distributed.is_initialized() else 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.args = args

        wq_linear = TempLinear if model_parallel_size == 1 else ColumnParallelLinear
        wk_linear = TempLinear if model_parallel_size == 1 else ColumnParallelLinear
        wv_linear = TempLinear if model_parallel_size == 1 else ColumnParallelLinear
        wo_linear = TempLinear if model_parallel_size == 1 else RowParallelLinear


        self.wq = wq_linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=torch.nn.init.kaiming_uniform_,
        )
        self.wk = wk_linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=torch.nn.init.kaiming_uniform_,
        )
        self.wv = wv_linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=torch.nn.init.kaiming_uniform_,
        )
        self.wo = wo_linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=torch.nn.init.kaiming_uniform_,
        )

        if self.args.mode == 'train':
            self.cache_k=None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )#.cuda()
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )#.cuda()

    def reset_kv_cache(self):
        self.cache_k = torch.zeros(
            (
                self.args.max_batch_size,
                self.args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            device = self.cache_k.device
        )
        self.cache_v = torch.zeros(
            (
                self.args.max_batch_size,
                self.args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            device=self.cache_v.device
        )
        # print("成功reset")

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            index_in_batch=None
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.args.mode == 'infer':
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            if index_in_batch is None:
                self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
                self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv
                keys = self.cache_k[:bsz, : start_pos + seqlen]
                values = self.cache_v[:bsz, : start_pos + seqlen]
            else:
                self.cache_k[index_in_batch, start_pos: start_pos + seqlen] = xk
                self.cache_v[index_in_batch, start_pos: start_pos + seqlen] = xv
                keys = self.cache_k[index_in_batch, : start_pos + seqlen]
                values = self.cache_v[index_in_batch, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        model_parallel_size = fs_init.get_model_parallel_world_size() if torch.distributed.is_initialized() else 1
        w1_linear = TempLinear if model_parallel_size == 1 else ColumnParallelLinear
        w2_linear = TempLinear if model_parallel_size == 1 else RowParallelLinear
        w3_linear = TempLinear if model_parallel_size == 1 else ColumnParallelLinear

        self.w1 = w1_linear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=torch.nn.init.kaiming_uniform_
        )
        self.w2 = w2_linear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=torch.nn.init.kaiming_uniform_
        )
        self.w3 = w3_linear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=torch.nn.init.kaiming_uniform_
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            index_in_batch=None
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, index_in_batch)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class TempEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        if 'init_method' in kwargs:
            del kwargs['init_method']
        super(TempEmbedding, self).__init__(*args, **kwargs)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        model_parallel_size = fs_init.get_model_parallel_world_size() if torch.distributed.is_initialized() else 1
        embedding_layer = TempEmbedding if model_parallel_size == 1 else VocabParallelEmbedding
        output_linear = TempLinear if model_parallel_size == 1 else ColumnParallelLinear

        self.tok_embeddings = embedding_layer(
            params.vocab_size, params.dim, init_method=torch.nn.init.kaiming_uniform_
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = output_linear(
            params.dim, params.vocab_size, bias=False, init_method=torch.nn.init.kaiming_uniform_
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos=None, index_in_batch=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        _time_begin_generate_mask = timer.mark()
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        _time_end_generate_mask = timer.mark()
        logger.debug(f'time of generate mask: {_time_end_generate_mask - _time_begin_generate_mask}')

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, index_in_batch)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_kv_cache()
