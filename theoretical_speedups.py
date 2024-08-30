"""Estimates the speedup that SparQ could achieve under ideal conditions.

The caluculation is based on computing
speedup of attention = transfers during dense attn / transfers during sparq attn,
estimating the time spent in attention compared to in the rest of the model,
and using this to convert the speedup to attention to an overall speedup.
Thus, this assumes that attention is fully bottlenecked by memory bandwidth.
"""

# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from benchmark import Benchmark, Results
from model import ModelArgs, transformer_configs


@dataclass
class XPU:
    name: str
    bytes_per_sec: int
    flop_per_sec: int
    capacity: float


# Numbers for fp16
a10 = XPU(name="A10", bytes_per_sec=600e9, flop_per_sec=125e12, capacity=24e9)
a10g = XPU(name="A10G", bytes_per_sec=600e9, flop_per_sec=31.52e12, capacity=24e9)
a100 = XPU(name="A100", bytes_per_sec=1560e9, flop_per_sec=312e12, capacity=40e9)
h100 = XPU(name="H100", bytes_per_sec=2040e9, flop_per_sec=756e12, capacity=80e9)


def speedup_theoretical_time_in_attn(
    b: Benchmark,
    sparq_results: Results,
    platform: XPU,
    model_config_name: str,
) -> float:
    """Estimates the speedup that SparQ will achieve.

    To estimate the time in attention, uses the theoretical FLOP count + memory
    transfers of attention vs the rest of the model. Assumes 2 byte parameters and kv
    cache.
    """
    model_config = ModelArgs(**transformer_configs[model_config_name])
    attn_flops, attn_mem = _get_attn_perf_numbers(
        batch_size=1,
        seq_len=b.prompt_length + 1,
        hidden_dim=model_config.dim,
        kv_group_size=model_config.n_head // model_config.n_local_heads,
        n_layers=model_config.n_layer,
        bytes_per_kv=2.0,
    )
    flops, mem = _get_perf_numbers(
        batch_size=1,
        seq_len=b.prompt_length + 1,
        hidden_dim=model_config.dim,
        kv_group_size=model_config.n_head // model_config.n_local_heads,
        n_layers=model_config.n_layer,
        vocab_size=model_config.vocab_size,
        bytes_per_param=2.0,
        bytes_per_kv=2.0,
    )
    attn_time = attn_flops / platform.flop_per_sec + attn_mem / platform.bytes_per_sec
    total_time = flops / platform.flop_per_sec + mem / platform.bytes_per_sec
    dense_frac_in_attention = attn_time / total_time
    return _get_speedup(
        b, sparq_results, model_config.head_dim, dense_frac_in_attention
    )


def speedup_measured_time_in_attn(
    b: Benchmark,
    dense_results: Results,
    sparq_results: Results,
    head_dim: int,
    secs_not_in_attention: float,
) -> float:
    """Estimates the speedup that SparQ will achieve.

    Uses the given secs_not_in_attention to calculate the fraction of the total time
    spent in attention.
    """
    dense_secs_per_token = dense_results.secs_per_token_mean
    dense_frac_in_attention = (
        dense_secs_per_token - secs_not_in_attention
    ) / dense_secs_per_token
    return _get_speedup(b, sparq_results, head_dim, dense_frac_in_attention)


def _get_speedup(
    b: Benchmark,
    sparq_results: Results,
    head_dim: int,
    dense_frac_in_attention: float,
) -> float:
    r, k = sparq_results.sparq_r, sparq_results.sparq_k
    sparq_transfers = b.prompt_length * r + 2 * k * head_dim + 4 * head_dim
    dense_transfers = 2 * b.prompt_length * head_dim + 2 * head_dim
    attention_speedup = dense_transfers / sparq_transfers
    frac_saved = dense_frac_in_attention - (dense_frac_in_attention / attention_speedup)
    return 1 / (1 - frac_saved)


def _get_perf_numbers(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    kv_group_size: int,
    n_layers: int = 1,
    bytes_per_param: float = 2,
    bytes_per_kv: float = 2,
    vocab_size: int | None = None,
) -> tuple[int, int]:
    """Get FLOP count and memory transfer in bytes for a single generation step."""
    if vocab_size is None:
        vocab_size = 0
    embed_params = hidden_dim * vocab_size
    model_params = n_layers * 12 * hidden_dim**2

    attn_flops, attn_mem = _get_attn_perf_numbers(
        batch_size, seq_len, hidden_dim, kv_group_size, n_layers, bytes_per_kv
    )

    # Multiply + add per param, count output projection as well
    flops = 2 * batch_size * (model_params + embed_params) + attn_flops

    memory_transfer = (model_params + 2 * embed_params) * bytes_per_param + attn_mem

    return flops, memory_transfer


def _get_attn_perf_numbers(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    kv_group_size: int,
    n_layers: int = 1,
    bytes_per_kv: float = 2,
) -> tuple[int, int]:
    n_kv_elements = batch_size * n_layers * 2 * seq_len * hidden_dim // kv_group_size

    flops = 2 * n_kv_elements * kv_group_size
    memory_transfer = n_kv_elements * bytes_per_kv

    return flops, memory_transfer
