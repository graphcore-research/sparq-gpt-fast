# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kv_cache import KVCache


@dataclass(frozen=True)
class RKForCompressionRatio:
    """Config option to set r and k to achieve to specified compression ratio.

    i.e. ratio = 8 means SparQ will transfer ~1/8 of the data that would be transferred
    by dense.
    """

    ratio: int = 8


@dataclass(frozen=True)
class SparQArgs:
    rk: RKForCompressionRatio = RKForCompressionRatio(ratio=8)

    reallocation: bool = True
    running_V_mean: bool = True
    K_mode: Literal["store_once", "store_twice"] = "store_twice"
    # Sorting the output of the top-k takes time, but might result in more contiguous
    # memory accesses. In our experiments we found it was faster not to sort.
    sort_stage_1_top_k: bool = False
    sort_stage_2_top_k: bool = False


def get_r_k_for_compression_ratio(
    ratio: int, sequence_length: int, head_dim: int
) -> tuple[int, int]:
    """Gets r, k to reduce memory transferred during attention by the given ratio."""
    r = round(head_dim / ratio)
    k = round(sequence_length / (2 * ratio))
    return r, k


class SparQAttention(nn.Module):
    def __init__(self, config: SparQArgs, n_head: int, n_local_heads: int) -> None:
        super().__init__()
        self.config = config
        self.n_head = n_head
        self.n_local_heads = n_local_heads
        self.kv_cache: DoubleKVCache | SingleKVCache | None = None
        self.V_mean: RunningVMean | None = None
        self.r: int | None = None
        self.k: int | None = None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
        input_pos: Tensor | None,
        prefill: bool,
    ) -> Tensor:
        if self.kv_cache is not None:
            K1, K2, V = self.kv_cache.update(input_pos, k, v)
        K1 = K1.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        K2 = K2.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        V = V.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        #    q: [batch x n heads         x target seq len x head embed dim]
        #    K: [batch x n grouped heads x source seq len x head embed dim]
        #    V: [batch x n grouped heads x source seq len x head embed dim]
        # mask: [1     x 1               x 1              x source seq len]

        if prefill:
            return self._prefill(q, K2, V, mask)
        else:
            return self._generate(q, K1, K2, V, v, mask)

    def _prefill(self, q: Tensor, K: Tensor, V: Tensor, mask: Tensor) -> Tensor:
        if self.config.reallocation and self.config.running_V_mean:
            self.V_mean.init(V)
        return F.scaled_dot_product_attention(q, K, V, mask)

    def _generate(
        self, q: Tensor, K1: Tensor, K2: Tensor, V: Tensor, v: Tensor, mask: Tensor
    ) -> Tensor:
        if self.config.reallocation and self.config.running_V_mean:
            V_mean = self.V_mean.update(v)
        elif self.config.reallocation and not self.config.running_V_mean:
            V_mean = _masked_V_mean(V, mask)
        else:
            V_mean = None

        assert self.r is not None and self.k is not None
        return sparq_attn(q, K1, K2, V, V_mean, mask, self.r, self.k, self.config)

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
    ) -> None:
        if self.config.K_mode == "store_twice":
            self.kv_cache = DoubleKVCache(
                max_batch_size, max_seq_length, n_heads, head_dim, dtype
            )
        elif self.config.K_mode == "store_once":
            self.kv_cache = SingleKVCache(
                max_batch_size, max_seq_length, n_heads, head_dim, dtype
            )

        self.V_mean = RunningVMean(max_batch_size, n_heads, head_dim)
        self.r, self.k = get_r_k_for_compression_ratio(
            self.config.rk.ratio, max_seq_length, head_dim
        )


class RunningVMean(nn.Module):
    """Maintains a running mean of V over sequence length.

    FIXME: As the mean is accumulated for each token generated and reset of prefill,
    this implementation is only correct if the nth token is only generated once per
    prefill. The rest of gpt-fast doesn't enforce this, you can set input_pos to
    generate the nth token as many times as you like, but we ignore this problem for
    now.
    """

    def __init__(self, max_batch_size: int, n_local_heads: int, head_dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "V_mean",
            torch.full(
                (max_batch_size, n_local_heads, 1, head_dim),
                float("nan"),
                dtype=torch.float32,
            ),
        )
        self.register_buffer("n", torch.tensor(0))

    def init(self, V: Tensor) -> None:
        self.V_mean[:, :, :, :] = V.mean(-2, dtype=torch.float32, keepdim=True)
        self.n.zero_().add_(V.shape[-2])

    def update(self, v: Tensor) -> Tensor:
        V_mean = (self.n * self.V_mean + v) / (self.n + 1)
        self.V_mean[:, :, :, :] = V_mean
        self.n.add_(1)
        return V_mean.to(v.dtype)


def _masked_V_mean(V: Tensor, mask: Tensor) -> Tensor:
    value_mask = mask.transpose(-2, -1)
    V_sum = (V * value_mask).sum(-2, dtype=torch.float32, keepdim=True)
    V_mean = V_sum / value_mask.sum(-2, dtype=torch.float32, keepdim=True)
    return V_mean.to(V.dtype)


class DoubleKVCache(nn.Module):
    """KV cache that stores both K and K transpose."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype,
    ) -> None:
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer(
            "kt_cache",
            torch.zeros(cache_shape, dtype=dtype).transpose(-1, -2).contiguous(),
        )
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out: Tensor = self.k_cache
        kt_out: Tensor = self.kt_cache
        v_out: Tensor = self.v_cache
        k_out[:, :, input_pos] = k_val
        kt_out[:, :, :, input_pos] = k_val.transpose(-1, -2)
        v_out[:, :, input_pos] = v_val

        return kt_out.transpose(-1, -2), k_out, v_out


class SingleKVCache(nn.Module):
    """Just wraps the default KV cache so it has the same interface as DoubleKVCache."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype,
    ) -> None:
        super().__init__()
        self.cache = KVCache(max_batch_size, max_seq_length, n_heads, head_dim, dtype)

    def update(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        K, V = self.cache.update(input_pos, k_val, v_val)
        return K, K, V


def sparq_attn(
    Q: Tensor,
    K1: Tensor,
    K2: Tensor,
    V: Tensor,
    V_mean: Tensor | None,
    mask: Tensor,
    r: int,
    k: int,
    config: SparQArgs,
) -> Tensor:
    # 1. Approximate attention scores using r largest components of Q
    absQ = torch.abs(Q)
    absQ_hat, i1 = torch.topk(absQ, r, dim=-1, sorted=config.sort_stage_1_top_k)
    QK_hat = _gather(Q, -1, i1) @ _gather(K1, -1, i1).transpose(-1, -2)
    masked_QK_hat = torch.where(mask, QK_hat, float("-inf"))
    scale = torch.sqrt(
        Q.shape[-1]
        * absQ_hat.sum(dim=-1, keepdim=True)
        / absQ.sum(dim=-1, keepdim=True)
    )
    s_hat = _scaled_softmax(masked_QK_hat, scale, dim=-1)

    # 2. Gather top k2 positions based on approximate attention scores & run attention
    # This min ensures that k <= sequence length, otherwise torch.compile() will crash.
    k = min(k, V.shape[-2])
    s_hat_i2, i2 = torch.topk(s_hat, k, dim=-1, sorted=config.sort_stage_2_top_k)
    iKV = i2[..., 0, :, None]
    QK = Q @ _gather(K2, -2, iKV).transpose(2, 3)
    masked_QK = torch.where(_gather(mask.expand_as(QK_hat), -1, i2), QK, float("-inf"))
    s = _scaled_softmax(masked_QK, Q.shape[-1] ** 0.5, dim=-1)
    y_ = s @ _gather(V, -2, iKV)

    # 3. Estimate the total score of the top k, and interpolate with V_mean
    if V_mean is not None:
        return torch.lerp(V_mean, y_, s_hat_i2.sum(-1, keepdim=True))
    else:
        return y_


def _gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


@torch.compile(disable=not torch.cuda.is_available())
def _scaled_softmax(x: Tensor, divscale: Tensor | float, dim: int) -> Tensor:
    return torch.softmax(x / divscale, dim=dim)
