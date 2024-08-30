# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.

import torch

from sparq import (
    RKForCompressionRatio,
    RunningVMean,
    SparQArgs,
    SparQAttention,
    get_r_k_for_compression_ratio,
)


def test__prefill_and_generate_one__does_not_crash() -> None:
    batch_size = 1
    n_heads = 8
    prompt_len = 14
    max_seq_len = 16
    head_dim = 128
    dtype = torch.bfloat16

    def run_config(**kwargs) -> None:
        attention = SparQAttention(
            SparQArgs(implementation="torch", rk=RKForCompressionRatio(2), **kwargs),
            n_head=n_heads,
            n_local_heads=n_heads,
        )
        attention.setup_caches(batch_size, max_seq_len, n_heads, head_dim)

        Q = torch.randn((batch_size, n_heads, prompt_len, head_dim), dtype=dtype)
        K = torch.randn((batch_size, n_heads, prompt_len, head_dim), dtype=dtype)
        V = torch.randn((batch_size, n_heads, prompt_len, head_dim), dtype=dtype)
        input_pos = torch.arange(0, prompt_len)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))[
            None, None, input_pos
        ]
        attention(Q, K, V, mask, input_pos, prefill=True)

        q = torch.randn((batch_size, n_heads, 1, head_dim), dtype=dtype)
        k = torch.randn((batch_size, n_heads, 1, head_dim), dtype=dtype)
        v = torch.randn((batch_size, n_heads, 1, head_dim), dtype=dtype)
        input_pos = torch.tensor([prompt_len])
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))[
            None, None, input_pos
        ]
        attention(q, k, v, mask, input_pos, prefill=False)

    run_config(reallocation=True, running_V_mean=False)
    run_config(reallocation=True, running_V_mean=True)
    run_config(reallocation=False, running_V_mean=False)
    run_config(reallocation=False, running_V_mean=True)


def test__store_K_once_vs_twice__results_equal() -> None:
    batch_size = 1
    n_heads = 8
    prompt_len = 14
    max_seq_len = 16
    head_dim = 128
    dtype = torch.bfloat16

    single_K = SparQAttention(
        SparQArgs(implementation="torch", rk=RKForCompressionRatio(2), K_mode="as_is"),
        n_head=n_heads,
        n_local_heads=n_heads,
    )
    double_K = SparQAttention(
        SparQArgs(
            implementation="torch", rk=RKForCompressionRatio(2), K_mode="store_twice"
        ),
        n_head=n_heads,
        n_local_heads=n_heads,
    )

    single_K.setup_caches(batch_size, max_seq_len, n_heads, head_dim)
    double_K.setup_caches(batch_size, max_seq_len, n_heads, head_dim)

    Q = torch.randn((batch_size, n_heads, prompt_len, head_dim), dtype=dtype)
    K = torch.randn((batch_size, n_heads, prompt_len, head_dim), dtype=dtype)
    V = torch.randn((batch_size, n_heads, prompt_len, head_dim), dtype=dtype)
    input_pos = torch.arange(0, prompt_len)
    mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))[
        None, None, input_pos
    ]
    prefill_result_1 = single_K(Q, K, V, mask, input_pos, prefill=True)
    prefill_result_2 = double_K(Q, K, V, mask, input_pos, prefill=True)
    assert torch.allclose(prefill_result_1, prefill_result_2)

    q = torch.randn((batch_size, n_heads, 1, head_dim), dtype=dtype)
    k = torch.randn((batch_size, n_heads, 1, head_dim), dtype=dtype)
    v = torch.randn((batch_size, n_heads, 1, head_dim), dtype=dtype)
    input_pos = torch.tensor([prompt_len])
    mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))[
        None, None, input_pos
    ]
    generate_result_1 = single_K(q, k, v, mask, input_pos, prefill=False)
    generate_result_2 = double_K(q, k, v, mask, input_pos, prefill=False)
    assert torch.allclose(generate_result_1, generate_result_2)


class TestRunningVMean:
    def test__init_and_update__returns_correct_mean(self) -> None:
        batch_size = 1
        n_heads = 8
        prompt_len = 14
        head_dim = 128

        V_mean = RunningVMean(batch_size, n_heads, head_dim)
        prompt_V = torch.randn(batch_size, n_heads, prompt_len, head_dim)
        V_mean.init(prompt_V)
        generate_1_v = torch.randn(batch_size, n_heads, 1, head_dim)
        V_mean_1 = V_mean.update(generate_1_v)
        generate_2_v = torch.randn(batch_size, n_heads, 1, head_dim)
        V_mean_2 = V_mean.update(generate_2_v)

        assert torch.allclose(
            V_mean_1,
            torch.concatenate([prompt_V, generate_1_v], dim=-2).mean(
                dim=-2, keepdim=True
            ),
        )
        assert torch.allclose(
            V_mean_2,
            torch.concatenate([prompt_V, generate_1_v, generate_2_v], dim=-2).mean(
                dim=-2, keepdim=True
            ),
        )

    def test__update__returns_correct_dtype(self) -> None:
        batch_size = 1
        n_heads = 8
        prompt_len = 14
        head_dim = 128
        dtype = torch.bfloat16

        V_mean = RunningVMean(batch_size, n_heads, head_dim)
        prompt_V = torch.randn(batch_size, n_heads, prompt_len, head_dim, dtype=dtype)
        V_mean.init(prompt_V)
        generate_1_v = torch.randn(batch_size, n_heads, 1, head_dim, dtype=dtype)
        V_mean_1 = V_mean.update(generate_1_v)

        assert V_mean_1.dtype == torch.bfloat16


def test__get_k1_k2_for_compression_ratio__returns_correct_ratio() -> None:
    def get_ratio(S: int, d: int, r: int, k: int) -> float:
        sparq_transfers = S * r + 2 * k * d + 4 * d
        dense_transfers = 2 * S * d + 2 * d
        return round(sparq_transfers / dense_transfers, ndigits=3)

    d = 128
    S = 10_000
    assert get_ratio(S, d, *get_r_k_for_compression_ratio(2, S, d)) == 0.5
    assert get_ratio(S, d, *get_r_k_for_compression_ratio(4, S, d)) == 0.25
    assert get_ratio(S, d, *get_r_k_for_compression_ratio(8, S, d)) == 0.125
    S = 20_000
    assert get_ratio(S, d, *get_r_k_for_compression_ratio(2, S, d)) == 0.5
    assert get_ratio(S, d, *get_r_k_for_compression_ratio(4, S, d)) == 0.25
    assert get_ratio(S, d, *get_r_k_for_compression_ratio(8, S, d)) == 0.125
