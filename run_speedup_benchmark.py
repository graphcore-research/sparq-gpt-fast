"""Benchmarks the speed of SparQ relative to the dense baseline."""

# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.


from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from benchmark import Benchmark, run_or_load
from sparq import RKForCompressionRatio, SparQArgs
from theoretical_speedups import h100, speedup_theoretical_time_in_attn

expected_gpu = "H100"

base_config = dict(model="llama27bchat", quant=None, compile=True, gpu=expected_gpu)
prompt_lengths = [2**x for x in [12, 13, 14, 15]]

benchmarks = {}
benchmarks["dense"] = [
    Benchmark(**base_config, attention="dense", prompt_length=prompt_length)
    for prompt_length in prompt_lengths
]
benchmarks["SparQ (1/8)"] = [
    Benchmark(
        **base_config,
        attention="sparq",
        prompt_length=prompt_length,
        sparq=SparQArgs(rk=RKForCompressionRatio(8)),
    )
    for prompt_length in prompt_lengths
]
results = {k: [run_or_load(b) for b in bs] for k, bs in benchmarks.items()}

fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
time_ax: Axes = axes[0]
speedup_ax: Axes = axes[1]
labels_done = False

for i, (label, rs) in enumerate(results.items()):
    color = f"C{i}"
    means = [r.secs_per_token_mean for r in rs]
    stds = [r.secs_per_token_std for r in rs]
    time_ax.errorbar(prompt_lengths, means, yerr=stds, label=label, color=color)

    if label != "dense":
        speedups = [
            dense_r.secs_per_token_mean / sparq_t
            for dense_r, sparq_t in zip(results["dense"], means)
        ]
        speedup_ax.plot(prompt_lengths, speedups, label=label, color=color)

        print(label)
        for prompt_length, speedup in zip(prompt_lengths, speedups):
            print(f"{prompt_length}, {speedup:.2f}")

    if "dense" not in label:
        (theoretical_speedup_line,) = speedup_ax.plot(
            prompt_lengths,
            [
                speedup_theoretical_time_in_attn(
                    b, sr, platform=h100, model_config_name="7B"
                )
                for b, sr in zip(benchmarks[label], rs)
            ],
            color=color,
            linestyle="--",
            label="estimated theoretical max",
        )


time_ax.set_xlabel("prompt length")
time_ax.set_ylabel("secs per token")
time_ax.set_xlim(left=min(prompt_lengths))
time_ax.set_ylim(bottom=0)
time_ax.legend()

speedup_ax.axhline(1.0, color="C0")
speedup_ax.set_xlabel("prompt length")
speedup_ax.set_ylabel("speedup over dense")
speedup_ax.set_xlim(left=min(prompt_lengths))
speedup_ax.legend(handles=[theoretical_speedup_line])

plt.tight_layout()
figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
plt.savefig(figure_dir / "speedup_benchmark.png")
plt.close()
