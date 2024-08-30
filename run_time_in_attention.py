"""Estimates the time the dense model spends in attention at different seq lengths."""

# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.

from pathlib import Path

import matplotlib.pyplot as plt

from benchmark import Benchmark, run_or_load
from sparq import SparQArgs

base_config = dict(model="llama27bchat", quant=None, compile=True, gpu="H100")
prompt_lengths = [1, 10_000, 20_000, 30_000, 40_000]

results = {}
results["dense"] = [
    run_or_load(
        Benchmark(**base_config, attention="dense", prompt_length=prompt_length)
    )
    for prompt_length in prompt_lengths
]
results["sparq"] = [
    run_or_load(
        Benchmark(
            **base_config,
            attention="sparq",
            prompt_length=prompt_length,
            sparq=SparQArgs(),
        )
    )
    for prompt_length in prompt_lengths
]

for label, means_stds in results.items():
    means = [m for m, _ in means_stds]
    # Ideally we want seq length 0, but that doesn't so 1 is probably close enough.
    time_not_in_attention = means[prompt_lengths.index(1)]
    frac_in_attention = [(m - time_not_in_attention) / m for m in means]
    plt.plot(prompt_lengths, frac_in_attention, label=label)

plt.xlabel("prompt length")
plt.ylabel("frac time in attention")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()

plt.tight_layout()
figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
plt.savefig(figure_dir / "time_spent_in_attention.png")
plt.close()
