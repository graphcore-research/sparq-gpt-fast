"""Defines and runs speed benchmarks of gpt-fast."""

# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.


import gc
import json
import subprocess
import time
import uuid
from contextlib import nullcontext
from dataclasses import InitVar, asdict, dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import torch
import typed_configs
from torch import Tensor
from torch.nn.attention import SDPBackend

from model import AttentionMethod, Transformer
from sparq import SparQArgs, SparQAttention

RESULTS_DIR = Path("results")
CONFIG_FILE_NAME = "benchmark"
RESULTS_FILE_NAME = "results"


@dataclass(frozen=True)
class Benchmark:
    model: Literal["llama27bchat"]
    attention: AttentionMethod
    prompt_length: int
    quant: Optional[Literal["int8", "int4"]]
    compile: bool
    gpu: str
    sparq: SparQArgs = SparQArgs()

    def to_config_str(self) -> str:
        return " ".join(_dict_to_config_args(asdict(self)))


def _dict_to_config_args(d: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _dict_to_config_args(v, prefix=f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}={v}"


@dataclass
class Results:
    samples_and_durations: list[tuple[int, float]]
    torch_version: str
    revision: str
    gpu_name: str
    sparq_r: int | None = None
    sparq_k: int | None = None
    model: InitVar[Transformer] = None

    def __post_init__(self, model: Transformer | None) -> None:
        if model is not None:
            f = model.layers[0].attention.attention_function
            if isinstance(f, SparQAttention):
                self.sparq_k = f.k
                self.sparq_r = f.r
            self.model = None

    @property
    def secs_per_token_mean(self) -> float:
        secs_per_token = torch.tensor(
            [duration / n_samples for n_samples, duration in self.samples_and_durations]
        )
        return secs_per_token.mean().item()

    @property
    def secs_per_token_std(self) -> float:
        secs_per_token = torch.tensor(
            [duration / n_samples for n_samples, duration in self.samples_and_durations]
        )
        return secs_per_token.std().item()


def run_or_load(b: Benchmark) -> Results:
    existing_runs = get_existing_runs()
    if b in existing_runs:
        print(f"Loading {b}")
        r = existing_runs[b]
    else:
        print(f"Running {b}")
        r = run_benchmark(b)
        save_results(b, r)

    # Try to avoid running out of memory between benchmarks (possibly due to
    # fragmentation?)
    gc.collect()
    torch.cuda.empty_cache()

    return r


@cache
def get_existing_runs() -> dict[Benchmark, Results]:
    RESULTS_DIR.mkdir(exist_ok=True)

    def get():
        for run_dir in RESULTS_DIR.iterdir():
            config_path = run_dir / CONFIG_FILE_NAME
            results_path = run_dir / RESULTS_FILE_NAME
            with open(config_path, "r") as f:
                try:
                    config = typed_configs.parse(
                        Benchmark, ["fakeprog"] + f.read().split(" ")
                    )
                except Exception as e:
                    print(f"Failed on {config_path}")
                    raise e
            with open(results_path, mode="r") as f:
                results = Results(**json.load(f))
            yield config, results

    return dict(get())


def save_results(benchmark: Benchmark, results: Results) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_id = str(uuid.uuid4())[:5]
    run_path = RESULTS_DIR / f"{timestamp}_{rand_id}"
    run_path.mkdir(exist_ok=False, parents=True)
    with open(run_path / CONFIG_FILE_NAME, mode="w") as f:
        f.write(benchmark.to_config_str())
    with open(run_path / RESULTS_FILE_NAME, mode="w") as f:
        f.write(json.dumps(asdict(results)))


@torch.no_grad
def run_benchmark(b: Benchmark, n_repeats: int = 5, n_samples_per_repeat=50) -> Results:
    if b.gpu not in torch.cuda.get_device_name():
        raise ValueError("Wrong GPU to run this benchmark")

    if b.model == "llama27bchat" and b.quant is None:
        checkpoint_path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
    elif b.model == "llama27bchat" and b.quant == "int8":
        checkpoint_path = Path(
            "checkpoints/meta-llama/Llama-2-7b-chat-hf/model_int8.pth"
        )
    elif b.model == "llama27bchat" and b.quant == "int4":
        checkpoint_path = Path(
            "checkpoints/meta-llama/Llama-2-7b-chat-hf/model_int4.g32.pth"
        )
    else:
        raise NotImplementedError

    device = "cuda"

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    print(f"Using device={device}")

    print("Loading model ...")
    t0 = time.time()
    if b.attention == "sparq":
        sparq_args = b.sparq
    else:
        sparq_args = SparQArgs()
    model = load_model(
        checkpoint_path,
        device,
        precision=torch.bfloat16,
        attention=b.attention,
        sparq=sparq_args,
        # Ensure we have enough position embeddings for longer sequences.
        block_size=2**17,
    )
    torch.cuda.synchronize(device)

    torch.manual_seed(1234)
    forward = forward_for_generate
    if b.compile:
        forward = torch.compile(forward, mode="reduce-overhead", fullgraph=True)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=b.prompt_length + 1)

    results = []
    for repeat in range(n_repeats):
        print(f"Starting repeat {repeat+1} / {n_repeats}")
        prompt = torch.randint(
            low=0,
            high=model.tok_embeddings.embedding_dim - 1,
            size=(b.prompt_length,),
            device=device,
            dtype=torch.int32,
        )
        input_pos = torch.arange(0, prompt.size(0), device=device)
        print("Prefilling... ", end="", flush=True)
        with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            next_token = prefill(model, prompt.view(1, -1), input_pos).clone()

        input_pos = torch.tensor([b.prompt_length], device=device, dtype=torch.int)
        print("Warming up... ", end="", flush=True)
        for _ in range(2):
            with _get_attention_context(b.attention):
                logits = forward(model, next_token, input_pos)
                assert not torch.any(torch.isnan(logits))

        print("Benchmarking...")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_samples_per_repeat):
            with _get_attention_context(b.attention):
                logits = forward(model, next_token, input_pos)
                assert not torch.any(torch.isnan(logits))

        torch.cuda.synchronize()
        duration = time.perf_counter() - t0
        results.append((n_samples_per_repeat, duration))

    revision = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().rstrip("\n")
    )
    return Results(
        results,
        torch_version=torch.__version__,
        revision=revision,
        gpu_name=torch.cuda.get_device_name(),
        model=model,
    )


def _get_attention_context(method: AttentionMethod):
    if method == "dense" or method == "dense_vanilla":
        return torch.nn.attention.sdpa_kernel(SDPBackend.MATH)
    elif method == "sparq":
        return nullcontext()


def forward_for_generate(
    model: Transformer, next_token: Tensor, input_pos: Tensor
) -> Tensor:
    return model(next_token.view(1, -1), input_pos, prefill=False)


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, prefill=True)
    return sample(logits, **sampling_kwargs)[0]


def sample(logits, temperature: float = 0.8, top_k: Optional[int] = 200):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def load_model(checkpoint_path, device, precision, **config_kwargs):
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name, **config_kwargs)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()
