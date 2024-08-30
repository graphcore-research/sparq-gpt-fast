"""Feeds a text file into the prompt, allowing you to ask the LLM about the file."""

# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.

import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch._dynamo.config
import torch._inductor.config
from tap import Tap
from torch import Tensor
from torch.nn.attention import SDPBackend

from benchmark import forward_for_generate, load_model, prefill, sample
from model import Transformer
from sparq import RKForCompressionRatio, SparQArgs
from tokenizer import TokenizerInterface, get_tokenizer


class CliArgs(Tap):
    checkpoint: Path
    max_new_tokens: int = 1000
    compile: bool = False
    document: Path | None = None
    sparq: bool = False
    sparq_compression_ratio: int = 8
    temperature: float = 0.7
    top_k: int = 50


@torch.no_grad
def main(args: CliArgs) -> None:
    device = torch.device("cuda")

    assert args.checkpoint.is_file()
    tokenizer_path = args.checkpoint.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_chat = "chat" in str(args.checkpoint) or "Instruct" in str(args.checkpoint)

    if args.document is not None:
        if not is_chat:
            raise NotImplementedError
        with open(args.document, "r") as file:
            document = file.read()
    else:
        document = None

    print("Loading model ...")
    t0 = time.time()
    model = load_model(
        args.checkpoint,
        device,
        precision,
        attention="sparq" if args.sparq else "dense",
        sparq=SparQArgs(rk=RKForCompressionRatio(args.sparq_compression_ratio)),
    )

    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint)

    forward = forward_for_generate
    if args.compile:
        forward = torch.compile(forward, mode="reduce-overhead", fullgraph=True)

    while True:
        prompt = _read_prompt(is_chat, document)
        encoded_prompt = _encode_tokens(tokenizer, prompt, bos=True, device=device)
        print(f"Total length of prompt: {len(encoded_prompt)}")

        with torch.device(device):
            max_seq_len = len(encoded_prompt) + args.max_new_tokens
            assert max_seq_len <= model.config.block_size
            model.setup_caches(max_batch_size=1, max_seq_length=max_seq_len)

        for output in _generate(
            model, tokenizer, forward, encoded_prompt, device, args
        ):
            print(output, end="", flush=True)


def _encode_tokens(tokenizer, string, bos: bool, device: torch.device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _read_prompt(is_chat: bool, document: str | None) -> str:
    # TODO: Prompt needs to be set correctly for the specific model in use.
    prompt = input("What is your question about the document?")
    if document is not None:
        document = document.replace("\\n", " ")
        prompt = f"Document:{document} Question:{prompt.strip()}"
    if is_chat:
        prompt = f"[INST]{prompt.strip()}[/INST]"
    return prompt


def _generate(
    model: Transformer,
    tokenizer: TokenizerInterface,
    forward,
    encoded_prompt: Tensor,
    device: torch.device,
    args: CliArgs,
) -> Iterator[str]:
    period_id = tokenizer.encode(".")[0]

    input_pos = torch.arange(0, encoded_prompt.size(0), device=device)
    with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        token = prefill(model, encoded_prompt.view(1, -1), input_pos).clone()
    decoded_token = tokenizer.decode([period_id] + token.tolist())[1:]
    yield decoded_token

    for n_generated in range(args.max_new_tokens):
        input_pos = torch.tensor(
            [encoded_prompt.size(0) + n_generated], device=device, dtype=torch.int
        )
        logits = forward(model, token.view(1, -1), input_pos)[0]
        token = sample(logits.view(1, 1, -1), args.temperature, args.top_k)[0]
        decoded_token = tokenizer.decode([period_id] + token.tolist())[1:]
        yield decoded_token
        if token.item() == tokenizer.eos_id():
            break


def _stop(output: Iterator[str], stop_str: str) -> Iterator[str]:
    buffer = ""
    for decoded in output:
        buffer += decoded
        if buffer.endswith(stop_str):
            buffer.removesuffix(stop_str)
            yield buffer
            return
        if len(buffer) >= len(stop_str):
            yield buffer
            buffer = ""


if __name__ == "__main__":
    main(CliArgs().parse_args())
