#!/usr/bin/env python3
"""Create the pinned CPU-fp32 oracle for Qwen3 raw-completion decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PINNED_TRANSFORMERS = "4.51.0"
PINNED_TORCH = "2.13.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Qwen3 snapshot or model id")
    parser.add_argument("--prompts", required=True, type=Path, help="JSONL {id,prompt}")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k-logits", type=int, default=5)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("prompt set is empty")
    return rows


def top_logits(scores: torch.Tensor, count: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(scores.float(), k=min(count, scores.shape[-1]))
    return [
        {"token_id": int(token), "logit": float(value)}
        for token, value in zip(indices.tolist(), values.tolist(), strict=True)
    ]


def main() -> None:
    args = parse_args()
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise RuntimeError(
            f"transformers {PINNED_TRANSFORMERS} is required, found {transformers.__version__}"
        )
    if torch.__version__ != PINNED_TORCH:
        raise RuntimeError(f"torch {PINNED_TORCH} is required, found {torch.__version__}")
    if args.max_new_tokens <= 0 or args.top_k_logits <= 0:
        raise ValueError("generation and top-k sizes must be positive")

    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=local_only,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    eos_token_id = model.generation_config.eos_token_id
    pad_token_id = model.generation_config.pad_token_id
    prompts = load_prompts(args.prompts)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as output, torch.inference_mode():
        for row in prompts:
            encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True)
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            prompt_length = encoded["input_ids"].shape[1]
            tokens = generated.sequences[0, prompt_length:].tolist()
            record = {
                "id": row["id"],
                "tokens": tokens,
                "top_logits": [
                    top_logits(step_scores[0], args.top_k_logits)
                    for step_scores in generated.scores
                ],
                "generation_config": {
                    "transformers": PINNED_TRANSFORMERS,
                    "torch": PINNED_TORCH,
                    "torch_dtype": "float32",
                    "device": "cpu",
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                    "temperature": model.generation_config.temperature,
                    "top_k": model.generation_config.top_k,
                    "top_p": model.generation_config.top_p,
                    "sampling_parameters_ignored": True,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": pad_token_id,
                    "add_special_tokens": True,
                    "chat_template": False,
                },
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
