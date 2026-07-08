#!/usr/bin/env python3
"""Dump Hugging Face MiniLM hidden-state summaries for MLX parity debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_TEXT = "debug minilm parity now"


def parse_token_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    token_ids = [int(part) for part in value.replace(",", " ").split()]
    if not token_ids:
        raise ValueError("--token-ids must contain at least one id")
    return token_ids


def summarize(name: str, tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().to(torch.float32).flatten()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "mean": float(values.mean()),
        "first8": [float(value) for value in values[:8]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="HF MiniLM snapshot directory")
    parser.add_argument("--tokenizer", type=Path, help="optional tokenizer snapshot or tokenizer.json")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to tokenize when --token-ids is omitted")
    parser.add_argument(
        "--token-ids",
        help="comma or whitespace separated ids; defaults to the tokenized debug text",
    )
    args = parser.parse_args()

    token_ids = parse_token_ids(args.token_ids)
    tokenizer_root = args.tokenizer
    if tokenizer_root is not None and tokenizer_root.name == "tokenizer.json":
        tokenizer_root = tokenizer_root.parent
    if token_ids is None:
        if tokenizer_root is None:
            tokenizer_root = args.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        encoding = tokenizer(args.text, return_tensors="pt", padding=False, truncation=False)
        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids))
    else:
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    summaries = [summarize("embeddings.hidden", output.hidden_states[0])]
    for layer_idx, hidden in enumerate(output.hidden_states[1:]):
        summaries.append(summarize(f"layer.{layer_idx}.hidden", hidden))

    payload = {
        "model": str(args.model),
        "token_ids": [int(value) for value in input_ids[0].tolist()],
        "summaries": summaries,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
