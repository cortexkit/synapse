#!/usr/bin/env python3
"""Create fixed-bucket Transformers CPU-fp32 hidden-state references for LFM2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Hugging Face snapshot")
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL rows with id and prompt")
    parser.add_argument("--seq-len", type=int, required=True, choices=(128, 256))
    parser.add_argument("--out", type=Path, required=True, help="Compressed NumPy reference archive")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 20:
        raise ValueError(f"the feasibility gate requires exactly 20 prompts, found {len(rows)}")
    for row in rows:
        if not isinstance(row.get("id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError("every prompt row must contain string id and prompt fields")
    return rows


def main() -> int:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite")
    prompts = load_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [row["prompt"] for row in prompts],
        add_special_tokens=True,
        truncation=True,
        max_length=args.seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(dtype=torch.long)
    attention_mask = encoded["attention_mask"].to(dtype=torch.long)
    if input_ids.shape != (20, args.seq_len):
        raise RuntimeError(f"tokenizer returned unexpected shape {tuple(input_ids.shape)}")

    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    load_seconds = time.monotonic() - started
    started = time.monotonic()
    with torch.inference_mode():
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.float()
    inference_seconds = time.monotonic() - started
    metadata = {
        "source_model": str(args.model),
        "source_revision": args.model.name,
        "source_model_sha256": sha256(args.model / "model.safetensors"),
        "seq_len": args.seq_len,
        "rows": len(prompts),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "checkpoint_dtype": "bfloat16",
        "compute_dtype": "float32",
        "device": "cpu",
        "padding_side": "left",
        "model_load_s": load_seconds,
        "inference_s": inference_seconds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        ids=np.asarray([row["id"] for row in prompts]),
        input_ids=input_ids.cpu().numpy().astype(np.int32),
        attention_mask=attention_mask.cpu().numpy().astype(np.int32),
        hidden_states=hidden.cpu().numpy().astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
