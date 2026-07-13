#!/usr/bin/env python3
"""Generate a bucket-correct fp32 Qwen3 embedding reference from prepared IDs.

The input must be left-padded with the terminal model EOS at the final position,
as produced by ``prep_qwen3_tokenized_jsonl.py``.  This reference runner is used
only for rows whose shorter fixed bucket changes active tokens; unchanged rows
can retain the canonical frozen Qwen3 reference vectors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF repo id or local snapshot directory")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row.get("id"), str):
            raise ValueError(f"{path}:{line_number} is missing string id")
        if not isinstance(row.get("input_ids"), list) or not isinstance(row.get("attention_mask"), list):
            raise ValueError(f"{path}:{line_number} is missing token arrays")
        if len(row["input_ids"]) != len(row["attention_mask"]):
            raise ValueError(f"{path}:{line_number} has mismatched token arrays")
        rows.append(row)
    if not rows:
        raise ValueError(f"input is empty: {path}")
    bucket = len(rows[0]["input_ids"])
    if bucket == 0 or any(len(row["input_ids"]) != bucket for row in rows):
        raise ValueError("all rows must use the same non-empty fixed bucket")
    return rows


def batch_tensors(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
        torch.tensor([row["attention_mask"] for row in batch], dtype=torch.long),
    )


def embed(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )[0]
    return functional.normalize(hidden[:, -1, :].float(), p=2, dim=-1)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    rows = read_rows(args.input)
    loaded = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval()
    with torch.inference_mode():
        warmup_ids, warmup_mask = batch_tensors(rows[:1])
        _ = embed(model, warmup_ids, warmup_mask)
    cold_load_s = time.perf_counter() - loaded

    args.output.parent.mkdir(parents=True, exist_ok=True)
    inference_started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as output_handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            input_ids, attention_mask = batch_tensors(batch)
            vectors = embed(model, input_ids, attention_mask).cpu().numpy().astype(np.float32)
            for row, vector in zip(batch, vectors, strict=True):
                json.dump({"id": row["id"], "vec": vector.tolist()}, output_handle, ensure_ascii=False)
                output_handle.write("\n")
    infer_wall_s = time.perf_counter() - inference_started
    input_tokens = sum(int(sum(row["attention_mask"])) for row in rows)
    stats = {
        "lane": "qwen3-hf-fp32-reference",
        "source_model": str(args.model),
        "items": len(rows),
        "bucket": len(rows[0]["input_ids"]),
        "input_tokens": input_tokens,
        "cold_load_s": cold_load_s,
        "infer_wall_s": infer_wall_s,
        "docs_per_s": len(rows) / infer_wall_s if infer_wall_s > 0 else 0.0,
        "tokens_per_s": input_tokens / infer_wall_s if infer_wall_s > 0 else 0.0,
        "batch_size": args.batch_size,
        "pooling": "terminal-eos-last-token+l2",
    }
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
