#!/usr/bin/env python3
"""Generate an ONNX Runtime fp32 embedding reference from pretokenized inputs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingImports]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to MiniLM model.onnx")
    parser.add_argument("--input", type=Path, required=True, help="Pretokenized JSONL from prep_tokenized_jsonl.py")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path for {id, vec}")
    parser.add_argument("--stats-out", type=Path, help="Optional stats JSON path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument(
        "--output-name",
        help="Optional explicit ONNX output name. Defaults to the session's first output.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row.get("id"), str):
            raise ValueError(f"{path}:{line_number} is missing string id")
        rows.append(row)
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


def pool_and_normalize(
    hidden_states: np.ndarray, attention_mask: np.ndarray, pooling: str
) -> np.ndarray:
    if pooling == "cls":
        pooled = hidden_states[:, 0, :].astype(np.float32)
    else:
        mask = attention_mask.astype(np.float32)[..., None]
        masked = hidden_states.astype(np.float32) * mask
        lengths = mask.sum(axis=1)
        if np.any(lengths <= 0):
            raise ValueError("attention mask contains an all-zero row")
        pooled = masked.sum(axis=1) / lengths
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return pooled / norms


def stats_dict(
    *,
    items: int,
    input_tokens: int,
    infer_wall_s: float,
    cold_load_s: float,
    batch_size: int,
    output_name: str,
) -> dict[str, object]:
    return {
        "lane": "ort-fp32-reference",
        "items": items,
        "input_tokens": input_tokens,
        "infer_wall_s": infer_wall_s,
        "cold_load_s": cold_load_s,
        "docs_per_s": items / infer_wall_s if infer_wall_s > 0 else 0.0,
        "tokens_per_s": input_tokens / infer_wall_s if infer_wall_s > 0 else 0.0,
        "batch_size": batch_size,
        "output_name": output_name,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    rows = read_rows(args.input)

    started = time.perf_counter()
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_info = {item.name: item for item in session.get_inputs()}
    if "input_ids" not in input_info or "attention_mask" not in input_info:
        raise RuntimeError(f"unexpected input names: {list(input_info)}")
    output_name = args.output_name or session.get_outputs()[0].name

    warmup_row = rows[0]
    warmup_ids = np.asarray([warmup_row["input_ids"]], dtype=np.int64)
    warmup_mask = np.asarray([warmup_row["attention_mask"]], dtype=np.int64)
    warmup_inputs: dict[str, np.ndarray] = {
        "input_ids": warmup_ids,
        "attention_mask": warmup_mask,
    }
    if "token_type_ids" in input_info:
        warmup_inputs["token_type_ids"] = np.zeros_like(warmup_ids)
    session.run([output_name], warmup_inputs)
    cold_load_s = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)

    infer_started = time.perf_counter()
    input_tokens = 0
    with args.output.open("w", encoding="utf-8") as output_handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            input_ids = np.asarray([row["input_ids"] for row in batch], dtype=np.int64)
            attention_mask = np.asarray([row["attention_mask"] for row in batch], dtype=np.int64)

            feeds: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in input_info:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            hidden_states = np.asarray(session.run([output_name], feeds)[0])
            pooled = pool_and_normalize(hidden_states, attention_mask, args.pooling)
            input_tokens += int(attention_mask.sum())

            for row, vector in zip(batch, pooled, strict=True):
                json.dump({"id": row["id"], "vec": vector.tolist()}, output_handle, ensure_ascii=False)
                output_handle.write("\n")
    infer_wall_s = time.perf_counter() - infer_started

    if args.stats_out is not None:
        stats = stats_dict(
            items=len(rows),
            input_tokens=input_tokens,
            infer_wall_s=infer_wall_s,
            cold_load_s=cold_load_s,
            batch_size=args.batch_size,
            output_name=output_name,
        )
        args.stats_out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            stats_dict(
                items=len(rows),
                input_tokens=input_tokens,
                infer_wall_s=infer_wall_s,
                cold_load_s=cold_load_s,
                batch_size=args.batch_size,
                output_name=output_name,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
