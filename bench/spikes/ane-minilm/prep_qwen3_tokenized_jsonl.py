#!/usr/bin/env python3
"""Prepare fixed-bucket, left-padded Qwen3 inputs with a terminal model EOS.

Each output row is ``{id,input_ids,attention_mask,token_count}``.  Left padding
keeps the terminal EOS in the final static position, allowing Core ML to pool
without a dynamic last-token gather while retaining Qwen3's causal semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF repo id or local snapshot directory")
    parser.add_argument("--bucket", type=int, required=True, choices=(128, 256, 512))
    parser.add_argument("--input", default="-", help="Input JSONL path or '-' for stdin")
    parser.add_argument("--output", default="-", help="Output JSONL path or '-' for stdout")
    parser.add_argument("--text-field", default="text", help="Input text field; default: text")
    parser.add_argument("--id-field", default="id", help="Input ID field; default: id")
    parser.add_argument("--limit", type=int, help="Optional output row cap")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def cached_model_snapshot(model_id: str) -> Path | None:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
    )
    if not cache_root.exists():
        return None
    snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def resolve_model_ref(requested: str) -> str:
    path = Path(requested).expanduser()
    if path.exists():
        return str(path.resolve())
    cached = cached_model_snapshot(requested)
    return str(cached.resolve()) if cached is not None else requested


def read_eos_token_id(model_ref: str) -> int:
    config_path = Path(model_ref) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    eos_token_id = config.get("eos_token_id")
    if not isinstance(eos_token_id, int):
        raise ValueError(f"{config_path} has no integer eos_token_id")
    return eos_token_id


def open_input(path: str) -> tuple[str, list[str]]:
    if path == "-":
        return "stdin", sys.stdin.read().splitlines()
    input_path = Path(path).expanduser()
    return str(input_path), input_path.read_text(encoding="utf-8").splitlines()


def open_output(path: str):
    if path == "-":
        return sys.stdout
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w", encoding="utf-8")


def tokenize(
    tokenizer: Any, text: str, bucket: int, eos_token_id: int, pad_token_id: int
) -> tuple[list[int], list[int], int]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=bucket - 1,
        padding=False,
        return_attention_mask=False,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    if ids and ids[-1] == eos_token_id:
        ids.pop()
    ids = ids[: bucket - 1]
    ids.append(eos_token_id)
    pad_count = bucket - len(ids)
    return [pad_token_id] * pad_count + ids, [0] * pad_count + [1] * len(ids), len(ids)


def main() -> int:
    args = parse_args()
    model_ref = resolve_model_ref(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=not args.allow_download)
    eos_token_id = read_eos_token_id(model_ref)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    source_name, lines = open_input(args.input)
    output_handle = open_output(args.output)
    produced = 0
    token_total = 0
    try:
        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            text = row.get(args.text_field)
            if not isinstance(text, str):
                raise ValueError(f"{source_name}:{line_number} is missing string field {args.text_field!r}")
            row_id = row.get(args.id_field)
            if not isinstance(row_id, str) or not row_id:
                row_id = f"line:{line_number:06d}"
            input_ids, attention_mask, token_count = tokenize(
                tokenizer, text, args.bucket, eos_token_id, int(pad_token_id)
            )
            output_handle.write(
                json.dumps(
                    {
                        "id": row_id,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_count": token_count,
                    },
                    ensure_ascii=False,
                )
            )
            output_handle.write("\n")
            produced += 1
            token_total += token_count
            if args.limit is not None and produced >= args.limit:
                break
    finally:
        if output_handle is not sys.stdout:
            output_handle.close()
    print(
        f"prepared {produced} Qwen3 rows from {source_name} at bucket {args.bucket} "
        f"({token_total} real tokens; terminal EOS={eos_token_id}; left padding)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
