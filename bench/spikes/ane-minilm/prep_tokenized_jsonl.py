#!/usr/bin/env python3
"""Prepare fixed-bucket MiniLM inputs as JSONL.

Input rows are read from stdin or a file and must contain the configured text
field (default: `embed_text`). Output rows are:

    {"id": "...", "input_ids": [...], "attention_mask": [...], "token_count": N}

The Qdrant tokenizer snapshot carries baked-in fixed padding to 128 tokens. This
script disables that padding policy, truncates to the requested bucket, then pads
manually to the exact bucket length so the Core ML runner receives consistent
fixed-shape inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

DEFAULT_TOKENIZER = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qdrant--all-MiniLM-L6-v2-onnx"
    / "snapshots"
    / "manual"
    / "tokenizer.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--bucket", type=int, required=True, choices=(256, 512))
    parser.add_argument(
        "--input",
        default="-",
        help="Input JSONL path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSONL path, or '-' for stdout. Default: stdout.",
    )
    parser.add_argument(
        "--text-field",
        default="embed_text",
        help="JSON field to tokenize. Default: embed_text",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Optional existing string id field to preserve; falls back to the 1-based input line number.",
    )
    parser.add_argument("--limit", type=int, help="Optional row cap.")
    parser.add_argument(
        "--id-prefix",
        default="line:",
        help="Prefix for generated ids when the input row has no string id. Default: line:",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    tokenizer_path = args.tokenizer.expanduser()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=args.bucket)

    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("tokenizer is missing a [PAD] token id")

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
                raise ValueError(
                    f"{source_name}:{line_number} is missing string field {args.text_field!r}"
                )

            row_id = row.get(args.id_field)
            if not isinstance(row_id, str) or not row_id:
                row_id = f"{args.id_prefix}{line_number:06d}"

            encoding = tokenizer.encode(text)
            input_ids = list(encoding.ids)
            token_count = len(input_ids)
            if token_count > args.bucket:
                raise AssertionError(
                    f"tokenizer returned {token_count} ids after truncation; bucket is {args.bucket}"
                )

            attention_mask = [1] * token_count + [0] * (args.bucket - token_count)
            input_ids.extend([pad_id] * (args.bucket - token_count))

            output_row = {
                "id": row_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_count": token_count,
            }
            output_handle.write(json.dumps(output_row, ensure_ascii=False))
            output_handle.write("\n")
            produced += 1
            token_total += token_count

            if args.limit is not None and produced >= args.limit:
                break
    finally:
        if output_handle is not sys.stdout:
            output_handle.close()

    print(
        f"prepared {produced} rows from {source_name} at bucket {args.bucket} ({token_total} real tokens)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
