#!/usr/bin/env python3
"""Flatten rerank requests into fixed-bucket pretokenized query/document pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", type=int, required=True, choices=(128, 256, 512))
    parser.add_argument("--request-limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer.expanduser()))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=args.bucket, strategy="longest_first")
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = tokenizer.token_to_id("<|padding|>")
    if pad_id is None:
        raise RuntimeError("tokenizer is missing a supported padding token id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    requests = 0
    pairs = 0
    with args.requests.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            request_id = row.get("id")
            query = row.get("query")
            documents = row.get("documents")
            if not isinstance(request_id, str) or not isinstance(query, str):
                raise ValueError(f"{args.requests}:{line_number} requires string id and query")
            if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
                raise ValueError(f"{args.requests}:{line_number} requires a documents string array")

            for document_index, document in enumerate(documents):
                encoding = tokenizer.encode(query, document)
                ids = list(encoding.ids)
                token_count = len(ids)
                if token_count > args.bucket:
                    raise AssertionError("tokenizer truncation exceeded the selected bucket")
                mask = [1] * token_count + [0] * (args.bucket - token_count)
                ids.extend([pad_id] * (args.bucket - token_count))
                destination.write(
                    json.dumps(
                        {
                            "id": f"{request_id}::{document_index}",
                            "input_ids": ids,
                            "attention_mask": mask,
                            "token_count": token_count,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                pairs += 1
            requests += 1
            if args.request_limit is not None and requests >= args.request_limit:
                break

    print(f"prepared {pairs} pairs from {requests} requests at bucket {args.bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
