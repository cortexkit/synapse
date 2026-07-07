#!/usr/bin/env python3
"""Model2Vec static embedding benchmark lane."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from model2vec import StaticModel  # pyright: ignore[reportMissingImports]

LANE = "potion"
WORKLOAD = "embed-corpus-v1"
MODEL_REPO = "minishlab/potion-code-16M"
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 1024


@dataclass(frozen=True)
class CorpusRow:
    id: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vectors-out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--prefix-document")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    model = load_model()
    _ = model.encode(
        [apply_prefix(args.prefix_document, "warmup")],
        show_progress_bar=False,
        batch_size=1,
        use_multiprocessing=False,
        max_length=MAX_LENGTH,
    )
    cold_load_s = time.perf_counter() - started

    rows = load_corpus(args.corpus, args.limit)

    if args.vectors_out is not None:
        args.vectors_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    input_tokens = 0
    for batch in iter_batches(rows, args.batch_size):
        texts = [apply_prefix(args.prefix_document, row.text) for row in batch]
        input_tokens += sum(len(ids) for ids in model.tokenize(texts, max_length=MAX_LENGTH))

    infer_started = time.perf_counter()
    items = 0

    vector_context = args.vectors_out.open("w", encoding="utf-8") if args.vectors_out is not None else nullcontext()
    with vector_context as handle:
        for batch in iter_batches(rows, args.batch_size):
            texts = [apply_prefix(args.prefix_document, row.text) for row in batch]
            vectors = model.encode(
                texts,
                show_progress_bar=False,
                batch_size=args.batch_size,
                use_multiprocessing=False,
                max_length=MAX_LENGTH,
            )
            items += len(batch)
            if handle is not None:
                for row, vector in zip(batch, vectors, strict=True):
                    json.dump({"id": row.id, "vec": vector.tolist()}, handle, ensure_ascii=False)
                    handle.write("\n")

    infer_wall_s = time.perf_counter() - infer_started

    result = {
        "lane": LANE,
        "workload": WORKLOAD,
        "model": args.model_label,
        "cold_load_s": cold_load_s,
        "infer_wall_s": infer_wall_s,
        "input_tokens": input_tokens,
        "tok_per_s": (input_tokens / infer_wall_s) if infer_wall_s > 0 else 0.0,
        "items": items,
        "parity_mean_cosine": None,
        "self_peak_rss_bytes": peak_rss_bytes(),
        "notes": (
            f"source={MODEL_REPO}; dim={model.dim}; normalize={model.normalize}; "
            f"tokenizer=model.tokenize; batch_size={args.batch_size}; max_len={MAX_LENGTH}; "
            f"prefix_document={format_prefix_note(args.prefix_document)}"
        ),
    }

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"{LANE}: {items} items, {input_tokens} tokens, {result['tok_per_s']:.1f} tok/s, "
        f"cold_load {cold_load_s:.2f}s, infer {infer_wall_s:.2f}s",
        file=sys.stderr,
    )
    return 0


def load_model() -> StaticModel:
    model = StaticModel.from_pretrained(MODEL_REPO, force_download=False)
    if not model.normalize:
        model.normalize = True
    return model


def apply_prefix(prefix_document: str | None, text: str) -> str:
    if prefix_document is None:
        return text
    return f"{prefix_document}{text}"


def format_prefix_note(prefix_document: str | None) -> str:
    if prefix_document is None:
        return "none"
    return repr(prefix_document)


def load_corpus(path: Path, limit: int | None) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                row_id = data["id"]
                text = data["text"]
            except KeyError as error:
                raise KeyError(f"{path}:{line_no} missing {error.args[0]!r}") from error
            if not isinstance(row_id, str):
                raise TypeError(f"{path}:{line_no} id must be a string")
            if not isinstance(text, str):
                raise TypeError(f"{path}:{line_no} text must be a string")
            rows.append(CorpusRow(id=row_id, text=text))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_batches(rows: Sequence[CorpusRow], batch_size: int) -> Iterator[list[CorpusRow]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


if __name__ == "__main__":
    raise SystemExit(main())
