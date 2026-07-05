#!/usr/bin/env python3
"""MLX MiniLM embedding benchmark lane."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import mlx.core as mx  # pyright: ignore[reportMissingImports]

LANE = "mlx-minilm"
WORKLOAD = "embed-corpus-v1"
PRIMARY_MODEL = "mlx-community/all-MiniLM-L6-v2-bf16"
SOURCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_CONVERTED_MODEL = Path.home() / ".cache" / "synapse" / "mlx-minilm" / "all-MiniLM-L6-v2-bf16"
MAX_LENGTH = 512
DEFAULT_TOKEN_BUDGET = 16_384
DEFAULT_ITEM_CAP = 64

_AUTO_TOKENIZER_REGISTER_PATCHED = False


@dataclass(frozen=True)
class CorpusRow:
    id: str
    text: str


@dataclass(frozen=True)
class EncodedRow:
    id: str
    ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vectors-out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--model",
        default=None,
        help="HF repo or local path for mlx_embeddings.load; default: MiniLM chain",
    )
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--item-cap", type=int, default=DEFAULT_ITEM_CAP)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    device_note = set_gpu_default_device()
    model, tokenizer, source_note = load_model_and_tokenizer(args.model)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    warmup_ids = encode_text(tokenizer, "warmup")
    _ = embed_batch(model, [EncodedRow(id="warmup", ids=warmup_ids)], pad_id)
    cold_load_s = time.perf_counter() - started

    rows = load_corpus(args.corpus, args.limit)
    encoded_rows = [EncodedRow(row.id, encode_text(tokenizer, row.text)) for row in rows]

    if args.vectors_out is not None:
        args.vectors_out.parent.mkdir(parents=True, exist_ok=True)

    infer_started = time.perf_counter()
    produced: list[dict[str, object]] = []
    input_tokens = 0
    items = 0

    for batch in iter_batches(encoded_rows, args.token_budget, args.item_cap):
        vectors = embed_batch(model, batch, pad_id)
        for row, vector in zip(batch, vectors, strict=True):
            produced.append({"id": row.id, "vec": vector})
            input_tokens += len(row.ids)
            items += 1

    infer_wall_s = time.perf_counter() - infer_started

    if args.vectors_out is not None:
        with args.vectors_out.open("w", encoding="utf-8") as handle:
            for item in produced:
                json.dump(item, handle, ensure_ascii=False)
                handle.write("\n")

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
            f"source={source_note}; device={device_note}; pooling=mean+l2; "
            f"token_budget={args.token_budget}; item_cap={args.item_cap}; max_len={MAX_LENGTH}"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"{LANE}: {items} items, {input_tokens} tokens, {result['tok_per_s']:.1f} tok/s, "
        f"cold_load {cold_load_s:.2f}s, infer {infer_wall_s:.2f}s",
        file=sys.stderr,
    )
    return 0


def set_gpu_default_device() -> str:
    setter = getattr(mx, "set_default_device", None)
    if setter is None:
        return "default"
    candidates: list[object] = []
    gpu_attr = getattr(mx, "gpu", None)
    if gpu_attr is not None:
        candidates.append(gpu_attr() if callable(gpu_attr) else gpu_attr)
    device_cls = getattr(mx, "Device", None)
    gpu_factory = getattr(device_cls, "gpu", None) if device_cls is not None else None
    if gpu_factory is not None:
        candidates.append(gpu_factory() if callable(gpu_factory) else gpu_factory)
    candidates.extend(["gpu", "mps"])
    for candidate in candidates:
        try:
            setter(candidate)
            return "gpu"
        except Exception:
            continue
    return "default"


def patch_transformers_autotokenizer_register() -> None:
    global _AUTO_TOKENIZER_REGISTER_PATCHED
    if _AUTO_TOKENIZER_REGISTER_PATCHED:
        return
    try:
        from transformers import AutoTokenizer
    except Exception:
        return

    original_register = AutoTokenizer.register

    def patched_register(cls, config_class, *args, **kwargs):
        if isinstance(config_class, str):
            config_class = type(config_class, (), {})
            config_class.__module__ = "transformers"
        return original_register(config_class, *args, **kwargs)

    setattr(AutoTokenizer, "register", classmethod(patched_register))
    _AUTO_TOKENIZER_REGISTER_PATCHED = True


def import_mlx_embeddings_helpers() -> tuple[Any, Any]:
    patch_transformers_autotokenizer_register()
    from mlx_embeddings.convert import convert  # pyright: ignore[reportMissingImports]
    from mlx_embeddings.utils import load  # pyright: ignore[reportMissingImports]

    return load, convert


def load_model_and_tokenizer(explicit_model: str | None = None) -> tuple[Any, Any, str]:
    load, convert = import_mlx_embeddings_helpers()

    if explicit_model is not None:
        model, tokenizer = load(explicit_model)
        return model, tokenizer, explicit_model

    try:
        model, tokenizer = load(PRIMARY_MODEL)
        return model, tokenizer, PRIMARY_MODEL
    except Exception:
        pass

    try:
        model, tokenizer = load(SOURCE_MODEL)
        return model, tokenizer, SOURCE_MODEL
    except Exception:
        pass

    if (LOCAL_CONVERTED_MODEL / "config.json").exists():
        model, tokenizer = load(str(LOCAL_CONVERTED_MODEL))
        return model, tokenizer, f"converted:{SOURCE_MODEL}"

    LOCAL_CONVERTED_MODEL.parent.mkdir(parents=True, exist_ok=True)
    convert(SOURCE_MODEL, mlx_path=str(LOCAL_CONVERTED_MODEL), dtype="bfloat16")
    model, tokenizer = load(str(LOCAL_CONVERTED_MODEL))
    return model, tokenizer, f"converted:{SOURCE_MODEL}"


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


def encode_text(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    return list(ids)


def iter_batches(
    rows: Sequence[EncodedRow], token_budget: int, item_cap: int
) -> Iterator[list[EncodedRow]]:
    batch: list[EncodedRow] = []
    token_sum = 0
    for row in rows:
        row_tokens = len(row.ids)
        if batch and (len(batch) >= item_cap or token_sum + row_tokens > token_budget):
            yield batch
            batch = []
            token_sum = 0
        batch.append(row)
        token_sum += row_tokens
        if len(batch) >= item_cap:
            yield batch
            batch = []
            token_sum = 0
    if batch:
        yield batch


def embed_batch(model: Any, batch: Sequence[EncodedRow], pad_id: int) -> list[list[float]]:
    max_len = max(len(row.ids) for row in batch)
    input_ids = []
    attention_mask = []
    for row in batch:
        pad = max_len - len(row.ids)
        input_ids.append(row.ids + [pad_id] * pad)
        attention_mask.append([1] * len(row.ids) + [0] * pad)
    outputs = model(
        input_ids=mx.array(input_ids, dtype=mx.int32),
        attention_mask=mx.array(attention_mask, dtype=mx.int32),
    )
    embeddings = outputs.text_embeds
    mx.eval(embeddings)
    return embeddings.tolist()


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


if __name__ == "__main__":
    raise SystemExit(main())
