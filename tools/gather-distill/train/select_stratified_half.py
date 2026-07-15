#!/usr/bin/env python3
"""Select a deterministic half-dataset stratified by request class and language."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SEED = 8918


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def source_prompt(row: dict[str, Any], row_index: int) -> str:
    trajectory = row.get("full_trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"metadata row {row_index} has no full_trajectory")
    first = trajectory[0]
    if not isinstance(first, dict) or first.get("role") != "user":
        raise ValueError(f"metadata row {row_index} does not begin with a user message")
    prompt = first.get("content")
    if not isinstance(prompt, str):
        raise ValueError(f"metadata row {row_index} has a non-string first user message")
    return prompt


def curated_prompt(row: dict[str, Any], row_index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"curated row {row_index} has no messages")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            prompt = message.get("content")
            if isinstance(prompt, str):
                return prompt
            raise ValueError(f"curated row {row_index} has a non-string first user message")
    raise ValueError(f"curated row {row_index} has no user message")


def stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def metadata_by_prompt(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        prompt = source_prompt(row, index)
        if prompt in result:
            raise ValueError(f"metadata prompt is not unique at row {index}")
        result[prompt] = row
    return result


def stratum_for(row: dict[str, Any], row_index: int) -> tuple[str, str]:
    tags = row.get("tags")
    if not isinstance(tags, dict):
        raise ValueError(f"metadata for curated row {row_index} has no tags")
    request_class = tags.get("request_class")
    language = tags.get("language")
    if not isinstance(request_class, str) or not request_class:
        raise ValueError(f"metadata for curated row {row_index} has no request_class")
    if not isinstance(language, str) or not language:
        raise ValueError(f"metadata for curated row {row_index} has no language")
    return request_class, language


def select_indices(
    curated: list[dict[str, Any]], metadata: list[dict[str, Any]], seed: int
) -> tuple[list[int], dict[tuple[str, str], list[int]]]:
    lookup = metadata_by_prompt(metadata)
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(curated):
        prompt = curated_prompt(row, index)
        source = lookup.get(prompt)
        if source is None:
            raise ValueError(f"curated row {index} has no exact metadata prompt match")
        strata[stratum_for(source, index)].append(index)

    target = len(curated) // 2
    quotas = {key: len(indices) // 2 for key, indices in strata.items()}
    remaining = target - sum(quotas.values())
    odd_strata = [key for key, indices in strata.items() if len(indices) % 2]
    odd_strata.sort(key=lambda key: stable_rank(seed, "quota:" + "\0".join(key)))
    if remaining < 0 or remaining > len(odd_strata):
        raise ValueError("cannot apportion exact half across strata")
    for key in odd_strata[:remaining]:
        quotas[key] += 1

    selected: list[int] = []
    for key, indices in strata.items():
        ranked = sorted(
            indices,
            key=lambda index: stable_rank(seed, f"row:{key[0]}\0{key[1]}:{index}"),
        )
        selected.extend(ranked[: quotas[key]])
    selected.sort()
    if len(selected) != target:
        raise AssertionError(f"selected {len(selected)} rows instead of {target}")
    return selected, strata


def write_jsonl(path: Path, rows: list[dict[str, Any]], indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in indices:
            handle.write(json.dumps(rows[index], separators=(",", ":"), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Curated training JSONL")
    parser.add_argument(
        "--metadata-source",
        type=Path,
        required=True,
        help="Source JSONL carrying tags.request_class and tags.language",
    )
    parser.add_argument("--output", type=Path, required=True, help="Ignored half-dataset JSONL")
    parser.add_argument("--indices", type=Path, required=True, help="Tracked zero-based row indices")
    parser.add_argument("--report", type=Path, required=True, help="Tracked selection report")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    curated = load_jsonl(args.input)
    metadata = load_jsonl(args.metadata_source)
    if not curated or len(curated) % 2:
        raise ValueError("curated dataset must contain a nonzero even number of rows")

    selected, strata = select_indices(curated, metadata, args.seed)
    write_jsonl(args.output, curated, selected)
    args.indices.parent.mkdir(parents=True, exist_ok=True)
    args.indices.write_text("".join(f"{index}\n" for index in selected), encoding="utf-8")

    selected_set = set(selected)
    report_strata = []
    for key, full_indices in sorted(strata.items()):
        half_count = sum(index in selected_set for index in full_indices)
        report_strata.append(
            {
                "request_class": key[0],
                "language": key[1],
                "full_rows": len(full_indices),
                "half_rows": half_count,
            }
        )
    report = {
        "method": "Within-stratum SHA-256 ranking with exact-half largest-remainder apportionment",
        "row_index_base": 0,
        "seed": args.seed,
        "stratification": ["tags.request_class", "tags.language"],
        "input": str(args.input),
        "input_rows": len(curated),
        "input_sha256": sha256_file(args.input),
        "metadata_source": str(args.metadata_source),
        "metadata_source_rows": len(metadata),
        "metadata_source_sha256": sha256_file(args.metadata_source),
        "output": str(args.output),
        "output_rows": len(selected),
        "output_sha256": sha256_file(args.output),
        "indices": str(args.indices),
        "indices_sha256": sha256_file(args.indices),
        "strata": report_strata,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
