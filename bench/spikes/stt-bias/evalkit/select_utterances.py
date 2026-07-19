#!/usr/bin/env python3
"""Select a deterministic, balanced subset from the full generated utterance set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, help="Take this many terms from each class.")
    parser.add_argument("--carrier", type=int, default=1, help="One-based carrier slot to retain per term.")
    parser.add_argument(
        "--omit-excluded",
        action="store_true",
        help="Drop TTS-excluded rows before balancing (useful for a scored dev split).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.utterances.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    chosen = [row for row in rows if row["id"].endswith(f"-{args.carrier}")]
    if args.omit_excluded:
        chosen = [row for row in chosen if not row.get("excluded", False)]
    if args.per_class is not None:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in chosen:
            grouped[row["term_class"]].append(row)
        chosen = [
            row
            for term_class in sorted(grouped)
            for row in grouped[term_class][: args.per_class]
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} seeded carrier rows to {args.out}")


if __name__ == "__main__":
    main()
