#!/usr/bin/env python3
"""Keep audio-manifest rows whose source utterance ids appear in a selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_id(identifier: str) -> str:
    return identifier.rsplit("-", 1)[0]


def main() -> None:
    args = parse_args()
    selected = {row["id"] for row in load_jsonl(args.utterances)}
    rows = [
        row for row in load_jsonl(args.audio_manifest) if source_id(row["id"]) in selected
    ]
    if len(rows) != len(selected):
        raise SystemExit(f"selected {len(selected)} sources but found {len(rows)} audio rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} selected audio rows to {args.out}")


if __name__ == "__main__":
    main()
