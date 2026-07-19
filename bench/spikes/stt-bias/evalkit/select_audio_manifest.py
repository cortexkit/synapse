#!/usr/bin/env python3
"""Choose one of several synthesized voices per source utterance reproducibly."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voices", default="Samantha,Daniel")
    return parser.parse_args()


def source_id(identifier: str) -> str:
    return identifier.rsplit("-", 1)[0]


def main() -> None:
    args = parse_args()
    requested_voices = [voice.strip() for voice in args.voices.split(",") if voice.strip()]
    with args.manifest.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[source_id(row["id"])][row["voice"]] = row

    selected = []
    for index, identifier in enumerate(sorted(grouped)):
        voice = requested_voices[index % len(requested_voices)]
        if voice not in grouped[identifier]:
            raise SystemExit(f"{identifier} has no synthesized {voice} clip")
        selected.append(grouped[identifier][voice])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} alternating-voice rows to {args.out}")


if __name__ == "__main__":
    main()
