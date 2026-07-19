#!/usr/bin/env python3
"""Attach per-request vocabulary context for one STT-bias evaluation arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("baseline", "prompt", "trie", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--terms", type=Path, default=Path(__file__).with_name("terms.jsonl"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    audio_rows = load_jsonl(args.audio_manifest)
    technical_terms = [
        row["term"]
        for row in load_jsonl(args.terms)
        if row["class"] == "technical" and not row.get("excluded", False)
    ]
    use_context = args.arm != "baseline"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in audio_rows:
            audio_path = (args.audio_manifest.parent / row["path"]).resolve()
            output = {
                "id": row["id"],
                "path": str(audio_path.relative_to(args.out.parent.resolve()))
                if args.out.parent.resolve() in audio_path.parents
                else str(audio_path),
            }
            if use_context:
                # Target terms model a code-view vocabulary for technical rows.
                # Controls receive the full technical list to expose unwanted
                # vocabulary insertion instead of giving them a trivial context.
                output["bias_terms"] = (
                    [row["expected_term"]]
                    if row["term_class"] == "technical"
                    else technical_terms
                )
                output["bias_prompt"] = (
                    "Transcribe the audio faithfully. Use vocabulary only when it is spoken."
                )
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"wrote {args.arm} manifest with {len(audio_rows)} requests to {args.out}")


if __name__ == "__main__":
    main()
