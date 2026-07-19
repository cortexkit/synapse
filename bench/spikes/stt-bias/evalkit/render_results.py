#!/usr/bin/env python3
"""Render reproducible STT-bias score JSON into the four-row Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> None:
    args = parse_args()
    scores = [json.loads(path.read_text(encoding="utf-8")) for path in args.scores]
    lines = [
        "| Arm | Term-exact (class A) | False insertion (class B) | WER | Case fidelity | Added prompt tokens | Prefill / decode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score in scores:
        runtime = score["runtime"]
        prefill = runtime.get("prefill_wall_s")
        decode = runtime.get("decode_wall_s")
        timing = "n/a" if prefill is None or decode is None else f"{prefill:.2f}s / {decode:.2f}s"
        lines.append(
            "| {arm} | {term} | {insertion} | {wer} | {case} | {tokens} | {timing} |".format(
                arm=score["arm"],
                term=percent(score["term_exact_accuracy"]),
                insertion=percent(score["false_insertion_rate"]),
                wer=percent(score["wer"]),
                case=percent(score["case_fidelity"]),
                tokens=runtime.get("bias_prompt_tokens", 0),
                timing=timing,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
