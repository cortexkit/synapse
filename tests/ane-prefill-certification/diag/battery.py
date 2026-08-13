#!/usr/bin/env python3
"""Run and summarize the 20 W128 CPU_AND_NE flip-density fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-bucket", type=int, default=512, choices=(512, 1024, 2048))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def control(analysis: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in analysis["controls"] if row["compute_units"] == "CPU_AND_NE")


def main() -> int:
    args = parse_args()
    run_script = Path(__file__).with_name("run.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index in range(20):
        case_id = f"w128-width-{index:02d}"
        case_dir = args.output_dir / case_id
        subprocess.run(
            [
                sys.executable,
                str(run_script),
                "--root",
                str(args.root),
                "--model",
                str(args.model),
                "--compiled",
                str(args.compiled),
                "--runner",
                str(args.runner),
                "--analyzer",
                str(args.analyzer),
                "--case-id",
                case_id,
                "--cache-bucket",
                str(args.cache_bucket),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--skip-cpu-control",
                "--output-dir",
                str(case_dir),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
        observation = control(analysis)
        divergence = observation["divergence"]
        rows.append(
            {
                "case_id": case_id,
                "token_exact": observation["token_exact"],
                "match_depth": observation["match_depth"],
                "divergence": divergence,
                "kv_vs_pure_gpu": observation["kv_vs_pure_gpu"]["overall"],
                "admission_roundtrip_bit_mismatches": observation["kv_vs_pure_gpu"][
                    "admission_roundtrip_bit_mismatches"
                ],
                "load_context": json.loads(
                    (case_dir / "load-context.json").read_text(encoding="utf-8")
                ),
            }
        )
    flips = [row for row in rows if not row["token_exact"]]
    summary = {
        "schema_revision": 1,
        "fixture_count": len(rows),
        "token_exact_count": len(rows) - len(flips),
        "flip_count": len(flips),
        "flip_density": len(flips) / len(rows),
        "all_flips_same_top2_token_set": all(
            row["divergence"]["same_top2_token_set"] for row in flips
        ),
        "rows": rows,
    }
    output = args.output_dir / "battery-summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
