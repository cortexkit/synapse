#!/usr/bin/env python3
"""Summarize locked-M1 power windows for rerank serving repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import fmean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_samples(path: Path) -> list[dict]:
    samples = []
    with path.open() as handle:
        for line in handle:
            sample = json.loads(line)
            sample["epoch_s"] = datetime.fromisoformat(sample["timestamp"]).timestamp()
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    cells = []
    for result_path in sorted(args.result_dir.glob("hit-r*.json")):
        stem = result_path.stem
        result = json.loads(result_path.read_text())
        duration = result["infer_wall_s"]
        macmon_path = args.raw_dir / f"{stem}-macmon.jsonl"
        samples = read_samples(macmon_path)
        powered = [
            sample
            for sample in samples
            if sample["gpu_power"] >= 1.0 and sample["gpu_usage"][1] >= 0.05
        ]
        if not powered:
            raise RuntimeError(f"no powered GPU sample found for {stem}")
        window_end = powered[-1]["epoch_s"]
        window_start = window_end - duration
        window = [
            sample
            for sample in samples
            if window_start <= sample["epoch_s"] <= window_end
        ]
        if not window:
            raise RuntimeError(f"no inference-window samples found for {stem}")
        cells.append(
            {
                "cell": stem,
                "result": result_path.name,
                "infer_wall_s": duration,
                "window_start": datetime.fromtimestamp(window_start)
                .astimezone()
                .isoformat(),
                "window_end": datetime.fromtimestamp(window_end).astimezone().isoformat(),
                "samples": len(window),
                "gpu_power_w_mean": fmean(sample["gpu_power"] for sample in window),
                "gpu_power_w_max": max(sample["gpu_power"] for sample in window),
                "gpu_effective_usage_mean": fmean(
                    sample["gpu_usage"][1] for sample in window
                ),
                "gpu_effective_usage_max": max(
                    sample["gpu_usage"][1] for sample in window
                ),
                "macmon_sha256": sha256(macmon_path),
            }
        )

    summary = {
        "host": "[bench-host]",
        "tool": "macmon 0.7.2",
        "interval_ms": 100,
        "metric_scope": "system-wide",
        "usage_metric": "frequency-scaled effective GPU utilization ratio",
        "method": (
            "Each inference window has the result JSON infer_wall_s duration and ends "
            "at the last sample with at least 1 W GPU power and 5% effective GPU usage."
        ),
        "raw_logs": "[bench-user-home]/bench-tools/unified-rt-serving/results/m1-rerank/",
        "cells": cells,
    }
    (args.result_dir / "power-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
