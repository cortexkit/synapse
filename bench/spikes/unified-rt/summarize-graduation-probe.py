#!/usr/bin/env python3
"""Summarize retained locked-M1 power captures for the graduation probe."""

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


def inference_duration(result: dict) -> float:
    passes = result.get("passes")
    if passes:
        return float(passes[-1]["infer_wall_s"])
    return float(result["infer_wall_s"])


def summarize(raw_dir: Path) -> dict:
    cells = []
    for result_path in sorted(raw_dir.glob("*.json")):
        if result_path.name.endswith("-parity.json") or result_path.name.endswith(
            "-prime.json"
        ):
            continue
        stem = result_path.stem
        macmon_path = raw_dir / f"{stem}-macmon.jsonl"
        start_path = raw_dir / f"{stem}-start-epoch.txt"
        end_path = raw_dir / f"{stem}-end-epoch.txt"
        if not (macmon_path.exists() and start_path.exists() and end_path.exists()):
            continue

        result = json.loads(result_path.read_text())
        duration = inference_duration(result)
        samples = read_samples(macmon_path)
        powered = [
            sample
            for sample in samples
            if sample["gpu_power"] >= 1.0 and sample["gpu_usage"][1] >= 0.05
        ]
        if not powered:
            raise RuntimeError(f"no powered GPU sample found for {stem}")
        inference_end = powered[-1]["epoch_s"]
        inference_start = inference_end - duration
        window = [
            sample
            for sample in samples
            if inference_start <= sample["epoch_s"] <= inference_end
        ]
        if not window:
            raise RuntimeError(f"no inference-window samples found for {stem}")

        cells.append(
            {
                "cell": stem,
                "result": result_path.name,
                "process_wall_s": float(end_path.read_text())
                - float(start_path.read_text()),
                "inference_wall_s": duration,
                "window_start": datetime.fromtimestamp(inference_start)
                .astimezone()
                .isoformat(),
                "window_end": datetime.fromtimestamp(inference_end)
                .astimezone()
                .isoformat(),
                "samples": len(window),
                "gpu_power_w_mean": fmean(sample["gpu_power"] for sample in window),
                "gpu_power_w_max": max(sample["gpu_power"] for sample in window),
                "gpu_effective_usage_mean": fmean(
                    sample["gpu_usage"][1] for sample in window
                ),
                "gpu_frequency_mhz_mean": fmean(
                    sample["gpu_usage"][0] for sample in window
                ),
                "macmon_sha256": sha256(macmon_path),
            }
        )

    return {
        "host": "[bench-host]",
        "tool": "macmon 0.7.2",
        "interval_ms": 100,
        "metric_scope": "system-wide",
        "usage_metric": "frequency-scaled effective GPU utilization ratio",
        "method": (
            "Each inference window has the result JSON's final-pass duration and ends "
            "at the final sample with at least 1 W GPU power and 5% effective usage."
        ),
        "raw_logs": "[bench-user-home]/bench-tools/graduation-probe/results/raw/",
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.raw_dir), indent=2) + "\n")


if __name__ == "__main__":
    main()
