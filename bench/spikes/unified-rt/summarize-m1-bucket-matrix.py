#!/usr/bin/env python3
"""Summarize macmon and package-inventory evidence for the M1 bucket matrix."""

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


def summarize_power(raw_dir: Path, result_dir: Path) -> dict:
    cells = []
    for result_path in sorted(result_dir.glob("*.json")):
        if result_path.name in {"package-invariants.json", "power-summary.json"}:
            continue
        stem = result_path.stem
        macmon_path = raw_dir / f"{stem}-macmon.jsonl"
        start_path = raw_dir / f"{stem}-start-epoch.txt"
        end_path = raw_dir / f"{stem}-end-epoch.txt"
        if not (macmon_path.exists() and start_path.exists() and end_path.exists()):
            continue

        result = json.loads(result_path.read_text())
        duration = result["passes"][-1]["infer_wall_s"]
        samples = read_samples(macmon_path)
        powered = [
            sample
            for sample in samples
            if sample["gpu_power"] >= 1.0 and sample["gpu_usage"][1] >= 0.05
        ]
        if not powered:
            raise RuntimeError(f"no powered GPU sample found for {stem}")
        steady_end = powered[-1]["epoch_s"]
        steady_start = steady_end - duration
        steady = [
            sample
            for sample in samples
            if steady_start <= sample["epoch_s"] <= steady_end
        ]
        if not steady:
            raise RuntimeError(f"no steady-window samples found for {stem}")

        cells.append(
            {
                "cell": stem,
                "result": result_path.name,
                "process_wall_s": float(end_path.read_text())
                - float(start_path.read_text()),
                "steady_infer_wall_s": duration,
                "steady_window_start": datetime.fromtimestamp(
                    steady_start
                ).astimezone().isoformat(),
                "steady_window_end": datetime.fromtimestamp(
                    steady_end
                ).astimezone().isoformat(),
                "samples": len(steady),
                "gpu_power_w_mean": fmean(sample["gpu_power"] for sample in steady),
                "gpu_power_w_max": max(sample["gpu_power"] for sample in steady),
                "gpu_effective_usage_mean": fmean(
                    sample["gpu_usage"][1] for sample in steady
                ),
                "gpu_effective_usage_max": max(
                    sample["gpu_usage"][1] for sample in steady
                ),
                "gpu_frequency_mhz_mean": fmean(
                    sample["gpu_usage"][0] for sample in steady
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
            "The steady window has the result JSON pass-3 duration and ends at the "
            "last sample with at least 1 W GPU power and 5% effective GPU usage."
        ),
        "raw_logs": "[bench-user-home]/bench-tools/unified-rt-serving/results/m1-bucket-matrix/",
        "cells": cells,
    }


def inventory_roots(path: Path) -> list[dict]:
    roots = []
    for line in path.read_text().splitlines():
        name, mtime, size = line.rsplit(r"\t", 2)
        if name.endswith(".mpsgraphpackage"):
            roots.append(
                {
                    "name": Path(name).name,
                    "mtime_epoch_s": int(mtime),
                    "directory_size_bytes": int(size),
                }
            )
    return roots


def summarize_packages(raw_dir: Path) -> dict:
    cells = []
    for before in sorted(raw_dir.glob("*-bucketed-*-packages-before.tsv")):
        stem = before.name.removesuffix("-packages-before.tsv")
        after = raw_dir / f"{stem}-packages-after.tsv"
        roots = inventory_roots(before)
        cells.append(
            {
                "cell": stem,
                "package_count": len(roots),
                "package_root_mtimes": roots,
                "recursive_inventory_entries": len(before.read_text().splitlines()),
                "before_sha256": sha256(before),
                "after_sha256": sha256(after),
                "unchanged_during_inference": before.read_bytes()
                == after.read_bytes(),
            }
        )
    return {
        "inventory_fields": ["path", "mtime_epoch_s", "size_bytes"],
        "method": (
            "MISS inventories were captured after ten package roots stabilized and "
            "before inference completed; HIT inventories were captured before launch. "
            "Each recursive inventory was compared byte-for-byte after the process."
        ),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    power = summarize_power(args.raw_dir, args.result_dir)
    packages = summarize_packages(args.raw_dir)
    (args.result_dir / "power-summary.json").write_text(
        json.dumps(power, indent=2) + "\n"
    )
    (args.result_dir / "package-invariants.json").write_text(
        json.dumps(packages, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
