#!/usr/bin/env python3
"""Summarize macmon power samples inside a benchmark epoch window."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macmon", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--end", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = float(args.start.read_text(encoding="utf-8"))
    end = float(args.end.read_text(encoding="utf-8"))
    rows = []
    for line in args.macmon.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        timestamp = datetime.fromisoformat(row["timestamp"]).timestamp()
        if start <= timestamp <= end:
            rows.append(row)
    if not rows:
        raise ValueError("no macmon samples fall inside the benchmark window")
    metrics = {}
    for name in ("ane_power", "gpu_power", "cpu_power", "all_power", "ram_power", "sys_power"):
        values = [float(row[name]) for row in rows]
        metrics[name] = {
            "mean_w": statistics.fmean(values),
            "p50_w": statistics.median(values),
            "p95_w": percentile(values, 0.95),
            "max_w": max(values),
        }
    report = {
        "sample_count": len(rows),
        "start_epoch": start,
        "end_epoch": end,
        "duration_s": end - start,
        "first_sample": rows[0]["timestamp"],
        "last_sample": rows[-1]["timestamp"],
        "power": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
