#!/usr/bin/env python3
"""Summarize metal_fixed_cost probe runs.

Usage: summarize.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR holds `<family>.json` (the probe's output) and `<family>.stderr`
(its stderr with SYNAPSE_EMBED_PROFILE=1). Only measured replays are counted;
the warm-up replay's lines are skipped. Prints a per-stage table and a
least-squares fit of each pass's GPU run time against the tokens the graph
computes (batch capacity x sequence bucket, padding included).
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIELD = re.compile(r"(\w+)=([^\s]+)")
MARKER = "[metal-fixed-cost] call "
PROFILE = "[synapse-embed-profile] "


def fields(line):
    return {key: value for key, value in FIELD.findall(line)}


def parse(stderr_path):
    """Returns a list of engine calls; each has its marker fields and its passes."""
    calls = []
    current = None
    for line in stderr_path.read_text().splitlines():
        if line.startswith(MARKER):
            current = fields(line)
            current["passes"] = []
            current["lines"] = defaultdict(list)
            if current["replay"] != "warmup":
                calls.append(current)
            continue
        if current is None or not line.startswith(PROFILE):
            continue
        body = line[len(PROFILE):]
        name = body.split(" ", 1)[0]
        values = fields(body)
        if name == "bucket_select":
            current["passes"].append({"select": values})
        elif name in ("pass_embed_in", "pass_native", "pass_metal", "pass_host"):
            current["passes"][-1][name] = values
        elif name in ("family_return", "bucket_total"):
            current["lines"][name].append(values)
    return calls


def ms(record, key):
    value = float(record.get(key, "nan"))
    return value


def fit(xs, ys):
    """Ordinary least squares y = a + b x; returns (a, b, r^2)."""
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    return intercept, slope, (1 - ss_res / ss_tot) if ss_tot else 1.0


def summarize(run_dir, family):
    output = json.loads((run_dir / f"{family}.json").read_text())
    calls = parse(run_dir / f"{family}.stderr")
    replays = len(output["replays"])
    passes = [p for call in calls for p in call["passes"]]
    totals = defaultdict(float)
    shapes = Counter()
    real_tokens = computed_tokens = padded_rows = 0
    run_points = []
    gpu_points = []
    for call in calls:
        totals["engine_call"] += sum(float(v["total_ms"]) for v in call["lines"]["bucket_total"])
    for p in passes:
        select = p["select"]
        batch, seq = (int(v) for v in select["shape"].split("x"))
        shapes[f"{batch}x{seq}"] += 1
        host = p["pass_host"]
        native = p["pass_native"]
        metal = p["pass_metal"]
        rows = int(host["rows"])
        real_tokens += int(host["tokens"])
        computed_tokens += batch * seq
        padded_rows += batch - rows
        totals["pad"] += ms(host, "pad_ms")
        totals["embed_in"] += ms(p["pass_embed_in"], "embed_in_ms")
        totals["host_prep"] += ms(metal, "prep_ms")
        totals["select"] += ms(native, "select_ms")
        totals["upload"] += ms(native, "upload_ms")
        totals["run"] += ms(native, "run_ms")
        encode = ms(native, "encode_ms")
        gpu = ms(native, "gpu_ms")
        if gpu >= 0:
            totals["encode"] += encode
            totals["gpu"] += gpu
            gpu_points.append((batch * seq, gpu))
        totals["readback"] += ms(native, "readback_ms")
        totals["native"] += ms(native, "native_ms")
        totals["metal"] += ms(metal, "native_ms")
        totals["decode"] += ms(metal, "decode_ms")
        totals["pool"] += ms(host, "pool_ms")
        totals["pass_total"] += ms(host, "total_ms")
        run_points.append((batch * seq, ms(native, "run_ms")))
        if native.get("cached") != "1":
            totals["uncached_passes"] += 1
    replay = output["replays"][0]
    return {
        "family": family,
        "replays": replays,
        "wall_s": statistics.fmean(r["wall_s"] for r in output["replays"]),
        "tokenize_ms": statistics.fmean(r["tokenize_ms"] for r in output["replays"]),
        "engine_ms": statistics.fmean(r["engine_ms"] for r in output["replays"]),
        "calls": len(replay["calls"]),
        "engine_calls": len(calls) // replays,
        "passes": len(passes) // replays,
        "rows": output["rows"],
        "real_tokens": real_tokens // replays,
        "computed_tokens": computed_tokens // replays,
        "padded_rows": padded_rows // replays,
        "shapes": {k: v // replays for k, v in sorted(shapes.items(), key=lambda kv: tuple(map(int, kv[0].split("x"))))},
        "totals_ms": {k: v / replays for k, v in totals.items()},
        "run_fit": fit(*zip(*run_points)) if run_points else None,
        "gpu_fit": fit(*zip(*gpu_points)) if gpu_points else None,
        "vectors_sha256": output["vectors_sha256"],
    }


def main():
    for run_dir in map(Path, sys.argv[1:]):
        for family in ("gte-modernbert", "qwen3-0.6b"):
            if not (run_dir / f"{family}.json").exists():
                continue
            summary = summarize(run_dir, family)
            print(f"== {run_dir.name} / {family}")
            print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
