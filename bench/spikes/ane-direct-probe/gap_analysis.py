#!/usr/bin/env python3
"""Summarize a `modernbert_full --blocks` dispatch-gap run.

Reads the per-call JSON written by `--per-call-out` and prints a Markdown
summary per arm and per block. With `--svg PATH` it also draws every block's
per-call series as one strip per block on a shared log-scale axis, so a stall
pattern is visible instead of averaged away. With `--csv PATH` it flattens the
per-call records (dispatch times joined by `;`).

A "stall-signature" call is one slower than three times its arm's median
per-call wall time, the shape of the scheduling stalls reported for Core AI.
"""

import argparse
import csv
import json
import math
import statistics
import sys


def percentile(values, fraction):
    """Nearest-rank percentile; values need not be sorted."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def arm_label(gap):
    return f"{gap:g} ms"


def summarize(data):
    arms = sorted({block["gap_ms"] for block in data["blocks"]})
    calls_by_arm = {arm: [c for c in data["calls"] if c["gap_ms"] == arm] for arm in arms}
    blocks_by_arm = {arm: [b for b in data["blocks"] if b["gap_ms"] == arm] for arm in arms}
    pooled_median = statistics.median(c["wall_ms"] for c in data["calls"])
    rows = []
    for arm in arms:
        calls = calls_by_arm[arm]
        walls = [c["wall_ms"] for c in calls]
        median = statistics.median(walls)
        dispatch_count = len(calls[0]["dispatch_ms"])
        dispatch_medians = [
            statistics.median(c["dispatch_ms"][i] for c in calls) for i in range(dispatch_count)
        ]
        dispatch_stalls = sum(
            1
            for c in calls
            for i, value in enumerate(c["dispatch_ms"])
            if value > 3 * dispatch_medians[i]
        )
        call_ms = sum(b["call_ms_sum"] for b in blocks_by_arm[arm])
        elapsed_ms = sum(b["elapsed_ms"] for b in blocks_by_arm[arm])
        rows.append(
            {
                "gap_ms": arm,
                "blocks": len(blocks_by_arm[arm]),
                "calls": len(calls),
                "median_ms": median,
                "p95_ms": percentile(walls, 0.95),
                "p99_ms": percentile(walls, 0.99),
                "max_ms": max(walls),
                "stalls_3x_arm_median": sum(1 for w in walls if w > 3 * median),
                "stalls_3x_pooled_median": sum(1 for w in walls if w > 3 * pooled_median),
                "over_1_5x_arm_median": sum(1 for w in walls if w > 1.5 * median),
                "dispatch_stalls_3x": dispatch_stalls,
                "dispatches": dispatch_count * len(calls),
                "max_dispatch_ms": max(max(c["dispatch_ms"]) for c in calls),
                "throughput_excl_gap_per_s": len(calls) / (call_ms / 1000.0),
                "throughput_incl_gap_per_s": len(calls) / (elapsed_ms / 1000.0),
                "mean_slept_ms": statistics.mean(c["slept_ms"] for c in calls),
                "mismatches": sum(b["output_mismatches"] for b in blocks_by_arm[arm]),
                "block_medians_ms": [
                    statistics.median(c["wall_ms"] for c in calls if c["block"] == b["block"])
                    for b in blocks_by_arm[arm]
                ],
            }
        )
    block_rows = []
    for block in data["blocks"]:
        walls = [c["wall_ms"] for c in data["calls"] if c["block"] == block["block"]]
        median = statistics.median(walls)
        block_rows.append(
            {
                "block": block["block"],
                "gap_ms": block["gap_ms"],
                "load_1m_start": block["load_1m_start"],
                "load_5m_start": block["load_5m_start"],
                "load_1m_end": block["load_1m_end"],
                "median_ms": median,
                "p95_ms": percentile(walls, 0.95),
                "p99_ms": percentile(walls, 0.99),
                "max_ms": max(walls),
                "stalls_3x_block_median": sum(1 for w in walls if w > 3 * median),
                "mismatches": block["output_mismatches"],
            }
        )
    return {"pooled_median_ms": pooled_median, "arms": rows, "blocks": block_rows}


def markdown(summary):
    out = [
        "| arm | blocks | calls | median | p95 | p99 | max | >3x arm median | >3x pooled median | >1.5x arm median | dispatches >3x their median | max dispatch | calls/s excl. gap | calls/s incl. gap | output mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        out.append(
            f"| {arm_label(arm['gap_ms'])} | {arm['blocks']} | {arm['calls']} | {arm['median_ms']:.2f} | "
            f"{arm['p95_ms']:.2f} | {arm['p99_ms']:.2f} | {arm['max_ms']:.2f} | "
            f"{arm['stalls_3x_arm_median']} | {arm['stalls_3x_pooled_median']} | {arm['over_1_5x_arm_median']} | "
            f"{arm['dispatch_stalls_3x']} / {arm['dispatches']} | {arm['max_dispatch_ms']:.2f} | "
            f"{arm['throughput_excl_gap_per_s']:.2f} | {arm['throughput_incl_gap_per_s']:.2f} | {arm['mismatches']} |"
        )
    out += [
        "",
        "| block | arm | 1m load start→end | 5m load start | median | p95 | p99 | max | >3x block median | mismatches |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for block in summary["blocks"]:
        out.append(
            f"| {block['block']} | {arm_label(block['gap_ms'])} | {block['load_1m_start']:.2f}→{block['load_1m_end']:.2f} | "
            f"{block['load_5m_start']:.2f} | {block['median_ms']:.2f} | {block['p95_ms']:.2f} | "
            f"{block['p99_ms']:.2f} | {block['max_ms']:.2f} | {block['stalls_3x_block_median']} | {block['mismatches']} |"
        )
    return "\n".join(out)


def svg(data, summary):
    """One strip per block, shared log y-axis from 10 ms to the slowest call."""
    strip_h, strip_w, left, top = 60, 800, 150, 30
    blocks = data["blocks"]
    floor = 10.0
    ceiling = max(max(c["wall_ms"] for c in data["calls"]), 100.0)
    log_span = math.log10(ceiling) - math.log10(floor)
    height = top + strip_h * len(blocks) + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + strip_w + 20}" height="{height}" font-family="monospace" font-size="11">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="18">per-call wall ms (log y, {floor:g} to {ceiling:.0f} ms) by call index; red line = 3x arm median</text>',
    ]
    medians = {arm["gap_ms"]: arm["median_ms"] for arm in summary["arms"]}
    colors = {}
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
    for index, arm in enumerate(sorted(medians)):
        colors[arm] = palette[index % len(palette)]

    def y_of(value, base):
        clamped = min(max(value, floor), ceiling)
        return base + strip_h - 6 - (math.log10(clamped) - math.log10(floor)) / log_span * (strip_h - 12)

    for row, block in enumerate(blocks):
        base = top + row * strip_h
        calls = [c for c in data["calls"] if c["block"] == block["block"]]
        n = max(len(calls) - 1, 1)
        points = " ".join(
            f"{left + i * strip_w / n:.1f},{y_of(c['wall_ms'], base):.1f}" for i, c in enumerate(calls)
        )
        parts.append(
            f'<rect x="{left}" y="{base}" width="{strip_w}" height="{strip_h - 2}" fill="none" stroke="#ccc"/>'
        )
        parts.append(
            f'<text x="4" y="{base + strip_h / 2 + 4}">block {block["block"]:>2} gap {block["gap_ms"]:g}  load {block["load_1m_start"]:.1f}</text>'
        )
        threshold = y_of(3 * medians[block["gap_ms"]], base)
        parts.append(
            f'<line x1="{left}" x2="{left + strip_w}" y1="{threshold:.1f}" y2="{threshold:.1f}" stroke="red" stroke-width="0.5"/>'
        )
        parts.append(
            f'<polyline fill="none" stroke="{colors[block["gap_ms"]]}" stroke-width="0.7" points="{points}"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_csv(data, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["block", "gap_ms", "index_in_block", "row", "started_ms", "wall_ms", "slept_ms", "identical", "dispatch_ms"]
        )
        for c in data["calls"]:
            writer.writerow(
                [
                    c["block"],
                    c["gap_ms"],
                    c["index_in_block"],
                    c["row"],
                    c["started_ms"],
                    c["wall_ms"],
                    c["slept_ms"],
                    int(c["identical"]),
                    ";".join(f"{v:g}" for v in c["dispatch_ms"]),
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_call_json")
    parser.add_argument("--svg")
    parser.add_argument("--csv")
    parser.add_argument("--summary-json")
    args = parser.parse_args()
    with open(args.per_call_json) as handle:
        data = json.load(handle)
    summary = summarize(data)
    print(markdown(summary))
    if args.svg:
        with open(args.svg, "w") as handle:
            handle.write(svg(data, summary))
    if args.csv:
        write_csv(data, args.csv)
    if args.summary_json:
        with open(args.summary_json, "w") as handle:
            json.dump(summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
