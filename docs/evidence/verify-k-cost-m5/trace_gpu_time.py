#!/usr/bin/env python3
"""Per-command-buffer GPU time from a Metal System Trace of `verify_k_cost`.

Run `verify_k_cost` in trace mode under `xctrace record --template
'Metal System Trace'` (see README.md), then:

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    python3 trace_gpu_time.py verify_k_trace.trace

What it reads: the trace's `metal-gpu-intervals` table, the GPU activity
intervals for every process. It keeps the compute intervals of the
`verify_k_cost` process and sums them per command buffer. When other
processes use the GPU at the same time, one command buffer shows up as
several intervals with gaps between them. Summing the intervals leaves out
those gaps, so the total is the time the GPU actually spent on our work.

In trace mode every verify call is exactly one command buffer, and each
(depth, K) phase is surrounded by two seconds of idle time. This script splits
the command buffers into clusters wherever the gap between them is larger
than `--gap` seconds, and prints the cluster count and GPU-time statistics for
each cluster. Match clusters to phases by order and by command-buffer count:
the program log prints one `TRACE PHASE ... calls=N` line per phase, in the
same order. Clusters whose count is not N are setup (model upload, prefill,
correctness guard, warm-up).

Limit: Metal System Trace merges all the compute encoders in one command
buffer into a single "Compute Command 0" interval. This gives GPU time per
verify call, not per kernel.
"""

import argparse
import math
import os
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET


def export_intervals(trace):
    xpath = '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'
    env = dict(os.environ)
    env.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    return subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath],
        check=True,
        capture_output=True,
        env=env,
    ).stdout


def rows(xml_bytes):
    root = ET.fromstring(xml_bytes)
    columns = [col.findtext("mnemonic") for col in root.iter("col")]
    # xctrace writes each distinct value once with an id and then refers back
    # to it with ref=; resolve refs before reading.
    by_id = {el.attrib["id"]: el for el in root.iter() if "id" in el.attrib}

    def resolve(el):
        return by_id[el.attrib["ref"]] if "ref" in el.attrib else el

    for row in root.iter("row"):
        cells = [resolve(el) for el in row]
        yield dict(zip(columns, cells))


def percentile(sorted_values, q):
    # Nearest rank, matching the Rust harness.
    rank = max(1, min(len(sorted_values), math.ceil(q * len(sorted_values))))
    return sorted_values[rank - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--process", default="verify_k_cost")
    parser.add_argument("--gap", type=float, default=1.0, help="cluster split, seconds")
    args = parser.parse_args()

    buffers = {}
    for row in rows(export_intervals(args.trace)):
        process = row.get("process")
        channel = row.get("channel-name")
        buffer = row.get("cmdbuffer-id")
        if process is None or channel is None or buffer is None:
            continue
        if not process.attrib.get("fmt", "").startswith(args.process):
            continue
        if channel.text != "Compute":
            continue
        start_ns = int(row["start"].text)
        duration_ns = int(row["duration"].text)
        entry = buffers.setdefault(buffer.text, [start_ns, 0])
        entry[0] = min(entry[0], start_ns)
        entry[1] += duration_ns

    if not buffers:
        sys.exit(f"no compute intervals for process {args.process!r} in {args.trace}")

    ordered = sorted(buffers.values())
    clusters = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] - previous[0] > args.gap * 1e9:
            clusters.append([])
        clusters[-1].append(current)

    print(f"{'cluster':>7} {'t_start_s':>10} {'buffers':>8} {'median_ms':>10} "
          f"{'p10_ms':>8} {'p90_ms':>8} {'min_ms':>8} {'max_ms':>8}")
    for index, cluster in enumerate(clusters):
        gpu_ms = sorted(duration / 1e6 for _, duration in cluster)
        print(f"{index:>7} {cluster[0][0] / 1e9:>10.3f} {len(cluster):>8} "
              f"{statistics.median(gpu_ms):>10.3f} {percentile(gpu_ms, 0.1):>8.3f} "
              f"{percentile(gpu_ms, 0.9):>8.3f} {gpu_ms[0]:>8.3f} {gpu_ms[-1]:>8.3f}")


if __name__ == "__main__":
    main()
