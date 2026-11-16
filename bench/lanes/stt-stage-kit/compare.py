#!/usr/bin/env python3
"""Join matched-stage records by sealed clip identity, never by clock or duration."""
import argparse
import json
import math
from pathlib import Path
import statistics

STAGES = ("encoder_ms", "readback_pack_ms", "predictor_ms", "joint_ms",
          "greedy_bookkeeping_ms", "remaining_model_ms")
COUNTS = ("valid_frames", "padded_frames", "encoder_calls", "joint_steps", "predictor_calls", "accepted_tokens")


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def aggregate(records, expected, tolerance_ms=1.0):
    groups = {}
    run_ids = set()
    for original in records:
        if (original.get("population") != "corpus" or original.get("invocation_id") != "primary"
                or original.get("warm") is not True):
            continue
        row = dict(original)
        if row.get("schema") != "matched-stages-v1":
            raise ValueError("unknown stage schema")
        run_ids.add(row["run_id"])
        if row["engine"] not in ("synapse", "fluidaudio") or type(row["instrumented"]) is not bool:
            raise ValueError("unknown engine/instrumentation mode")
        if row["clip_id"] not in expected:
            raise ValueError(f"unsealed clip: {row['clip_id']}")
        key = row["engine"], row["instrumented"], row["clip_id"]
        group = groups.setdefault(key, {})
        if row["segment_id"] in group:
            raise ValueError(f"duplicate primary segment: {key}")
        group[row["segment_id"]] = row
    if len(run_ids) != 1:
        raise ValueError(f"expected one common run ID, got {run_ids}")
    result = {}
    for (engine, instrumented, clip), segments in groups.items():
        values = list(segments.values())
        count = values[0]["segment_count"]
        if type(count) is not int or count < 1:
            raise ValueError(f"invalid segment count: {clip}")
        aggregated = values[0].get("aggregated_segments", False)
        if aggregated:
            if len(values) != 1:
                raise ValueError(f"mixed aggregated/individual segments: {clip}")
        elif set(segments) != set(range(count)) or any(v["segment_count"] != count for v in values):
            raise ValueError(f"incomplete segments: {clip}")
        row = {"clip_id": clip, "segment_count": count,
               "rungs": [v["rung"] for v in values], "observation": values[0]["observation"]}
        fields = ("model_total_ms", "request_total_ms", *STAGES, *COUNTS)
        for field in fields:
            if all(field in v for v in values):
                row[field] = sum(finite(v[field], field) for v in values)
        if "model_total_ms" not in row:
            raise ValueError(f"missing model total: {clip}")
        # Old host records call joint evaluations decode_steps; retain the measured count.
        if all("decode_steps" in v for v in values):
            row["joint_steps"] = sum(finite(v["decode_steps"], "decode_steps") for v in values)
        if instrumented and engine == "synapse":
            if any(stage not in row for stage in STAGES):
                raise ValueError(f"missing stage: {clip}")
            # Validate each invocation as well as the aggregate: opposite errors must not cancel.
            residuals = [v["model_total_ms"] - sum(v[s] for s in STAGES) for v in values]
            if any(abs(r) > tolerance_ms for r in residuals):
                raise ValueError(f"model stage residual outside tolerance: {clip}: {residuals}")
            row["residual_ms"] = sum(residuals)
        if instrumented and engine == "fluidaudio":
            calls = ("encoder_ms", "predictor_ms", "joint_ms")
            if any(stage not in row for stage in calls):
                raise ValueError(f"missing FluidAudio call timers: {clip}")
            row["other_unsplit_ms"] = row["model_total_ms"] - sum(row[s] for s in calls)
            if row["other_unsplit_ms"] < -tolerance_ms:
                raise ValueError(f"FluidAudio calls exceed model span: {clip}")
        if instrumented and any(field not in row for field in COUNTS):
            raise ValueError(f"missing work counts: {clip}")
        result.setdefault((engine, instrumented), {})[clip] = row
    for engine in ("synapse", "fluidaudio"):
        for instrumented in (False, True):
            actual = set(result.get((engine, instrumented), {}))
            if actual != set(expected):
                raise ValueError(f"unmatched clips: {engine} instrumented={instrumented}: missing={sorted(set(expected)-actual)}")
    return result


def stats(values):
    values = sorted(values)
    return {"median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1],
            "mean": statistics.mean(values)}


def summarize(joined):
    summary = {}
    for engine in ("synapse", "fluidaudio"):
        rows = joined[engine, True]
        fields = ("model_total_ms", *STAGES, "other_unsplit_ms", "residual_ms", "request_total_ms", *COUNTS, "segment_count")
        stages = {field: stats([row[field] for row in rows.values()]) for field in fields
                  if all(field in row for row in rows.values())}
        on = sum(row["model_total_ms"] for row in rows.values())
        off = sum(row["model_total_ms"] for row in joined[engine, False].values())
        if off <= 0:
            raise ValueError("baseline warm total must be positive")
        median = stages["model_total_ms"]["median"]
        near = min(rows.values(), key=lambda row: (abs(row["model_total_ms"] - median), row["clip_id"]))
        summary[engine] = {"clips": len(rows), "stages": stages, "near_total_median": near,
                           "warm_total_off_ms": off, "warm_total_on_ms": on,
                           "instrumentation_overhead_percent": (on / off - 1) * 100}
    return summary


def table(summary):
    lines = ["| Stage (ms unless count) | Syn median | p95 | mean | Fluid median | p95 | mean | mean gap |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for field in ("model_total_ms", *STAGES, "other_unsplit_ms", "residual_ms", "request_total_ms", *COUNTS, "segment_count"):
        a = summary["synapse"]["stages"].get(field)
        b = summary["fluidaudio"]["stages"].get(field)
        cells = [f"{s[k]:.3f}" for s in (a, b) for k in ("median", "p95", "mean")] if a and b else [
            f"{s[k]:.3f}" if s else "unmeasured" for s in (a, b) for k in ("median", "p95", "mean")]
        gap = f"{a['mean'] - b['mean']:.3f}" if a and b else "—"
        lines.append("| " + " | ".join([field, *cells, gap]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=1.0)
    args = parser.parse_args()
    finite(args.tolerance_ms, "tolerance")
    request = json.loads(args.request.read_text())
    expected = [v["id"] for v in request["utterances"]]
    if len(expected) != 1200 or len(set(expected)) != 1200:
        raise ValueError("comparison requires 1200 unique sealed corpus IDs")
    records = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    summary = summarize(aggregate(records, expected, args.tolerance_ms))
    evidence = args.request.parent
    exits = json.loads((evidence / "exits.json").read_text())
    synapse = json.loads((evidence / "synapse-on.json").read_text())
    self_check = json.loads((evidence / "self-check.json").read_text())
    summary["synapse"]["accuracy"] = {"status": synapse["status"], "first_load_status": self_check["status"],
        "gate_failures": synapse.get("gate_failures"), "split_wer": synapse.get("split_wer")}
    summary["fluidaudio"]["accuracy"] = json.loads((evidence / "fluid-on-wer.json").read_text())
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(table(summary))
    print("\nBoundary: model-only, segment-summed. Fluid request_total_ms is the original WAV→text boundary;")
    print("Synapse frontend/IPC/formatting are unmeasured. Synapse observations ON; Fluid observations OFF.")
    print("Fluid other_unsplit_ms is a derived remainder, not a separately measured host-stage split.")
    print("Synapse residual_ms is independently checked per invocation; stage medians are not additive.")
    for engine, result in summary.items():
        print(f"{engine}: {result['clips']} clips; warm totals off/on {result['warm_total_off_ms']:.3f}/{result['warm_total_on_ms']:.3f} ms; "
              f"instrumentation overhead {result['instrumentation_overhead_percent']:+.3f}%")
        print(f"{engine} near-total-median clip: {json.dumps(result['near_total_median'], sort_keys=True)}")
        print(f"{engine} accuracy (not inferred from timing): {json.dumps(result['accuracy'], sort_keys=True)}")
    print("Native exit codes: " + json.dumps(exits, sort_keys=True))
    metadata = [dict(row.get("metadata", {}), call_compute_units=row.get("call_compute_units"))
                for row in records if row.get("engine") == "fluidaudio" and row.get("instrumented")]
    if metadata:
        print("FluidAudio configuration: " + json.dumps(metadata[0], sort_keys=True))


if __name__ == "__main__":
    main()
