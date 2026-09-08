#!/usr/bin/env python3
"""Compare two serial embed_bucket_probe runs and retain engine-reported shapes."""

import argparse
import json
import math
import re
from pathlib import Path

BUCKET = re.compile(
    r"bucket_select call=\d+ items=(\d+) max_tokens=(\d+) shape=(\d+)x(\d+)"
)
TOTAL = re.compile(r"bucket_total items=\d+ bucket_calls=\d+ total_ms=([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-stderr", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-stderr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def invocations(path):
    result = []
    current = []
    for line in path.read_text().splitlines():
        match = BUCKET.search(line)
        if match:
            items, max_tokens, batch, seq = map(int, match.groups())
            current.append(
                {
                    "items": items,
                    "max_tokens": max_tokens,
                    "batch": batch,
                    "seq": seq,
                }
            )
        match = TOTAL.search(line)
        if match:
            result.append(
                {"shapes": current, "engine_wall_s": float(match.group(1)) / 1000.0}
            )
            current = []
    return result


def load_run(data_path, stderr_path):
    data = json.loads(data_path.read_text())
    calls = invocations(stderr_path)
    repeats = data["metadata"]["measured_repeats_per_case"]
    calls_per_case = data["metadata"]["warmup_runs_per_case"] + repeats
    expected = len(data["cases"]) * calls_per_case
    if len(calls) != expected:
        raise ValueError(f"{stderr_path}: expected {expected} calls, found {len(calls)}")
    return data, calls, calls_per_case


def case_metrics(run, index):
    data, calls, calls_per_case = run
    case = data["cases"][index]
    first_call = calls[index * calls_per_case]
    measured_call = calls[index * calls_per_case + 1]
    shapes = measured_call["shapes"]
    padded = sum(shape["batch"] * shape["seq"] for shape in shapes)
    real = case["real_tokens"]
    walls = case["warm_engine_wall_s"]
    rates = [real / wall for wall in walls]
    return {
        "shapes": shapes,
        "real_tokens": real,
        "padded_tokens": padded,
        "padding_ratio_padded_over_real": padded / real,
        "padding_waste_fraction": (padded - real) / padded,
        "first_use_engine_wall_s": first_call["engine_wall_s"],
        "warm_engine_wall_s": walls,
        "mean_warm_engine_wall_s": sum(walls) / len(walls),
        "real_tokens_per_s": rates,
        "mean_real_tokens_per_s": sum(rates) / len(rates),
    }


def compare_vectors(left, right):
    max_abs = 0.0
    min_cosine = 1.0
    for left_row, right_row in zip(left, right):
        max_abs = max(max_abs, max(abs(x - y) for x, y in zip(left_row, right_row)))
        dot = sum(x * y for x, y in zip(left_row, right_row))
        left_norm = math.sqrt(sum(x * x for x in left_row))
        right_norm = math.sqrt(sum(y * y for y in right_row))
        min_cosine = min(min_cosine, dot / (left_norm * right_norm))
    return {
        "exact_vectors": left == right,
        "max_abs_diff": max_abs,
        "min_cosine": min_cosine,
    }


def main():
    args = parse_args()
    baseline = load_run(args.baseline, args.baseline_stderr)
    candidate = load_run(args.candidate, args.candidate_stderr)
    if baseline[0]["metadata"]["case_order"] != candidate[0]["metadata"]["case_order"]:
        raise ValueError("case order differs")
    output = {
        "baseline_metadata": baseline[0]["metadata"],
        "candidate_metadata": candidate[0]["metadata"],
        "cold_load_s": {
            "baseline": baseline[0]["cold_load_s"],
            "candidate": candidate[0]["cold_load_s"],
        },
        "cases": [],
    }
    for index, name in enumerate(baseline[0]["metadata"]["case_order"]):
        baseline_metrics = case_metrics(baseline, index)
        candidate_metrics = case_metrics(candidate, index)
        comparison = compare_vectors(
            baseline[0]["cases"][index]["vectors"],
            candidate[0]["cases"][index]["vectors"],
        )
        output["cases"].append(
            {
                "name": name,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "candidate_speedup_real_tokens_per_s": candidate_metrics[
                    "mean_real_tokens_per_s"
                ]
                / baseline_metrics["mean_real_tokens_per_s"],
                "vector_comparison": comparison,
            }
        )
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    for case in output["cases"]:
        baseline_metrics = case["baseline"]
        candidate_metrics = case["candidate"]
        print(
            f"{case['name']:14} "
            f"{baseline_metrics['mean_warm_engine_wall_s'] * 1000:.3f}ms -> "
            f"{candidate_metrics['mean_warm_engine_wall_s'] * 1000:.3f}ms; "
            f"{baseline_metrics['mean_real_tokens_per_s']:.1f} -> "
            f"{candidate_metrics['mean_real_tokens_per_s']:.1f} tok/s; "
            f"{case['candidate_speedup_real_tokens_per_s']:.2f}x"
        )


if __name__ == "__main__":
    main()
