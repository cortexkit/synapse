#!/usr/bin/env python3
"""Validate and compare two serial embed_bucket_probe runs."""

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

BUCKET = re.compile(
    r"bucket_select call=\d+ items=(\d+) max_tokens=(\d+) shape=(\d+)x(\d+)"
)
TOTAL = re.compile(r"bucket_total items=\d+ bucket_calls=\d+ total_ms=([0-9.]+)")
MATCHED_METADATA = (
    "probe_version",
    "family",
    "dtype",
    "model_config_sha256",
    "model_weights_sha256",
    "tokenizer_sha256",
    "max_tokens",
    "attention_units",
    "execution",
    "first_use_runs_per_case",
    "measured_repeats_per_case",
    "case_order",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-stderr", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-stderr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def parse_invocations(path):
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
    if current:
        raise ValueError(f"{path}: unterminated profile invocation")
    return result


def vector_sha256(vector):
    digest = hashlib.sha256()
    for value in vector:
        digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def validate_run(data, calls, label):
    metadata = data.get("metadata", {})
    repeats = metadata.get("measured_repeats_per_case")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError(f"{label}: measured repeat count must be positive")
    if metadata.get("first_use_runs_per_case") != 1:
        raise ValueError(f"{label}: exactly one first-use run is required")
    cases = data.get("cases")
    if not isinstance(cases, list) or metadata.get("case_order") != [
        case.get("name") for case in cases
    ]:
        raise ValueError(f"{label}: case order metadata does not match cases")
    calls_per_case = repeats + 1
    if len(calls) != len(cases) * calls_per_case:
        raise ValueError(
            f"{label}: expected {len(cases) * calls_per_case} profile calls, found {len(calls)}"
        )

    vector_dims = set()
    for case_index, case in enumerate(cases):
        rows = case.get("rows")
        vectors = case.get("vectors")
        walls = case.get("warm_engine_wall_s")
        digests = case.get("repeat_vector_sha256")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{label}/{case['name']}: rows must be non-empty")
        if not isinstance(vectors, list) or len(vectors) != len(rows):
            raise ValueError(f"{label}/{case['name']}: vector row count mismatch")
        if not isinstance(walls, list) or len(walls) != repeats:
            raise ValueError(f"{label}/{case['name']}: measured wall count mismatch")
        if not all(isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0 for wall in walls):
            raise ValueError(f"{label}/{case['name']}: measured walls must be finite and positive")
        if not isinstance(case.get("first_use_engine_wall_s"), (int, float)) or case[
            "first_use_engine_wall_s"
        ] <= 0:
            raise ValueError(f"{label}/{case['name']}: first-use wall must be positive")
        if not isinstance(digests, list) or len(digests) != repeats:
            raise ValueError(f"{label}/{case['name']}: repeat digest count mismatch")

        real_tokens = 0
        expected_digests = []
        for row_index in range(len(rows)):
            row = rows[row_index]
            vector = vectors[row_index]
            input_ids = row.get("input_ids")
            if not isinstance(input_ids, list) or row.get("target_tokens") != len(input_ids):
                raise ValueError(f"{label}/{case['name']}: input length mismatch at row {row_index}")
            real_tokens += len(input_ids)
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"{label}/{case['name']}: empty vector at row {row_index}")
            vector_dims.add(len(vector))
            expected_digests.append(vector_sha256(vector))
        if case.get("real_tokens") != real_tokens:
            raise ValueError(f"{label}/{case['name']}: real token total mismatch")
        for repeat_index, repeat_digests in enumerate(digests):
            if repeat_digests != expected_digests:
                raise ValueError(
                    f"{label}/{case['name']}: repeat {repeat_index} vector digest mismatch"
                )

        for invocation_index in range(calls_per_case):
            invocation = calls[case_index * calls_per_case + invocation_index]
            if sum(shape["items"] for shape in invocation["shapes"]) != len(rows):
                raise ValueError(
                    f"{label}/{case['name']}: profile row count mismatch in invocation {invocation_index}"
                )
            if any(shape["batch"] < shape["items"] for shape in invocation["shapes"]):
                raise ValueError(f"{label}/{case['name']}: profile shape does not cover rows")
    if len(vector_dims) != 1:
        raise ValueError(f"{label}: inconsistent vector dimensions: {sorted(vector_dims)}")
    return calls_per_case


def validate_compatible(baseline, candidate):
    for field in MATCHED_METADATA:
        if baseline["metadata"].get(field) != candidate["metadata"].get(field):
            raise ValueError(f"metadata mismatch: {field}")
    baseline_cases = baseline["cases"]
    candidate_cases = candidate["cases"]
    if len(baseline_cases) != len(candidate_cases):
        raise ValueError("case count mismatch")
    for case_index in range(len(baseline_cases)):
        left = baseline_cases[case_index]
        right = candidate_cases[case_index]
        if left["name"] != right["name"]:
            raise ValueError(f"case name mismatch at index {case_index}")
        if left["rows"] != right["rows"]:
            raise ValueError(f"input rows or token IDs mismatch for {left['name']}")
        if left["real_tokens"] != right["real_tokens"]:
            raise ValueError(f"real token mismatch for {left['name']}")
        if len(left["vectors"]) != len(right["vectors"]):
            raise ValueError(f"vector row count mismatch for {left['name']}")
        for row_index in range(len(left["vectors"])):
            if len(left["vectors"][row_index]) != len(right["vectors"][row_index]):
                raise ValueError(f"vector dimension mismatch for {left['name']} row {row_index}")


def load_run(data_path, stderr_path, label):
    data = json.loads(data_path.read_text())
    calls = parse_invocations(stderr_path)
    calls_per_case = validate_run(data, calls, label)
    return data, calls, calls_per_case


def case_metrics(run, index):
    data, calls, calls_per_case = run
    case = data["cases"][index]
    first_call = calls[index * calls_per_case]
    measured_calls = calls[index * calls_per_case + 1 : (index + 1) * calls_per_case]
    shapes = measured_calls[0]["shapes"]
    if any(call["shapes"] != shapes for call in measured_calls[1:]):
        raise ValueError(f"{case['name']}: measured profile shapes changed between repeats")
    padded = sum(shape["batch"] * shape["seq"] for shape in shapes)
    real = case["real_tokens"]
    walls = case["warm_engine_wall_s"]
    total_time = sum(walls)
    return {
        "shapes": shapes,
        "real_tokens": real,
        "padded_tokens": padded,
        "padding_ratio_padded_over_real": padded / real,
        "padding_waste_fraction": (padded - real) / padded,
        "first_use_engine_wall_s": case["first_use_engine_wall_s"],
        "profile_first_use_engine_wall_s": first_call["engine_wall_s"],
        "warm_engine_wall_s": walls,
        "mean_warm_engine_wall_s": total_time / len(walls),
        "aggregate_real_tokens_per_s": real * len(walls) / total_time,
    }


def compare_vectors(left, right):
    if len(left) != len(right):
        raise ValueError("vector row count mismatch")
    exact = left == right
    max_abs = 0.0
    min_cosine = 1.0
    for row_index in range(len(left)):
        left_row = left[row_index]
        right_row = right[row_index]
        if len(left_row) != len(right_row):
            raise ValueError(f"vector dimension mismatch at row {row_index}")
        max_abs = max(max_abs, max(abs(left_row[i] - right_row[i]) for i in range(len(left_row))))
        dot = sum(left_row[i] * right_row[i] for i in range(len(left_row)))
        left_norm = math.sqrt(sum(value * value for value in left_row))
        right_norm = math.sqrt(sum(value * value for value in right_row))
        if left_norm == 0 or right_norm == 0:
            raise ValueError(f"zero-norm vector at row {row_index}")
        min_cosine = min(min_cosine, dot / (left_norm * right_norm))
    return {"exact_vectors": exact, "max_abs_diff": max_abs, "min_cosine": min_cosine}


def build_comparison(baseline, candidate):
    validate_compatible(baseline[0], candidate[0])
    output = {
        "baseline_metadata": baseline[0]["metadata"],
        "candidate_metadata": candidate[0]["metadata"],
        "cold_load_s": {
            "baseline": baseline[0]["cold_load_s"],
            "candidate": candidate[0]["cold_load_s"],
        },
        "cases": [],
    }
    baseline_total_tokens = 0
    candidate_total_tokens = 0
    baseline_total_time = 0.0
    candidate_total_time = 0.0
    repeats = baseline[0]["metadata"]["measured_repeats_per_case"]
    for index, name in enumerate(baseline[0]["metadata"]["case_order"]):
        baseline_metrics = case_metrics(baseline, index)
        candidate_metrics = case_metrics(candidate, index)
        comparison = compare_vectors(
            baseline[0]["cases"][index]["vectors"],
            candidate[0]["cases"][index]["vectors"],
        )
        baseline_total_tokens += baseline_metrics["real_tokens"] * repeats
        candidate_total_tokens += candidate_metrics["real_tokens"] * repeats
        baseline_total_time += sum(baseline_metrics["warm_engine_wall_s"])
        candidate_total_time += sum(candidate_metrics["warm_engine_wall_s"])
        output["cases"].append(
            {
                "name": name,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "candidate_speedup_real_tokens_per_s": candidate_metrics[
                    "aggregate_real_tokens_per_s"
                ]
                / baseline_metrics["aggregate_real_tokens_per_s"],
                "vector_comparison": comparison,
            }
        )
    output["aggregate"] = {
        "baseline_real_tokens_per_s": baseline_total_tokens / baseline_total_time,
        "candidate_real_tokens_per_s": candidate_total_tokens / candidate_total_time,
        "baseline_real_tokens": baseline_total_tokens,
        "candidate_real_tokens": candidate_total_tokens,
        "baseline_measured_wall_s": baseline_total_time,
        "candidate_measured_wall_s": candidate_total_time,
    }
    return output


def main():
    args = parse_args()
    baseline = load_run(args.baseline, args.baseline_stderr, "baseline")
    candidate = load_run(args.candidate, args.candidate_stderr, "candidate")
    output = build_comparison(baseline, candidate)
    args.out.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    for case in output["cases"]:
        baseline_metrics = case["baseline"]
        candidate_metrics = case["candidate"]
        print(
            f"{case['name']:24} "
            f"{baseline_metrics['mean_warm_engine_wall_s'] * 1000:.3f}ms -> "
            f"{candidate_metrics['mean_warm_engine_wall_s'] * 1000:.3f}ms; "
            f"{baseline_metrics['aggregate_real_tokens_per_s']:.1f} -> "
            f"{candidate_metrics['aggregate_real_tokens_per_s']:.1f} tok/s; "
            f"{case['candidate_speedup_real_tokens_per_s']:.2f}x"
        )


if __name__ == "__main__":
    main()
