#!/usr/bin/env python3
"""Build the certifiable A/B summary from graduation-probe result JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


FAMILIES = ("minilm", "qwen3", "gte-modernbert")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def mean(values: list[float]) -> float:
    return fmean(values)


def result_paths(root: Path, family: str, engine: str) -> list[Path]:
    if engine == "owned":
        return [root / f"owned-{family}-hit-r1.json", root / f"owned-{family}-hit-r2.json"]
    return [root / f"{engine}-{family}-r1.json", root / f"{engine}-{family}-r2.json"]


def parity_paths(root: Path, family: str, engine: str) -> list[Path]:
    return [root / f"{engine}-{family}-r1-parity.json", root / f"{engine}-{family}-r2-parity.json"]


def power_for(power: dict, names: list[str]) -> tuple[float, int]:
    by_name = {cell["cell"]: cell for cell in power["cells"]}
    cells = [by_name[name] for name in names]
    return mean([cell["gpu_power_w_mean"] for cell in cells]), sum(
        cell["samples"] for cell in cells
    )


def owned_row(root: Path, power: dict, family: str) -> dict:
    repeats = [load(path) for path in result_paths(root, family, "owned")]
    steady_path = root / f"owned-{family}-steady.json"
    steady = load(steady_path)
    final_pass = steady["passes"][-1]
    watts, samples = power_for(power, [f"owned-{family}-steady"])
    return {
        "engine": "owned-runtime-metal",
        "result_files": [path.name for path in result_paths(root, family, "owned")]
        + [steady_path.name],
        "load_boot_s_repeats": [result["cold_load_s"] for result in repeats],
        "load_boot_s_mean": mean([result["cold_load_s"] for result in repeats]),
        "timing_basis": "pass 3 (steady) of an in-process three-pass package-HIT run",
        "infer_wall_s_repeats": [final_pass["infer_wall_s"]],
        "tok_per_s_repeats": [final_pass["tok_per_s"]],
        "steady_tok_per_s": final_pass["tok_per_s"],
        "real_tokens": final_pass["input_tokens"],
        "gpu_power_w_mean": watts,
        "power_samples": samples,
        "parity_mean_cosine": final_pass["parity_mean_cosine"],
        "top10_rank_overlap": final_pass["top10_rank_overlap"],
        "parity_queries": 400,
    }


def incumbent_row(root: Path, power: dict, family: str, engine: str) -> dict:
    paths = result_paths(root, family, engine)
    repeats = [load(path) for path in paths]
    parities = [load(path) for path in parity_paths(root, family, engine)]
    watts, samples = power_for(power, [f"{engine}-{family}-r2"])
    return {
        "engine": "llama-server-metal" if engine == "llama" else "mlx-python",
        "result_files": [path.name for path in paths]
        + [path.name for path in parity_paths(root, family, engine)],
        "load_boot_s_repeats": [result["cold_load_s"] for result in repeats],
        "load_boot_s_mean": mean([result["cold_load_s"] for result in repeats]),
        "timing_basis": (
            "repeat 2 steady throughput after repeat 1 populated framework caches; "
            "each fresh process excluded one warmup batch"
        ),
        "infer_wall_s_repeats": [result["infer_wall_s"] for result in repeats],
        "tok_per_s_repeats": [result["tok_per_s"] for result in repeats],
        "steady_tok_per_s": repeats[-1]["tok_per_s"],
        "real_tokens": repeats[0]["input_tokens"],
        "gpu_power_w_mean": watts,
        "power_samples": samples,
        "parity_mean_cosine": mean([report["mean_cosine"] for report in parities]),
        "top10_rank_overlap": mean(
            [report["rank"]["mean_topk_overlap"] for report in parities]
        ),
        "parity_queries": parities[0]["rank"]["queries"],
    }


def summarize(root: Path, power_path: Path) -> dict:
    power = load(power_path)
    workloads = []
    for family in FAMILIES:
        rows = [owned_row(root, power, family), incumbent_row(root, power, family, "llama")]
        if family != "gte-modernbert":
            rows.append(incumbent_row(root, power, family, "mlx"))
        owned_tokens = rows[0]["real_tokens"]
        token_reconciliation = []
        for row in rows:
            reported_tokens = row["real_tokens"]
            token_reconciliation.append(
                {
                    "engine": row["engine"],
                    "reported_tokens": reported_tokens,
                    "canonical_real_tokens": owned_tokens,
                    "difference_vs_canonical_fraction": (
                        reported_tokens - owned_tokens
                    )
                    / owned_tokens,
                }
            )
            row["reported_tokens"] = reported_tokens
            row["reported_tok_per_s_repeats"] = row["tok_per_s_repeats"]
            row["real_tokens"] = owned_tokens
            row["tok_per_s_repeats"] = [
                owned_tokens / wall for wall in row["infer_wall_s_repeats"]
            ]
            row["steady_tok_per_s"] = row["tok_per_s_repeats"][-1]
        by_engine = {row["engine"]: row for row in rows}
        ratios = {
            "owned_vs_llama": by_engine["owned-runtime-metal"]["steady_tok_per_s"]
            / by_engine["llama-server-metal"]["steady_tok_per_s"]
        }
        if "mlx-python" in by_engine:
            ratios["owned_vs_mlx"] = by_engine["owned-runtime-metal"][
                "steady_tok_per_s"
            ] / by_engine["mlx-python"]["steady_tok_per_s"]
        workloads.append(
            {
                "family": family,
                "rows": rows,
                "token_reconciliation": token_reconciliation,
                "certified_throughput_ratios": ratios,
            }
        )

    return {
        "schema": "synapse-graduation-probe-v1",
        "host": "<bench-host> (Apple M1 Max, 64 GiB)",
        "corpus_rows": 400,
        "token_accounting": "real sanitized-tokenizer tokens",
        "power": {
            "tool": power["tool"],
            "interval_ms": power["interval_ms"],
            "metric_scope": power["metric_scope"],
        },
        "workloads": workloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("power_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(summarize(args.result_dir, args.power_summary), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
