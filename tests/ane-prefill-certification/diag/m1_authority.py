#!/usr/bin/env python3
"""Measure the M1 authority token battery without updating certification evidence.

The production certification driver owns the real worker and sidecar. This
measurement wrapper preserves its raw observations while continuing after a
first token fork so a report can describe every arm before an owner decision.
"""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SOURCE_CHECKPOINT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()


def load_context() -> dict[str, Any]:
    return {
        "captured_at_epoch_seconds": time.time(),
        "load_average": command_output(["sysctl", "-n", "vm.loadavg"]),
        "power": command_output(["pmset", "-g", "batt"]),
        "runner_worker": command_output(["pgrep", "-fl", "Runner.Worker"]),
    }


def cases_for_arm(symbols: dict[str, Any], bucket: int) -> list[dict[str, Any]]:
    cases = [case for case in symbols["fixture_cases"]() if case["bucket"] == bucket]
    cases.append(
        {
            "case_id": f"w{bucket}-chain-k-16",
            "kind": "chain_k",
            "bucket": bucket,
            "prompt_token_ids": symbols["deterministic_prompt"](
                f"w{bucket}-chain-k-16", min(96, bucket)
            ),
            "chain_k": 16,
        }
    )
    return cases


def require_generation_invariants(
    symbols: dict[str, Any], arm: Any, case: dict[str, Any], split: dict[str, Any]
) -> None:
    symbols["require"](
        split.get("padded_width") == arm.bucket,
        f"{case['case_id']} used the wrong fixed-width graph",
    )
    prompt_length = len(case["prompt_token_ids"])
    symbols["require"](
        split.get("first_token_index") == prompt_length - 1,
        f"{case['case_id']} read logits from a padded position",
    )
    symbols["require"](
        split.get("active_cache_positions") == prompt_length,
        f"{case['case_id']} imported padded cache positions",
    )
    symbols["require"](
        split.get("decode_cache_position") == prompt_length,
        f"{case['case_id']} started decode at the wrong cache position",
    )
    if arm.decode_config == "q8-step":
        symbols["require"](
            split.get("cache_handoff") == "engine_to_engine",
            f"{case['case_id']} bypassed q8 cache handoff",
        )


def driver_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.root / "tests/ane-prefill-certification/machine_driver.py"),
        "--root",
        str(args.root),
        "--checkpoint",
        str(args.checkpoint),
        "--artifacts",
        str(args.artifacts),
        "--worker",
        str(args.worker),
        "--constraint-compiler",
        str(args.constraint_compiler),
        "--sidecar",
        str(args.sidecar),
    ]


def run_arm(args: argparse.Namespace, symbols: dict[str, Any], arm: Any) -> dict[str, Any]:
    driver = symbols["JsonlDriver"](driver_command(args))
    started_context = load_context()
    try:
        certifier = symbols["Certifier"](args.root, driver)
        metadata = certifier.metadata()
        symbols["require"](
            metadata["source_checkpoint_digest"] == SOURCE_CHECKPOINT_SHA256,
            "driver source checkpoint digest does not match the pinned authority source",
        )
        artifact = certifier.validate_artifact(
            certifier.call("precondition", arm=arm.wire()), metadata["source_checkpoint_digest"]
        )
        rows: list[dict[str, Any]] = []
        for case in cases_for_arm(symbols, arm.bucket):
            row: dict[str, Any] = {
                "case_id": case["case_id"],
                "kind": case["kind"],
            }
            try:
                options = {
                    key: value
                    for key, value in (
                        ("grammar", case.get("grammar")),
                        ("chain_k", case.get("chain_k")),
                    )
                    if value is not None
                }
                oracle = certifier.generated_tokens(arm, case, "gpu", **options)
                split = certifier.generated_tokens(arm, case, "ane-split", **options)
                require_generation_invariants(symbols, arm, case, split)
                oracle_tokens = oracle["generated_token_ids"]
                split_tokens = split["generated_token_ids"]
                first_difference = next(
                    (
                        index
                        for index, (oracle_token, split_token) in enumerate(
                            zip(oracle_tokens, split_tokens)
                        )
                        if oracle_token != split_token
                    ),
                    None,
                )
                row["token_exact"] = first_difference is None
                if first_difference is not None:
                    row.update(
                        {
                            "first_divergent_generated_index": first_difference,
                            "pure_gpu_token": oracle_tokens[first_difference],
                            "ane_split_token": split_tokens[first_difference],
                        }
                    )
            except Exception as error:
                row.update({"token_exact": None, "measurement_error": str(error)})
            rows.append(row)

        divergences = [row for row in rows if row.get("token_exact") is False]
        measurement_errors = [row for row in rows if row.get("token_exact") is None]
        result: dict[str, Any] = {
            "arm": arm.wire(),
            "artifact_triple": artifact,
            "token_battery": {
                "case_count": len(rows),
                "exact_case_count": sum(row.get("token_exact") is True for row in rows),
                "divergence_count": len(divergences),
                "measurement_error_count": len(measurement_errors),
                "cases": rows,
            },
        }
        if divergences or measurement_errors:
            gate = "token-exactness failed" if divergences else "token battery was incomplete"
            result["ttft"] = {
                "status": "not_measured",
                "reason": f"{gate} before the ordered timing gate",
            }
            result["worst_case_fallback"] = {
                "status": "not_measured",
                "reason": f"{gate} before the ordered fallback timing gate",
            }
        else:
            quiet = not started_context["runner_worker"] and parse_load_average(
                started_context["load_average"]
            ) < args.max_timing_load
            if quiet:
                result["ttft"] = {"status": "measured", **certifier.ttft(arm)}
                fallback_responses = [
                    certifier.call("worst_case_fallback", case_id=case_id)
                    for case_id in (
                        "artifact_warm",
                        "cold_ready_compile_failure",
                        "cold_ready_load_failure",
                    )
                ]
                result["worst_case_fallback"] = {
                    "status": "driver_responses",
                    "responses": fallback_responses,
                }
            else:
                result["ttft"] = {
                    "status": "not_measured",
                    "reason": "quiet timing precondition was not met",
                    "load_context": started_context,
                }
                result["worst_case_fallback"] = {
                    "status": "not_measured",
                    "reason": "quiet timing precondition was not met",
                }
        return result
    finally:
        driver.close()


def parse_load_average(value: str) -> float:
    return float(value.strip("{} ").split()[0])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--artifacts", type=Path, default=root / "bench/spikes/ane-prefill-split/artifacts"
    )
    parser.add_argument("--worker", type=Path, default=root / "target/release/ck-synapse-worker-decode")
    parser.add_argument(
        "--constraint-compiler", type=Path, default=root / "target/release/compile_constraint"
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=root / "workers/ane-prefill-sidecar/.build/release/ane-prefill-sidecar",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-timing-load", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.root = args.root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.artifacts = args.artifacts.resolve()
    args.worker = args.worker.resolve()
    args.constraint_compiler = args.constraint_compiler.resolve()
    args.sidecar = args.sidecar.resolve()
    symbols = runpy.run_path(str(args.root / "tests/ane-prefill-certification/certify.py"))
    metadata_driver = symbols["JsonlDriver"](driver_command(args))
    try:
        metadata = symbols["Certifier"](args.root, metadata_driver).metadata()
    finally:
        metadata_driver.close()
    arms = symbols["arms_from_manifest"](symbols["contract_and_manifest"](args.root)[1], metadata["machine_profile"])
    result = {
        "schema_revision": 1,
        "measurement_kind": "m1_authority_token_battery",
        "metadata": metadata,
        "load_context_before": load_context(),
        "arms": [run_arm(args, symbols, arm) for arm in arms],
        "load_context_after": load_context(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
