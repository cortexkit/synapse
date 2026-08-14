#!/usr/bin/env python3
"""Deterministic acceptance tests for the ANE-prefill hardware certification harness."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("certify.py")
SPEC = importlib.util.spec_from_file_location("ane_prefill_certify", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
certify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = certify
SPEC.loader.exec_module(certify)

DIGEST = "a" * 64


class FakeDriver:
    """Models raw driver observations without recreating harness verdict logic."""

    def __init__(self, forks: dict[str, dict[str, Any]] | None = None) -> None:
        self.forks = forks or {}
        self.requests: list[dict[str, Any]] = []

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        operation = request["operation"]
        if operation == "metadata":
            return {
                "status": "ok",
                "machine_profile": "test-m5-profile",
                "source_checkpoint_digest": DIGEST,
            }
        if operation == "precondition":
            arm = request["arm"]
            if arm["bucket"] == 256 and arm["decode_config"] == "q8-step":
                return {
                    "status": "absent",
                    "absence_reason": "capacity_precondition_unmet",
                    "detail": "the test q8 cache has no W320 capacity",
                }
            return {
                "status": "ready",
                "artifact_triple": {
                    "source_checkpoint_digest": DIGEST,
                    "derived_or_compiled_artifact_digest": DIGEST,
                    "certification_recorded_artifact_digest": DIGEST,
                },
            }
        if operation == "generate":
            prompt = request["prompt_token_ids"]
            arm = request["arm"]
            token_seed = sum(prompt) % 10_000
            tokens = [token_seed + offset for offset in range(certify.CONTINUATION_TOKENS)]
            fork = self.forks.get(request["case_id"])
            if request["engine"] == "ane-split" and fork is not None:
                tokens[int(fork.get("position", 0))] += 1
            response: dict[str, Any] = {
                "status": "ok",
                "generated_token_ids": tokens,
                "padded_width": arm["bucket"],
                "first_token_index": len(prompt) - 1,
                "active_cache_positions": len(prompt),
                "decode_cache_position": len(prompt),
            }
            if arm["decode_config"] == "q8-step":
                response["cache_handoff"] = "engine_to_engine"
            return response
        if operation == "band_gate_observation":
            prompt = request["prompt_token_ids"]
            token_seed = sum(prompt) % 10_000
            fork = self.forks.get(request["case_id"])
            first_fork = None
            if fork is not None:
                position = int(fork.get("position", 0))
                oracle_token = token_seed + position
                split_token = oracle_token + 1
                oracle_top2 = [oracle_token, split_token]
                split_top2 = [split_token, oracle_token]
                if fork.get("kind") == "non_swap":
                    split_top2 = [split_token, split_token + 1]
                first_fork = {
                    "position": position,
                    "oracle_selected_token": oracle_token,
                    "split_selected_token": split_token,
                    "oracle_top2_token_ids": oracle_top2,
                    "split_top2_token_ids": split_top2,
                    "oracle_top2_gap": float(fork.get("oracle_gap", 0.01)),
                    "split_top2_gap": float(fork.get("split_gap", 0.01)),
                }
            return {
                "status": "ok",
                "case_id": request["case_id"],
                "first_fork": first_fork,
                "kv_admission": {
                    "active_positions": len(prompt),
                    "p95_abs_difference": float((fork or {}).get("kv_p95", 0.07)),
                    "roundtrip_bit_mismatches": int((fork or {}).get("bit_mismatches", 0)),
                },
            }
        if operation in {"routing_battery", "warmup"}:
            response = {"status": "ok"}
            if operation == "routing_battery":
                response["executed_case_ids"] = request["case_ids"]
            return response
        if operation == "measure_ttft":
            return {
                "status": "ok",
                "worker_ttft_ms": 10 if request["engine"] == "ane-split" else 60,
                "wire_ttft_ms": 12 if request["engine"] == "ane-split" else 62,
            }
        if operation == "exercise":
            kind = request["kind"]
            case_id = request["case_id"]
            if kind == "bypass":
                return {
                    "status": "ok",
                    "observed": {
                        "prefill_engine": "gpu",
                        "prefill_bypass_reason": case_id,
                        "prefill_fallback_reason": None,
                        "split_attempt_started": False,
                        "arm_health_debit": 0,
                        "decode_lane_health_debit": 0,
                    },
                }
            if kind == "fallback":
                return {
                    "status": "ok",
                    "observed": {
                        "prefill_engine": "gpu",
                        "prefill_fallback_from": "ane-w128",
                        "prefill_fallback_reason": certify.FALLBACK_FAULTS[case_id],
                        "split_attempt_started": True,
                        "arm_health_debit": 1,
                        "decode_lane_health_debit": 0,
                    },
                }
            if kind == "protocol":
                return {
                    "status": "ok",
                    "observed": {
                        "initial_exact_arm_health_debit": 1,
                        "later_prefill_bypass_reason": "quarantined",
                        "later_request_health_debit": 0,
                    },
                }
            return {
                "status": "ok",
                "observed": certify.STATE_EXPECTATIONS[case_id],
            }
        if operation == "worst_case_fallback":
            components = {
                "artifact_warm": {"guard_ms": 10, "prediction_ms": 10, "handoff_ms": 20, "gpu_prefill_ms": 60},
                "cold_ready_compile_failure": {"guard_ms": 10, "readiness_ms": 100, "gpu_prefill_ms": 60},
                "cold_ready_load_failure": {"guard_ms": 10, "readiness_ms": 100, "gpu_prefill_ms": 60},
            }[request["case_id"]]
            attempt_budget_spend = (
                components["prediction_ms"]
                if "prediction_ms" in components
                else components["readiness_ms"]
            )
            return {
                "status": "ok",
                "forced_fault": certify.TIMING_FORCED_FAULTS[request["case_id"]],
                "consumed_components_ms": components,
                "attempt_budget_spend_ms": attempt_budget_spend,
                "fallback_trigger_latency_ms": sum(components.values()) - components["gpu_prefill_ms"],
                "gpu_prefill_ms": components["gpu_prefill_ms"],
                "total_ttft_ms": sum(components.values()),
            }
        raise AssertionError(f"unhandled driver request: {request}")


class BrokenLifecycleDriver(FakeDriver):
    """Returns a plausible but unsafe lifecycle observation for the guard test."""

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        response = super().exchange(request)
        if request["operation"] == "exercise" and request["kind"] == "state" and request["case_id"] == "postselection_load_triple_mismatch":
            response["observed"] = dict(response["observed"])
            response["observed"]["certification_row_preserved"] = False
        return response


class BrokenTimingDriver(FakeDriver):
    """Omits one warm-path component to prove the timing completeness guard is live."""

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        response = super().exchange(request)
        if request["operation"] == "worst_case_fallback" and request["case_id"] == "artifact_warm":
            response["consumed_components_ms"] = dict(response["consumed_components_ms"])
            del response["consumed_components_ms"]["handoff_ms"]
        return response


class DisabledFallbackSeamDriver(FakeDriver):
    """Models a production worker that did not opt into the certification probe."""

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["operation"] == "worst_case_fallback":
            return {
                "status": "absent",
                "absence_reason": "compile_or_load_failure",
                "detail": "certification ANE timing probe is disabled",
            }
        return super().exchange(request)


class CertificationHarnessTests(unittest.TestCase):
    def test_fixture_material_is_digest_pinned(self) -> None:
        certify.verify_fixture_digest()
        cases = certify.fixture_cases()
        self.assertEqual(len([case for case in cases if case["kind"] == "width_exact"]), 60)
        self.assertEqual(len([case for case in cases if case["kind"] == "variable_length"]), 17)

    def test_full_matrix_requires_w128_green_and_records_allowed_absence(self) -> None:
        driver = FakeDriver()
        result = certify.Certifier(ROOT, driver).run()
        outcomes = {(row["arm"]["bucket"], row["arm"]["decode_config"]): row for row in result["attempts"]}
        self.assertEqual(outcomes[(128, "f16-step")]["outcome"], "certified")
        self.assertEqual(outcomes[(128, "q8-step")]["outcome"], "certified")
        self.assertEqual(outcomes[(256, "q8-step")]["absence_reason"], "capacity_precondition_unmet")
        self.assertEqual(result["maximum_worst_case_fallback_ttft_ms"], 170.0)
        fallback_rows = result["worst_case_fallback_rows"]
        self.assertEqual([row["case_id"] for row in fallback_rows], [case_id for case_id, _ in certify.TIMING_CASES])
        self.assertTrue(all(row["arm"]["bucket"] == 128 for row in fallback_rows))
        self.assertTrue(all(row["gpu_prefill_ms"] == row["consumed_components_ms"]["gpu_prefill_ms"] for row in fallback_rows))
        self.assertEqual(len(result["semantic_exercises"]), len(certify.BYPASS_REASONS) + len(certify.FALLBACK_FAULTS) + len(certify.LIFECYCLE_CASES) + len(certify.QUARANTINE_CASES) + len(certify.PIN_CASES) + 1)

    def test_legal_top2_swap_inside_band_passes(self) -> None:
        driver = FakeDriver({"w128-width-00": {"position": 7}})
        battery = certify.Certifier(ROOT, driver).run_token_battery(
            certify.Arm("test-m5-profile", "qwen3-0.6b", 128, "f16-step", True)
        )
        gate = battery["band_gate"]
        self.assertEqual(gate["fork_count"], 1)
        self.assertTrue(gate["fork_evidence"][0]["swap_verdict"])

    def test_non_swap_divergence_fails_closed(self) -> None:
        driver = FakeDriver({"w128-width-00": {"kind": "non_swap"}})
        with self.assertRaisesRegex(certify.TokenDivergence, "not an ordered top-2 swap") as raised:
            certify.Certifier(ROOT, driver).run_token_battery(
                certify.Arm("test-m5-profile", "qwen3-0.6b", 128, "f16-step", True)
            )
        self.assertFalse(raised.exception.evidence["fork_evidence"][0]["swap_verdict"])

    def test_four_in_band_swaps_fail_the_twenty_prompt_cap(self) -> None:
        forks = {f"w128-width-{index:02d}": {} for index in range(4)}
        with self.assertRaisesRegex(certify.TokenDivergence, "above the 3-fork cap") as raised:
            certify.Certifier(ROOT, FakeDriver(forks)).run_token_battery(
                certify.Arm("test-m5-profile", "qwen3-0.6b", 128, "f16-step", True)
            )
        self.assertEqual(len(raised.exception.evidence["fork_evidence"]), 4)

    def test_swap_with_gap_point_zero_six_fails_the_band(self) -> None:
        driver = FakeDriver({"w128-width-00": {"split_gap": 0.06}})
        with self.assertRaisesRegex(certify.TokenDivergence, "outside the < 0.05 band"):
            certify.Certifier(ROOT, driver).run_token_battery(
                certify.Arm("test-m5-profile", "qwen3-0.6b", 128, "f16-step", True)
            )

    def test_real_driver_dispatch_covers_every_harness_operation(self) -> None:
        certify_tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        requested = {
            call.args[0].value
            for call in ast.walk(certify_tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "call"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        }

        driver_path = MODULE_PATH.with_name("machine_driver.py")
        driver_tree = ast.parse(driver_path.read_text(encoding="utf-8"))
        driver_class = next(
            node
            for node in driver_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Driver"
        )
        handle = next(
            node
            for node in driver_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "handle"
        )
        dispatched: set[str] = set()
        for comparison in (
            node for node in ast.walk(handle) if isinstance(node, ast.Compare)
        ):
            if not isinstance(comparison.left, ast.Name) or comparison.left.id != "operation":
                continue
            for comparator in comparison.comparators:
                values = (
                    comparator.elts
                    if isinstance(comparator, (ast.Set, ast.Tuple, ast.List))
                    else [comparator]
                )
                dispatched.update(
                    value.value
                    for value in values
                    if isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                )
        self.assertEqual(requested - dispatched, set())

    def test_every_requested_semantic_case_reaches_the_driver(self) -> None:
        driver = FakeDriver()
        certify.Certifier(ROOT, driver).run()
        bypasses = {request["case_id"] for request in driver.requests if request["operation"] == "exercise" and request["kind"] == "bypass"}
        fallbacks = {request["case_id"] for request in driver.requests if request["operation"] == "exercise" and request["kind"] == "fallback"}
        self.assertEqual(bypasses, set(certify.BYPASS_REASONS))
        self.assertEqual(fallbacks, set(certify.FALLBACK_FAULTS))

    def test_lifecycle_guard_rejects_a_runtime_mutation_of_a_green_row(self) -> None:
        with self.assertRaisesRegex(certify.CertificationFailure, "certification_row_preserved=True"):
            certify.Certifier(ROOT, BrokenLifecycleDriver()).run()

    def test_warm_fallback_requires_every_consumed_component(self) -> None:
        with self.assertRaisesRegex(certify.CertificationFailure, "components do not cover"):
            certify.Certifier(ROOT, BrokenTimingDriver()).run()

    def test_disabled_fallback_seam_emits_no_timing_rows(self) -> None:
        certifier = certify.Certifier(ROOT, DisabledFallbackSeamDriver())
        fallback_arm = certify.Arm("test-m5-profile", "qwen3-0.6b", 128, "f16-step", True)
        with self.assertRaisesRegex(certify.CertificationFailure, "artifact_warm fallback measurement failed"):
            certifier.worst_case_fallbacks(fallback_arm)


if __name__ == "__main__":
    unittest.main()
