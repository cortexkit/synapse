#!/usr/bin/env python3
"""Run the hardware certification battery for ANE-prefill split arms.

The harness deliberately owns the acceptance checks rather than trusting a driver to
return a green verdict.  A machine-specific driver supplies raw generated tokens,
timing samples, and fault-injection observations over newline-delimited JSON.  This
keeps the checked-in battery portable while making the machine evidence reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


CONTRACT_PATH = Path("contracts/ane-prefill-split/ane-prefill-split-contract-v2.json")
MANIFEST_PATH = Path("manifests/ane-prefill-split/ane-prefill-split-manifest-v1.json")
FAMILY = "qwen3-0.6b"
CONTINUATION_TOKENS = 64
WIDTH_CASE_COUNT = 20
TTFT_SAMPLE_COUNT = 20
W128_HEADLINE_RATIO = 5.0
BAND_GATE_REVISION = "ane-prefill-split-band-gate-v2"
TOP2_GAP_LIMIT = 0.05
WIDTH_FORK_LIMIT = 3
KV_P95_LIMIT = 0.10

VARIABLE_LENGTHS = {
    128: (1, 2, 16, 17, 64, 127, 128),
    256: (129, 130, 192, 255, 256),
    512: (257, 258, 384, 511, 512),
}

BYPASS_REASONS = (
    "disabled",
    "platform_unsupported",
    "family_unsupported",
    "sampling_uncertified",
    "identity_pinned_gpu",
    "prompt_over_max_bucket",
    "no_fitting_cache_bucket",
    "bucket_absent",
    "not_certified",
    "artifact_digest_mismatch",
    "quarantined",
    "ane_busy",
    "deadline_too_tight",
)

FALLBACK_FAULTS = {
    "compile_error_after_selection": "compile_failure",
    "load_error_or_load_timeout_after_selection": "load_failure",
    "dispatch_start_or_acknowledgement_failure": "dispatch_failure",
    "acknowledged_stage_exit_before_prediction_result": "prediction_failure",
    "prediction_budget_expiry_while_stage_unresponsive": "prediction_timeout",
    "kv_layout_or_conversion_failure": "kv_conversion_failure",
    "cache_payload_publication_or_receipt_failure": "ipc_handoff_failure",
    "prefill_engine_to_decoding_engine_cache_conversion_failure": "cache_handoff_failure",
    "metal_cache_upload_failure": "metal_upload_failure",
    "handoff_budget_expiry": "transfer_budget_exceeded",
    "readiness_budget_expiry": "readiness_budget_exhausted",
    "load_completion_artifact_triple_mismatch": "artifact_mismatch",
    "logits_copy_or_first_token_publication_failure": "logits_publication_failure",
}

ABSENCE_REASONS = frozenset(
    (
        "capacity_precondition_unmet",
        "correctness_divergence",
        "placement_failure",
        "cert_transfer_budget_exceeded",
        "ttft_not_lower",
        "compile_or_load_failure",
        "disk_headroom_unmet",
    )
)

ROUTING_CASES = (
    "global_precedence",
    "bucket_escalation",
    "smallest_terminal_state",
    "unloaded_matching_artifact",
    "present_compiled_digest_mismatch",
    "capacity_boundaries",
    "guard_timeout",
    "deadline_after_guard",
)

LIFECYCLE_CASES = (
    "unloaded_matching_certified_arm",
    "present_artifact_triple_mismatch",
    "postselection_load_triple_mismatch",
    "readiness_budget_expired",
    "runtime_failure_preserves_certification_row",
)

QUARANTINE_CASES = (
    "consecutive_failures_quarantine_exact_arm",
    "success_resets_only_exact_arm",
    "expiry_enters_probation",
    "probation_failure_requarantines",
    "probation_success_clears",
)

PIN_CASES = (
    "gpu_pin_bypasses_without_attempt",
    "ane_split_pin_pre_attempt_refuses_substitution",
    "ane_split_pin_in_attempt_failure_preserves_identity",
)

STATE_EXPECTATIONS = {
    "unloaded_matching_certified_arm": {
        "split_selected": True,
        "readiness_started": True,
        "prefill_bypass_reason": None,
    },
    "present_artifact_triple_mismatch": {
        "prefill_engine": "gpu",
        "prefill_bypass_reason": "artifact_digest_mismatch",
        "split_attempt_started": False,
        "arm_health_debit": 0,
    },
    "postselection_load_triple_mismatch": {
        "prefill_engine": "gpu",
        "prefill_fallback_from": "ane-w128",
        "prefill_fallback_reason": "artifact_mismatch",
        "split_attempt_started": True,
        "arm_health_debit": 1,
        "certification_row_preserved": True,
    },
    "readiness_budget_expired": {
        "prefill_engine": "gpu",
        "prefill_fallback_reason": "readiness_budget_exhausted",
        "split_attempt_started": True,
        "arm_health_debit": 1,
        "certification_row_preserved": True,
    },
    "runtime_failure_preserves_certification_row": {
        "certification_row_preserved": True,
        "runtime_fault_is_not_certification_mutation": True,
    },
    "consecutive_failures_quarantine_exact_arm": {
        "target_arm_quarantined": True,
        "other_arm_strikes": 0,
        "decode_lane_health_debit": 0,
    },
    "success_resets_only_exact_arm": {
        "successful_arm_strikes": 0,
        "other_arm_strikes_unchanged": True,
    },
    "expiry_enters_probation": {"probation": True, "strikes": "max_strikes_minus_one"},
    "probation_failure_requarantines": {"target_arm_quarantined": True, "failure_count": 1},
    "probation_success_clears": {"probation": False, "strikes": 0},
    "gpu_pin_bypasses_without_attempt": {
        "prefill_engine": "gpu",
        "prefill_bypass_reason": "identity_pinned_gpu",
        "split_attempt_started": False,
        "arm_health_debit": 0,
    },
    "ane_split_pin_pre_attempt_refuses_substitution": {
        "identity_preserving_failure": True,
        "prefill_engine": None,
        "split_attempt_started": False,
        "arm_health_debit": 0,
    },
    "ane_split_pin_in_attempt_failure_preserves_identity": {
        "identity_preserving_failure": True,
        "prefill_engine": None,
        "split_attempt_started": True,
        "arm_health_debit": 1,
    },
}

TIMING_CASES = (
    ("artifact_warm", ("guard_ms", "prediction_ms", "handoff_ms", "gpu_prefill_ms")),
    ("cold_ready_compile_failure", ("guard_ms", "readiness_ms", "gpu_prefill_ms")),
    ("cold_ready_load_failure", ("guard_ms", "readiness_ms", "gpu_prefill_ms")),
)
TIMING_FORCED_FAULTS = {
    "artifact_warm": "handoff_budget",
    "cold_ready_compile_failure": "compile",
    "cold_ready_load_failure": "load",
}


class CertificationFailure(RuntimeError):
    """Raised when a driver observation cannot issue a valid certification row."""


class ArmAbsent(CertificationFailure):
    """An attempted non-headline arm ended with an allowed deterministic absence."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


class TokenDivergence(CertificationFailure):
    """The split execution failed its width-battery structural correctness gate."""

    def __init__(self, message: str, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificationFailure(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def deterministic_prompt(case_id: str, length: int) -> list[int]:
    """Create stable valid-range token IDs without relying on host tokenization."""

    seed = hashlib.sha256(case_id.encode("utf-8")).digest()
    return [1024 + ((seed[index % len(seed)] * 257 + index * 313) % 120_000) for index in range(length)]


def fixture_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for bucket in (128, 256, 512):
        for index in range(WIDTH_CASE_COUNT):
            case_id = f"w{bucket}-width-{index:02d}"
            cases.append(
                {
                    "case_id": case_id,
                    "kind": "width_exact",
                    "bucket": bucket,
                    "prompt_token_ids": deterministic_prompt(case_id, bucket),
                }
            )
        for length in VARIABLE_LENGTHS[bucket]:
            case_id = f"w{bucket}-variable-{length}"
            cases.append(
                {
                    "case_id": case_id,
                    "kind": "variable_length",
                    "bucket": bucket,
                    "prompt_token_ids": deterministic_prompt(case_id, length),
                }
            )
    return cases


# Updating generated prompt material must intentionally rotate this digest to trigger
# re-certification rather than silently changing the expected certification cases.
FIXTURE_SHA256 = "aff1d65ff22b758e4a4e3676e5fd95dfe024f66b8ef23587cffaf2e704281157"


def verify_fixture_digest() -> None:
    actual = sha256_hex(fixture_cases())
    require(actual == FIXTURE_SHA256, f"fixture digest changed: expected {FIXTURE_SHA256}, got {actual}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CertificationFailure(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must be a JSON object")
    return value


@dataclass(frozen=True)
class Arm:
    machine_profile: str
    family: str
    bucket: int
    decode_config: str
    headline_gate: bool

    @property
    def id(self) -> str:
        return f"{self.family}-w{self.bucket}-{self.decode_config}"

    def wire(self) -> dict[str, Any]:
        return {
            "machine_profile": self.machine_profile,
            "family": self.family,
            "bucket": self.bucket,
            "decode_config": self.decode_config,
        }


class Driver(Protocol):
    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return the one JSON response for a certification request."""


class JsonlDriver:
    """A strict single-process JSONL bridge for the machine-specific test adapter."""

    def __init__(self, command: list[str]) -> None:
        require(command, "a driver command is required")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        require(self.process.stdin is not None and self.process.stdout is not None, "driver pipes are unavailable")
        self.process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise CertificationFailure(f"driver closed its response stream: {stderr.strip()}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise CertificationFailure(f"driver returned invalid JSON: {line!r}") from error
        require(isinstance(response, dict), "driver response must be a JSON object")
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=10)
        if return_code:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise CertificationFailure(f"driver exited {return_code}: {stderr.strip()}")


def contract_and_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(root / CONTRACT_PATH)
    manifest = read_json(root / MANIFEST_PATH)
    require(contract.get("contract_revision") == "ane-prefill-split-contract-v2", "unexpected ANE split contract revision")
    gate = contract.get("certification", {}).get("certification_requirements", {}).get("split_arm_correctness_gate", {})
    require(gate.get("gate_revision") == BAND_GATE_REVISION, "unexpected ANE split correctness gate revision")
    require(gate.get("logit_near_tie", {}).get("maximum_exclusive") == TOP2_GAP_LIMIT, "ANE split near-tie band drifted")
    require(gate.get("fork_cap", {}).get("maximum_inclusive") == WIDTH_FORK_LIMIT, "ANE split fork cap drifted")
    require(gate.get("fork_cap", {}).get("battery_prompt_count") == WIDTH_CASE_COUNT, "ANE split width battery size drifted")
    require(gate.get("kv_admission_fidelity", {}).get("active_position_p95_abs_maximum_inclusive") == KV_P95_LIMIT, "ANE split K/V p95 limit drifted")
    require(gate.get("kv_admission_fidelity", {}).get("cache_admission_roundtrip_bit_mismatches") == 0, "ANE split cache admission fidelity drifted")
    require(manifest.get("manifest_revision") == "ane-prefill-split-manifest-v1", "unexpected ANE split manifest revision")
    expected_fallbacks = set(FALLBACK_FAULTS.values())
    actual_fallbacks = set(contract.get("reason_vocabularies", {}).get("prefill_fallback_reason", []))
    require(actual_fallbacks == expected_fallbacks, "contract fallback vocabulary drifted from the certification fault map")
    require(set(contract.get("reason_vocabularies", {}).get("prefill_bypass_reason", [])) == set(BYPASS_REASONS), "contract bypass vocabulary drifted")
    require(set(contract.get("reason_vocabularies", {}).get("absence_reason", [])) == ABSENCE_REASONS, "contract absence vocabulary drifted")
    return contract, manifest


def arms_from_manifest(manifest: dict[str, Any], machine_profile: str) -> list[Arm]:
    entries = manifest.get("arm_matrix")
    require(isinstance(entries, list), "manifest arm_matrix must be a list")
    arms: list[Arm] = []
    for entry in entries:
        require(isinstance(entry, dict), "manifest arm entry must be an object")
        key = entry.get("arm_key_fields")
        require(isinstance(key, dict), "manifest arm entry is missing arm_key_fields")
        arm = Arm(
            machine_profile=machine_profile,
            family=key.get("family"),
            bucket=key.get("bucket"),
            decode_config=key.get("decode_config"),
            headline_gate=entry.get("headline_ttft_gate_required") is True,
        )
        require(arm.family == FAMILY, f"unsupported manifest family for {entry.get('arm_id')}")
        require(arm.bucket in (128, 256, 512), f"unexpected manifest bucket for {entry.get('arm_id')}")
        require(arm.decode_config in ("f16-step", "q8-step"), f"unexpected decode config for {entry.get('arm_id')}")
        arms.append(arm)
    arms.sort(key=lambda arm: (arm.bucket, arm.decode_config))
    require(
        {(arm.bucket, arm.decode_config) for arm in arms}
        == {(bucket, config) for bucket in (128, 256, 512) for config in ("f16-step", "q8-step")},
        "manifest must provide exactly all six certification arms",
    )
    return arms


class Certifier:
    def __init__(self, root: Path, driver: Driver) -> None:
        self.root = root
        self.driver = driver
        self.contract, self.manifest = contract_and_manifest(root)

    def call(self, operation: str, **payload: Any) -> dict[str, Any]:
        response = self.driver.exchange({"operation": operation, **payload})
        require(response.get("status") in {"ok", "ready", "absent"}, f"{operation} returned an invalid status")
        return response

    def metadata(self) -> dict[str, Any]:
        response = self.call("metadata")
        require(response.get("status") == "ok", "driver metadata must succeed")
        require(isinstance(response.get("machine_profile"), str) and response["machine_profile"], "metadata lacks machine profile")
        require(is_sha256(response.get("source_checkpoint_digest")), "metadata lacks source checkpoint SHA-256")
        return response

    def validate_artifact(self, response: dict[str, Any], source_digest: str) -> dict[str, str]:
        artifact = response.get("artifact_triple")
        require(isinstance(artifact, dict), "ready arm lacks the artifact digest triple")
        for field in (
            "source_checkpoint_digest",
            "derived_or_compiled_artifact_digest",
            "certification_recorded_artifact_digest",
        ):
            require(is_sha256(artifact.get(field)), f"artifact triple has invalid {field}")
        require(artifact["source_checkpoint_digest"] == source_digest, "arm source digest differs from driver metadata")
        require(
            artifact["derived_or_compiled_artifact_digest"] == artifact["certification_recorded_artifact_digest"],
            "loaded compiled artifact does not match the certification row",
        )
        return artifact

    def generated_tokens(self, arm: Arm, case: dict[str, Any], engine: str, **options: Any) -> dict[str, Any]:
        response = self.call(
            "generate",
            arm=arm.wire(),
            engine=engine,
            case_id=case["case_id"],
            kind=case["kind"],
            prompt_token_ids=case["prompt_token_ids"],
            max_tokens=CONTINUATION_TOKENS,
            greedy_top1=True,
            **options,
        )
        if response.get("status") == "absent":
            require(engine == "ane-split", f"{case['case_id']} GPU oracle cannot be recorded absent")
            reason = response.get("absence_reason")
            require(reason in ABSENCE_REASONS, f"{case['case_id']} {engine} reported an invalid absence reason")
            raise ArmAbsent(reason, str(response.get("detail", f"{case['case_id']} {engine} is absent")))
        require(response.get("status") == "ok", f"{case['case_id']} {engine} generation failed")
        tokens = response.get("generated_token_ids")
        valid_length = (
            isinstance(tokens, list)
            and all(isinstance(token, int) and not isinstance(token, bool) for token in tokens)
            and (
                0 < len(tokens) <= CONTINUATION_TOKENS
                if case["kind"] == "grammar_constrained"
                else len(tokens) == CONTINUATION_TOKENS
            )
        )
        require(valid_length, f"{case['case_id']} {engine} returned an invalid generated-token sequence")
        return response

    def band_gate_observation(
        self,
        arm: Arm,
        case: dict[str, Any],
        oracle: dict[str, Any],
        split: dict[str, Any],
    ) -> dict[str, Any]:
        oracle_tokens = oracle["generated_token_ids"]
        split_tokens = split["generated_token_ids"]
        first_fork_position = next(
            (index for index, pair in enumerate(zip(oracle_tokens, split_tokens)) if pair[0] != pair[1]),
            None,
        )
        response = self.call(
            "band_gate_observation",
            arm=arm.wire(),
            case_id=case["case_id"],
            prompt_token_ids=case["prompt_token_ids"],
            max_tokens=CONTINUATION_TOKENS,
            common_prefix_token_ids=(
                oracle_tokens[:first_fork_position]
                if first_fork_position is not None
                else None
            ),
        )
        if response.get("status") == "absent":
            reason = response.get("absence_reason")
            require(reason in ABSENCE_REASONS, f"{case['case_id']} band observation reported an invalid absence reason")
            raise ArmAbsent(reason, str(response.get("detail", f"{case['case_id']} band observation is absent")))
        require(response.get("status") == "ok", f"{case['case_id']} band observation failed")
        require(response.get("case_id") == case["case_id"], f"{case['case_id']} band observation names the wrong fixture")

        kv = response.get("kv_admission")
        require(isinstance(kv, dict), f"{case['case_id']} lacks K/V admission evidence")
        active_positions = kv.get("active_positions")
        p95 = kv.get("p95_abs_difference")
        bit_mismatches = kv.get("roundtrip_bit_mismatches")
        require(active_positions == len(case["prompt_token_ids"]), f"{case['case_id']} K/V evidence covers the wrong active positions")
        require(
            isinstance(p95, (int, float)) and not isinstance(p95, bool) and math.isfinite(p95) and p95 >= 0,
            f"{case['case_id']} has invalid K/V p95 evidence",
        )
        require(
            isinstance(bit_mismatches, int) and not isinstance(bit_mismatches, bool) and bit_mismatches >= 0,
            f"{case['case_id']} has invalid cache-admission mismatch evidence",
        )
        kv_evidence = {
            "case_id": case["case_id"],
            "active_positions": active_positions,
            "p95_abs_difference": float(p95),
            "roundtrip_bit_mismatches": bit_mismatches,
        }
        if p95 > KV_P95_LIMIT:
            raise TokenDivergence(
                f"{arm.id} {case['case_id']} K/V p95 {p95:.6f} exceeds {KV_P95_LIMIT:.2f}",
                {"kv_admission_evidence": [kv_evidence]},
            )
        if bit_mismatches != 0:
            raise TokenDivergence(
                f"{arm.id} {case['case_id']} cache admission changed {bit_mismatches} bits",
                {"kv_admission_evidence": [kv_evidence]},
            )

        raw_fork = response.get("first_fork")
        result: dict[str, Any] = {
            "case_id": case["case_id"],
            "active_positions": active_positions,
            "kv_p95_abs_difference": float(p95),
            "admission_roundtrip_bit_mismatches": bit_mismatches,
            "token_exact": first_fork_position is None,
        }
        if first_fork_position is None:
            return result

        require(isinstance(raw_fork, dict), f"{case['case_id']} divergence lacks first-fork evidence")
        position = raw_fork.get("position")
        oracle_selected = raw_fork.get("oracle_selected_token")
        split_selected = raw_fork.get("split_selected_token")
        require(position == first_fork_position, f"{case['case_id']} diagnostic fork position differs from the production paths")
        require(oracle_selected == oracle_tokens[position], f"{case['case_id']} diagnostic oracle token differs from the production path")
        require(split_selected == split_tokens[position], f"{case['case_id']} diagnostic split token differs from the production path")

        def top2(field: str) -> list[int]:
            value = raw_fork.get(field)
            require(
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(token, int) and not isinstance(token, bool) for token in value)
                and len(set(value)) == 2,
                f"{case['case_id']} has invalid {field}",
            )
            return value

        def gap(field: str) -> float:
            value = raw_fork.get(field)
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0,
                f"{case['case_id']} has invalid {field}",
            )
            return float(value)

        oracle_top2 = top2("oracle_top2_token_ids")
        split_top2 = top2("split_top2_token_ids")
        oracle_gap = gap("oracle_top2_gap")
        split_gap = gap("split_top2_gap")
        swap_verdict = (
            oracle_top2 == [oracle_selected, split_selected]
            and split_top2 == [split_selected, oracle_selected]
        )
        fork = {
            "case_id": case["case_id"],
            "position": position,
            "oracle_selected_token": oracle_selected,
            "split_selected_token": split_selected,
            "oracle_top2_token_ids": oracle_top2,
            "split_top2_token_ids": split_top2,
            "oracle_top2_gap": oracle_gap,
            "split_top2_gap": split_gap,
            "swap_verdict": swap_verdict,
        }
        result["first_fork"] = fork
        if not swap_verdict:
            raise TokenDivergence(
                f"{arm.id} {case['case_id']} first fork is not an ordered top-2 swap",
                {"fork_evidence": [fork], "kv_admission_evidence": [kv_evidence]},
            )
        if oracle_gap >= TOP2_GAP_LIMIT or split_gap >= TOP2_GAP_LIMIT:
            raise TokenDivergence(
                f"{arm.id} {case['case_id']} first-fork gaps {oracle_gap:.6f}/{split_gap:.6f} are outside the < {TOP2_GAP_LIMIT:.2f} band",
                {"fork_evidence": [fork], "kv_admission_evidence": [kv_evidence]},
            )
        return result

    def compare_case(
        self,
        arm: Arm,
        case: dict[str, Any],
        *,
        apply_band_gate: bool = False,
        **options: Any,
    ) -> dict[str, Any] | None:
        oracle = self.generated_tokens(arm, case, "gpu", **options)
        split = self.generated_tokens(arm, case, "ane-split", **options)
        prompt_length = len(case["prompt_token_ids"])
        require(split.get("padded_width") == arm.bucket, f"{case['case_id']} used the wrong fixed-width graph")
        require(split.get("first_token_index") == prompt_length - 1, f"{case['case_id']} read logits from a padded position")
        require(split.get("active_cache_positions") == prompt_length, f"{case['case_id']} imported padded cache positions")
        require(split.get("decode_cache_position") == prompt_length, f"{case['case_id']} started decode at the wrong cache position")
        if arm.decode_config == "q8-step":
            require(split.get("cache_handoff") == "engine_to_engine", f"{case['case_id']} bypassed q8 cache handoff")
        if apply_band_gate:
            return self.band_gate_observation(arm, case, oracle, split)
        return None

    def run_token_battery(self, arm: Arm) -> dict[str, Any]:
        cases = [case for case in fixture_cases() if case["bucket"] == arm.bucket]
        width_cases = [case for case in cases if case["kind"] == "width_exact"]
        variable_cases = [case for case in cases if case["kind"] == "variable_length"]
        require(len(width_cases) == WIDTH_CASE_COUNT, f"W{arm.bucket} corpus does not have 20 width-exact prompts")
        require(tuple(len(case["prompt_token_ids"]) for case in variable_cases) == VARIABLE_LENGTHS[arm.bucket], f"W{arm.bucket} variable-length battery drifted")

        width_rows = [
            self.compare_case(arm, case, apply_band_gate=True) for case in width_cases
        ]
        require(all(isinstance(row, dict) for row in width_rows), f"{arm.id} width battery lacks band-gate evidence")
        forks = [row["first_fork"] for row in width_rows if "first_fork" in row]
        gaps = [
            gap
            for fork in forks
            for gap in (fork["oracle_top2_gap"], fork["split_top2_gap"])
        ]
        band_gate = {
            "gate_revision": BAND_GATE_REVISION,
            "prompt_count": len(width_rows),
            "fork_count": len(forks),
            "fork_limit": WIDTH_FORK_LIMIT,
            "gap_limit_exclusive": TOP2_GAP_LIMIT,
            "worst_gap": max(gaps, default=0.0),
            "kv_p95_limit_inclusive": KV_P95_LIMIT,
            "worst_kv_p95_abs_difference": max(row["kv_p95_abs_difference"] for row in width_rows),
            "fork_evidence": forks,
            "kv_admission_evidence": [
                {
                    "case_id": row["case_id"],
                    "active_positions": row["active_positions"],
                    "p95_abs_difference": row["kv_p95_abs_difference"],
                    "roundtrip_bit_mismatches": row["admission_roundtrip_bit_mismatches"],
                }
                for row in width_rows
            ],
        }
        if len(forks) > WIDTH_FORK_LIMIT:
            raise TokenDivergence(
                f"{arm.id} has {len(forks)} first forks in {WIDTH_CASE_COUNT} prompts, above the {WIDTH_FORK_LIMIT}-fork cap",
                band_gate,
            )

        for case in variable_cases:
            self.compare_case(arm, case)
        grammar_case = {
            "case_id": f"w{arm.bucket}-grammar",
            "kind": "grammar_constrained",
            "bucket": arm.bucket,
            "prompt_token_ids": deterministic_prompt(f"w{arm.bucket}-grammar", min(64, arm.bucket)),
        }
        self.compare_case(arm, grammar_case, grammar="json-object")
        chain_case = {
            "case_id": f"w{arm.bucket}-chain-k-16",
            "kind": "chain_k",
            "bucket": arm.bucket,
            "prompt_token_ids": deterministic_prompt(f"w{arm.bucket}-chain-k-16", min(96, arm.bucket)),
        }
        self.compare_case(arm, chain_case, chain_k=16)
        return {
            "width_exact": len(width_cases),
            "variable_length": len(variable_cases),
            "grammar": 1,
            "chain_k_16": 1,
            "band_gate": band_gate,
        }

    def run_routing_battery(self, arm: Arm) -> None:
        response = self.call("routing_battery", arm=arm.wire(), case_ids=list(ROUTING_CASES))
        require(response.get("status") == "ok", f"routing battery did not complete for {arm.id}")
        require(response.get("executed_case_ids") == list(ROUTING_CASES), f"routing battery coverage drifted for {arm.id}")

    def warmup(self, arm: Arm, engine: str) -> None:
        for attempt in range(3):
            response = self.call("warmup", arm=arm.wire(), engine=engine, attempt=attempt)
            require(response.get("status") == "ok", f"{arm.id} {engine} warmup {attempt} failed")

    def ttft(self, arm: Arm) -> dict[str, Any]:
        self.warmup(arm, "gpu")
        self.warmup(arm, "ane-split")
        samples: dict[str, list[dict[str, float]]] = {"gpu": [], "ane-split": []}
        for sample in range(TTFT_SAMPLE_COUNT):
            for engine in ("ane-split", "gpu"):
                response = self.call("measure_ttft", arm=arm.wire(), engine=engine, sample=sample, request_cold=True, artifact_warm=True)
                if response.get("status") == "absent":
                    require(engine == "ane-split", f"{arm.id} GPU TTFT oracle cannot be recorded absent")
                    reason = response.get("absence_reason")
                    require(reason in ABSENCE_REASONS, f"{arm.id} {engine} reported an invalid TTFT absence reason")
                    raise ArmAbsent(reason, str(response.get("detail", f"{arm.id} {engine} TTFT is absent")))
                require(response.get("status") == "ok", f"{arm.id} {engine} TTFT sample {sample} failed")
                worker_ms = response.get("worker_ttft_ms")
                wire_ms = response.get("wire_ttft_ms")
                require(isinstance(worker_ms, (int, float)) and worker_ms > 0, f"{arm.id} {engine} has invalid worker TTFT")
                require(isinstance(wire_ms, (int, float)) and wire_ms > 0, f"{arm.id} {engine} has invalid wire TTFT")
                samples[engine].append({"worker_ttft_ms": float(worker_ms), "wire_ttft_ms": float(wire_ms)})
        split_p50 = statistics.median(sample["worker_ttft_ms"] for sample in samples["ane-split"])
        gpu_p50 = statistics.median(sample["worker_ttft_ms"] for sample in samples["gpu"])
        ratio = gpu_p50 / split_p50
        if split_p50 >= gpu_p50:
            raise ArmAbsent("ttft_not_lower", f"{arm.id} split worker TTFT p50 is not lower than GPU")
        if arm.headline_gate:
            require(ratio >= W128_HEADLINE_RATIO, f"{arm.id} headline worker TTFT ratio is {ratio:.3f}, below {W128_HEADLINE_RATIO:.1f}x")
        return {
            "samples": samples,
            "split_worker_p50_ms": split_p50,
            "gpu_worker_p50_ms": gpu_p50,
            "worker_ttft_ratio": ratio,
        }

    def certify_arm(self, arm: Arm, source_digest: str) -> dict[str, Any]:
        precondition = self.call("precondition", arm=arm.wire())
        if precondition.get("status") == "absent":
            reason = precondition.get("absence_reason")
            require(reason in ABSENCE_REASONS, f"{arm.id} reported an invalid absence reason")
            if arm.headline_gate:
                raise CertificationFailure(f"{arm.id} is headline-required but absent: {reason}")
            return {"arm": arm.wire(), "outcome": "absent", "absence_reason": reason, "detail": precondition.get("detail", "")}
        require(precondition.get("status") == "ready", f"{arm.id} precondition status is invalid")
        artifact = self.validate_artifact(precondition, source_digest)
        try:
            battery = self.run_token_battery(arm)
            self.run_routing_battery(arm)
            ttft = self.ttft(arm)
        except ArmAbsent as absence:
            if arm.headline_gate:
                raise CertificationFailure(f"{arm.id} is headline-required but absent: {absence.reason}") from absence
            return {"arm": arm.wire(), "outcome": "absent", "absence_reason": absence.reason, "detail": str(absence), "artifact_triple": artifact}
        except TokenDivergence as divergence:
            if arm.headline_gate:
                raise
            return {
                "arm": arm.wire(),
                "outcome": "absent",
                "absence_reason": "correctness_divergence",
                "detail": str(divergence),
                "artifact_triple": artifact,
                "correctness_evidence": divergence.evidence,
            }
        return {"arm": arm.wire(), "outcome": "certified", "artifact_triple": artifact, "battery": battery, "ttft": ttft}

    def exercise(self, kind: str, case_id: str, arm: Arm | None = None, **expectations: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": kind, "case_id": case_id}
        if arm is not None:
            payload["arm"] = arm.wire()
        response = self.call("exercise", **payload)
        require(response.get("status") == "ok", f"{kind}/{case_id} did not complete")
        observed = response.get("observed")
        require(isinstance(observed, dict), f"{kind}/{case_id} did not return an observation")
        for key, expected in expectations.items():
            require(observed.get(key) == expected, f"{kind}/{case_id} expected {key}={expected!r}, got {observed.get(key)!r}")
        return {"kind": kind, "case_id": case_id, "observed": observed}

    def run_semantic_exercises(self, arms: list[Arm]) -> list[dict[str, Any]]:
        state_arm = next(arm for arm in arms if arm.bucket == 128 and arm.decode_config == "f16-step")
        q8_arm = next(arm for arm in arms if arm.bucket == 128 and arm.decode_config == "q8-step")
        results: list[dict[str, Any]] = []
        for reason in BYPASS_REASONS:
            results.append(
                self.exercise(
                    "bypass",
                    reason,
                    arm=state_arm,
                    prefill_engine="gpu",
                    prefill_bypass_reason=reason,
                    prefill_fallback_reason=None,
                    split_attempt_started=False,
                    arm_health_debit=0,
                    decode_lane_health_debit=0,
                )
            )
        for fault, reason in FALLBACK_FAULTS.items():
            fault_arm = q8_arm if reason == "cache_handoff_failure" else state_arm
            results.append(
                self.exercise(
                    "fallback",
                    fault,
                    arm=fault_arm,
                    prefill_engine="gpu",
                    prefill_fallback_from="ane-w128",
                    prefill_fallback_reason=reason,
                    split_attempt_started=True,
                    arm_health_debit=1,
                    decode_lane_health_debit=0,
                )
            )
        for case_id in LIFECYCLE_CASES + QUARANTINE_CASES + PIN_CASES:
            results.append(self.exercise("state", case_id, arm=state_arm, **STATE_EXPECTATIONS[case_id]))
        results.append(
            self.exercise(
                "protocol",
                "connect_mismatch",
                arm=state_arm,
                initial_exact_arm_health_debit=1,
                later_prefill_bypass_reason="quarantined",
                later_request_health_debit=0,
            )
        )
        return results

    def worst_case_fallbacks(self, arm: Arm) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case_id, components in TIMING_CASES:
            response = self.call("worst_case_fallback", case_id=case_id, arm=arm.wire())
            require(response.get("status") == "ok", f"{case_id} fallback measurement failed")
            consumed = response.get("consumed_components_ms")
            require(isinstance(consumed, dict), f"{case_id} lacks consumed timing components")
            require(set(consumed) == set(components), f"{case_id} components do not cover the complete attempted path")
            require(
                all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                    for value in consumed.values()
                ),
                f"{case_id} has invalid component timing",
            )
            attempt_budget_spend = response.get("attempt_budget_spend_ms")
            fallback_trigger_latency = response.get("fallback_trigger_latency_ms")
            gpu_prefill = response.get("gpu_prefill_ms")
            for field, value in (
                ("attempt_budget_spend_ms", attempt_budget_spend),
                ("fallback_trigger_latency_ms", fallback_trigger_latency),
                ("gpu_prefill_ms", gpu_prefill),
            ):
                require(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0,
                    f"{case_id} has invalid {field}",
                )
            expected_attempt_spend = consumed.get("prediction_ms", consumed.get("readiness_ms"))
            require(
                math.isclose(attempt_budget_spend, expected_attempt_spend, rel_tol=0.0, abs_tol=1e-9),
                f"{case_id} attempt budget spend differs from its measured ANE stage",
            )
            require(
                fallback_trigger_latency >= attempt_budget_spend,
                f"{case_id} fallback triggered before its measured attempt budget spend",
            )
            require(
                math.isclose(gpu_prefill, consumed["gpu_prefill_ms"], rel_tol=0.0, abs_tol=1e-9),
                f"{case_id} reports a GPU prefill time that differs from its consumed stage",
            )
            forced_fault = response.get("forced_fault")
            require(forced_fault == TIMING_FORCED_FAULTS[case_id], f"{case_id} reported the wrong forced fault")
            total = response.get("total_ttft_ms")
            require(
                isinstance(total, (int, float))
                and not isinstance(total, bool)
                and math.isfinite(total)
                and total >= sum(consumed.values())
                and total >= fallback_trigger_latency + gpu_prefill,
                f"{case_id} total TTFT is less than its observed fallback chain",
            )
            rows.append(
                {
                    "case_id": case_id,
                    "arm": arm.wire(),
                    "forced_fault": forced_fault,
                    "consumed_components_ms": consumed,
                    "attempt_budget_spend_ms": float(attempt_budget_spend),
                    "fallback_trigger_latency_ms": float(fallback_trigger_latency),
                    "gpu_prefill_ms": float(gpu_prefill),
                    "total_ttft_ms": float(total),
                }
            )
        return rows

    def run(self) -> dict[str, Any]:
        verify_fixture_digest()
        metadata = self.metadata()
        arms = arms_from_manifest(self.manifest, metadata["machine_profile"])
        attempts = [self.certify_arm(arm, metadata["source_checkpoint_digest"]) for arm in arms]
        headline = [attempt for attempt in attempts if attempt["arm"]["bucket"] == 128]
        require(len(headline) == 2 and all(attempt["outcome"] == "certified" for attempt in headline), "both W128 arms must be green")
        require(all(attempt["outcome"] == "certified" or attempt.get("absence_reason") in ABSENCE_REASONS for attempt in attempts), "each non-headline arm must be green or deterministically absent")
        exercises = self.run_semantic_exercises(arms)
        fallback_arm = next(
            arm for arm in arms if arm.bucket == 128 and arm.decode_config == "f16-step"
        )
        fallback_rows = self.worst_case_fallbacks(fallback_arm)
        return {
            "schema_revision": 2,
            "record_kind": "ane_prefill_hardware_certification",
            "correctness_gate_revision": BAND_GATE_REVISION,
            "fixture_sha256": FIXTURE_SHA256,
            "machine_profile": metadata["machine_profile"],
            "source_checkpoint_digest": metadata["source_checkpoint_digest"],
            "attempts": attempts,
            "semantic_exercises": exercises,
            "worst_case_fallback_rows": fallback_rows,
            "maximum_worst_case_fallback_ttft_ms": max(row["total_ttft_ms"] for row in fallback_rows),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True, help="write the machine evidence record to this path")
    parser.add_argument("--driver", nargs=argparse.REMAINDER, required=True, help="JSONL certification driver command, after --driver")
    args = parser.parse_args()
    driver = JsonlDriver(args.driver)
    try:
        result = Certifier(args.root.resolve(), driver).run()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        driver.close()
    print(f"ANE prefill certification passed; evidence written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
