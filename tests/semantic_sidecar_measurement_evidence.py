#!/usr/bin/env python3
"""Validate and summarize constrained-decode sidecar measurement JSONL.

An arm is one fixed runtime configuration, such as B1 or SC-METAL-DIRECT. The
runner must emit one JSON object for each request, arm, and repetition, timed
from handler entry after arm selection until the response is ready after sidecar
cancellation is signalled. This tool consumes those captured rows; it does not
benchmark a model itself. It rejects missing arm coverage, changed request IDs,
token mismatches, and statistics that are not paired by request ID.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKLOADS = ("athena-classify-json", "dreamer-contract-transform")
BASELINE_ARM = "B1"
STRUCTURE_ONLY_ARM = "SO-BANK"
SIDECAR_ARMS = ("SC-METAL-DIRECT", "SC-ANE", "SC-METAL-SCHEDULED")
OUTCOMES = (
    "cancelled",
    "failed",
    "completed_late",
    "completed_invalid",
    "completed_usable_used",
    "completed_usable_unused",
)
BOUNDARY = "handler_entry_after_arm_identified_to_response_ready_after_cancellation_signal"


class EvidenceError(ValueError):
    """Raised when measurement rows are malformed, incomplete, or contradict the frozen contract."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON record {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON record {path} must be an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"cannot read measurement rows {path}: {error}") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise EvidenceError(f"{path}:{line_number}: each row must be an object")
        rows.append(row)
    if not rows:
        raise EvidenceError(f"{path}: measurement file has no rows")
    return rows


def positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise EvidenceError(f"{label} must be a finite positive number")
    return number


def nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EvidenceError(f"{label} must be a finite non-negative number")
    return number


def nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{label} must be a non-negative integer count")
    return value


def token_ids(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise EvidenceError(f"{label} must be a non-empty token-ID list")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
        raise EvidenceError(f"{label} contains an invalid token ID")
    return tuple(value)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvidenceError("cannot compute a percentile of no observations")
    if not 0 < fraction <= 1:
        raise EvidenceError("percentile fraction must be in (0, 1]")
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def measured_workloads(manifest: dict[str, Any]) -> tuple[str, ...]:
    if manifest.get("measurement_phase") != "deterministic_arms_phase1":
        return WORKLOADS
    workloads = manifest.get("workloads")
    if not isinstance(workloads, list):
        raise EvidenceError("phase-1 manifest workloads must be a list")
    selected = tuple(
        workload.get("workload_id")
        for workload in workloads
        if isinstance(workload, dict) and workload.get("selected_for_measurement") is True
    )
    if selected != ("athena-classify-json",):
        raise EvidenceError("deterministic phase 1 must select only athena-classify-json")
    return selected


def supported_arms(manifest: dict[str, Any]) -> list[str]:
    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise EvidenceError("manifest arms must be a list")
    selected: list[str] = []
    seen: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise EvidenceError("each manifest arm must be an object")
        arm_id = arm.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise EvidenceError("each manifest arm needs a non-empty arm_id")
        if arm_id in seen:
            raise EvidenceError(f"manifest duplicates arm {arm_id}")
        seen.add(arm_id)
        if arm.get("selected_for_measurement") is True:
            selected.append(arm_id)
    if BASELINE_ARM not in selected or STRUCTURE_ONLY_ARM not in selected:
        raise EvidenceError("manifest must select B1 and SO-BANK")
    return selected


def validate_manifest(manifest: dict[str, Any], require_ready: bool) -> None:
    if manifest.get("record_id") != "semantic-sidecar-measurement-contract-v1":
        raise EvidenceError("unexpected measurement contract record_id")
    if manifest.get("revision") != 1:
        raise EvidenceError("unexpected measurement contract revision")
    if require_ready and manifest.get("measurement_state") != "ready":
        raise EvidenceError("manifest is not ready for primary measurement")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise EvidenceError("manifest protocol must be an object")
    if protocol.get("wall_clock_boundary") != BOUNDARY:
        raise EvidenceError("manifest changes the common wall-clock boundary")
    if protocol.get("repetitions") is None or int(protocol["repetitions"]) < 5:
        raise EvidenceError("manifest requires fewer than five repetitions")
    if protocol.get("bootstrap_resamples") is None or int(protocol["bootstrap_resamples"]) < 10_000:
        raise EvidenceError("manifest requires fewer than 10,000 bootstrap resamples")
    if protocol.get("minimum_eligible_requests_per_workload") is None or int(
        protocol["minimum_eligible_requests_per_workload"]
    ) < 100:
        raise EvidenceError("manifest requires fewer than 100 eligible requests per workload")
    if protocol.get("request_statistic") != "median_of_repetitions":
        raise EvidenceError("manifest must use request medians")
    if protocol.get("percentiles") != {"p50": "nearest_rank", "p95": "nearest_rank"}:
        raise EvidenceError("manifest must freeze nearest-rank p50 and p95")
    if protocol.get("paired_difference") != "B1_request_median_minus_arm_request_median":
        raise EvidenceError("manifest changes the paired-difference direction")
    if protocol.get("bootstrap_unit") != "request_id":
        raise EvidenceError("repetitions must not be bootstrap units")
    if protocol.get("adversarial_schema_count") is None or int(protocol["adversarial_schema_count"]) < 30:
        raise EvidenceError("manifest requires fewer than 30 adversarial schemas")
    selected = supported_arms(manifest)
    measured_workloads(manifest)
    unsupported = [
        arm
        for arm in manifest["arms"]
        if arm.get("support_state") == "unsupported" and arm.get("selected_for_measurement")
    ]
    if unsupported:
        raise EvidenceError("unsupported placements must not enter measured-arm denominators")
    phase = manifest.get("measurement_phase")
    if phase == "deterministic_arms_phase1":
        if set(selected) != {"B0", BASELINE_ARM, STRUCTURE_ONLY_ARM}:
            raise EvidenceError("deterministic phase 1 must select exactly B0, B1, and SO-BANK")
    elif not any(arm in SIDECAR_ARMS for arm in selected):
        raise EvidenceError("at least one supported sidecar arm must be selected outside deterministic phase 1")


def validate_frozen_tree(root: Path) -> None:
    evidence = root / "evidence" / "semantic-sidecar-v1"
    manifest_path = evidence / "measurement-contract-v1.json"
    status_path = evidence / "measurement-status-v1.json"
    calibration_path = evidence / "render-calibration-v1.json"
    row_contract_path = evidence / "measurement-row-contract-v1.json"
    ready_manifest_path = evidence / "measurement-manifest-phase1-v1.json"
    rows_path = evidence / "measurement-rows-phase1-v1.jsonl"
    analysis_path = evidence / "measurement-analysis-phase1-v1.json"
    required_paths = (
        manifest_path,
        status_path,
        calibration_path,
        row_contract_path,
        ready_manifest_path,
        rows_path,
        analysis_path,
        evidence / "cohort-classify-v1.jsonl",
        evidence / "cohort-adversarial-v1.jsonl",
        evidence / "SHA256SUMS",
    )
    for path in required_paths:
        if not path.is_file():
            raise EvidenceError(f"frozen evidence file is missing: {path}")
    manifest = read_json(manifest_path)
    validate_manifest(manifest, require_ready=False)
    status = read_json(status_path)
    calibration = read_json(calibration_path)
    row_contract = read_json(row_contract_path)
    if row_contract.get("record_id") != "semantic-sidecar-measurement-row-contract-v1":
        raise EvidenceError("unexpected measurement row-contract record_id")
    if status.get("result") != "negative_not_graduated":
        raise EvidenceError("checked-in status must be an honest negative result")
    if status.get("measurement_execution") != "phase1_executed":
        raise EvidenceError("checked-in status must record the executed deterministic phase")
    if status.get("graduated_arms") != []:
        raise EvidenceError("checked-in status contradicts the negative graduation result")
    if calibration.get("measurement_execution") != "not_run":
        raise EvidenceError("unrun calibration must be recorded as unrun")
    expected_records = {
        "measurement-contract-v1.json",
        "measurement-status-v1.json",
        "render-calibration-v1.json",
        "measurement-row-contract-v1.json",
        "measurement-manifest-phase1-v1.json",
        "measurement-analysis-phase1-v1.json",
        "measurement-rows-phase1-v1.jsonl",
        "cohort-classify-v1.jsonl",
        "cohort-adversarial-v1.jsonl",
    }
    entries: dict[str, str] = {}
    for line_number, line in enumerate((evidence / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        parts = line.split("  ")
        if len(parts) != 2 or len(parts[0]) != 64 or any(char not in "0123456789abcdef" for char in parts[0]):
            raise EvidenceError(f"SHA256SUMS:{line_number}: invalid digest line")
        digest, name = parts
        if name in entries:
            raise EvidenceError(f"SHA256SUMS:{line_number}: duplicate record {name}")
        entries[name] = digest
    if set(entries) != expected_records:
        raise EvidenceError("SHA256SUMS must cover exactly the frozen evidence records")
    for name, expected in entries.items():
        actual = sha256((evidence / name).read_bytes())
        if actual != expected:
            raise EvidenceError(f"digest mismatch for {name}: expected {expected}, got {actual}")

    ready_manifest = read_json(ready_manifest_path)
    validate_manifest(ready_manifest, require_ready=True)
    rows = read_jsonl(rows_path)
    validate_manifest_cohorts(rows, ready_manifest, evidence)
    report = summarize(rows, ready_manifest)
    if canonical_json(report) != analysis_path.read_bytes():
        raise EvidenceError("checked-in phase-1 analysis does not reproduce from the raw rows")
    if report.get("result") != status.get("result") or report.get("graduated_arms") != status.get("graduated_arms"):
        raise EvidenceError("checked-in status disagrees with reproduced phase-1 analysis")


def validate_manifest_cohorts(
    rows: list[dict[str, Any]], manifest: dict[str, Any], evidence_dir: Path
) -> None:
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, dict):
        raise EvidenceError("ready manifest must bind frozen cohort records")

    def cohort_rows(partition: str) -> list[dict[str, Any]]:
        record = cohorts.get(partition)
        if not isinstance(record, dict):
            raise EvidenceError(f"manifest lacks the {partition} cohort record")
        name = record.get("file")
        digest = record.get("sha256")
        if not isinstance(name, str) or Path(name).name != name:
            raise EvidenceError(f"manifest {partition} cohort file must be a local basename")
        path = evidence_dir / name
        if not path.is_file() or not isinstance(digest, str) or sha256(path.read_bytes()) != digest:
            raise EvidenceError(f"manifest {partition} cohort digest does not match {name}")
        frozen = read_jsonl(path)
        for item in frozen:
            if not isinstance(item.get("request_id"), str) or not item["request_id"]:
                raise EvidenceError(f"{name}: frozen request_id is missing")
            if not isinstance(item.get("prompt"), str) or not item["prompt"]:
                raise EvidenceError(f"{name}: frozen prompt is missing")
            if not isinstance(item.get("grammar"), str) or not item["grammar"]:
                raise EvidenceError(f"{name}: frozen grammar is missing")
            nonnegative_count(item.get("max_tokens"), f"{name}.max_tokens")
        return frozen

    primary = cohort_rows("primary")
    expected_primary = {item["request_id"] for item in primary}
    if len(expected_primary) != len(primary):
        raise EvidenceError("primary cohort duplicates a request_id")
    actual_primary = {row["request_id"] for row in rows if row.get("partition") == "primary"}
    if actual_primary != expected_primary:
        raise EvidenceError("measurement primary rows do not match the frozen cohort file")

    adversarial = cohort_rows("adversarial")
    expected_adversarial = {
        (item.get("request_id"), item.get("adversarial_schema_id")) for item in adversarial
    }
    if len(expected_adversarial) != len(adversarial) or any(
        not isinstance(schema_id, str) or not schema_id
        for _, schema_id in expected_adversarial
    ):
        raise EvidenceError("adversarial cohort has duplicate or missing schema identities")
    actual_adversarial = {
        (row["request_id"], row.get("adversarial_schema_id"))
        for row in rows
        if row.get("partition") == "adversarial"
    }
    if actual_adversarial != expected_adversarial:
        raise EvidenceError("measurement adversarial rows do not match the frozen cohort file")


def validate_phase1_arm_order(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    if manifest.get("measurement_phase") != "deterministic_arms_phase1":
        return
    protocol = manifest["protocol"]
    if protocol.get("arm_order_algorithm") != "sha256_u64_be_sort_v1":
        raise EvidenceError("deterministic phase 1 must bind its arm-order algorithm")
    seed = protocol.get("arm_order_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvidenceError("deterministic phase 1 must bind an integer arm-order seed")
    selected = supported_arms(manifest)
    for offset in range(0, len(rows), len(selected)):
        block = rows[offset : offset + len(selected)]
        if len(block) != len(selected):
            raise EvidenceError("measurement rows end inside an arm-order block")
        identity = {(row.get("partition"), row.get("request_id"), row.get("repetition")) for row in block}
        if len(identity) != 1:
            raise EvidenceError("measurement rows do not preserve request/repetition arm blocks")
        _, request_id, repetition = next(iter(identity))
        expected = sorted(
            selected,
            key=lambda arm: hashlib.sha256(
                f"{seed}\n{request_id}\nmeasured\n{repetition}\n{arm}".encode()
            ).digest()[:8],
        )
        actual = [row.get("arm_id") for row in block]
        if actual != expected:
            raise EvidenceError(f"{request_id}/{repetition}: measured arm order is not the seeded order")


def validate_row_shape(row: dict[str, Any], selected: set[str], repetitions: int) -> None:
    workload = row.get("workload")
    if workload not in WORKLOADS:
        raise EvidenceError(f"unknown workload {workload!r}")
    request_id = row.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise EvidenceError("measurement row needs a non-empty request_id")
    arm_id = row.get("arm_id")
    if arm_id not in selected:
        raise EvidenceError(f"row names an unselected or unsupported arm {arm_id!r}")
    repetition = row.get("repetition")
    if isinstance(repetition, bool) or not isinstance(repetition, int) or not 1 <= repetition <= repetitions:
        raise EvidenceError(f"{workload}/{request_id}/{arm_id}: repetition is outside frozen bounds")
    partition = row.get("partition")
    if partition not in {"primary", "adversarial"}:
        raise EvidenceError(f"{workload}/{request_id}/{arm_id}: partition must be primary or adversarial")
    if partition == "adversarial" and (not isinstance(row.get("adversarial_schema_id"), str) or not row["adversarial_schema_id"]):
        raise EvidenceError(f"{workload}/{request_id}/{arm_id}: adversarial row lacks its schema identity")
    if row.get("wall_clock_boundary") != BOUNDARY:
        raise EvidenceError(f"{workload}/{request_id}/{arm_id}: wall-clock boundary is not common")
    positive_number(row.get("wall_clock_ms"), f"{workload}/{request_id}/{arm_id}.wall_clock_ms")
    token_ids(row.get("generated_token_ids"), f"{workload}/{request_id}/{arm_id}.generated_token_ids")
    decoded = row.get("decoded_response_sha256")
    if not isinstance(decoded, str) or len(decoded) != 64 or any(char not in "0123456789abcdef" for char in decoded):
        raise EvidenceError(f"{workload}/{request_id}/{arm_id}: decoded_response_sha256 is invalid")
    if not isinstance(row.get("finish_reason"), str) or not row["finish_reason"]:
        raise EvidenceError(f"{workload}/{request_id}/{arm_id}: finish_reason is missing")
    instrument = row.get("instrumentation")
    if not isinstance(instrument, dict):
        raise EvidenceError(f"{workload}/{request_id}/{arm_id}: instrumentation is missing")
    for field in ("target_decode_ms", "response_ready_ms"):
        nonnegative_number(instrument.get(field), f"{workload}/{request_id}/{arm_id}.instrumentation.{field}")
    if arm_id == STRUCTURE_ONLY_ARM:
        bank = row.get("so_bank")
        if not isinstance(bank, dict):
            raise EvidenceError(f"{workload}/{request_id}: SO-BANK row lacks bank evidence")
        if bank.get("source") != "longest_common_legal_completion_byte_prefix":
            raise EvidenceError(f"{workload}/{request_id}: SO-BANK source is not the legal-completion prefix")
        if bank.get("automaton_state_visit_cap") != 4096:
            raise EvidenceError(f"{workload}/{request_id}: SO-BANK state-visit cap is not frozen")
        if bank.get("max_suffix_match_tokens") != 7 or bank.get("max_proposal_tokens") != 16:
            raise EvidenceError(f"{workload}/{request_id}: SO-BANK bounds do not match the frozen contract")
        digest = bank.get("content_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            raise EvidenceError(f"{workload}/{request_id}: SO-BANK digest is invalid")
    if arm_id == STRUCTURE_ONLY_ARM or arm_id in SIDECAR_ARMS:
        verification = row.get("verification")
        if not isinstance(verification, dict) or verification.get("grammar_masked") is not True:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: grammar-masked verification evidence is missing")
        for field in ("proposed_tokens", "verified_tokens", "accepted_tokens", "rejected_proposal_attempts"):
            nonnegative_count(verification.get(field), f"{workload}/{request_id}/{arm_id}.verification.{field}")
        proposed = verification["proposed_tokens"]
        verified = verification["verified_tokens"]
        accepted = verification["accepted_tokens"]
        if accepted > verified or verified > proposed:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: verification token counts are inconsistent")
        spans = verification.get("accepted_tokens_by_span")
        if not isinstance(spans, dict) or set(spans) != {"structural", "value"}:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: accepted span attribution is missing")
        if sum(nonnegative_count(spans[name], f"{workload}/{request_id}/{arm_id}.verification.{name}") for name in spans) != accepted:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: accepted span attribution does not sum to accepted tokens")
        divergences = verification.get("first_divergence_categories")
        if not isinstance(divergences, dict) or set(divergences) != {"semantic_value", "json_structure", "whitespace", "tokenization_boundary"}:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: first-divergence categories are incomplete")
        if sum(nonnegative_count(divergences[name], f"{workload}/{request_id}/{arm_id}.verification.{name}") for name in divergences) > verification["rejected_proposal_attempts"]:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: divergence count exceeds rejected attempts")
    if arm_id in SIDECAR_ARMS:
        sidecar = row.get("sidecar")
        if not isinstance(sidecar, dict):
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: sidecar accounting is missing")
        outcome = sidecar.get("outcome")
        if outcome not in OUTCOMES:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: sidecar outcome is not terminal")
        matched_zero_acceptance = sidecar.get("matched_zero_acceptance")
        if not isinstance(matched_zero_acceptance, bool):
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: unused-bank sub-counter is missing")
        if matched_zero_acceptance and outcome != "completed_usable_unused":
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: matched-zero acceptance requires a usable unused outcome")
        for field in (
            "job_latency_ms",
            "pickup_delay_ms",
            "cancellation_to_termination_ms",
            "post_response_occupancy_ms",
        ):
            nonnegative_number(sidecar.get(field), f"{workload}/{request_id}/{arm_id}.sidecar.{field}")
        for field in ("prompt_tokens", "generated_tokens"):
            nonnegative_count(sidecar.get(field), f"{workload}/{request_id}/{arm_id}.sidecar.{field}")
        components = sidecar.get("component_latency_ms")
        if not isinstance(components, dict):
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: component latency is missing")
        for field in ("dispatch", "generation", "transport", "render", "target_tokenization"):
            nonnegative_number(components.get(field), f"{workload}/{request_id}/{arm_id}.sidecar.{field}")


def request_medians(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    protocol = manifest["protocol"]
    repetitions = int(protocol["repetitions"])
    selected = set(supported_arms(manifest))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    observed_keys: set[tuple[str, str, str, int]] = set()
    for row in rows:
        validate_row_shape(row, selected, repetitions)
        key = (row["workload"], row["request_id"], row["arm_id"], row["repetition"])
        if key in observed_keys:
            raise EvidenceError(f"duplicate measurement row {key}")
        observed_keys.add(key)
        grouped[key[:3]].append(row)
    request_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for workload, request_id, arm_id in grouped:
        request_sets[(workload, arm_id)].add(request_id)
    minimum = int(protocol["minimum_eligible_requests_per_workload"])
    for workload in measured_workloads(manifest):
        baseline_ids = request_sets[(workload, BASELINE_ARM)]
        if len(baseline_ids) < minimum:
            raise EvidenceError(f"{workload}: B1 has {len(baseline_ids)} eligible requests, needs {minimum}")
        for arm_id in selected:
            arm_ids = request_sets[(workload, arm_id)]
            if arm_ids != baseline_ids:
                raise EvidenceError(f"{workload}: {arm_id} did not use B1's frozen eligible request IDs")
    medians: dict[tuple[str, str, str], float] = {}
    b1_outputs: dict[tuple[str, str], tuple[tuple[int, ...], str, str]] = {}
    so_bank_digests: dict[tuple[str, str], str] = {}
    for key, group in grouped.items():
        workload, request_id, arm_id = key
        if len(group) != repetitions or {row["repetition"] for row in group} != set(range(1, repetitions + 1)):
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: missing measured repetition")
        outputs = {
            (
                token_ids(row["generated_token_ids"], "generated_token_ids"),
                row["decoded_response_sha256"],
                row["finish_reason"],
            )
            for row in group
        }
        if len(outputs) != 1:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: an arm is not deterministic across repetitions")
        output = next(iter(outputs))
        if arm_id == BASELINE_ARM:
            b1_outputs[(workload, request_id)] = output
        if arm_id == STRUCTURE_ONLY_ARM:
            digests = {row["so_bank"]["content_digest"] for row in group}
            if len(digests) != 1:
                raise EvidenceError(f"{workload}/{request_id}: SO-BANK is not deterministic across repetitions")
            so_bank_digests[(workload, request_id)] = next(iter(digests))
        medians[key] = float(statistics.median(row["wall_clock_ms"] for row in group))
    for (workload, request_id, arm_id), group in grouped.items():
        if arm_id == BASELINE_ARM:
            continue
        expected = b1_outputs.get((workload, request_id))
        actual = (
            token_ids(group[0]["generated_token_ids"], "generated_token_ids"),
            group[0]["decoded_response_sha256"],
            group[0]["finish_reason"],
        )
        if actual != expected:
            raise EvidenceError(f"{workload}/{request_id}/{arm_id}: token, response, or finish mismatch against B1")
    if len(so_bank_digests) != sum(
        len(request_sets[(workload, BASELINE_ARM)]) for workload in measured_workloads(manifest)
    ):
        raise EvidenceError("SO-BANK evidence is incomplete")
    return medians


def validate_adversarial_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Require every selected arm to cover each adversarial schema and repetition with B1-exact output."""
    protocol = manifest["protocol"]
    repetitions = int(protocol["repetitions"])
    required_schema_count = int(protocol["adversarial_schema_count"])
    selected = supported_arms(manifest)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_row_shape(row, set(selected), repetitions)
        if row["partition"] != "adversarial":
            raise EvidenceError("adversarial validation received a primary row")
        key = (row["workload"], row["adversarial_schema_id"], row["arm_id"])
        grouped[key].append(row)
    schemas_by_arm: dict[str, set[tuple[str, str]]] = defaultdict(set)
    b1_outputs: dict[tuple[str, str], tuple[tuple[int, ...], str, str]] = {}
    for (workload, schema_id, arm_id), group in grouped.items():
        if len(group) != repetitions or {row["repetition"] for row in group} != set(range(1, repetitions + 1)):
            raise EvidenceError(f"adversarial {workload}/{schema_id}/{arm_id}: missing measured repetition")
        outputs = {
            (
                token_ids(row["generated_token_ids"], "generated_token_ids"),
                row["decoded_response_sha256"],
                row["finish_reason"],
            )
            for row in group
        }
        if len(outputs) != 1:
            raise EvidenceError(f"adversarial {workload}/{schema_id}/{arm_id}: output changes across repetitions")
        schemas_by_arm[arm_id].add((workload, schema_id))
        if arm_id == BASELINE_ARM:
            b1_outputs[(workload, schema_id)] = next(iter(outputs))
    baseline_schemas = schemas_by_arm[BASELINE_ARM]
    if len(baseline_schemas) < required_schema_count:
        raise EvidenceError(f"adversarial battery has {len(baseline_schemas)} schemas, needs {required_schema_count}")
    for arm_id in selected:
        if schemas_by_arm[arm_id] != baseline_schemas:
            raise EvidenceError(f"adversarial {arm_id} did not run B1's frozen schema battery")
    for (workload, schema_id, arm_id), group in grouped.items():
        if arm_id == BASELINE_ARM:
            continue
        actual = (
            token_ids(group[0]["generated_token_ids"], "generated_token_ids"),
            group[0]["decoded_response_sha256"],
            group[0]["finish_reason"],
        )
        if actual != b1_outputs[(workload, schema_id)]:
            raise EvidenceError(f"adversarial {workload}/{schema_id}/{arm_id}: mismatch against B1")
    return {"schema_count": len(baseline_schemas), "arms": selected, "grammar_masked": True}


def bootstrap_median_ci(differences: list[float], resamples: int, seed: int) -> tuple[float, float]:
    if not differences:
        raise EvidenceError("cannot bootstrap no paired request differences")
    generator = random.Random(seed)
    count = len(differences)
    statistics_samples = [
        float(statistics.median(differences[generator.randrange(count)] for _ in range(count)))
        for _ in range(resamples)
    ]
    return percentile(statistics_samples, 0.025), percentile(statistics_samples, 0.975)


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest, require_ready=True)
    validate_phase1_arm_order(rows, manifest)
    primary_rows = [row for row in rows if row.get("partition") == "primary"]
    adversarial_rows = [row for row in rows if row.get("partition") == "adversarial"]
    if len(primary_rows) + len(adversarial_rows) != len(rows):
        raise EvidenceError("measurement rows contain an unknown partition")
    if not primary_rows or not adversarial_rows:
        raise EvidenceError("a result needs both primary workload rows and adversarial battery rows")
    adversarial = validate_adversarial_rows(adversarial_rows, manifest)
    medians = request_medians(primary_rows, manifest)
    target_decode_samples: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in primary_rows:
        target_decode_samples[(row["workload"], row["request_id"], row["arm_id"])].append(
            float(row["instrumentation"]["target_decode_ms"])
        )
    target_decode_medians = {
        key: float(statistics.median(values)) for key, values in target_decode_samples.items()
    }
    protocol = manifest["protocol"]
    selected = supported_arms(manifest)
    resamples = int(protocol["bootstrap_resamples"])
    seed = int(protocol["bootstrap_seed"])
    per_workload: dict[str, dict[str, Any]] = {}
    arm_graduation: dict[str, bool] = {}
    for workload in measured_workloads(manifest):
        workload_report: dict[str, Any] = {}
        request_ids = sorted(request_id for w, request_id, arm in medians if w == workload and arm == BASELINE_ARM)
        b1 = [medians[(workload, request_id, BASELINE_ARM)] for request_id in request_ids]
        b1_p95 = percentile(b1, 0.95)
        for arm_id in selected:
            values = [medians[(workload, request_id, arm_id)] for request_id in request_ids]
            entry: dict[str, Any] = {
                "request_count": len(values),
                "request_median_wall_clock_ms": {
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "mean": statistics.fmean(values),
                },
            }
            if arm_id != BASELINE_ARM:
                differences = [
                    medians[(workload, request_id, BASELINE_ARM)] - medians[(workload, request_id, arm_id)]
                    for request_id in request_ids
                ]
                low, high = bootstrap_median_ci(differences, resamples, seed)
                entry["paired_bootstrap_median_improvement_ms"] = {
                    "resamples": resamples,
                    "seed": seed,
                    "estimate": statistics.median(differences),
                    "ci95_percentile": [low, high],
                }
                if arm_id == STRUCTURE_ONLY_ARM or arm_id in SIDECAR_ARMS:
                    arm_graduation.setdefault(arm_id, True)
                    entry["graduation_gates"] = {
                        "median_at_least_five_percent_lower_than_b1": percentile(values, 0.5) <= percentile(b1, 0.5) * 0.95,
                        "paired_ci_lower_bound_above_zero": low > 0,
                        "p95_no_more_than_five_percent_worse_than_b1": percentile(values, 0.95) <= b1_p95 * 1.05,
                    }
                    arm_graduation[arm_id] = arm_graduation[arm_id] and all(entry["graduation_gates"].values())
                if arm_id in SIDECAR_ARMS:
                    baseline_target = [target_decode_medians[(workload, request_id, BASELINE_ARM)] for request_id in request_ids]
                    arm_target = [target_decode_medians[(workload, request_id, arm_id)] for request_id in request_ids]
                    entry["target_slowdown_ms"] = {
                        "B1_target_decode": {"p50": percentile(baseline_target, 0.5), "p95": percentile(baseline_target, 0.95), "mean": statistics.fmean(baseline_target)},
                        "sidecar_arm_target_decode": {"p50": percentile(arm_target, 0.5), "p95": percentile(arm_target, 0.95), "mean": statistics.fmean(arm_target)},
                        "paired_arm_minus_B1_median": statistics.median([arm - baseline for arm, baseline in zip(arm_target, baseline_target)]),
                    }

            workload_report[arm_id] = entry
        per_workload[workload] = workload_report
    outcomes: dict[str, dict[str, Any]] = {}
    latency: dict[str, dict[str, Any]] = {}
    verification_evidence: dict[str, dict[str, Any]] = {}
    for arm_id in selected:
        arm_rows = [row for row in primary_rows if row["arm_id"] == arm_id]
        if arm_id == STRUCTURE_ONLY_ARM or arm_id in SIDECAR_ARMS:
            verification_rows = [row["verification"] for row in arm_rows]
            verified = sum(int(row["verified_tokens"]) for row in verification_rows)
            accepted = sum(int(row["accepted_tokens"]) for row in verification_rows)
            verification_evidence[arm_id] = {
                "grammar_masked_rows": len(verification_rows),
                "proposed_tokens": sum(int(row["proposed_tokens"]) for row in verification_rows),
                "verified_tokens": verified,
                "accepted_tokens": accepted,
                "rejected_proposal_attempts": sum(int(row["rejected_proposal_attempts"]) for row in verification_rows),
                "acceptance_rate": None if verified == 0 else accepted / verified,
                "accepted_tokens_by_span": {
                    "structural": sum(int(row["accepted_tokens_by_span"]["structural"]) for row in verification_rows),
                    "value": sum(int(row["accepted_tokens_by_span"]["value"]) for row in verification_rows),
                },
                "first_divergence_categories": {
                    category: sum(int(row["first_divergence_categories"][category]) for row in verification_rows)
                    for category in ("semantic_value", "json_structure", "whitespace", "tokenization_boundary")
                },
            }
        if arm_id in SIDECAR_ARMS:
            histogram = Counter(str(row["sidecar"]["outcome"]) for row in arm_rows)
            if set(histogram) - set(OUTCOMES):
                raise EvidenceError(f"{arm_id}: unknown terminal outcome in histogram")
            counts = {outcome: histogram.get(outcome, 0) for outcome in OUTCOMES}
            launched = len(arm_rows)
            outcomes[arm_id] = {
                "launched": launched,
                "counts": counts,
                "rates": {outcome: count / launched for outcome, count in counts.items()},
                "sidecar_success_rate": (counts["completed_usable_used"] + counts["completed_usable_unused"]) / launched,
                "completed_usable_unused": {
                    "matched_zero_acceptance": sum(
                        int(row["sidecar"]["matched_zero_acceptance"])
                        for row in arm_rows
                        if row["sidecar"]["outcome"] == "completed_usable_unused"
                    )
                },
            }
            components = {
                component: [float(row["sidecar"]["component_latency_ms"][component]) for row in arm_rows]
                for component in ("dispatch", "generation", "transport", "render", "target_tokenization")
            }
            job_latency = [float(row["sidecar"]["job_latency_ms"]) for row in arm_rows]
            occupancy = [float(row["sidecar"]["post_response_occupancy_ms"]) for row in arm_rows]
            cancellation = [float(row["sidecar"]["cancellation_to_termination_ms"]) for row in arm_rows]
            latency[arm_id] = {
                "sidecar_job_latency_ms": {"p50": percentile(job_latency, 0.5), "p95": percentile(job_latency, 0.95)},
                "component_latency_ms": {
                    component: {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}
                    for component, values in components.items()
                },
                "prompt_tokens": {"p50": percentile([float(row["sidecar"]["prompt_tokens"]) for row in arm_rows], 0.5), "p95": percentile([float(row["sidecar"]["prompt_tokens"]) for row in arm_rows], 0.95)},
                "generated_tokens": {"p50": percentile([float(row["sidecar"]["generated_tokens"]) for row in arm_rows], 0.5), "p95": percentile([float(row["sidecar"]["generated_tokens"]) for row in arm_rows], 0.95)},
                "cancellation_to_termination_ms": {"p50": percentile(cancellation, 0.5), "p95": percentile(cancellation, 0.95)},
                "post_response_occupancy_ms": {"p50": percentile(occupancy, 0.5), "p95": percentile(occupancy, 0.95)},
            }
    graduated = [arm for arm, passed in arm_graduation.items() if passed]
    return {
        "record_id": "semantic-sidecar-measurement-result-v1",
        "result": "graduated" if graduated else "negative_not_graduated",
        "graduated_arms": graduated,
        "common_wall_clock_boundary": BOUNDARY,
        "per_workload": per_workload,
        "adversarial_masked_verification": adversarial,
        "outcome_histograms": outcomes,
        "verification_evidence": verification_evidence,
        "latency_and_resource_evidence": latency,
        "notes": [
            "All timing statistics use request medians; bootstrap units are request IDs.",
            "An accelerated arm is eligible to graduate only when every measured workload passes every performance gate and token exactness has already been validated.",
        ],
    }


def fake_row(
    workload: str,
    request_id: str,
    arm_id: str,
    repetition: int,
    wall_clock_ms: float,
    partition: str = "primary",
    adversarial_schema_id: str | None = None,
) -> dict[str, Any]:
    tokens = [ord("{") + repetition * 0, len(request_id), ord("}")]
    row: dict[str, Any] = {
        "workload": workload,
        "request_id": request_id,
        "arm_id": arm_id,
        "repetition": repetition,
        "partition": partition,
        "wall_clock_boundary": BOUNDARY,
        "wall_clock_ms": wall_clock_ms,
        "generated_token_ids": tokens,
        "decoded_response_sha256": sha256(f"{workload}/{request_id}".encode()),
        "finish_reason": "stop",
        "instrumentation": {"target_decode_ms": wall_clock_ms * 0.8, "response_ready_ms": wall_clock_ms},
    }
    if arm_id == STRUCTURE_ONLY_ARM:
        row["so_bank"] = {
            "source": "longest_common_legal_completion_byte_prefix",
            "automaton_state_visit_cap": 4096,
            "max_suffix_match_tokens": 7,
            "max_proposal_tokens": 16,
            "content_digest": sha256(f"bank/{workload}/{request_id}".encode()),
        }
    if arm_id == STRUCTURE_ONLY_ARM or arm_id in SIDECAR_ARMS:
        row["verification"] = {
            "grammar_masked": True,
            "proposed_tokens": 3,
            "verified_tokens": 3,
            "accepted_tokens": 3,
            "rejected_proposal_attempts": 0,
            "accepted_tokens_by_span": {"structural": 1, "value": 2},
            "first_divergence_categories": {
                "semantic_value": 0,
                "json_structure": 0,
                "whitespace": 0,
                "tokenization_boundary": 0,
            },
        }
    if partition == "adversarial":
        row["adversarial_schema_id"] = adversarial_schema_id or request_id
    if arm_id in SIDECAR_ARMS:
        row["sidecar"] = {
            "outcome": "completed_usable_used",
            "matched_zero_acceptance": False,
            "job_latency_ms": 12.0,
            "prompt_tokens": 40,
            "generated_tokens": 12,
            "pickup_delay_ms": 0.2,
            "cancellation_to_termination_ms": 0.0,
            "post_response_occupancy_ms": 0.0,
            "component_latency_ms": {
                "dispatch": 1.0,
                "generation": 9.0,
                "transport": 0.5,
                "render": 0.7,
                "target_tokenization": 0.8,
            },
        }
    return row


def self_test(root: Path) -> None:
    validate_frozen_tree(root)
    manifest = read_json(root / "evidence" / "semantic-sidecar-v1" / "measurement-contract-v1.json")
    manifest = copy.deepcopy(manifest)
    manifest["measurement_state"] = "ready"
    rows: list[dict[str, Any]] = []
    for workload in WORKLOADS:
        for request_number in range(100):
            request_id = f"{workload}-{request_number:03d}"
            for repetition in range(1, 6):
                rows.append(fake_row(workload, request_id, BASELINE_ARM, repetition, 100.0))
                rows.append(fake_row(workload, request_id, STRUCTURE_ONLY_ARM, repetition, 101.0))
                rows.append(fake_row(workload, request_id, "SC-METAL-DIRECT", repetition, 94.0))
    for schema_number in range(30):
        request_id = f"adversarial-{schema_number:03d}"
        schema_id = f"schema-{schema_number:03d}"
        for repetition in range(1, 6):
            for arm_id, wall_clock_ms in (
                (BASELINE_ARM, 100.0),
                (STRUCTURE_ONLY_ARM, 101.0),
                ("SC-METAL-DIRECT", 94.0),
            ):
                rows.append(
                    fake_row(
                        WORKLOADS[0],
                        request_id,
                        arm_id,
                        repetition,
                        wall_clock_ms,
                        partition="adversarial",
                        adversarial_schema_id=schema_id,
                    )
                )
    report = summarize(rows, manifest)
    if report["result"] != "graduated" or report["graduated_arms"] != ["SC-METAL-DIRECT"]:
        raise AssertionError("known-good paired fixture must satisfy the graduation gate")

    phase1_manifest = copy.deepcopy(manifest)
    phase1_manifest["measurement_phase"] = "deterministic_arms_phase1"
    phase1_manifest["protocol"]["arm_order_seed"] = 20260810
    phase1_manifest["protocol"]["arm_order_algorithm"] = "sha256_u64_be_sort_v1"
    phase1_manifest["workloads"] = [
        {"workload_id": "athena-classify-json", "selected_for_measurement": True}
    ]
    for arm in phase1_manifest["arms"]:
        arm["selected_for_measurement"] = arm["arm_id"] in {"B0", BASELINE_ARM, STRUCTURE_ONLY_ARM}
    phase1_rows = [
        row
        for row in rows
        if row["workload"] == "athena-classify-json"
        and row["arm_id"] in {BASELINE_ARM, STRUCTURE_ONLY_ARM}
    ]
    for row in copy.deepcopy([row for row in phase1_rows if row["arm_id"] == BASELINE_ARM]):
        row["arm_id"] = "B0"
        phase1_rows.append(row)
    phase1_rows.sort(
        key=lambda row: (
            row["partition"] != "primary",
            row["request_id"],
            row["repetition"],
            hashlib.sha256(
                f"20260810\n{row['request_id']}\nmeasured\n{row['repetition']}\n{row['arm_id']}".encode()
            ).digest()[:8],
        )
    )
    phase1_report = summarize(phase1_rows, phase1_manifest)
    if phase1_report["result"] != "negative_not_graduated" or phase1_report["graduated_arms"]:
        raise AssertionError("deterministic phase-1 fixture must analyze without a sidecar graduation claim")
    corrupted = copy.deepcopy(rows)
    for row in corrupted:
        if (
            row["workload"] == WORKLOADS[-1]
            and row["request_id"] == f"{WORKLOADS[-1]}-099"
            and row["arm_id"] == "SC-METAL-DIRECT"
        ):
            row["generated_token_ids"] = [0]
    try:
        summarize(corrupted, manifest)
    except EvidenceError as error:
        if "mismatch against B1" not in str(error):
            raise AssertionError(f"wrong rejection reason for token mismatch: {error}") from error
    else:
        raise AssertionError("token mismatch must reject a claimed result")
    insufficient = copy.deepcopy(rows)
    insufficient = [row for row in insufficient if row["repetition"] != 5]
    try:
        summarize(insufficient, manifest)
    except EvidenceError as error:
        if "missing measured repetition" not in str(error):
            raise AssertionError(f"wrong rejection reason for missing repetition: {error}") from error
    else:
        raise AssertionError("four repetitions must reject a claimed result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-frozen", action="store_true", help="validate committed evidence records and digests")
    parser.add_argument("--self-test", action="store_true", help="exercise exactness, repetition, and graduation guards")
    parser.add_argument("--manifest", type=Path, help="ready measurement manifest for analysis")
    parser.add_argument("--input", type=Path, help="JSONL request-repetition rows captured by the arm runner")
    parser.add_argument("--out", type=Path, help="output path for an analyzed result")
    args = parser.parse_args()
    try:
        if args.validate_frozen:
            validate_frozen_tree(args.repo)
        if args.self_test:
            self_test(args.repo)
        analyze = any(value is not None for value in (args.manifest, args.input, args.out))
        if analyze:
            if not all(value is not None for value in (args.manifest, args.input, args.out)):
                raise EvidenceError("--manifest, --input, and --out must be supplied together")
            rows = read_jsonl(args.input)
            manifest = read_json(args.manifest)
            validate_manifest_cohorts(rows, manifest, args.manifest.resolve().parent)
            report = summarize(rows, manifest)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(canonical_json(report))
        if not args.validate_frozen and not args.self_test and not analyze:
            parser.error("choose --validate-frozen, --self-test, or --manifest/--input/--out")
    except EvidenceError as error:
        print(f"semantic-sidecar evidence rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
