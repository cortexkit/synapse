#!/usr/bin/env python3
"""Validate the ANE-prefill split contract and its complete first-cut manifest matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


CONTRACT_PATH = Path("contracts/ane-prefill-split/ane-prefill-split-contract-v1.json")
MANIFEST_PATH = Path("manifests/ane-prefill-split/ane-prefill-split-manifest-v1.json")
NAMED_PATHS = {
    "epic_design_note": "docs/design-ane-prefill-split.md",
    "evidence_record": "evidence/ane-prefill-split/evidence-record-v1.json",
    "operations_document": "docs/operations-ane-prefill-split.md",
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
FALLBACK_REASONS = (
    "compile_failure",
    "load_failure",
    "dispatch_failure",
    "prediction_failure",
    "prediction_timeout",
    "kv_conversion_failure",
    "ipc_handoff_failure",
    "cache_handoff_failure",
    "metal_upload_failure",
    "transfer_budget_exceeded",
    "readiness_budget_exhausted",
    "artifact_mismatch",
    "logits_publication_failure",
)
ABSENCE_REASONS = (
    "capacity_precondition_unmet",
    "correctness_divergence",
    "placement_failure",
    "cert_transfer_budget_exceeded",
    "ttft_not_lower",
    "compile_or_load_failure",
    "disk_headroom_unmet",
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
EXPECTED_BRANCHES = {
    "OQ-2-decode-cache-capacity": "enumerate_existing_decode_cache_capacities_and_record_absent_when_bucket_plus_64_is_unmet",
    "OQ-3-worker-stage-embodiment": "separately_supervised_swift_coreml_sidecar",
    "OQ-4-scheduler-accounting": "pure_gpu_equivalent_admission_accounting",
    "OQ-5-gpu-freed-calibration-failure": "calibration_failure_blocks_enable_pending_owner_decision",
}


class ContractError(ValueError):
    """Raised when a frozen contract invariant is absent or changes."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_equal(actual: Any, expected: Any, message: str) -> None:
    require(actual == expected, f"{message}: expected {expected!r}, got {actual!r}")


def require_exact_list(value: Any, expected: tuple[str, ...], label: str) -> None:
    require(isinstance(value, list), f"{label} must be a list")
    require_equal(tuple(value), expected, label)
    require_equal(len(value), len(set(value)), f"{label} contains a duplicate")


def require_disjoint(reason_sets: dict[str, set[str]]) -> None:
    names = tuple(reason_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            shared = reason_sets[left_name] & reason_sets[right_name]
            require(not shared, f"{left_name} and {right_name} overlap: {sorted(shared)}")


def validate_contract(contract: dict[str, Any]) -> None:
    require_equal(contract.get("contract_revision"), "ane-prefill-split-contract-v1", "contract revision")
    require_equal(contract.get("schema_revision"), 1, "schema revision")
    named_artifacts = contract.get("named_artifacts")
    require(isinstance(named_artifacts, dict), "named_artifacts must be an object")
    for artifact, path in NAMED_PATHS.items():
        entry = named_artifacts.get(artifact)
        require(isinstance(entry, dict), f"named artifact {artifact} is missing")
        require_equal(entry.get("path"), path, f"named artifact path for {artifact}")
        gates = entry.get("must_exist_before")
        require(isinstance(gates, list) and gates, f"named artifact {artifact} has no creation gate")
    design_records = named_artifacts["epic_design_note"].get("records")
    require(isinstance(design_records, list), "epic design note records must be a list")
    require(
        "handshake-mismatch -> unavailable-arm mapping and its bypass reason" in design_records,
        "epic design note must record the handshake-mismatch mapping",
    )

    handshake = contract.get("worker_handshake")
    require(isinstance(handshake, dict), "worker_handshake must be an object")
    require_equal(handshake.get("phase"), "sidecar CONNECT before any request frame", "worker handshake phase")
    require_equal(
        handshake.get("protocol"),
        "existing strict handshake negotiates the worker protocol version and engine identity",
        "worker handshake protocol",
    )
    require_equal(handshake.get("bypass_reason"), "quarantined", "worker handshake bypass reason")
    require_equal(handshake.get("request_health_debit"), False, "worker handshake request health debit")
    require("marks the exact arm unavailable" in str(handshake.get("mismatch")), "worker handshake unavailable consequence")
    require("debits that arm once" in str(handshake.get("mismatch")), "worker handshake health debit")
    require("dispatch_failure" in str(handshake.get("mid_attempt_defense_in_depth")), "mid-attempt mismatch fallback")

    arm_schema = contract.get("arm_schema")
    require(isinstance(arm_schema, dict), "arm_schema must be an object")
    require_exact_list(
        arm_schema.get("identity_fields"),
        ("machine_profile", "family", "bucket", "decode_config"),
        "arm identity fields",
    )
    require_equal(arm_schema.get("canonical_key"), "(machine_profile, family, bucket, decode_config)", "arm key")
    require_equal(arm_schema.get("compiled_package_key"), "(family, bucket)", "compiled package key")
    require_equal(arm_schema.get("inheritance"), "none across decode_config, bucket, or machine_profile", "arm inheritance")

    labels = contract.get("decode_config_labels")
    require(isinstance(labels, dict), "decode config labels must be an object")
    require_equal(set(labels), {"f16-step", "q8-step"}, "decode config labels")
    require_equal(labels["f16-step"].get("weight_quant"), "f16", "f16-step weight quant")
    require_equal(labels["f16-step"].get("engine_to_engine_cache_handoff"), False, "f16-step handoff")
    require_equal(labels["q8-step"].get("weight_quant"), "q8_0", "q8-step weight quant")
    require_equal(labels["q8-step"].get("engine_to_engine_cache_handoff"), True, "q8-step handoff")

    triple = contract.get("artifact_triple")
    require(isinstance(triple, dict), "artifact_triple must be an object")
    require_exact_list(
        triple.get("fields"),
        (
            "source_checkpoint_digest",
            "derived_or_compiled_artifact_digest",
            "certification_recorded_artifact_digest",
        ),
        "artifact triple fields",
    )
    require_equal(triple.get("preselection_mismatch"), "artifact_digest_mismatch bypass", "preselection artifact mismatch")
    require_equal(
        triple.get("postselection_load_completion_mismatch"),
        "artifact_mismatch fallback with one exact-arm health debit and no certification-row mutation",
        "postselection artifact mismatch",
    )

    certification = contract.get("certification")
    require(isinstance(certification, dict), "certification must be an object")
    require_exact_list(certification.get("row_outcomes"), ("certified", "absent"), "certification row outcomes")
    require_exact_list(
        certification.get("runtime_states"),
        ("certified", "bucket_absent", "not_certified"),
        "runtime certification states",
    )
    require_equal(
        certification.get("not_runtime_states"),
        ["artifact_digest_mismatch", "quarantined", "readiness", "load_state"],
        "non-state observables",
    )

    vocabularies = contract.get("reason_vocabularies")
    require(isinstance(vocabularies, dict), "reason_vocabularies must be an object")
    require_exact_list(vocabularies.get("prefill_bypass_reason"), BYPASS_REASONS, "bypass vocabulary")
    require_exact_list(vocabularies.get("prefill_fallback_reason"), FALLBACK_REASONS, "fallback vocabulary")
    require_exact_list(vocabularies.get("absence_reason"), ABSENCE_REASONS, "absence vocabulary")
    require_disjoint(
        {
            "prefill_bypass_reason": set(BYPASS_REASONS),
            "prefill_fallback_reason": set(FALLBACK_REASONS),
            "absence_reason": set(ABSENCE_REASONS),
        }
    )

    bypass_table = contract.get("bypass_table")
    require(isinstance(bypass_table, list), "bypass_table must be a list")
    require_equal(len(bypass_table), 13, "bypass table cardinality")
    bypass_entries = {entry.get("reason"): entry for entry in bypass_table if isinstance(entry, dict)}
    require_equal(set(bypass_entries), set(BYPASS_REASONS), "bypass table coverage")
    require(all(entry.get("split_attempt_started") is False for entry in bypass_entries.values()), "a bypass table row starts an attempt")
    require(all(entry.get("health_debit") is False for entry in bypass_entries.values()), "a bypass table row debits health")
    quarantined = bypass_entries["quarantined"]
    require(
        quarantined.get("unavailable_arm_mapping")
        == "A CONNECT-time protocol-version or engine-identity mismatch has already charged this exact arm once and made it quarantine-eligible; later requests bypass without an additional debit.",
        "quarantined bypass table handshake mapping",
    )

    fallback_table = contract.get("fallback_table")
    require(isinstance(fallback_table, list), "fallback_table must be a list")
    require_equal(len(fallback_table), 13, "fallback table cardinality")
    fallback_entries = {entry.get("reason"): entry for entry in fallback_table if isinstance(entry, dict)}
    require_equal(set(fallback_entries), set(FALLBACK_REASONS), "fallback table coverage")
    require(all(entry.get("health_debit") == "exact_arm_once" for entry in fallback_entries.values()), "a fallback row lacks exact-arm health accounting")
    fault_map = {entry.get("fault"): entry.get("reason") for entry in fallback_table if isinstance(entry, dict)}
    require_equal(fault_map, FALLBACK_FAULTS, "fault-to-fallback table")

    routing = contract.get("routing")
    require(isinstance(routing, dict), "routing must be an object")
    require_exact_list(routing.get("global_bypass_precedence"), BYPASS_REASONS[:5], "global bypass precedence")
    selection = routing.get("selection_algorithm")
    require(isinstance(selection, list) and len(selection) == 7, "selection algorithm must have seven steps")
    require(str(selection[0]).startswith("evaluate global_bypass_precedence before window enumeration"), "global gates must precede window work")
    require("smallest remaining window alone" in str(selection[4]), "terminal state must come from the smallest remaining window")

    timing = contract.get("timing")
    require(isinstance(timing, dict), "timing must be an object")
    values = timing.get("derived_values")
    require(isinstance(values, dict), "timing derived_values must be an object")
    require_equal(values.get("ane_attempt_budget_ms"), "2 * calibrated_coreml_prediction_p95_ms", "attempt budget derivation")
    require_equal(values.get("prediction_budget_ms"), "ane_attempt_budget_ms", "prediction budget derivation")
    require_equal(values.get("handoff_budget_ms"), "2 * calibrated_full_handoff_p95_ms", "handoff budget derivation")
    require("guard_wait_budget is retired" in str(timing.get("guard_rule")), "retired guard knob must be named")
    require_equal(
        timing.get("full_split_ceiling_ms"),
        "ane_attempt_budget_ms + readiness_budget_ms + prediction_budget_ms + handoff_budget_ms",
        "full split ceiling",
    )

    fingerprint = contract.get("processing_fingerprint")
    require(isinstance(fingerprint, dict), "processing_fingerprint must be an object")
    require_exact_list(fingerprint.get("engine_classes"), ("gpu", "ane-split"), "engine classes")
    require_equal(fingerprint.get("bucket_included"), False, "fingerprint bucket exclusion")
    require_exact_list(
        fingerprint.get("canonical_payload_fields_in_order"),
        (
            "decode_fingerprint",
            "prefill_engine_class",
            "tokenizer_sanitized_digest",
            "prompt_template_revision",
            "special_token_policy_revision",
            "stop_token_policy_revision",
            "detokenizer_revision",
        ),
        "processing fingerprint payload",
    )

    persistence = contract.get("persistence")
    require(isinstance(persistence, dict), "persistence must be an object")
    for name, expected_table in (
        ("certification_table", "split_prefill_certification_rows"),
        ("health_table", "split_prefill_arm_health"),
    ):
        table = persistence.get(name)
        require(isinstance(table, dict), f"{name} must be an object")
        require_equal(table.get("name"), expected_table, f"{name} name")
        require_equal(table.get("primary_key"), ["machine_profile", "family", "bucket", "decode_config"], f"{name} primary key")

    branches = contract.get("recorded_owner_branches")
    require(isinstance(branches, list), "recorded_owner_branches must be a list")
    branch_map = {entry.get("id"): entry.get("selected") for entry in branches if isinstance(entry, dict)}
    require_equal(branch_map, EXPECTED_BRANCHES, "recorded owner branches")


def validate_manifest(contract: dict[str, Any], manifest: dict[str, Any], contract_bytes: bytes) -> None:
    require_equal(manifest.get("manifest_revision"), "ane-prefill-split-manifest-v1", "manifest revision")
    reference = manifest.get("contract")
    require(isinstance(reference, dict), "manifest contract reference must be an object")
    require_equal(reference.get("path"), str(CONTRACT_PATH), "manifest contract path")
    require_equal(reference.get("sha256"), hashlib.sha256(contract_bytes).hexdigest(), "manifest contract digest")
    require_equal(manifest.get("fixed_artifact_paths"), NAMED_PATHS, "manifest fixed artifact paths")
    require_equal(manifest.get("family"), "qwen3-0.6b", "manifest family")
    require_equal(manifest.get("included_buckets"), [128, 256, 512], "manifest included buckets")
    require_equal(manifest.get("decode_configs"), ["f16-step", "q8-step"], "manifest decode configs")

    packages = manifest.get("packages")
    arms = manifest.get("arm_matrix")
    require(isinstance(packages, list) and len(packages) == 3, "manifest must include three packages")
    require(isinstance(arms, list) and len(arms) == 6, "manifest must include six arms")
    package_by_id = {package.get("package_id"): package for package in packages if isinstance(package, dict)}
    require_equal(len(package_by_id), 3, "manifest package identities")
    expected_arms = {
        (128, "f16-step"),
        (128, "q8-step"),
        (256, "f16-step"),
        (256, "q8-step"),
        (512, "f16-step"),
        (512, "q8-step"),
    }
    actual_arms: set[tuple[int, str]] = set()
    arm_ids: set[str] = set()
    for arm in arms:
        require(isinstance(arm, dict), "every arm matrix row must be an object")
        key = arm.get("arm_key_fields")
        require(isinstance(key, dict), "every arm matrix row needs arm_key_fields")
        require_equal(key.get("family"), "qwen3-0.6b", "arm family")
        pair = (key.get("bucket"), key.get("decode_config"))
        require(pair in expected_arms, f"unexpected arm {pair!r}")
        require(pair not in actual_arms, f"duplicate arm {pair!r}")
        actual_arms.add(pair)
        arm_id = arm.get("arm_id")
        require(isinstance(arm_id, str) and arm_id, "every arm must have an arm_id")
        arm_ids.add(arm_id)
        package = package_by_id.get(arm.get("package_id"))
        require(package is not None, f"arm {arm_id} names an unknown package")
        require_equal(package.get("bucket"), pair[0], f"arm {arm_id} package bucket")
        require(arm.get("certification_attempt_required") is True, f"arm {arm_id} may be omitted from certification")
        require_equal(arm.get("activation_requirement"), "current exact arm is certified", f"arm {arm_id} activation requirement")
    require_equal(actual_arms, expected_arms, "manifest arm matrix coverage")

    for bucket in (128, 256, 512):
        package = next((item for item in packages if item.get("bucket") == bucket), None)
        require(isinstance(package, dict), f"missing W{bucket} package")
        require(package.get("buildable") is True, f"W{bucket} is not buildable")
        require(package.get("routing_branch_required") is True, f"W{bucket} has no routing branch")
        require(package.get("certification_attempt_required") is True, f"W{bucket} has no certification attempt")
        expected_package_arms = {
            arm_id
            for arm_id, arm in ((item.get("arm_id"), item) for item in arms)
            if arm["arm_key_fields"]["bucket"] == bucket
        }
        require_equal(set(package.get("serves_arms", [])), expected_package_arms, f"W{bucket} package-to-arm mapping")

    completion = manifest.get("completion_rule")
    require(isinstance(completion, dict), "completion_rule must be an object")
    require("both decode-config arms are certified" in str(completion.get("w128")), "W128 completion gate")
    require("recorded absent attempt" in str(completion.get("w256_w512")), "W256/W512 absence rule")


def validate_all(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = root / CONTRACT_PATH
    manifest_path = root / MANIFEST_PATH
    contract_bytes = contract_path.read_bytes()
    contract = read_json(contract_path)
    manifest = read_json(manifest_path)
    validate_contract(contract)
    validate_manifest(contract, manifest, contract_bytes)
    return contract, manifest


def require_rejected(action: Callable[[], None], label: str) -> None:
    try:
        action()
    except ContractError:
        return
    raise AssertionError(f"self-test mutation was accepted: {label}")


def self_test(root: Path) -> None:
    contract, manifest = validate_all(root)

    missing_handshake = copy.deepcopy(contract)
    del missing_handshake["worker_handshake"]
    require_rejected(lambda: validate_contract(missing_handshake), "missing worker handshake")

    overlapping = copy.deepcopy(contract)
    overlapping["reason_vocabularies"]["prefill_fallback_reason"][0] = "disabled"
    require_rejected(lambda: validate_contract(overlapping), "reason-vocabulary overlap")

    bypass_gap = copy.deepcopy(contract)
    bypass_gap["bypass_table"].pop()
    require_rejected(lambda: validate_contract(bypass_gap), "missing bypass table row")

    fallback_gap = copy.deepcopy(contract)
    fallback_gap["fallback_table"].pop()
    require_rejected(lambda: validate_contract(fallback_gap), "missing fallback table row")

    missing_arm = copy.deepcopy(manifest)
    missing_arm["arm_matrix"].pop()
    require_rejected(
        lambda: validate_manifest(contract, missing_arm, (root / CONTRACT_PATH).read_bytes()),
        "missing manifest arm",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    validate_all(root)
    if args.self_test:
        self_test(root)
    print("ANE-prefill split contract and manifest matrix validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
