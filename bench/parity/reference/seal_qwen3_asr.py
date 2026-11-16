#!/usr/bin/env python3
"""Seal a complete public reference, refusing missing tensors and uncalibrated caps."""

import argparse
import json
from pathlib import Path
import hashlib
import subprocess

from qwen3_asr_corpus import (PROVISIONAL_SLOPE, SLUG, calibration_trace, dump, expand,
                              max_new_tokens, token_rate_report)
from qwen3_asr_reference import artifact, canonical, model_pin, sha256
from run_qwen3_asr_full import (SOURCE_FILES, check_case, check_termination, compare_determinism,
                                determinism_plan, validate_policy)

HERE = Path(__file__).resolve().parent
GENERATION_PIN = HERE / "qwen3-asr-generation-76cc6a4b.json"
REGRADE_REASON = "name-list traces excluded from calibration; measured slope <=25, generation and serving slope retained at 25"


def verify_generation_revision(provenance):
    pin = json.loads(GENERATION_PIN.read_text())
    if any(provenance.get(key) != pin[key] for key in ("source_sha256", "corpus_sha256", "reference_rules_sha256")):
        raise ValueError("generation inputs do not match revision " + pin["revision"])
    return pin["revision"]


def grading_identity():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    files = (*SOURCE_FILES, "seal_qwen3_asr.py", "qwen3-asr-reference.json", GENERATION_PIN.name)
    hashes = {}
    for name in files:
        committed = subprocess.check_output(["git", "show", revision + ":bench/parity/reference/" + name], cwd=HERE)
        digest = hashlib.sha256(committed).hexdigest()
        if sha256(HERE / name) != digest:
            raise ValueError("commit grading source before offline regrade: " + name)
        hashes[name] = digest
    return {"revision": revision, "source_sha256": hashes}


def verify_process_reports(generated, corpus, provenance, policy, external_cache):
    by_id = {case["id"]: case for case in corpus["cases"]}
    plan = determinism_plan(corpus["cases"])
    workers = provenance["workers"]
    if not isinstance(workers, int) or not 1 <= workers <= len(by_id):
        raise ValueError("invalid shard count")
    labels = [f"shard-{index}" for index in range(workers)] + ["check-a", "check-b"]
    if {p.stem for p in (generated / "shards").glob("*.json")} != set(labels):
        raise ValueError("process report coverage differs")
    tensors = []
    lineage = hashlib.sha256(canonical(provenance)).hexdigest()
    policy_digest = hashlib.sha256(canonical(policy)).hexdigest()
    for index, label in enumerate(labels):
        ids = list(by_id)[index::workers] if index < workers else plan[label]
        report = json.loads((generated / "shards" / (label + ".json")).read_text())
        if (set(report["cases"]) != set(ids) or [m["id"] for m in report["metrics"]] != ids or
                report["generation_policy"] != policy):
            raise ValueError("process order, coverage or generation policy differs: " + label)
        for id in ids:
            path = verify(generated, report["cases"][id], external_cache)
            expected = f"traces/{id}.json" if index < workers else f"checks/{label}/traces/{id}.json"
            if report["cases"][id]["path"] != expected:
                raise ValueError("process trace path differs")
            row = json.loads(path.read_text())
            check_case(row, by_id[id])
            if (row["run_provenance_sha256"] != lineage or row["corpus_sha256"] != provenance["corpus_sha256"]
                    or row["reference_rules_sha256"] != provenance["reference_rules_sha256"]):
                raise ValueError("process trace lineage differs")
            if not by_id[id].get("refusal"):
                check_termination(row)
                if (row["generation_policy_sha256"] != policy_digest or row["tokens_per_second"] != PROVISIONAL_SLOPE
                        or row["cap"] != max_new_tokens(row["sample_count"], PROVISIONAL_SLOPE)
                        or row["generated_ids"].count(151704) != 1
                        or row["language_prefix_tokens"] != row["generated_ids"].index(151704) + 1):
                    raise ValueError("process trace generation evidence differs")
                calibration_trace(row, by_id[id])
                tensors.extend(row[key] for key in ("mel", "encoder"))
    return len(labels), tensors


def source_bundle_receipts(generated):
    receipts = {}
    for path in sorted(generated.rglob("*")):
        if path.is_symlink() or ".partial-" in path.name:
            raise ValueError("returned bundle contains a symbolic link or partial file")
        if path.is_file():
            relative = path.relative_to(generated).as_posix()
            receipts[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return receipts


def verify(root, seal, external_cache=None):
    relative = Path(seal.get("path", seal.get("cache_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe artifact path")
    base = (external_cache if external_cache is not None else root / "external") if "cache_path" in seal else root
    file = base / relative
    if not file.resolve().is_relative_to(base.resolve()):
        raise ValueError("unsafe artifact path")
    if file.stat().st_size != seal["bytes"] or sha256(file) != seal["sha256"]:
        raise ValueError("artifact digest mismatch: " + str(relative))
    return file


def verify_full_run(generated, external_cache):
    if (generated / "FAILED.json").exists():
        raise ValueError("cannot seal a failed run")
    seen = set()
    for line in (generated / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid full-run checksum")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in seen:
            raise ValueError("unsafe or duplicate full-run artifact path")
        seen.add(relative)
        if path.parts and path.parts[0] == "external":
            receipt = {"cache_path": str(Path(*path.parts[1:])), "sha256": digest}
            base = external_cache if external_cache is not None else generated / "external"
            receipt["bytes"] = (base / receipt["cache_path"]).stat().st_size
        else:
            receipt = {"path": relative, "sha256": digest, "bytes": 0}
            receipt["bytes"] = (generated / path).stat().st_size
        verify(generated, receipt, external_cache)
    return seen


def seal(corpus_path, generated, destination, external_cache=None, offline_regrade=False):
    corpus = expand(json.loads(corpus_path.read_text()))
    expected = {c["id"] for c in corpus["cases"]}
    found = {p.stem for p in (generated / "traces").glob("*.json")}
    if expected != found:
        raise ValueError(f"reference coverage differs: missing {sorted(expected-found)}, extra {sorted(found-expected)}")
    original_bundle, grading, failure = {}, {}, {}
    process_count = 0
    process_tensors = []
    provenance = json.loads((generated / "provenance.json").read_text())
    generation_revision = verify_generation_revision(provenance) if offline_regrade else None
    if offline_regrade:
        failure = json.loads((generated / "FAILED.json").read_text())
        if failure != {"error": "speech slope 34 exceeds 25 at earnings22-0024-tier-100; stop before sealing", "type": "ValueError"}:
            raise ValueError("offline regrade requires the named calibration failure")
        if any((generated / name).exists() for name in ("SHA256SUMS", "run-complete.json", "calibration.json")):
            raise ValueError("offline regrade must not replace venue success artifacts")
        original_bundle = source_bundle_receipts(generated)
        seen = set(original_bundle)
        grading = grading_identity()
    else:
        seen = verify_full_run(generated, external_cache)
    required = {"provenance.json", "generation-policy.json", "run-complete.json", "calibration.json", "capped-fixtures.json",
                "loaded-weight-facts.json", "checks/determinism.json", "shards/check-a.json", "shards/check-b.json"}
    required.update("traces/" + id + ".json" for id in expected)
    if offline_regrade:
        required -= {"run-complete.json", "calibration.json"}
    if not required <= seen:
        raise ValueError("full-run checksum manifest is incomplete")
    provenance = json.loads((generated / "provenance.json").read_text())
    policy = json.loads((generated / "generation-policy.json").read_text())
    complete = json.loads((generated / "run-complete.json").read_text()) if not offline_regrade else {}
    validate_policy(policy)
    lineage = hashlib.sha256(canonical(provenance)).hexdigest()
    policy_digest = hashlib.sha256(canonical(policy)).hexdigest()
    if provenance["model"] != model_pin() or provenance["corpus_sha256"] != sha256(corpus_path):
        raise ValueError("full-run provenance differs")
    if not offline_regrade and (provenance["reference_rules_sha256"] != sha256(HERE / "qwen3-asr-reference.json")
            or complete["run_provenance_sha256"] != lineage or complete["generation_policy_sha256"] != policy_digest
            or complete["cases"] != len(expected)):
        raise ValueError("full-run provenance differs")
    if set(provenance["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("full-run source manifest is incomplete")
    for name, digest in provenance["source_sha256"].items():
        if Path(name).name != name or (not offline_regrade and sha256(HERE / name) != digest):
            raise ValueError("full-run source changed: " + name)
    checks = [json.loads((generated / "shards" / (label + ".json")).read_text())
              for label in ("check-a", "check-b")]
    determinism = compare_determinism(determinism_plan(corpus["cases"]), *checks)
    if offline_regrade:
        process_count, process_tensors = verify_process_reports(generated, corpus, provenance, policy, external_cache)
    for report in checks:
        for id in determinism:
            verify(generated, report["cases"][id], external_cache)
    if determinism != json.loads((generated / "checks/determinism.json").read_text()):
        raise ValueError("determinism evidence differs")
    facts = json.loads((generated / "loaded-weight-facts.json").read_text())
    if facts != json.loads((HERE / "qwen3-asr-reference.json").read_text())["checkpoint_weight_facts"]:
        raise ValueError("loaded head/embedding facts differ")
    rows, inventory = [], {}
    all_rows = []
    tensors = {}
    for case in corpus["cases"]:
        path = generated / "traces" / (case["id"] + ".json")
        row = json.loads(path.read_text())
        check_case(row, case)
        if (row["corpus_sha256"] != provenance["corpus_sha256"]
                or row["reference_rules_sha256"] != provenance["reference_rules_sha256"]
                or row["run_provenance_sha256"] != lineage):
            raise ValueError("reference lineage mismatch")
        if case["id"] in determinism and sha256(path) != determinism[case["id"]]["sha256"]:
            raise ValueError("full-shard trace differs from model-reuse checks")
        if "refusal" not in row:
            check_termination(row)
            if (row["generation_policy_sha256"] != policy_digest or row["tokens_per_second"] != PROVISIONAL_SLOPE
                    or row["max_new_tokens"] != max_new_tokens(case["sample_count"], PROVISIONAL_SLOPE)
                    or row["generated_ids"].count(151704) != 1
                    or row["language_prefix_tokens"] != row["generated_ids"].index(151704) + 1):
                raise ValueError("reference generation evidence differs")
            for key in ("mel", "encoder"):
                if not offline_regrade and "external/" + row[key]["cache_path"] not in seen:
                    raise ValueError("tensor missing from full-run checksum manifest")
                tensor_key = (row[key]["cache_path"], row[key]["sha256"], row[key]["bytes"])
                if tensor_key not in tensors:
                    verify(generated, row[key], external_cache)
                    tensors[tensor_key] = row[key]
            if row["encoder"]["bytes"] != row["span"]["len"] * 2048 * 4:
                raise ValueError("reference encoder shape differs")
            rows.append(calibration_trace(row, case))
        elif row.get("refusal") != case.get("refusal"):
            raise ValueError("unexpected reference refusal")
        all_rows.append((case, row, sha256(path)))
    for tensor in process_tensors:
        tensor_key = (tensor["cache_path"], tensor["sha256"], tensor["bytes"])
        if tensor_key not in tensors:
            verify(generated, tensor, external_cache)
            tensors[tensor_key] = tensor
    rate = token_rate_report(rows)
    if rate["capped_fixtures"] != json.loads((generated / "capped-fixtures.json").read_text()):
        raise ValueError("full-run capped fixture counts differ")
    if not offline_regrade and (rate != json.loads((generated / "calibration.json").read_text()) or rate["tokens_per_second"] != complete["sealed_tokens_per_second"]):
        raise ValueError("full-run calibration differs")
    destination.mkdir(parents=True, exist_ok=True)
    regrade = None
    if offline_regrade:
        if source_bundle_receipts(generated) != original_bundle:
            raise ValueError("returned bundle changed during offline validation")
        receipt = {"schema": "synapse-qwen3-asr-offline-regrade-v1", "reason": REGRADE_REASON,
                   "generation_revision": generation_revision, "grading": grading,
                   "original_failure": failure, "source_bundle": original_bundle,
                   "external_tensors": list(tensors.values()), "calibration": rate,
                   "validated_cases": len(all_rows), "validated_processes": process_count,
                   "determinism": determinism, "generation_policy_sha256": policy_digest}
        regrade = artifact(destination, canonical(receipt), "offline-regrade.json")
    for case, row, source_digest in all_rows:
        if not case.get("refusal") and row["stop_reason"] == "stop":
            row["sealed_max_new_tokens"] = max_new_tokens(case["sample_count"], rate["tokens_per_second"])
            row["sealed_tokens_per_second"] = rate["tokens_per_second"]
        receipt = artifact(destination, canonical(row), "traces/" + case["id"] + ".json")
        inventory[case["id"]] = {**receipt, "source_trace_sha256": source_digest}
    dump(destination / "reference.json", {"schema": "synapse-qwen3-asr-fixtures-v1", "model": SLUG,
                                           "corpus_sha256": sha256(corpus_path),
                                          "provenance": provenance, "generation_policy": policy,
                                          "source_run_checksums_sha256": None if offline_regrade else sha256(generated / "SHA256SUMS"),
                                          "offline_regrade": regrade,
                                          "checkpoint_weight_facts": facts, "determinism": determinism,
                                          "capped_parity": "compare the exact recorded generated_ids prefix at its fixed reference cap, independently of any later serving cap",
                                           "token_rate": rate, "cases": inventory})
    return rate


def register_reference(destination, parity_dir):
    index_path = parity_dir / "fixtures" / "index.json"
    index = json.loads(index_path.read_text())
    reference = destination / "reference.json"
    index[SLUG + ".ref-v1.transformers-4.57.6.seed-0"] = {
        "model": SLUG, "path": str(reference.relative_to(parity_dir)),
        "sha256": sha256(reference), "seed": 0, "transformers_version": "4.57.6"}
    dump(index_path, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=HERE.parent / "fixtures" / SLUG)
    parser.add_argument("--external-cache", type=Path)
    parser.add_argument("--offline-regrade", action="store_true",
                        help="regrade the named 76cc6a4b calibration failure without generation or changing its bundle")
    args = parser.parse_args()
    rate = seal(args.corpus, args.generated, args.destination, args.external_cache, args.offline_regrade)
    if args.destination == HERE.parent / "fixtures" / SLUG:
        register_reference(args.destination, HERE.parent)
    print(json.dumps(rate, indent=2))


if __name__ == "__main__":
    main()
