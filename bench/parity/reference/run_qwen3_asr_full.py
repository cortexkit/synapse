#!/usr/bin/env python3
"""Single-venue full reference: one model load per ordered, single-threaded shard."""

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import uuid
from typing import Any

from generate_qwen3_asr import pilot_cases
from qwen3_asr_corpus import (PROVISIONAL_SLOPE, SLUG, audio_tokens, calibration_trace, capped_fixture_report,
                              dump, expand, token_rate_report)
from qwen3_asr_reference import (Reference, artifact, canonical, checkpoint_weight_facts,
                                fetch_model, generation_stop_reason, model_pin, pcm, sha256, VERSIONS)

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "qwen3-asr-corpus.json"
RULES = HERE / "qwen3-asr-reference.json"
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                   "BLIS_NUM_THREADS")}
THREAD_ENV.update(TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED="0")
SOURCE_FILES = ("full_run.sh", "run_qwen3_asr_full.py", "qwen3_asr_reference.py",
                "qwen3_asr_corpus.py", "prepare_qwen3_asr.py", "generate_qwen3_asr.py",
                "prepare_parakeet.py", "generate_parakeet.py", "requirements-qwen3-asr.txt",
                "qwen3-asr-name-curation.json", "parakeet-corpus.json")


def cpu_facts():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("the full reference must run on one x86 Linux venue")
    blocks = Path("/proc/cpuinfo").read_text().strip().split("\n\n")
    processors = [dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                  for block in blocks]
    processors = [{key.strip(): value.strip() for key, value in p.items()} for p in processors]
    models = sorted({p["model name"] for p in processors})
    flags = sorted({tuple(sorted(p["flags"].split())) for p in processors})
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {"cpu_models": models, "cpu_flag_sets": [list(f) for f in flags],
            "logical_cpus": len(processors), "platform": platform.platform(),
            "physical_memory_bytes": int(memory["MemTotal"].split()[0]) * 1024}


def ordered_shard(cases, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard index outside [0, count)")
    return [case for i, case in enumerate(cases) if i % count == index]


def determinism_plan(cases):
    selected = pilot_cases(cases)
    targets = [selected[0], selected[9], next(id for id in selected if "tier-300" in id)]
    return {"targets": targets, "check-a": ["near-silence-00", *targets],
            "check-b": ["noise-00", *reversed(targets)]}


def check_termination(row):
    reason = generation_stop_reason(row["generated_ids"], row["cap"], (151643, 151645))
    if row["stop_reason"] != reason or row["cap"] != row["max_new_tokens"]:
        raise ValueError("trace stop reason or cap differs: " + row["id"])
    if row["generation_max_length"] != len(row["prompt_ids"]) + row["max_new_tokens"]:
        raise ValueError("trace generation length budget differs: " + row["id"])
    if len(row["generated_ids"]) != len(row["top_two_margins"]) or any(
            not math.isfinite(m) or m < 0 for m in row["top_two_margins"]):
        raise ValueError("invalid top-two margins: " + row["id"])


def check_case(row, case):
    if any(row.get(key) != case[key] for key in ("id", "split", "sample_count", "ground_truth")):
        raise ValueError("trace case identity differs: " + case["id"])
    if row.get("tier") != case.get("tier"):
        raise ValueError("trace case tier differs: " + case["id"])
    if case.get("refusal"):
        if row.get("refusal") != case["refusal"] or any(key in row for key in ("generated_ids", "mel", "encoder")):
            raise ValueError("over-limit fixture was not refused")
        return
    if "refusal" in row:
        raise ValueError("unexpected reference refusal")
    frames = case["sample_count"] // 160
    count = audio_tokens(frames)
    span = row["span"]
    positions = [i for i, token in enumerate(row["prompt_ids"]) if token == 151676]
    if (row["valid_mel_frames"] != frames or row["mel_frames"] != frames or
            row["encoder_width"] != 2048 or span["len"] != count or
            positions != list(range(span["start"], span["start"] + count)) or
            row["mel"]["bytes"] != frames * 128 * 4 or row["encoder"]["bytes"] != count * 2048 * 4):
        raise ValueError("reference tensor/prompt shape differs: " + case["id"])


def run_sequence(output, cases, label, provenance,
                 reference_factory: Callable[[Path], Any] = Reference, audio_loader=pcm):
    """Reuse one model; deterministic trace bytes never include order or timings."""
    start = time.monotonic()
    cache = output / "cache"
    reference = reference_factory(cache / "model" / SLUG)
    load_seconds = time.monotonic() - start
    receipts, metrics = {}, []
    prefix = "traces" if label.startswith("shard-") else f"checks/{label}/traces"
    for case in cases:
        begin = time.monotonic()
        audio = audio_loader(case, cache)
        if case.get("refusal"):
            row = {"id": case["id"], "split": case["split"], "ground_truth": case["ground_truth"],
                   "refusal": case["refusal"], "sample_count": case["sample_count"]}
        else:
            row = reference.run(case, audio, output, PROVISIONAL_SLOPE)
            check_termination(row)
            if row["tokens_per_second"] != PROVISIONAL_SLOPE:
                raise ValueError("reference generation slope changed")
        check_case(row, case)
        row.update(model_input_audio_sha256=hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest(),
                   corpus_sha256=provenance["corpus_sha256"],
                   reference_rules_sha256=provenance["reference_rules_sha256"],
                   run_provenance_sha256=hashlib.sha256(canonical(provenance)).hexdigest())
        receipt = artifact(output, canonical(row), f"{prefix}/{case['id']}.json")
        receipts[case["id"]] = receipt
        metric = {"id": case["id"], "wall_seconds": time.monotonic() - begin,
                  "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "trace_bytes": receipt["bytes"]}
        metrics.append(metric)
        print(json.dumps({"worker": label, **metric}), flush=True)
    report = {"cases": receipts, "generation_policy": reference.generation_policy,
              "model_load_seconds": load_seconds, "wall_seconds": time.monotonic() - start,
              "metrics": metrics}
    dump(output / "shards" / (label + ".json"), report)
    return report


def compare_determinism(plan, first, second):
    checks = {}
    for id in plan["targets"]:
        a, b = first["cases"][id], second["cases"][id]
        if a["sha256"] != b["sha256"] or a["bytes"] != b["bytes"]:
            raise ValueError("model-reuse determinism check failed: " + id)
        predecessors = {}
        for label in ("check-a", "check-b"):
            sequence = plan[label]
            predecessors[label] = sequence[sequence.index(id) - 1]
        if predecessors["check-a"] == predecessors["check-b"]:
            raise ValueError("determinism targets need different preceding fixtures")
        checks[id] = {"sha256": a["sha256"], "bytes": a["bytes"],
                      "byte_identical": True, "preceding_fixtures": predecessors}
    return checks


def validate_policy(policy):
    pinned = json.loads((HERE.parent / "checkpoints" / SLUG / "generation_config.json").read_text())
    if (policy["pinned_generation_config"] != pinned or
            policy["pinned_generation_config_sha256"] != model_pin()["files"]["generation_config.json"] or
            policy["actual_logits_processors"] != [] or policy["do_sample"] is not False or
            policy["generation_mode"] != "greedy_search"):
        raise ValueError("generation processor evidence changed")
    effective = policy["effective_generation_config_without_length_limits"]
    if effective["do_sample"] or effective["num_beams"] != 1 or effective["num_return_sequences"] != 1:
        raise ValueError("reference generation is not plain greedy decoding")


def child(output, label):
    provenance = json.loads((output / "provenance.json").read_text())
    if cpu_facts() != provenance["machine"]:
        raise ValueError("worker machine differs from the recorded venue")
    cases = expand(json.loads(CORPUS.read_text()))["cases"]
    if label.startswith("shard-"):
        cases = ordered_shard(cases, int(label.removeprefix("shard-")), provenance["workers"])
    else:
        by_id = {case["id"]: case for case in cases}
        cases = [by_id[id] for id in determinism_plan(cases)[label]]
    def checked_factory(directory):
        reference = Reference(directory)
        if label == "check-a":
            facts = checkpoint_weight_facts(directory, reference.model)
            if facts != json.loads(RULES.read_text())["checkpoint_weight_facts"]:
                raise ValueError("loaded head/embedding differ from the pinned reference facts")
            dump(output / "loaded-weight-facts.json", facts)
        return reference
    report = run_sequence(output, cases, label, provenance, checked_factory)
    validate_policy(report["generation_policy"])


def run_children(output, labels):
    processes = []
    try:
        for label in labels:
            log = output / "logs" / (label + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("wb") as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                            "--output", str(output), "--worker", label],
                                           stdout=stream, stderr=subprocess.STDOUT)
            processes.append((label, process))
        remaining = list(processes)
        while remaining:
            for label, process in remaining[:]:
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{label} failed (exit {code}); see {output / 'logs' / (label + '.log')}")
                    remaining.remove((label, process))
            if remaining:
                time.sleep(0.25)
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def checksums(output):
    retained = sorted(path for path in output.rglob("*") if path.is_file()
                      and path.relative_to(output).parts[0] != "cache"
                      and path.name != "SHA256SUMS")
    if any(".partial-" in path.name or path.is_symlink() for path in retained):
        raise ValueError("incomplete or linked output artifact")
    lines = [sha256(path) + "  " + path.relative_to(output).as_posix() for path in retained]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return len(lines)


def finish(output, cases, provenance, determinism):
    expected = {case["id"] for case in cases}
    found = {path.stem for path in (output / "traces").glob("*.json")}
    if expected != found:
        raise ValueError("full-run trace coverage differs from the corpus")
    policies = []
    for index in range(provenance["workers"]):
        report = json.loads((output / "shards" / f"shard-{index}.json").read_text())
        ids = [case["id"] for case in ordered_shard(cases, index, provenance["workers"])]
        if set(report["cases"]) != set(ids) or [m["id"] for m in report["metrics"]] != ids:
            raise ValueError("shard fixtures were not run in sealed order")
        policies.append(report["generation_policy"])
    policies += [json.loads((output / "shards" / (label + ".json")).read_text())["generation_policy"]
                 for label in ("check-a", "check-b")]
    if not policies or any(policy != policies[0] for policy in policies):
        raise ValueError("generation processor evidence differs between processes")
    policy = policies[0]
    validate_policy(policy)
    dump(output / "generation-policy.json", policy)
    policy_digest = hashlib.sha256(canonical(policy)).hexdigest()
    lineage = hashlib.sha256(canonical(provenance)).hexdigest()
    rows = []
    for case in cases:
        row = json.loads((output / "traces" / (case["id"] + ".json")).read_text())
        check_case(row, case)
        if case["id"] in determinism and sha256(output / "traces" / (case["id"] + ".json")) != determinism[case["id"]]["sha256"]:
            raise ValueError("full-shard trace differs from model-reuse checks: " + case["id"])
        if (row["run_provenance_sha256"] != lineage or row["corpus_sha256"] != provenance["corpus_sha256"]
                or row["reference_rules_sha256"] != provenance["reference_rules_sha256"]):
            raise ValueError("trace did not come from this venue/run")
        if not case.get("refusal"):
            check_termination(row)
            if row["generation_policy_sha256"] != policy_digest or row["tokens_per_second"] != PROVISIONAL_SLOPE:
                raise ValueError("trace generation policy differs")
            if row["language_prefix_tokens"] != row["generated_ids"].index(151704) + 1:
                raise ValueError("language-prefix evidence differs")
            rows.append(calibration_trace(row, case))
        elif row.get("refusal") != case["refusal"]:
            raise ValueError("over-limit fixture was not refused")
    dump(output / "capped-fixtures.json", capped_fixture_report(rows))
    rate = token_rate_report(rows)
    dump(output / "calibration.json", rate)
    dump(output / "checks" / "determinism.json", determinism)
    dump(output / "run-complete.json", {"schema": "synapse-qwen3-asr-full-run-v1", "cases": len(cases),
                                       "run_provenance_sha256": lineage,
                                       "generation_policy_sha256": policy_digest,
                                       "provisional_tokens_per_second": PROVISIONAL_SLOPE,
                                       "sealed_tokens_per_second": rate["tokens_per_second"]})
    count = checksums(output)
    print(json.dumps({"status": "complete", "checksummed_files": count, "calibration": rate}), flush=True)


def full_run(output, workers):
    if workers < 1 or workers > (os.cpu_count() or 1):
        raise ValueError("workers must be between one and the venue's logical CPU count")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be new or empty; partial runs cannot be mixed or resumed")
    output.mkdir(parents=True, exist_ok=True)
    cases = expand(json.loads(CORPUS.read_text()))["cases"]
    if workers > len(cases) or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("empty shards or duplicate corpus ids")
    provenance = {"schema": "synapse-qwen3-asr-venue-v1", "run_id": str(uuid.uuid4()),
                  "started_at": datetime.now(timezone.utc).isoformat(), "machine": cpu_facts(),
                  "python": sys.version, "workers": workers, "thread_environment": THREAD_ENV,
                  "packages": {key: importlib.metadata.version(key) for key in VERSIONS},
                  "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
                  "model": model_pin(), "corpus_sha256": sha256(CORPUS),
                  "reference_rules_sha256": sha256(RULES),
                  "source_sha256": {name: sha256(HERE / name) for name in SOURCE_FILES}}
    dump(output / "provenance.json", provenance)
    try:
        cache = output / "cache"
        cache.mkdir()
        prepared = cache / "prepared-corpus.json"
        subprocess.run([sys.executable, str(HERE / "prepare_qwen3_asr.py"), "--cache", str(cache),
                        "--output", str(prepared), "--annotations", str(HERE / "qwen3-asr-name-curation.json")], check=True)
        if sha256(prepared) != sha256(CORPUS):
            raise ValueError("public preparation did not reproduce the sealed corpus")
        fetch_model(cache)
        plan = determinism_plan(cases)
        # These are distinct processes, each with one model load and different
        # warm-up/preceding fixtures. They must pass before any full shard starts.
        for label in ("check-a", "check-b"):
            run_children(output, [label])
        reports = [json.loads((output / "shards" / (label + ".json")).read_text())
                   for label in ("check-a", "check-b")]
        determinism = compare_determinism(plan, *reports)
        dump(output / "checks" / "determinism.json", determinism)
        run_children(output, [f"shard-{index}" for index in range(workers)])
        finish(output, cases, provenance, determinism)
    except BaseException as error:
        dump(output / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--worker")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda _signal, _frame: sys.exit(128 + signal.SIGTERM))
    os.environ.update(THREAD_ENV)
    output = args.output.resolve()
    if args.worker is not None:
        child(output, args.worker)
    elif args.workers is not None:
        full_run(output, args.workers)
    else:
        parser.error("--workers or --worker is required")


if __name__ == "__main__":
    main()
