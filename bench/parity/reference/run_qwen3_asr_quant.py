#!/usr/bin/env python3
"""Run a public fail-fast Q8 pilot; a pilot never admits engine work."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import resource
import subprocess
import tarfile
import time

from generate_qwen3_asr import pilot_cases
from qwen3_asr_corpus import dump, expand
from qwen3_asr_quant import PROFILE, QuantReference
from qwen3_asr_quant_gate import calibration, cap_failure, probe_failure
from qwen3_asr_reference import VERSIONS, canonical, fetch_model, model_pin, pcm, sha256

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "qwen3-asr-corpus.json"
FP32 = HERE.parent / "fixtures/qwen3-asr-1.7b"


def archive_output(output):
    """Publish a complete small evidence archive after each completed stage."""
    destination = output.with_suffix(".tar.gz")
    temporary = destination.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    os.replace(temporary, destination)
    return destination


def source_key(case):
    # The clean-corpus curation supplies single-name probes; concatenation parts
    # retain their clean split. Earnings rows are all from the one pinned parquet.
    if case.get("source"):
        return case["source"]
    return {"single-name": "librispeech-test-clean",
            "earnings22-test-200": "earnings22"}.get(case["split"], case["split"])


def prepare_audio(case, sources, cache):
    """Materialize pinned public members lazily; generated probes need no fetch."""
    from prepare_parakeet import fetch
    if case.get("recipe") == "concatenate":
        for part in case["parts"]:
            prepare_audio(part, sources, cache)
        return
    if "audio_sha256" not in case:
        return
    destination = cache / "audio" / case["audio_sha256"]
    if destination.exists():
        if sha256(destination) != case["audio_sha256"]:
            raise ValueError("corrupt public audio member")
        return
    source = sources[source_key(case)]
    downloaded = fetch(source["url"], source["sha256"], cache / "sources")

    def store(raw):
        path = cache / "audio" / hashlib.sha256(raw).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    if source["url"].endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(downloaded).iter_batches(batch_size=16, columns=["audio"]):
            for row in batch.to_pylist():
                store(row["audio"]["bytes"])
    else:
        with tarfile.open(downloaded, "r:gz") as archive:
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("public archive member has no payload")
                    with stream:
                        store(stream.read())
    if not destination.is_file():
        raise ValueError("pinned source lacks requested audio member")


def pilot_order(cases):
    ids = pilot_cases(cases)
    # Cheap, zero-tolerance probes go first; WER and coverage fractions cannot
    # reject a partial split, so those gates wait for complete fixture sets.
    probes = [c["id"] for c in cases if c["split"] in ("near-silence", "noise", "single-name")]
    return probes + [id for id in ids if id not in probes]


def run_pilot(cache, output, derive_binary, start_at=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty: never mix independent runs")
    output.mkdir(parents=True, exist_ok=True)
    corpus = expand(json.loads(CORPUS.read_text()))
    cases = corpus["cases"]
    by_id = {c["id"]: c for c in cases}
    full_order = pilot_order(cases)
    order = full_order[full_order.index(start_at):] if start_at is not None else full_order
    report = {"schema": "qwen3-asr-quant-pilot-v1", "status": "running",
              "engine_slices_allowed": False, "planned_cases": order, "metrics": [],
              "omitted_prefix_cases": full_order[:len(full_order) - len(order)],
              "separate_pilot_phase": start_at is not None,
              "corpus_sha256": sha256(CORPUS), "profile_sha256": sha256(PROFILE),
              "model": model_pin(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in VERSIONS},
              "source_sha256": {name: sha256(HERE / name) for name in
                  ("qwen3_asr_quant.py", "qwen3_asr_quant_gate.py", "run_qwen3_asr_quant.py",
                   "qwen3_asr_reference.py", "qwen3_asr_corpus.py")}}

    def finish(status, failure=None):
        report.update(status=status, failure=failure,
                      completed_cases=len(report["metrics"]),
                      process_wall_seconds=time.monotonic() - started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        dump(output / "report.json", report)
        archive_output(output)
        print(json.dumps({key: report[key] for key in ("status", "failure", "completed_cases",
                         "process_wall_seconds", "peak_rss_bytes", "engine_slices_allowed")}), flush=True)
        return report

    started = time.monotonic()
    try:
        model_directory = fetch_model(cache)
        derived = cache / "quant-derived"
        if derived.exists() and any(derived.iterdir()):
            raise ValueError("Rust derivation requires an empty output cache")
        result = subprocess.run([str(derive_binary), "qwen3-asr-1.7b", str(model_directory), str(derived)],
                                text=True, capture_output=True, check=True)
        report["derivation"] = json.loads(result.stdout)
        artifacts = list(derived.glob("owned-decode-q8/*/weights.q8_0"))
        if len(artifacts) != 1:
            raise ValueError("expected one Rust-produced Q8 object")
        load_start = time.monotonic()
        reference = QuantReference(model_directory, artifacts[0].parent)
        report["model_load_seconds"] = time.monotonic() - load_start
        teachers, profiles, partial = {}, {}, None
        for id in order:
            case = by_id[id]
            prepare_audio(case, corpus["sources"], cache)
            audio = pcm(case, cache)
            begin = time.monotonic()
            teacher = reference.run(case, audio)
            teacher["audio_sha256"] = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
            baseline_path = FP32 / "traces" / (id + ".json")
            baseline = json.loads(baseline_path.read_text())
            if (teacher["prompt_ids"] != baseline["prompt_ids"] or teacher["span"] != baseline["span"]
                    or teacher["audio_sha256"] != baseline["model_input_audio_sha256"]):
                raise ValueError("Q8 inputs differ from sealed fp32 baseline: " + id)
            teacher["fp32_trace_sha256"] = sha256(baseline_path)
            dump(output / "teacher" / (id + ".json"), teacher)
            metric = {"id": id, "teacher_seconds": time.monotonic() - begin,
                      "teacher_tokens": len(teacher["generated_ids"]),
                      "teacher_sha256": hashlib.sha256(canonical(teacher)).hexdigest()}
            report["metrics"].append(metric)
            teachers[id] = teacher
            dump(output / "report.json", report)
            archive_output(output)
            failure = probe_failure(case, teacher)
            if failure:
                return finish("gate-failed", failure)
            failure = cap_failure(teacher, baseline)
            if failure:
                return finish("gate-failed", failure)
            begin = time.monotonic()
            profile = reference.run(case, audio, forced_ids=teacher["generated_ids"][:-1], profile=True)
            dump(output / "forced-profile" / (id + ".json"), profile)
            archive_output(output)
            profiles[id] = profile
            metric["profile_seconds"] = time.monotonic() - begin
            partial = calibration([by_id[id] for id in teachers], teachers, profiles)
            report["partial_calibration"] = partial
            tau_failures = [f for f in partial["failures"] if f["gate"] == "tau-maximum"]
            if tau_failures:
                return finish("gate-failed", {**tau_failures[0], "id": id})
            complete_sets = {split for split in ("near-silence", "noise", "single-name")
                             if sum(c["split"] == split for c in cases) ==
                             sum(by_id[key]["split"] == split for key in teachers)}
            coverage_failures = [f for f in partial["failures"] if f.get("split") in complete_sets]
            if coverage_failures:
                return finish("gate-failed", coverage_failures[0])
            dump(output / "report.json", report)
            archive_output(output)
            print(json.dumps(metric), flush=True)
        # Only the probe sets have complete coverage in this pilot. A partial
        # speech split's graded fraction cannot decide the corpus-wide gate.
        if partial is None:
            raise ValueError("empty pilot selection")
        coverage_failures = [f for f in partial["failures"] if f.get("split") in
                             ("near-silence", "noise", "single-name")]
        if coverage_failures:
            return finish("gate-failed", coverage_failures[0])
        report["cost_projection"] = {
            "measured_pilot_process_hours": sum(m["teacher_seconds"] + m["profile_seconds"] for m in report["metrics"]) / 3600,
            "full_public_cases": sum(not c.get("refusal") for c in cases),
            "warning": "Representative pilot is not a reliable linear extrapolation; the previous full fp32 run took 399 process-hours versus about 95 projected. Rent approval required."}
        measured = report["cost_projection"]["measured_pilot_process_hours"]
        naive = measured / len(report["metrics"]) * report["cost_projection"]["full_public_cases"]
        report["cost_projection"].update(naive_process_hours=naive,
              naive_wall_hours_at_four_processes=naive / 4,
              previous_underestimate_factor=399 / 95,
              adjusted_process_hours=naive * 399 / 95,
              adjusted_wall_hours_at_four_processes=naive * 399 / 95 / 4,
              note="Both estimates are projections, not upper bounds. Four single-threaded processes require at least four cores and four model working sets.")
        return finish("partial-pilot-passed-full-run-not-authorized" if start_at is not None else
                      "pilot-passed-full-run-not-authorized")
    except Exception as error:
        finish("execution-error", {"type": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derive-binary", type=Path, required=True)
    parser.add_argument("--start-at", help="Separate pilot phase starting at this planned id; never merges prior traces")
    args = parser.parse_args()
    report = run_pilot(args.cache.resolve(), args.output.resolve(), args.derive_binary.resolve(), args.start_at)
    if report["status"] == "gate-failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
