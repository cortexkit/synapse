#!/usr/bin/env python3
"""Run isolated single-threaded CPU references, or a measured 20-case pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

from qwen3_asr_corpus import dump, expand
from qwen3_asr_reference import (Reference, artifact, canonical, checkpoint_weight_facts,
                                deterministic_runtime, fetch_model, pcm, sha256)

HERE = Path(__file__).resolve().parent


def pilot_cases(cases):
    chosen = []
    for split in ("librispeech-test-clean", "librispeech-test-other", "fleurs-de-test",
                  "fleurs-fr-test", "fleurs-es-test", "fleurs-it-test", "fleurs-pl-test",
                  "fleurs-nl-test", "fleurs-ru-test", "fleurs-el-test"):
        eligible = [c for c in cases if c["split"] == split and not c.get("terms")]
        chosen.append(min(eligible, key=lambda c: (c["sample_count"], c["id"]))["id"])
    chosen += [f"frontend-{kind}" for kind in ("short", "sub-100-remainder", "over-30s")]
    chosen += ["long-form-120", "long-form-300", "long-form-570-list"]
    chosen.append(next(c["id"] for c in cases if c.get("tier") == 300 and c["split"] == "earnings22-test-200" and not c.get("baseline")))
    chosen += ["near-silence-00", "noise-00", next(c["id"] for c in cases if c["split"] == "single-name")]
    if len(set(chosen)) != 20:
        raise ValueError("pilot must contain exactly 20 distinct cases")
    return chosen


def one(args, case):
    start = time.monotonic()
    deterministic_runtime()
    audio = pcm(case, args.cache)
    if case.get("refusal"):
        raw = {"id": case["id"], "refusal": case["refusal"], "sample_count": case["sample_count"]}
        load_seconds = 0
        inference_seconds = 0
    else:
        load_start = time.monotonic()
        reference = Reference(args.cache / "model" / "qwen3-asr-1.7b")
        load_seconds = time.monotonic() - load_start
        inference_start = time.monotonic()
        raw = reference.run(case, audio, args.output, args.tokens_per_second)
        inference_seconds = time.monotonic() - inference_start
        if args.weight_facts:
            raw["checkpoint_weight_facts"] = checkpoint_weight_facts(
                args.cache / "model" / "qwen3-asr-1.7b", reference.model)
    raw["model_input_audio_sha256"] = hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest()
    raw["corpus_sha256"] = sha256(args.corpus)
    raw["reference_rules_sha256"] = sha256(HERE / "qwen3-asr-reference.json")
    trace = artifact(args.output, canonical(raw), "traces/" + case["id"] + ".json")
    metric = {"id": case["id"], "wall_seconds": time.monotonic() - start,
              "model_load_seconds": load_seconds,
              "inference_seconds": inference_seconds,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "committed_bytes": trace["bytes"],
              "external_tensor_bytes": sum(raw[key]["bytes"] for key in ("mel", "encoder") if key in raw),
              "trace": trace}
    dump(args.output / "metrics" / (case["id"] + ".json"), metric)
    print(json.dumps(metric), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--case", action="append")
    parser.add_argument("--one")
    parser.add_argument("--weight-facts", action="store_true")
    parser.add_argument("--tokens-per-second", type=int, default=6)
    args = parser.parse_args()
    corpus = expand(json.loads(args.corpus.read_text()))
    cases = corpus["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("duplicate corpus ids")
    if args.one:
        one(args, next(c for c in cases if c["id"] == args.one))
        return
    if args.partitions < 1 or not 0 <= args.partition < args.partitions:
        parser.error("partition outside [0, partitions)")
    if args.tokens_per_second < 6:
        parser.error("tokens-per-second cannot be below six")
    fetch_model(args.cache)
    selected = pilot_cases(cases) if args.pilot else args.case
    if selected:
        unknown = set(selected) - {c["id"] for c in cases}
        if unknown:
            parser.error(f"unknown fixture ids {unknown}")
    selected = [c for i, c in enumerate(cases) if i % args.partitions == args.partition
                and (not selected or c["id"] in selected)]
    metrics = []
    repeats = {}
    for index, case in enumerate(selected):
        command = [sys.executable, __file__, "--cache", str(args.cache), "--corpus", str(args.corpus),
                   "--output", str(args.output), "--one", case["id"],
                   "--tokens-per-second", str(args.tokens_per_second)]
        if args.weight_facts or (args.pilot and index == 0):
            command.append("--weight-facts")
        subprocess.run(command, check=True)
        metric = json.loads((args.output / "metrics" / (case["id"] + ".json")).read_text())
        metrics.append(metric)
        if args.pilot and index < 2:
            before = metric["trace"]["sha256"]
            subprocess.run(command, check=True)
            after = json.loads((args.output / "metrics" / (case["id"] + ".json")).read_text())
            if before != after["trace"]["sha256"]:
                raise ValueError("non-deterministic reference: " + case["id"])
            repeats[case["id"]] = {"byte_identical": True, "sha256": before,
                                    "second_run": after}
    dump(args.output / ("pilot.json" if args.pilot else f"partition-{args.partition}.json"),
         {"metrics": metrics, "repeat_checks": repeats, "corpus_sha256": sha256(args.corpus)})


if __name__ == "__main__":
    main()
