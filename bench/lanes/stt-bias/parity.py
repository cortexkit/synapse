#!/usr/bin/env python3
"""Prepare verified full-corpus inputs and grade through the actual parity evaluator."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from prepare import sha256
from safety import ROOT, TARGET, private_path
from score import load_jsonl


LANGUAGES = {"librispeech-test-clean": "English", "librispeech-test-other": "English",
             **{f"fleurs-{code}-test": name for code, name in
                [("de", "German"), ("el", "Greek"), ("es", "Spanish"), ("fr", "French"),
                 ("it", "Italian"), ("nl", "Dutch"), ("pl", "Polish"), ("ru", "Russian")]}}


def prepare_full(wavs, output):
    import soundfile as sf
    if sys.platform != "darwin":
        raise RuntimeError("corpus preparation must stay on the Mac")
    corpus = ROOT / "bench/parity/reference/parakeet-corpus.json"
    cases = json.loads(corpus.read_text())["cases"]
    index = json.loads((wavs / "inputs.json").read_text())
    if len(cases) != 1200 or len({row["id"] for row in cases}) != 1200:
        raise ValueError("expected the sealed 1200-utterance corpus")
    rows = []
    for case in cases:
        entry = index[case["id"]]
        path = (wavs / entry["path"]).resolve()
        if not path.is_relative_to(wavs.resolve()) or sha256(path) != entry["sha256"]:
            raise ValueError("WAV path/hash mismatch")
        info = sf.info(path)
        if (info.samplerate, info.channels, info.frames, info.subtype) != (16000, 1, case["sample_count"], "FLOAT"):
            raise ValueError("WAV format/sample-count mismatch")
        rows.append({"id": case["id"], "dataset": case["split"], "path": str(path),
                     "audio_sha256": entry["sha256"], "lists": {"none": []}})
    output = private_path(output)
    with output.open("x") as stream:
        output.chmod(0o600)
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "corpus_sha256": sha256(corpus), "wav_index_sha256": sha256(wavs / "inputs.json"),
        "manifest_sha256": sha256(output)}, indent=2))
    print(f"Prepared {len(rows)} verified WAVs")


def language_counts(rows, predictions):
    """Count a clip as mismatched if any chunk names a different language.

    Missing/empty detected labels are reported separately, not silently treated
    as correct. The expected language is used only after inference, for scoring.
    """
    groups = defaultdict(lambda: {"clips": 0, "language_mismatch_clips": 0, "language_unknown_clips": 0})
    for row in rows:
        split = row["dataset"]
        labels = predictions[row["id"]].get("detected_languages")
        value = groups[split]
        value["clips"] += 1
        if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label for label in labels):
            value["language_unknown_clips"] += 1
        if isinstance(labels, list) and any(isinstance(label, str) and label and
                                          label.casefold() != LANGUAGES[split].casefold() for label in labels):
            value["language_mismatch_clips"] += 1
    return dict(groups)


def score_full(wavs, manifest, predictions, scorer, output):
    rows = load_jsonl(private_path(manifest))
    values = load_jsonl(private_path(predictions))
    indexed = {row["id"]: row for row in values}
    if len(indexed) != len(values) or set(indexed) != {row["id"] for row in rows}:
        raise ValueError("duplicate/missing/unexpected corpus predictions")
    provenance = json.loads(predictions.with_suffix(".provenance.json").read_text())
    if provenance["status"] != "complete" or provenance["language"] != "auto":
        raise ValueError("full corpus requires completed automatic-language run")
    if provenance["manifest_sha256"] != sha256(manifest):
        raise ValueError("manifest differs from measured run")
    if any(row["list"] != "none" or row["system"] != provenance["system"] for row in values):
        raise ValueError("full corpus must use one system with no list")
    output = private_path(output)
    if output.exists():
        raise FileExistsError(output)
    worker = private_path(output.with_suffix(".worker.json"))
    raw = private_path(output.with_suffix(".parity.json"))
    with worker.open("x") as stream:
        worker.chmod(0o600)
        json.dump({"engine": provenance["system"], "version": provenance["packages"]["mlx-audio"],
                   "commit": provenance["pins"][provenance["system"]]["revision"], "backend": "mlx-gpu",
                   "cold_start_seconds": provenance["load_and_warmup_seconds"],
                   "warm_wall_seconds": sum(row["seconds"] for row in values),
                   "peak_rss_bytes": provenance["peak_rss_bytes"],
                   "transcripts": [{"id": row["id"], "text": row["text"], "seconds": row["seconds"]}
                                   for row in values]}, stream)
    subprocess.run([str(scorer.resolve()), str(wavs), str(worker), str(raw)], check=True)
    splits = json.loads(raw.read_text())
    languages = language_counts(rows, indexed)
    for split, value in splits.items():
        value.update(languages[split])
        times = [indexed[row["id"]]["seconds"] for row in rows if row["dataset"] == split]
        value["median_seconds"] = statistics.median(times)
    times = sorted(row["seconds"] for row in values)
    result = {"system": provenance["system"], "scorer_sha256": sha256(scorer), "splits": splits,
              "clips": len(values), "median_seconds": statistics.median(times),
              "p95_seconds": times[math.ceil(0.95 * len(times)) - 1],
              "inference_seconds": sum(times), "peak_phys_footprint_bytes": provenance["peak_phys_footprint_bytes"],
              "model_bytes": provenance["model_bytes"]}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Graded {len(values)} clips; aggregate-only output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "score"])
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=TARGET / "parity-manifest.jsonl")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_full(args.wavs, args.manifest)
    else:
        if not all([args.predictions, args.scorer, args.output]):
            parser.error("score requires --predictions, --scorer and --output")
        score_full(args.wavs, args.manifest, args.predictions, args.scorer, args.output)


if __name__ == "__main__":
    main()
