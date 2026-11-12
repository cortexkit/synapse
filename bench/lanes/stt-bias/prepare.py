#!/usr/bin/env python3
"""Prepare private manifests locally; only aggregate counts reach stdout."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import random
import re
import sys
from pathlib import Path

from safety import ROOT, TARGET, private_path
from score import norm, spans


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def colli(root):
    spec = importlib.util.spec_from_file_location("colli_score", root / "score.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import corpus-local COLLI scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (root / "phrases.tsv").open() as source:
        phrases = {row["id"]: row["text"] for row in csv.DictReader(source, delimiter="\t")}
    bias = [term.strip() for term in re.split(r"[,\n]", (root / "bias.txt").read_text()) if term.strip()]
    vocabulary = list(dict.fromkeys(term for terms in module.TERMS.values() for term in terms))
    aliases = {norm(alias) for term in vocabulary for alias in term.split("|")}
    vocabulary.extend(term for term in bias if norm(term) not in aliases)
    for text in [*phrases.values(), *bias, *vocabulary]:
        if norm(text) != module.norm(text):
            raise ValueError("normalizer differs from supplied COLLI scorer")
    name = "colli-real" if "ufuk" in root.name else "colli-tts"
    rows = []
    for audio in sorted((root / "audio").glob("*.wav")):
        phrase = audio.stem.split("-")[0]
        rows.append({"id": f"{name}/{audio.stem}", "dataset": name,
                     "path": str(audio.resolve()), "audio_sha256": sha256(audio),
                     "reference": phrases[phrase], "expected": module.TERMS[phrase],
                     "vocabulary": vocabulary,
                     "lists": {"none": [], "local": [s.split("|")[0] for s in module.TERMS[phrase]],
                               "global": bias}})
    expected_count = 20 if name == "colli-real" else 100
    if len(rows) != expected_count or len(bias) != 30:
        raise ValueError("unexpected COLLI clip/list count")
    return rows, bias, {"scorer_sha256": sha256(root / "score.py"),
                        "phrases_sha256": sha256(root / "phrases.tsv"),
                        "bias_sha256": sha256(root / "bias.txt")}


def parity(wavs, bias):
    document_path = ROOT / "bench/parity/reference/parakeet-corpus.json"
    document = json.loads(document_path.read_text())
    index = json.loads((wavs / "inputs.json").read_text())
    rows = []
    for case in document["cases"]:
        if not case["split"].startswith("librispeech"):
            continue
        entry = index[case["id"]]
        path = (wavs / entry["path"]).resolve()
        if not path.is_relative_to(wavs.resolve()) or sha256(path) != entry["sha256"]:
            raise ValueError("parity WAV boundary/hash mismatch")
        rows.append({"id": case["id"], "dataset": case["split"], "path": str(path),
                     "audio_sha256": entry["sha256"], "reference": case["ground_truth"],
                     "expected": [], "vocabulary": bias,
                     "lists": {"none": [], "unrelated": bias}})
    if len(rows) != 400:
        raise ValueError("expected 400 parity LibriSpeech clips")
    return rows, {"corpus_sha256": sha256(document_path), "wav_index_sha256": sha256(wavs / "inputs.json")}


def earnings():
    import pyarrow.parquet as pq
    import soundfile as sf
    ledger = json.loads((TARGET / "downloads.json").read_text())
    pin = json.loads(Path(__file__).with_name("pins.json").read_text())["earnings22"]
    parquet = Path(ledger["earnings22"]["snapshot"]) / pin["file"]
    table = pq.read_table(parquet)
    eligible = [i for i, flagged in enumerate(table["is_flagged"].to_pylist()) if not flagged]
    indices = sorted(random.Random(pin["seed"]).sample(eligible, pin["subset_size"]))
    rows = []
    for i in indices:
        item = table.slice(i, 1).to_pylist()[0]
        path = private_path(TARGET / "earnings-audio" / f"{i:04d}.wav")
        pcm, rate = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
        if rate != 16000 or pcm.ndim != 1:
            raise ValueError("unexpected Earnings audio format")
        sf.write(path, pcm, rate, subtype="FLOAT")
        local, global_list = item["dictionary"], item["keywords"]
        rows.append({"id": f"earnings22/{i:04d}", "dataset": "earnings22-test-200",
                     "path": str(path), "audio_sha256": sha256(path),
                     "reference": item["text"], "expected": local,
                     "vocabulary": list(dict.fromkeys([*global_list, *local])),
                     "lists": {"none": [], "local": local, "global": global_list,
                               "global64": global_list[:64]}})
    return rows, {"parquet_sha256": sha256(parquet), "revision": pin["revision"],
                  "seed": pin["seed"], "selection": "sorted random.Random(seed).sample(unflagged row indices, 200)",
                  "eligible": len(eligible), "rows": table.num_rows, "row_indices": indices,
                  "global_over_64": sum(len(row["lists"]["global"]) > 64 for row in rows)}


def main():
    if sys.platform != "darwin":
        raise RuntimeError("private corpus preparation must run on the Mac")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--tts", type=Path, required=True)
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=TARGET / "manifest.jsonl")
    args = parser.parse_args()
    real, bias, real_meta = colli(args.real)
    tts, tts_bias, tts_meta = colli(args.tts)
    if bias != tts_bias:
        raise ValueError("COLLI corpora have different bias lists")
    controls, controls_meta = parity(args.wavs, bias)
    public, public_meta = earnings()
    rows = [*real, *tts, *public, *controls]
    path = private_path(args.output)
    with path.open("x") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    path.chmod(0o600)
    metadata = {"colli-real": real_meta, "colli-tts": tts_meta,
                "parity": controls_meta, "earnings22": public_meta,
                "manifest_sha256": sha256(path)}
    path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2))
    missing = sum(not spans(norm(row["reference"]).split(), [term])
                  for row in rows for term in row["expected"])
    print(f"Prepared {len(rows)} clips; {missing} expected term annotations have no normalized reference span")
    print(f"Earnings global lists >64 terms: {public_meta['global_over_64']}/{len(public)}")


if __name__ == "__main__":
    main()
