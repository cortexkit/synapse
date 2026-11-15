#!/usr/bin/env python3
"""Prepare sealed 16 kHz WAV inputs without loading the neural reference model.

Reuses the pinned reference's audio recipes and hash-checked corpus sources.
The generated WAVs are local certification inputs, not replacement references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def prepare(repo, cache, output):
    sys.path.insert(0, str(repo / "bench/parity/reference"))
    import generate_parakeet as generator
    from prepare_parakeet import fetch
    import pyarrow.parquet as pq

    generator.ref.configure_runtime()

    parity = repo / "bench/parity"
    document = json.loads((parity / "fixtures/parakeet-tdt-0.6b-v3/reference.json").read_text())
    corpus = json.loads((parity / "reference/parakeet-corpus.json").read_text())
    audio = cache / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    for name, source in corpus["sources"].items():
        if name.endswith("-transcripts"):
            continue
        wanted = {row["member"]: row["audio_sha256"] for row in corpus["cases"] if row["source"] == name}
        if all((audio / digest).is_file() and hashlib.sha256((audio / digest).read_bytes()).hexdigest() == digest for digest in wanted.values()):
            continue
        artifact = fetch(source["url"], source["sha256"], cache / "sources")
        if name.startswith("librispeech"):
            rows = ((row["audio"]["path"], row["audio"]["bytes"]) for row in pq.read_table(artifact).to_pylist())
            for member, raw in rows:
                if member in wanted:
                    assert hashlib.sha256(raw).hexdigest() == wanted[member]
                    (audio / wanted[member]).write_bytes(raw)
        else:
            with tarfile.open(artifact, "r:gz") as archive:
                for member in archive:
                    if member.name in wanted:
                        raw = archive.extractfile(member).read()
                        assert hashlib.sha256(raw).hexdigest() == wanted[member.name]
                        (audio / wanted[member.name]).write_bytes(raw)
        for digest in wanted.values():
            assert hashlib.sha256((audio / digest).read_bytes()).hexdigest() == digest
        print("verified audio source", name, flush=True)
    generator.CACHE = cache
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for row in document["cases"]:
        # Tensor fixtures are consumed directly from their sealed little-endian
        # mel files. No source waveform is needed for those engine checks.
        if row.get("log_mel") is not None:
            continue
        recipe = row["input"]
        samples = generator.pcm(recipe)
        assert len(samples) == row["sample_count"]
        raw = generator.wav_bytes(samples, 16000, "FLOAT")
        target = row["id"] + ".wav"
        (output / target).write_bytes(raw)
        inputs[row["id"]] = {"path": target, "sha256": hashlib.sha256(raw).hexdigest()}
        print("prepared", row["id"], flush=True)
    (output / "inputs.json").write_text(json.dumps(inputs, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo.resolve(), args.cache.resolve(), args.out.resolve())
