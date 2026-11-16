#!/usr/bin/env python3
"""Seal a prepared, verified corpus order and build/asset provenance before the owner window."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
FLUID_ASSETS = "7dd20fe6b1797d35f5e3307e8b1732d9a178edfe"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_at(path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "wavs", "fluid-model", "snapshot", "package", "ane-executable", "fluid-executable", "scorer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = {k: v.resolve() for k, v in vars(args).items()}
    if paths["fluid_model"].name != FLUID_ASSETS:
        raise ValueError("use the pinned Hugging Face snapshot directory, not an unversioned model folder")
    for model in ("Preprocessor.mlmodelc", "Encoder.mlmodelc", "Decoder.mlmodelc", "JointDecisionv3.mlmodelc", "parakeet_vocab.json"):
        if not (paths["fluid_model"] / model).exists():
            raise ValueError(f"missing pinned FluidAudio asset: {model}")
    manifest = [json.loads(line) for line in paths["manifest"].read_text().splitlines()]
    by_id = {row["id"]: row for row in manifest}
    if len(manifest) != 1200 or len(by_id) != 1200:
        raise ValueError("expected 1200 unique verified manifest rows")
    reference_path = ROOT / "bench/parity/fixtures/parakeet-tdt-0.6b-v3/reference.json"
    cases = [c for c in json.loads(reference_path.read_text())["cases"] if c.get("split") and c.get("log_mel") is None]
    if {c["id"] for c in cases} != set(by_id):
        raise ValueError("manifest does not match the sealed WAV fixture population")
    # This stable sort is the signed hardware harness's order, including ties.
    cases.sort(key=lambda c: c["valid_encoder_frames"])
    for row in manifest:
        if sha256(row["path"]) != row["audio_sha256"]:
            raise ValueError(f"WAV checksum mismatch: {row['id']}")
    request = {"utterances": [{"id": c["id"], "path": by_id[c["id"]]["path"]} for c in cases]}
    (args.out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    shutil.copy2(paths["manifest"], args.out / "manifest.jsonl")
    hashes = {}
    for key in ("fluid_model", "snapshot", "package", "ane_executable", "fluid_executable", "scorer"):
        for path in files_at(paths[key]):
            hashes[str(path)] = sha256(path)
    for path in (reference_path, ROOT / "bench/parity/reference/parakeet-corpus.json", ROOT / "bench/parity/models.json",
                 paths["wavs"] / "inputs.json", args.out / "request.json", args.out / "manifest.jsonl",
                 ROOT / "bench/lanes/stt-fluidaudio/timing-patch.json", ROOT / "bench/lanes/stt-fluidaudio/StageKit.swift"):
        hashes[str(path.resolve())] = sha256(path)
    for row in manifest:
        hashes[row["path"]] = row["audio_sha256"]
    provenance = {"paths": {k: str(v) for k, v in paths.items()}, "sha256": hashes,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "os": platform.platform(), "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "scope": "81850 ms selected envelope; not 600-second certification", "precision": "Synapse fp16 encoder/fp32 host; pinned Fluid pal6",
        "cache": "existing CoreML/ANE compilation caches retained; per-rung inventory in Synapse receipts; one unmeasured same-clip warmup per pass"}
    (args.out / "prepared.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {len(cases)} sealed clips and {len(hashes)} file hashes in {args.out}")


if __name__ == "__main__":
    main()
