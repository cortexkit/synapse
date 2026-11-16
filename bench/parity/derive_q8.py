#!/usr/bin/env python3
"""Fetch pinned public checkpoints and measure the production derivation CLI.

Run on Linux with an empty job-local output directory. Only the small JSON
receipt should leave the job: neither original nor derived weights are evidence
to copy back. Build q8-derive with --locked --release --features q8-derive first.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def fetch(repo, revision, name, expected, destination):
    digest = hashlib.sha256()
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true"
    with urllib.request.urlopen(url, timeout=300) as response, destination.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            out.write(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"source digest mismatch: {repo}/{name}: {actual} != {expected}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["qwen3-asr-1.7b", "qwen3-embedding-0.6b", "qwen3-0.6b", "lfm2-1.2b"])
    parser.add_argument("--baseline-binary", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(__file__).with_name("models.json").read_text())
    asr = manifest["models"]["qwen3-asr-1.7b"]
    sources = [
        ("qwen3-asr-1.7b", asr["hf_repo"], asr["hf_revision"], asr["files"]),
        ("qwen3-embedding-0.6b", manifest["models"]["qwen3-embedding-0.6b"]["hf_repo"], manifest["models"]["qwen3-embedding-0.6b"]["hf_revision"], manifest["models"]["qwen3-embedding-0.6b"]["files"]),
        ("qwen3-0.6b", "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", None),
        ("lfm2-1.2b", "LiquidAI/LFM2-1.2B", "933cee00d754fb3bfe06c644c0cb95453f2d8bb2", None),
    ]
    receipt = {"cpu_count": os.cpu_count(), "cpu_model": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "runs": []}
    for slug, repo, revision, files in sources:
        if slug not in args.models:
            continue
        root = args.output / slug
        root.mkdir()
        if files is None:
            metadata = get_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
            if metadata["sha"] != revision:
                raise ValueError("upstream returned another revision")
            siblings = {s["rfilename"]: s for s in metadata["siblings"]}
            # Configs are Git blobs, weights are SHA-256 LFS objects. Both are
            # addressed by the immutable revision; record the config SHA below.
            files = {"model.safetensors": siblings["model.safetensors"]["lfs"]["sha256"], "config.json": None}
            if slug == "lfm2-1.2b" and files["model.safetensors"] != "60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd":
                raise ValueError("LFM2 weight differs from existing source pin")
        verified = {}
        for name, digest in files.items():
            print(f"fetch {slug}/{name}", flush=True)
            if digest is None:
                with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=120) as response:
                    data = response.read()
                (root / name).write_bytes(data)
                verified[name] = hashlib.sha256(data).hexdigest()
            else:
                verified[name] = fetch(repo, revision, name, digest, root / name)
        start = time.perf_counter()
        process = subprocess.Popen([str(args.binary.resolve()), slug, str(root), str(root / "derived")], stdout=subprocess.PIPE)
        assert process.stdout is not None
        output = process.stdout.read()
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
        if process.returncode:
            raise RuntimeError(f"derivation failed for {slug}: exit {process.returncode}")
        result = json.loads(output)
        result.update(hf_repo=repo, hf_revision=revision, verified_files=verified,
                      process_wall_seconds=time.perf_counter() - start,
                      peak_host_rss_bytes=usage.ru_maxrss * 1024)
        if args.baseline_binary and slug != "qwen3-asr-1.7b":
            baseline = subprocess.check_output([str(args.baseline_binary.resolve()), slug, str(root), str(root / "baseline-derived")])
            result["baseline"] = json.loads(baseline)
            if result["q8_0_digest"] != result["baseline"]["q8_0_digest"]:
                raise ValueError(f"baseline/current Q8 mismatch: {slug}")
        receipt["runs"].append(result)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
