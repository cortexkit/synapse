#!/usr/bin/env python3
"""Run one local model at a time; private predictions stay under target/."""

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import time
from pathlib import Path

from prepare import sha256
from safety import TARGET, Watchdog, private_path
from score import load_jsonl


def pcm(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    value = value.mean(axis=1)
    if rate != 16000:
        factor = gcd(rate, 16000)
        value = resample_poly(value, 16000 // factor, rate // factor)
    if not len(value) or not np.isfinite(value).all():
        raise ValueError("empty or non-finite audio")
    return np.ascontiguousarray(value, dtype="float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True,
                        choices=["transcribe", "qwen_0_6b", "qwen_1_7b", "granite_2b"])
    parser.add_argument("--boost", type=float, choices=[2.5, 4.0], default=2.5)
    parser.add_argument("--manifest", type=Path, default=TARGET / "manifest.jsonl")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--lists", nargs="+", default=["none", "local", "global", "unrelated"])
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = private_path(args.output)
    if output_path.exists():
        raise FileExistsError("never overwrite a measured run")
    rows = load_jsonl(private_path(args.manifest))
    if args.datasets:
        rows = [row for row in rows if row["dataset"] in args.datasets]
    if not rows:
        raise ValueError("no selected clips")
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    for row in rows:
        if sha256(row["path"]) != row["audio_sha256"]:
            raise ValueError("audio changed after manifest preparation")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    watchdog = Watchdog(output_path.with_suffix(".resources.jsonl"))
    watchdog.start()
    system_name = f"transcribe-b{args.boost:g}" if args.system == "transcribe" else args.system
    previous = load_jsonl(private_path(args.resume_from)) if args.resume_from else []
    wanted = {(row["id"], name) for row in rows for name in args.lists if name in row["lists"]}
    finished = set()
    for item in previous:
        key = (item["id"], item["list"])
        if item["system"] != system_name or key not in wanted or key in finished:
            raise ValueError("resume inputs do not match selected run")
        finished.add(key)
    provenance = {"system": system_name, "host": platform.machine(), "os": platform.platform(),
                  "python": platform.python_version(), "manifest_sha256": sha256(args.manifest),
                  "pins": pins, "started": time.time(), "packages": {}}
    provenance["resume_source_sha256"] = sha256(args.resume_from) if args.resume_from else None
    provenance["resumed_predictions"] = len(previous)
    provenance["status"] = "started"
    output_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    for name in ["mlx", "mlx-audio", "mlx-lm", "transformers", "numpy", "soundfile", "transcribe-cpp"]:
        try:
            provenance["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    model = None
    completed = 0
    try:
        before = time.perf_counter()
        if args.system == "transcribe":
            from transcribe_adapter import Transcribe
            if not args.gguf or sha256(args.gguf) != pins["transcribe_cpp"]["sha256"]:
                raise ValueError("GGUF hash differs from lane pin")
            model = Transcribe(args.gguf, args.boost)
        else:
            from mlx_adapter import MLX
            ledger = json.loads((TARGET / "downloads.json").read_text())
            if ledger[args.system]["revision"] != pins[args.system]["revision"]:
                raise ValueError("model snapshot revision does not match pin")
            provenance["model_artifacts"] = {
                name: sha256(Path(ledger[args.system]["snapshot"]) / name)
                for name in ledger[args.system]["files"]}
            model = MLX(Path(ledger[args.system]["snapshot"]), args.system)
        model.generate(pcm(rows[0]["path"]), [])
        provenance["load_and_warmup_seconds"] = time.perf_counter() - before
        with output_path.open("x") as output:
            output_path.chmod(0o600)
            for item in previous:
                output.write(json.dumps(item) + "\n")
            # Interleave list arms by clip to reduce drift from shared-host load.
            for row in rows:
                for list_name in args.lists:
                    if list_name not in row["lists"]:
                        continue
                    if (row["id"], list_name) in finished:
                        continue
                    before = time.perf_counter()
                    text = model.generate(pcm(row["path"]), row["lists"][list_name])
                    seconds = time.perf_counter() - before
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 20 * 2**30:
                        raise RuntimeError("20 GiB resident-memory high-water limit exceeded")
                    result = {"id": row["id"], "system": system_name, "list": list_name,
                              "text": text, "seconds": seconds}
                    output.write(json.dumps(result) + "\n")
                    output.flush()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"{system_name}: {completed} predictions", flush=True)
        provenance["status"] = "complete"
    except BaseException:
        provenance["status"] = "failed"
        raise
    finally:
        if model:
            model.close()
        watchdog.close()
        provenance.update(completed=completed, finished=time.time(),
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"{system_name}: complete ({completed} predictions)")


if __name__ == "__main__":
    main()
