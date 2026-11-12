#!/usr/bin/env python3
"""Supervise the scheduled local ANE reference; refuse silent CTC failures."""

import argparse
import fcntl
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from prepare import sha256
from run import pcm
from safety import ROOT, TARGET, Watchdog, private_path
from score import aggregate, load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("STT_BIAS_ANE") != "1":
        raise RuntimeError("schedule an ANE window with its owner before setting STT_BIAS_ANE=1")
    if "TMPDIR" in os.environ:
        raise RuntimeError("unset custom TMPDIR before ANE execution")
    output = private_path(args.output)
    if output.exists():
        raise FileExistsError("never overwrite a measured run")
    binary = TARGET / "fluid-build/release/stt-bias-fluid"
    manifest_path = TARGET / "manifest.jsonl"
    rows = load_jsonl(manifest_path)
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    ledger = json.loads((TARGET / "downloads.json").read_text())
    model_hashes = {}
    for key in ["fluidaudio_tdt", "fluidaudio_ctc"]:
        if ledger[key]["revision"] != pins[key]["revision"]:
            raise RuntimeError("unexpected CoreML model pin")
        model_hashes[key] = {name: sha256(Path(ledger[key]["snapshot"]) / name)
                             for name in ledger[key]["files"]}
    clips = []
    for row in rows:
        if sha256(row["path"]) != row["audio_sha256"]:
            raise RuntimeError("audio changed after preparation")
        # The sequential numeric filename does not disclose an utterance's identity.
        path = private_path(TARGET / "fluid-pcm" / f"{len(clips):04d}.f32")
        pcm(row["path"]).astype("<f4").tofile(path)
        path.chmod(0o600)
        clips.append({"id": row["id"], "pcm": str(path), "lists": row["lists"]})
    request = {"clips": clips, "output": str(output),
               "tdt": ledger["fluidaudio_tdt"]["snapshot"], "ctc": ledger["fluidaudio_ctc"]["snapshot"]}
    request_path = private_path(output.with_suffix(".request.json"))
    request_path.write_text(json.dumps(request))
    request_path.chmod(0o600)
    provenance = {"system": "fluidaudio-ctc-default", "pins": pins, "os": platform.platform(),
                  "manifest_sha256": sha256(manifest_path), "worker_sha256": sha256(binary),
                  "model_artifacts": model_hashes, "started": time.time(), "status": "started",
                  "timing": "native PCM read, fresh TDT decode, vocabulary setup, CTC and rescore; no resampling"}
    native_log = private_path(output.with_suffix(".native.log"))
    # Open the existing shared lock read-only; never replace or unlink its inode.
    with args.lane_lock.open("rb") as lane_lock:
        fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watchdog = Watchdog(output.with_suffix(".resources.jsonl"))
        watchdog.start()
        try:
            with native_log.open("xb") as log:
                subprocess.run([str(binary), str(request_path)], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            log_text = native_log.read_text(errors="replace")
            if "Vocabulary rescoring failed" in log_text or "no log probs from CTC" in log_text:
                raise RuntimeError("CTC fallback warning invalidates arm; inspect target native log")
            results = aggregate(rows, load_jsonl(output))
            expected_groups = sum(len({row["dataset"] for row in rows if name in row["lists"]})
                                  for name in ["none", "local", "global", "unrelated"])
            if len(results) != expected_groups:
                raise RuntimeError("missing complete FluidAudio list arms")
            provenance["status"] = "complete"
            provenance["predictions"] = len(load_jsonl(output))
        except BaseException:
            provenance["status"] = "failed"
            raise
        finally:
            watchdog.close()
            provenance["finished"] = time.time()
            output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print("FluidAudio complete; private artifacts remain in target/stt-bias")


if __name__ == "__main__":
    main()
