#!/usr/bin/env python3
"""Fetch only public pinned assets; never read or upload private audio."""

import argparse
import json
import os
import shutil
from pathlib import Path

from safety import TARGET


def fetch(key):
    from huggingface_hub import HfApi, snapshot_download
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    pin = pins[key]
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    cache = TARGET / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    kind = "dataset" if key == "earnings22" else "model"
    info = HfApi().repo_info(pin["repo"], revision=pin["revision"], repo_type=kind,
                              files_metadata=True)
    if info.sha != pin["revision"]:
        raise RuntimeError("revision did not resolve to exact pin")
    if info.siblings is None:
        raise RuntimeError("missing HF file inventory")
    if key.startswith("fluidaudio_"):
        names = [s.rfilename for s in info.siblings if s.rfilename in pin["files"] or
                 any(s.rfilename.startswith(folder + "/") for folder in pin["directories"])]
    else:
        names = [s.rfilename for s in info.siblings if
                 s.rfilename == pin.get("file") or
                 (kind == "model" and s.rfilename.endswith((".json", ".safetensors", ".txt", ".model"))
                  and s.rfilename != "out_llm.safetensors")]
    size = sum(s.size or 0 for s in info.siblings if s.rfilename in names)
    ledger_path = TARGET / "downloads.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    # Reserve 1 GB for Python wheels, source trees, and metadata outside HF.
    projected = sum(v["bytes"] for k, v in ledger.items() if k != key) + size + 10**9
    if projected > 15 * 10**9:
        raise RuntimeError("15 GB cumulative download budget exceeded; ask owner")
    if shutil.disk_usage(TARGET).free < 20 * 2**30:
        raise RuntimeError("less than 20 GiB free before download")
    ledger[key] = {"bytes": size, "revision": info.sha, "files": names, "status": "reserved"}
    ledger_path.write_text(json.dumps(ledger, indent=2))
    snapshot = snapshot_download(pin["repo"], revision=pin["revision"], repo_type=kind,
                                 allow_patterns=names, cache_dir=cache, max_workers=2)
    ledger[key].update(status="complete", snapshot=snapshot)
    ledger_path.write_text(json.dumps(ledger, indent=2))
    print(f"{key}: pinned snapshot ready ({size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["qwen_0_6b", "qwen_1_7b", "granite_2b", "earnings22",
                                         "fluidaudio_tdt", "fluidaudio_ctc"])
    fetch(parser.parse_args().asset)
