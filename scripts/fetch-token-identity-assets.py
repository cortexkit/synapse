#!/usr/bin/env python3
"""Fetch only the checksum-pinned tokenizer/config inputs for token identity tests."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "crates/synapse-module/src/catalog/models.json")
    parser.add_argument("--output", type=Path, default=Path("target/token-identity-assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checked = 0
    for entry in json.loads(args.catalog.read_text())["models"]:
        for member in entry["files"]:
            if member["role"] not in ("tokenizer", "config"):
                continue
            output = args.output / member["sha256"]
            if not output.exists():
                url = "https://huggingface.co/{}/resolve/{}/{}".format(entry["upstream"]["hf_repo"], entry["upstream"]["revision"], member["path"])
                with urllib.request.urlopen(url, timeout=120) as response:
                    output.write_bytes(response.read())
            data = output.read_bytes()
            if len(data) != member["size_bytes"] or hashlib.sha256(data).hexdigest() != member["sha256"]:
                raise SystemExit("pinned input mismatch: {} {}".format(entry["id"], member["path"]))
            checked += 1
            print("verified {} {} ({})".format(entry["id"], member["role"], member["sha256"]))
    print("Verified {} pinned inputs; no model weights downloaded.".format(checked))


if __name__ == "__main__":
    main()
