#!/usr/bin/env python3
"""Derive compact, sealed load-time checks from the committed parity references."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "bench/parity"
DESTINATION = Path(__file__).resolve().parent / "src/self_check"
BASE_CASES = ["short-0", "boundary-127", "boundary-128", "boundary-129"]
POOL_CASES = ["pool10-000", "pool10-001", "pool10-002"]


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def main():
    source_index = json.loads((PARITY / "fixtures/index.json").read_text())
    index = {}
    DESTINATION.mkdir(exist_ok=True)
    for fixture_id, entry in source_index.items():
        source = (PARITY / entry["path"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != entry["sha256"]:
            raise ValueError("source fixture seal mismatch: " + fixture_id)
        document = json.loads(source)
        selected = BASE_CASES + (POOL_CASES if document["operation"] == "rerank" else [])
        cases = {case["id"]: case for case in document["cases"]}
        document["cases"] = [cases[case_id] for case_id in selected]
        subset = encoded(document)
        file = entry["model"] + ".json"
        (DESTINATION / file).write_bytes(subset)
        index[entry["model"]] = {
            "fixture_set_id": fixture_id,
            "source_sha256": entry["sha256"],
            "sha256": hashlib.sha256(subset).hexdigest(),
            "case_ids": selected,
        }
        print(file + ": " + str(len(subset)) + " bytes; " + str(len(selected)) + " cases")
    (DESTINATION / "index.json").write_bytes(encoded(index))


if __name__ == "__main__":
    main()
