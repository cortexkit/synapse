#!/usr/bin/env python3
"""Prepare an ignored converter input without changing any sealed profile or pin."""

import argparse
import copy
import json
from pathlib import Path
import unittest

PROFILE = "parakeet-tdt-0.6b-v3.owned-cuda"
ROOT = Path(__file__).resolve().parents[3]


def measurement_manifest(sealed):
    result = copy.deepcopy(sealed)
    result["profiles"][PROFILE]["fp32_tensors"] = "all"
    return result


class PreparationTests(unittest.TestCase):
    def test_only_scratch_precision_selection_changes(self):
        sealed = {"profiles": {PROFILE: {"fp32_tensors": [], "converted_package_digest": "sealed"}}}
        result = measurement_manifest(sealed)
        self.assertEqual(sealed["profiles"][PROFILE]["fp32_tensors"], [])
        self.assertEqual(result["profiles"][PROFILE]["fp32_tensors"], "all")
        result["profiles"][PROFILE]["fp32_tensors"] = []
        self.assertEqual(result, sealed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="target/scratch/cuda-speech-f32")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
        return
    destination = (ROOT / args.out_dir).resolve()
    scratch = (ROOT / "target" / "scratch").resolve()
    if not destination.is_relative_to(scratch):
        parser.error("measurement inputs must stay under this worktree's target/scratch")
    sealed = json.loads((ROOT / "bench/parity/models.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "models.json"
    output.write_text(json.dumps(measurement_manifest(sealed), indent=2) + "\n", encoding="utf-8")
    print(f"measurement-only converter manifest: {output}")


if __name__ == "__main__":
    main()
