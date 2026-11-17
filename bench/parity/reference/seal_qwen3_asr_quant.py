#!/usr/bin/env python3
"""Offline full-corpus grading; partial pilots cannot produce an admitted profile."""

import argparse
import json
from pathlib import Path

from qwen3_asr_corpus import dump, expand
from qwen3_asr_quant_gate import seal_profile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--private-aggregate", type=Path, required=True)
    parser.add_argument("--wer-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new: never overwrite a previous grading report")
    cases = expand(json.loads((HERE / "qwen3-asr-corpus.json").read_text()))["cases"]
    wanted = {c["id"] for c in cases if not c.get("refusal")}

    def load(directory):
        if {p.stem for p in directory.glob("*.json")} != wanted:
            raise ValueError("incomplete or unexpected quant fixture coverage")
        return {id: json.loads((directory / (id + ".json")).read_text()) for id in wanted}

    teachers = load(args.generated / "teacher")
    profiles = load(args.generated / "forced-profile")
    baseline = {id: json.loads((HERE.parent / "fixtures/qwen3-asr-1.7b/traces" / (id + ".json")).read_text()) for id in wanted}
    result = seal_profile(cases, teachers, profiles, baseline, args.wer_binary.resolve(),
                          json.loads(args.private_aggregate.read_text()))
    dump(args.output, result)
    print(json.dumps({"status": result["status"], "tau": result["tau"],
                      "engine_slices_allowed": result["engine_slices_allowed"]}))
    if not result["engine_slices_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
