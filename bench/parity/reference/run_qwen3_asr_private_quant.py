#!/usr/bin/env python3
"""Operator-only local Q8 voice-set check. Stdout contains aggregate counts only."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import sys


@contextlib.contextmanager
def private_output():
    """Suppress Python and native-library diagnostics without creating a log."""
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with open(os.devnull, "w") as sink:
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def aggregate(manifest, reference):
    import numpy as np
    import soundfile as sf
    from qwen3_asr_quant_gate import name_scorer
    scorer = name_scorer()
    rows = json.loads(Path(manifest).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("invalid manifest")
    correct, total, clips, false = 0, 0, 0, 0
    for row in rows:
        audio, rate = sf.read(Path(manifest).parent / row["audio"], dtype="float32", always_2d=True)
        if rate != 16000 or audio.shape[1] != 1 or not 0 < len(audio) <= 9600000:
            raise ValueError("invalid local audio")
        expected = row["expected"]
        if not expected or not all(isinstance(t, str) and t for t in expected):
            raise ValueError("invalid expected names")
        trace = reference.run({"id": "private", "split": "private", "terms": row["terms"]},
                              np.ascontiguousarray(audio[:, 0]))
        if trace["stop_reason"] != "stop":
            raise ValueError("decode limit")
        score = scorer.score_clip({"reference": row["reference"], "vocabulary": row["terms"],
                                   "expected": expected}, trace["text"])
        correct += score["colli_hits"]
        total += score["colli_expected"]
        false += score["false_insertions"]
        clips += 1
    return {"correct": correct, "total": total, "clips": clips, "false_occurrences": false,
            "names_right_percent": 100 * correct / total, "passed": correct * 10 >= total * 9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    # No downloads, no saved traces and no exception payloads. Dependency/model
    # messages can contain caller text, so suppress them for the whole local run.
    try:
        if platform.system() not in ("Darwin", "Windows"):
            raise ValueError("private set is restricted to the owner's Mac or Windows machine")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        with private_output():
            from qwen3_asr_quant import QuantReference
            result = aggregate(args.manifest, QuantReference(args.model, args.artifacts))
    except Exception:
        print('{"status":"local-check-error","private_details_suppressed":true}')
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
