"""Does Laya classify Athena consult requests well enough to be worth a lane?

Measures zero-shot `choice` accuracy against two numbers we already have on the
same 233 real rows: a trained Qwen3.5-2B student at 92.0% class-exact and stock
Qwen3.5-2B at 87.7%. Laya is 421M and answers in one encoder pass with no
training, so the question is how close it lands.

Two properties of the data decide the design, both measured before this ran:

  * `request_prose` is p50 2441 characters, so the MEDIAN request overflows the
    English checkpoint's 512-token budget (of which ~192 is the question head).
    A single accuracy number would be "laya on the first ~1300 characters"
    reported as "laya on this task", so every arm reports its truncation rate
    and splits accuracy by whether the row fit.
  * the real label set is skewed enough that a constant "AUDIT" predictor scores
    52.4%, so that floor is printed beside every real-set number.

The criteria below are PLACEHOLDER-CRITERIA-V0: one neutral sentence per class
derived from the class name alone. They were authored by a worker that never saw
an accuracy number (its run died on an unrelated tool failure), and are reused
verbatim here so the wording cannot have been tuned against the labels. The
authoritative Athena class descriptions are not in this repo; if they arrive they
run as a separate V1 arm and V0 stays in the report.
"""

import json
import math
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter, defaultdict

import torch

import laya
from laya.common import build_sequence, render_options

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
CLASSES = ["AUDIT", "EVALUATE", "PLAN", "DIAGNOSE", "EXPLAIN", "SPEC"]

PLACEHOLDER_CRITERIA_V0 = {
    "AUDIT": "Inspect something against requirements or standards to identify issues.",
    "EVALUATE": "Assess the quality, value, or suitability of something.",
    "PLAN": "Organize proposed actions into a plan for achieving a goal.",
    "DIAGNOSE": "Determine the cause of a problem from the available evidence.",
    "EXPLAIN": "Clarify how or why something works or happens.",
    "SPEC": "Define the requirements and intended behavior of something.",
}

CHECKPOINTS = {
    "english-512": {"subfolder": None},
    "typed-decisions-1024": {"subfolder": "typed-decisions"},
}


def load_rows(name):
    rows = []
    for line in (DATA / name).open():
        row = json.loads(line)
        gold = (row.get("reply_json") or {}).get("class")
        if gold is None:
            continue
        # 14 real rows carry lowercase `diagnose`; case is not a prediction error.
        rows.append({"id": row["id"], "prose": row["request_prose"], "gold": gold.upper()})
    return rows


def question(criteria):
    return {"type": "choice", "instructions": "Which kind of work does this request ask for?", "criteria": criteria}


def measure(agent, rows, criteria, label):
    """One arm. Returns per-row records; accuracy is computed by the caller."""
    q = question(criteria)
    # The budgets belong to the checkpoint, not to the English defaults.
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    internal = agent._to_internal({"type": "choice", "instructions": q["instructions"], "criteria": criteria})
    records = []
    started = time.time()
    for i, row in enumerate(rows):
        # Token accounting first: how much of the request actually reached the model.
        full = agent.tok(row["prose"], add_special_tokens=False)["input_ids"]
        seq, markers = build_sequence(agent.tok, row["prose"], internal, max_len, head_max_len)
        head_and_specials = len(seq) - sum(1 for _ in range(0))  # placeholder, replaced below
        # The state occupies whatever is left after the head, options and separators.
        state_room = max_len - (markers[0] if markers else 0) - sum(
            len(agent.tok(" " + o.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"][:48]) + 1
            for o in render_options(internal)
        ) - 2
        kept = min(len(full), max(0, state_room))
        truncated = kept < len(full)

        out = agent.predict(row["prose"], {"kind": q})["answers"]["kind"]
        records.append(
            {
                "id": row["id"],
                "gold": row["gold"],
                "predicted": out["choice"],
                "correct": out["choice"] == row["gold"],
                "probabilities": out["probabilities"],
                "confidence": out["confidence"],
                "act_probability": out["action"]["act_probability"],
                "prose_tokens": len(full),
                "kept_tokens": kept,
                "truncated": truncated,
                "surviving_fraction": (kept / len(full)) if full else 1.0,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"   {label}: {i + 1}/{len(rows)}", flush=True)
    elapsed = time.time() - started
    return records, elapsed


def accuracy(records):
    return (sum(r["correct"] for r in records) / len(records)) if records else float("nan")


def confusion(records):
    table = defaultdict(Counter)
    for r in records:
        table[r["gold"]][r["predicted"]] += 1
    return table


def ece(records, bins=10):
    """Expected calibration error over the library's confidence, with bins printed."""
    edges = [i / bins for i in range(bins + 1)]
    rows = []
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        bucket = [r for r in records if (lo < r["confidence"] <= hi) or (lo == 0 and r["confidence"] == 0)]
        if not bucket:
            rows.append((lo, hi, 0, None, None))
            continue
        acc = accuracy(bucket)
        conf = sum(r["confidence"] for r in bucket) / len(bucket)
        total += (len(bucket) / len(records)) * abs(acc - conf)
        rows.append((lo, hi, len(bucket), acc, conf))
    return total, rows


def retention(records, thresholds=(0.5, 0.7, 0.85, 0.95)):
    out = []
    for t in thresholds:
        kept = [r for r in records if r["confidence"] >= t]
        out.append((t, len(kept), len(kept) / len(records), accuracy(kept) if kept else float("nan")))
    return out


def main():
    loadavg_before = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    results = {}
    for name, spec in CHECKPOINTS.items():
        print(f"== loading {name}", flush=True)
        agent = laya.load("convaiinnovations/laya", subfolder=spec["subfolder"])
        print(f"   max_len={agent.cfg.get('max_len')} head_max_len={agent.cfg.get('head_max_len')}", flush=True)
        for corpus in ("real-gold.jsonl", "classify-gold-v1.jsonl"):
            rows = load_rows(corpus)
            label = f"{name}/{corpus}"
            records, elapsed = measure(agent, rows, PLACEHOLDER_CRITERIA_V0, label)
            results[label] = {"records": records, "elapsed_s": elapsed, "n": len(records)}
            out = ROOT / f"rows-{name}-{corpus.replace('.jsonl', '')}.jsonl"
            with out.open("w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            print(f"   wrote {out.name}: {len(records)} rows in {elapsed:.1f}s", flush=True)
        del agent

    loadavg_after = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    summary = {
        "criteria_version": "PLACEHOLDER-CRITERIA-V0",
        "criteria": PLACEHOLDER_CRITERIA_V0,
        "loadavg_before": loadavg_before,
        "loadavg_after": loadavg_after,
        "arms": {},
    }
    for label, data in results.items():
        recs = data["records"]
        fitting = [r for r in recs if not r["truncated"]]
        truncated = [r for r in recs if r["truncated"]]
        surviving = sorted(r["surviving_fraction"] for r in truncated)
        e, bins = ece(recs)
        summary["arms"][label] = {
            "n": len(recs),
            "accuracy": accuracy(recs),
            "accuracy_fitting": accuracy(fitting),
            "n_fitting": len(fitting),
            "accuracy_truncated": accuracy(truncated),
            "n_truncated": len(truncated),
            "median_surviving_fraction_of_truncated": (surviving[len(surviving) // 2] if surviving else None),
            "confusion": {g: dict(c) for g, c in confusion(recs).items()},
            "ece": e,
            "ece_bins": bins,
            "retention": retention(recs),
            "act_saturated_at_1": sum(1 for r in recs if r["act_probability"] >= 0.9999),
            "median_wall_ms": 1000 * data["elapsed_s"] / max(1, data["n"]),
        }
    (ROOT / "quality-v0.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("confusion", "ece_bins", "retention")}
                      for k, v in summary["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()


def run_v1():
    """Second arm: source-grounded criteria, same rows, same everything else."""
    from criteria_v1 import CRITERIA_V1
    out = {}
    for name, spec in CHECKPOINTS.items():
        agent = laya.load("convaiinnovations/laya", subfolder=spec["subfolder"])
        for corpus in ("real-gold.jsonl", "classify-gold-v1.jsonl"):
            rows = load_rows(corpus)
            recs, elapsed = measure(agent, rows, CRITERIA_V1, f"v1 {name}/{corpus}")
            fitting = [r for r in recs if not r["truncated"]]
            trunc = [r for r in recs if r["truncated"]]
            e, _ = ece(recs)
            out[f"{name}/{corpus}"] = {
                "n": len(recs), "accuracy": accuracy(recs),
                "accuracy_fitting": accuracy(fitting), "n_fitting": len(fitting),
                "accuracy_truncated": accuracy(trunc) if trunc else None, "n_truncated": len(trunc),
                "confusion": {g: dict(c) for g, c in confusion(recs).items()},
                "ece": e, "retention": retention(recs),
                "act_saturated_at_1": sum(1 for r in recs if r["act_probability"] >= 0.9999),
            }
            with (ROOT / f"rows-v1-{name}-{corpus.replace('.jsonl','')}.jsonl").open("w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"   v1 {name}/{corpus}: {accuracy(recs):.3f}", flush=True)
        del agent
    (ROOT / "quality-v1.json").write_text(json.dumps({"criteria_version": "SOURCE-GROUNDED-V1", "criteria": CRITERIA_V1, "arms": out}, indent=2))
