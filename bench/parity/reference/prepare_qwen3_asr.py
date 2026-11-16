#!/usr/bin/env python3
"""Materialize only immutable public audio; never ingest a caller's voice set."""

import argparse
import io
import json
from pathlib import Path
import random

import prepare_parakeet as parakeet
from qwen3_asr_corpus import (EARNINGS_SOURCE, SEED, TIERS, compact, control_lists,
                             distractors, dump, tier_list, validate_annotations)

HERE = Path(__file__).resolve().parent


def public_rows(cache):
    import pyarrow.parquet as pq
    corpus = json.loads((HERE / "parakeet-corpus.json").read_text())
    source = corpus["sources"]["librispeech-test-clean"]
    table = pq.read_table(parakeet.fetch(source["url"], source["sha256"], cache / "sources"))
    candidates = []
    for item in table.to_pylist():
        candidates.append({"id": "librispeech-test-clean-" + item["id"],
                           "utterance_id": item["id"], "split": "librispeech-test-clean",
                           "ground_truth": item["text"],
                           **parakeet.audio_member(item["audio"]["bytes"], cache)})
    source = EARNINGS_SOURCE
    table = pq.read_table(parakeet.fetch(source["url"], source["sha256"], cache / "sources"))
    eligible = [i for i, flagged in enumerate(table["is_flagged"].to_pylist()) if not flagged]
    selected = sorted(random.Random(SEED).sample(eligible, 200))
    earnings = []
    for i in selected:
        item = table.slice(i, 1).to_pylist()[0]
        earnings.append({"id": f"earnings22-{i:04d}", "split": "earnings22-test-200",
                         "source_row": i, "ground_truth": item["text"],
                         "global_terms": item["keywords"], "local_terms": item["dictionary"],
                         "vocabulary": list(dict.fromkeys(item["keywords"] + item["dictionary"])),
                         **parakeet.audio_member(item["audio"]["bytes"], cache)})
    return corpus, candidates, earnings


def concatenation(candidates, seconds):
    """Keep whole utterances in chapter/id order; silence fills the final gap."""
    parts, count = [], 0
    for row in sorted(candidates, key=lambda r: r["utterance_id"]):
        if count + row["sample_count"] > seconds * 16000:
            break
        parts.append(row)
        count += row["sample_count"]
    return {"recipe": "concatenate", "parts": parts, "sample_count": seconds * 16000,
            "sample_rate": 16000, "channels": 1,
            "ground_truth": " ".join(row["ground_truth"] for row in parts),
            "trailing_silence_samples": seconds * 16000 - count}


def plans(corpus, candidates, earnings, annotations):
    cases = [{**row, "terms": []} for row in corpus["cases"]]
    controls = [row for row in cases if row["split"].startswith("librispeech-")]
    if len(controls) != 400:
        raise ValueError("C4 requires all 400 Parakeet LibriSpeech controls")
    pool = sorted({term for row in earnings for term in row["global_terms"]})
    lists = {}
    for row in earnings:
        cases.append({**row, "terms": [], "id": row["id"] + "-none"})
        cases.append({**row, "terms": row["global_terms"], "id": row["id"] + "-global"})
        for tier in TIERS:
            arm = tier_list(row, pool, tier)
            # The baseline uses precisely the same extended vocabulary as its tier arm.
            for baseline in (False, True):
                cases.append({**row, **arm, "terms": [] if baseline else arm["terms"],
                              "id": f"{row['id']}-tier-{tier}" + ("-none" if baseline else ""),
                              "tier": tier, "baseline": baseline})
    for tier in TIERS:
        for i, terms in enumerate(control_lists(controls, pool, tier)):
            key = f"control-{tier}-{i}"
            lists[key] = terms
            for row in controls:
                cases.append({**row, "id": f"{row['id']}-{key}", "list_id": key,
                              "terms": terms, "vocabulary": terms, "tier": tier,
                              "split": row["split"], "probe_set": "unrelated-list"})
    for kind in ("near-silence", "noise"):
        for i in range(50):
            cases.append({"id": f"{kind}-{i:02d}", "split": kind, "recipe": kind,
                          "seed": SEED + i, "sample_count": (1 + i % 10) * 16000,
                          "sample_rate": 16000, "channels": 1, "ground_truth": "",
                          "terms": distractors(pool, [], 30, f"{kind}:{i}")})
    for row, review in validate_annotations(annotations, candidates):
        cases.append({**row, "id": "single-name-" + row["utterance_id"],
                      "split": "single-name", "noun": review["noun"],
                      "word_span": review["word_span"],
                      "terms": [review["noun"]] + distractors(pool, [row["ground_truth"]], 29,
                                                               "single:" + row["id"])})
    for seconds in (120, 300, 570):
        base = concatenation(candidates, seconds)
        for biased in (False, True):
            cases.append({**base, "id": f"long-form-{seconds}" + ("-list" if biased else ""),
                          "split": "long-form", "terms": lists["control-300-0"] if biased else []})
    for name, samples in (("short", 1600), ("sub-100-remainder", 21920), ("over-30s", 501920)):
        cases.append({"id": "frontend-" + name, "split": "frontend", "recipe": "noise",
                      "seed": SEED, "sample_count": samples, "sample_rate": 16000,
                      "channels": 1, "ground_truth": "", "terms": []})
    cases.append({"id": "over-limit", "split": "over-limit", "recipe": "near-silence",
                  "seed": SEED, "sample_count": 9600160, "sample_rate": 16000,
                  "channels": 1, "ground_truth": "", "terms": [], "refusal": "audio_too_long"})
    return {"schema": "synapse-qwen3-asr-corpus-v1", "seed": SEED,
            "parakeet_corpus_sha256": parakeet.sha256_file(HERE / "parakeet-corpus.json"),
            "sources": {**corpus["sources"], "earnings22": EARNINGS_SOURCE},
            "earnings_selection": "sorted Random(20261119).sample(unflagged indices, 200)",
            "curation": annotations, "control_lists": lists, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--review-candidates", action="store_true")
    args = parser.parse_args()
    corpus, candidates, earnings = public_rows(args.cache)
    if args.review_candidates:
        order = sorted(candidates, key=lambda r: r["utterance_id"])
        random.Random(SEED).shuffle(order)
        dump(args.output, {"seed": SEED, "candidates": order, "earnings": earnings})
    else:
        if not args.annotations:
            parser.error("--annotations is required after manual public-name curation")
        # Reuse the sealed Parakeet selection, verifying all cached source/member bytes.
        actual = parakeet.prepare(args.cache, args.cache / "parakeet-corpus.json")
        if actual != corpus:
            raise ValueError("public Parakeet corpus changed")
        dump(args.output, compact(plans(corpus, candidates, earnings,
                                       json.loads(args.annotations.read_text()))))


if __name__ == "__main__":
    main()
