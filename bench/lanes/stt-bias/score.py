#!/usr/bin/env python3
"""Aggregate local STT runs without emitting references or hypotheses."""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def norm(text):
    """Match COLLI speech-bias-v1/score.py, including its limited digit map."""
    numbers = {"1": "one", "4": "four", "28": "twenty eight", "20": "twenty"}
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " "))
    return " ".join(numbers.get(word, word) for word in text.split())


def alignment(ref, hyp):
    """Minimum word edits; ties prefer diagonal, deletion, then insertion."""
    table = [list(range(len(hyp) + 1))]
    for i, word in enumerate(ref, 1):
        row = [i]
        for j, other in enumerate(hyp, 1):
            row.append(min(table[-1][j - 1] + (word != other),
                           table[-1][j] + 1, row[-1] + 1))
        table.append(row)
    i, j, pairs = len(ref), len(hyp), []
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
            pairs.append((i, j))
        elif i and table[i][j] == table[i - 1][j] + 1:
            i -= 1
            pairs.append((i, None))
        else:
            j -= 1
            pairs.append((None, j))
    return list(reversed(pairs))


def spans(words, terms):
    """Return non-overlapping longest-first term spans with alias-group IDs."""
    candidates = []
    for term in dict.fromkeys(terms):
        for alias in term.split("|"):
            tokens = norm(alias).split()
            if not tokens:
                raise ValueError("empty normalized term")
            for start in range(len(words) - len(tokens) + 1):
                if words[start:start + len(tokens)] == tokens:
                    candidates.append((start, start + len(tokens), term))
    occupied, result = set(), []
    for start, end, term in sorted(candidates, key=lambda s: (s[0] - s[1], s[0], s[2])):
        if not occupied.intersection(range(start, end)):
            result.append((start, end, term))
            occupied.update(range(start, end))
    return sorted(result)


def score_clip(row, text):
    ref, hyp = norm(row["reference"]).split(), norm(text).split()
    pairs = alignment(ref, hyp)
    vocabulary = row["vocabulary"]
    expected = row["expected"]
    ref_spans, hyp_spans = spans(ref, vocabulary), spans(hyp, vocabulary)
    mapping = {j: i for i, j in pairs if j is not None}
    ref_keys = {(start, end, term) for start, end, term in ref_spans}
    hits, false = 0, 0
    for start, end, term in hyp_spans:
        indices = [mapping[j] for j in range(start, end)]
        valid = (None not in indices and
                 indices == list(range(indices[0], indices[0] + len(indices))) and
                 (indices[0], indices[-1] + 1, term) in ref_keys)
        hits += valid
        false += not valid
    ref_mask = {i for start, end, _ in ref_spans for i in range(start, end)}
    hyp_mask = {j for start, end, _ in hyp_spans for j in range(start, end)}
    edits, nonlist_edits = 0, 0
    for i, j in pairs:
        error = i is None or j is None or ref[i] != hyp[j]
        edits += error
        if i is not None:
            nonlist_edits += error and i not in ref_mask
        else:
            nonlist_edits += j not in hyp_mask
    anywhere = sum(any(" " + norm(alias) + " " in " " + norm(text) + " "
                       for alias in term.split("|")) for term in expected)
    return {"term_hits": hits, "term_occurrences": len(ref_spans),
            "term_predictions": len(hyp_spans), "false_insertions": false,
            "colli_hits": anywhere, "colli_expected": len(expected),
            "word_edits": edits, "reference_words": len(ref),
            "nonlist_edits": nonlist_edits, "nonlist_words": len(ref) - len(ref_mask)}


def load_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def aggregate(manifest, predictions):
    if not manifest or not predictions:
        raise ValueError("empty manifest or predictions")
    wanted = {row["id"]: row for row in manifest}
    if len(wanted) != len(manifest):
        raise ValueError("duplicate manifest ID")
    arms = defaultdict(dict)
    for result in predictions:
        key = (result["system"], result["list"])
        if result["id"] in arms[key]:
            raise ValueError("duplicate prediction")
        if not isinstance(result.get("text"), str):
            raise ValueError("missing transcript: incomplete arms cannot be scored")
        seconds = result["seconds"]
        if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid latency")
        arms[key][result["id"]] = result
    output = []
    for (system, list_name), results in sorted(arms.items()):
        applicable = {key: row for key, row in wanted.items() if list_name in row["lists"]}
        if results.keys() != applicable.keys():
            raise ValueError("missing or unexpected clip IDs; do not score partial arms")
        groups = defaultdict(list)
        for clip_id, result in results.items():
            row = applicable[clip_id]
            groups[row["dataset"]].append((score_clip(row, result["text"]), result["seconds"]))
        for dataset, rows in sorted(groups.items()):
            totals = {key: sum(item[key] for item, _ in rows) for key in rows[0][0]}
            def ratio(numerator, denominator):
                return totals[numerator] / totals[denominator] if totals[denominator] else None
            output.append({"system": system, "dataset": dataset, "list": list_name,
                           "clips": len(rows), **totals,
                           "aligned_recall": ratio("term_hits", "term_occurrences"),
                           "precision": ratio("term_hits", "term_predictions"),
                           "colli_recall": ratio("colli_hits", "colli_expected"),
                           "wer": ratio("word_edits", "reference_words"),
                           "nonlist_wer": ratio("nonlist_edits", "nonlist_words"),
                           "median_seconds": statistics.median(time for _, time in rows),
                           "p95_seconds": sorted(time for _, time in rows)[math.ceil(0.95 * len(rows)) - 1]})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(load_jsonl(args.manifest),
                       [row for path in args.predictions for row in load_jsonl(path)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(f"Scored {len(result)} aggregate rows; no per-clip text emitted")


if __name__ == "__main__":
    main()
