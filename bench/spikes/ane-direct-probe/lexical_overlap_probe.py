"""Does the retrieval advantage ride on lexical overlap, or survive it?

A pooled blind comparison found the ANE lane outranking a production embedder
by a wide margin on real queries. Those queries are keyword bags — median six
words, none of them questions — and an LLM judge grading a keyword bag against
a document tends to reward surface term overlap. If the winning system also
retrieves more lexically-overlapping rows, retrieval and judging reward the same
thing and the margin is partly an artifact rather than a quality difference.

This measures that directly, three ways:

  1. Do the two systems differ in the lexical overlap of what they retrieve?
  2. Does the judge's grade track overlap, holding the system constant?
  3. Does the advantage survive on the low-overlap rows, where a judge cannot be
     rewarding surface match?

Question 3 is the one that matters. If the advantage holds where overlap is
lowest, it is not an overlap artifact.

Reads private judging artifacts from the spike scratch directory and prints
aggregates only. No query text, row text, or identifier is emitted.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict

SCRATCH = os.path.expanduser("~/.local/share/cortexkit/synapse-spikes/broca-ane/rig-v1")

# Words carrying no retrieval signal. Kept short deliberately: an aggressive
# list would do the embedding model's job for it and flatter the overlap score.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
    "was", "are", "be", "it", "that", "this", "as", "at", "by", "from", "we",
}

TOKEN = re.compile(r"[a-z0-9_]+")


def terms(text: str) -> set[str]:
    """Content words, lowercased, short tokens dropped."""
    return {t for t in TOKEN.findall(text.lower()) if len(t) > 2 and t not in STOPWORDS}


def overlap(query: str, document: str) -> float:
    """Share of the query's content words that appear in the document.

    Coverage of the query rather than Jaccard: a long document should not be
    penalised for containing much else, since the question is whether the
    query's words are present at all — which is what a judge reading the pair
    would notice.
    """
    q = terms(query)
    if not q:
        return 0.0
    return len(q & terms(document)) / len(q)


def read_jsonl(name: str) -> list[dict]:
    path = os.path.join(SCRATCH, name)
    if not os.path.exists(path):
        sys.exit(f"missing input: {name} (run this only where the spike scratch exists)")
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def analyse(arm: str) -> None:
    rankings = read_jsonl(f"{arm}-rankings-second.jsonl")
    batches = {b["batch_index"]: b for b in read_jsonl(f"{arm}-judge-batches-second.jsonl")}
    labels = {l["batch_index"]: l["labels"] for l in read_jsonl(f"{arm}-judge-labels-second.jsonl")}

    # Per system: overlap of retrieved rows, and grade beside overlap.
    retrieved_overlap: dict[str, list[float]] = defaultdict(list)
    graded: list[tuple[str, float, int]] = []

    for ranking in rankings:
        index = ranking["batch_index"]
        batch, label = batches.get(index), labels.get(index)
        if batch is None or label is None:
            continue

        query = batch["query"]
        text_of = {item["id"]: item["text"] for item in batch["items"]}
        blind_of = {entry["key"]: entry["blind_id"] for entry in ranking["pool"]}

        for system in ("production", "ane"):
            for hit in ranking[f"{system}_top"]:
                blind = blind_of.get(hit["key"])
                if blind is None or blind not in text_of:
                    continue
                score = overlap(query, text_of[blind])
                retrieved_overlap[system].append(score)
                grade = label.get(blind)
                if grade is not None:
                    graded.append((system, score, int(grade)))

    print(f"\n=== {arm} arm ===")
    print("1. lexical overlap of what each system retrieved")
    for system in ("production", "ane"):
        values = retrieved_overlap[system]
        print(f"   {system:<11} mean {mean(values):.3f}  n={len(values)}")
    difference = mean(retrieved_overlap["ane"]) - mean(retrieved_overlap["production"])
    print(f"   ane minus production: {difference:+.3f}")

    print("2. does the judge reward overlap, holding system constant?")
    for system in ("production", "ane"):
        rows = [(s, g) for sys_, s, g in graded if sys_ == system]
        if len(rows) < 2:
            continue
        xs = [s for s, _ in rows]
        ys = [float(g) for _, g in rows]
        mx, my = mean(xs), mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        vy = math.sqrt(sum((y - my) ** 2 for y in ys))
        r = cov / (vx * vy) if vx and vy else float("nan")
        print(f"   {system:<11} correlation(overlap, grade) = {r:+.3f}  n={len(rows)}")

    print("3. relevance rate by overlap band (grade >= 2 counts as relevant)")
    bands = [(0.0, 0.001, "none"), (0.001, 0.25, "low"), (0.25, 0.5, "mid"), (0.5, 1.01, "high")]
    print(f"   {'band':<8}{'production':>12}{'ane':>10}{'gap':>10}{'n prod':>9}{'n ane':>8}")
    for low, high, name in bands:
        rates = {}
        counts = {}
        for system in ("production", "ane"):
            rows = [g for sys_, s, g in graded if sys_ == system and low <= s < high]
            counts[system] = len(rows)
            rates[system] = mean([1.0 if g >= 2 else 0.0 for g in rows]) if rows else float("nan")
        gap = rates["ane"] - rates["production"]
        print(
            f"   {name:<8}{rates['production']:>12.3f}{rates['ane']:>10.3f}"
            f"{gap:>+10.3f}{counts['production']:>9}{counts['ane']:>8}"
        )


if __name__ == "__main__":
    for arm in ("pure", "hybrid"):
        analyse(arm)
    print(
        "\nReading: if the gap in band 'none' and 'low' is close to the gap overall,\n"
        "the advantage does not depend on surface term overlap. If it collapses\n"
        "there and lives only in 'high', the judge is rewarding what the retrieval\n"
        "is matching and the margin is inflated."
    )
