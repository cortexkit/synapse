from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import cast

import pytrec_eval  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coir_eval import mean_reciprocal_rank_at_k, ndcg_at_k_from_ranked_ids, score_task


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id, doc_id, relevance in rows:
            handle.write(f"{query_id}\t{doc_id}\t{relevance}\n")


def test_score_task_matches_pytrec_eval_and_hand_computed_ndcg(tmp_path: Path) -> None:
    corpus_vectors = tmp_path / "corpus-vectors.jsonl"
    query_vectors = tmp_path / "query-vectors.jsonl"
    qrels_path = tmp_path / "toy" / "qrels.tsv"
    qrels_path.parent.mkdir(parents=True)

    write_jsonl(
        corpus_vectors,
        [
            {"id": "d1", "vec": [1.0, 0.0, 0.0]},
            {"id": "d2", "vec": [0.0, 1.0, 0.0]},
            {"id": "d3", "vec": [0.0, 0.0, 1.0]},
        ],
    )
    write_jsonl(
        query_vectors,
        [
            {"id": "q1", "vec": [1.0, 0.0, 0.0]},
            {"id": "q2", "vec": [0.6, 0.8, 0.0]},
            {"id": "q3", "vec": [0.0, 0.8, 0.6]},
        ],
    )
    write_qrels(
        qrels_path,
        [
            ("q1", "d1", 3),
            ("q1", "d2", 2),
            ("q2", "d1", 1),
            ("q3", "d2", 1),
            ("q3", "d3", 3),
        ],
    )

    report = score_task(
        corpus_vectors_path=corpus_vectors,
        query_vectors_path=query_vectors,
        qrels_path=qrels_path,
        k=2,
    )

    run = {
        "q1": {"d1": 1.0, "d2": 0.0},
        "q2": {"d2": 0.8, "d1": 0.6},
        "q3": {"d2": 0.8, "d3": 0.6},
    }
    qrels = {
        "q1": {"d1": 3, "d2": 2},
        "q2": {"d1": 1},
        "q3": {"d2": 1, "d3": 3},
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_2", "recall_2"})
    per_query = evaluator.evaluate(run)
    expected_ndcg = sum(per_query[query_id]["ndcg_cut_2"] for query_id in sorted(qrels)) / 3
    expected_recall = sum(per_query[query_id]["recall_2"] for query_id in sorted(qrels)) / 3

    ranked_doc_ids = {
        "q1": ["d1", "d2"],
        "q2": ["d2", "d1"],
        "q3": ["d2", "d3"],
    }
    hand_ndcg = (
        ndcg_at_k_from_ranked_ids(qrels["q1"], ranked_doc_ids["q1"], 2)
        + ndcg_at_k_from_ranked_ids(qrels["q2"], ranked_doc_ids["q2"], 2)
        + ndcg_at_k_from_ranked_ids(qrels["q3"], ranked_doc_ids["q3"], 2)
    ) / 3
    expected_mrr = mean_reciprocal_rank_at_k(qrels, ranked_doc_ids, 2)

    assert report["task"] == "toy"
    assert report["n_corpus"] == 3
    assert report["n_queries"] == 3
    ndcg = cast(float, report["ndcg_at_10"])
    recall = cast(float, report["recall_at_10"])
    mrr = cast(float, report["mrr_at_10"])

    assert math.isclose(ndcg, expected_ndcg, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(ndcg, hand_ndcg, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(recall, expected_recall, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mrr, expected_mrr, rel_tol=0, abs_tol=1e-9)
