from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coir_eval import ndcg_at_k_from_ranked_ids
from rerank_eval import emit_rerank_requests, score_rerank_delta



def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")



def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id, doc_id, relevance in rows:
            handle.write(f"{query_id}\t{doc_id}\t{relevance}\n")



def test_emit_requests_and_score_rerank_delta(tmp_path: Path) -> None:
    corpus_vectors = tmp_path / "corpus-vectors.jsonl"
    query_vectors = tmp_path / "query-vectors.jsonl"
    qrels_path = tmp_path / "toy" / "qrels.tsv"
    queries_jsonl = tmp_path / "queries.jsonl"
    corpus_jsonl = tmp_path / "corpus.jsonl"
    rerank_requests = tmp_path / "rerank-requests.jsonl"
    rerank_scores = tmp_path / "rerank-scores.jsonl"
    qrels_path.parent.mkdir(parents=True)

    write_jsonl(
        corpus_vectors,
        [
            {"id": "d1", "vec": [1.0, 0.0]},
            {"id": "d2", "vec": [0.8, 0.6]},
            {"id": "d3", "vec": [0.0, 1.0]},
        ],
    )
    write_jsonl(
        query_vectors,
        [
            {"id": "q1", "vec": [0.6, 0.8]},
        ],
    )
    write_qrels(
        qrels_path,
        [
            ("q1", "d1", 3),
            ("q1", "d2", 1),
        ],
    )
    write_jsonl(
        queries_jsonl,
        [
            {"id": "q1", "text": "find the best matching answer"},
        ],
    )
    write_jsonl(
        corpus_jsonl,
        [
            {"id": "d1", "text": "gold answer"},
            {"id": "d2", "text": "partial answer"},
            {"id": "d3", "text": "distractor"},
        ],
    )
    write_jsonl(
        rerank_scores,
        [
            {"id": "q1", "scores": [0.2, 0.1, 0.9]},
        ],
    )

    emit_rerank_requests(
        corpus_vectors_path=corpus_vectors,
        query_vectors_path=query_vectors,
        qrels_path=qrels_path,
        queries_jsonl_path=queries_jsonl,
        corpus_jsonl_path=corpus_jsonl,
        top_k=3,
        out_path=rerank_requests,
    )

    emitted_rows = [json.loads(line) for line in rerank_requests.read_text(encoding="utf-8").splitlines()]
    assert emitted_rows == [
        {
            "id": "q1",
            "query": "find the best matching answer",
            "documents": ["partial answer", "distractor", "gold answer"],
            "document_ids": ["d2", "d3", "d1"],
        }
    ]

    report = score_rerank_delta(
        corpus_vectors_path=corpus_vectors,
        query_vectors_path=query_vectors,
        qrels_path=qrels_path,
        rerank_scores_path=rerank_scores,
        top_k=3,
    )

    dense_ndcg = ndcg_at_k_from_ranked_ids({"d1": 3, "d2": 1}, ["d2", "d3", "d1"], 10)
    reranked_ndcg = ndcg_at_k_from_ranked_ids({"d1": 3, "d2": 1}, ["d1", "d2", "d3"], 10)

    assert report["task"] == "toy"
    assert report["k"] == 3
    assert report["n_queries"] == 1
    assert math.isclose(cast(float, report["ndcg_dense"]), dense_ndcg, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(cast(float, report["ndcg_reranked"]), reranked_ndcg, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(
        cast(float, report["delta"]),
        reranked_ndcg - dense_ndcg,
        rel_tol=0,
        abs_tol=1e-9,
    )
