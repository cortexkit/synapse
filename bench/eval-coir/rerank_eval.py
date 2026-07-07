from __future__ import annotations

import argparse
import json
from pathlib import Path

from coir_eval import (
    brute_force_search,
    evaluate_run,
    infer_task_label,
    load_qrels,
    load_vectors,
    score_task,
)

METRIC_K = 10
DEFAULT_TOP_K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit rerank requests from dense retrieval or score reranked top-k "
            "documents against qrels."
        )
    )
    parser.add_argument("--corpus-vectors", type=Path, required=True)
    parser.add_argument("--query-vectors", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path)
    parser.add_argument("--corpus-jsonl", type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--emit-rerank-requests", type=Path)
    parser.add_argument("--rerank-scores", type=Path)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.emit_rerank_requests is None and args.rerank_scores is None:
        parser.error("pass --emit-rerank-requests and/or --rerank-scores")
    if args.emit_rerank_requests is not None and args.queries_jsonl is None:
        parser.error("--queries-jsonl is required when emitting rerank requests")
    if args.emit_rerank_requests is not None and args.corpus_jsonl is None:
        parser.error("--corpus-jsonl is required when emitting rerank requests")

    return args


def load_texts(path: Path) -> dict[str, str]:
    rows_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError(f"{path}:{line_no} must decode to an object")
            row_id = data.get("id")
            text = data.get("text")
            if not isinstance(row_id, str):
                raise TypeError(f"{path}:{line_no} id must be a string")
            if not isinstance(text, str):
                raise TypeError(f"{path}:{line_no} text must be a string")
            if row_id in rows_by_id:
                raise ValueError(f"{path}:{line_no} duplicate text id {row_id!r}")
            rows_by_id[row_id] = text

    if not rows_by_id:
        raise ValueError(f"no text rows loaded from {path}")
    return rows_by_id


def load_dense_top_k(
    *, corpus_vectors_path: Path, query_vectors_path: Path, qrels_path: Path, top_k: int
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    corpus_vectors = load_vectors(corpus_vectors_path)
    query_vectors = load_vectors(query_vectors_path)
    qrels = load_qrels(qrels_path)

    qrel_query_ids = set(qrels)
    query_ids = set(query_vectors.ids)
    if query_ids != qrel_query_ids:
        missing_queries = sorted(qrel_query_ids - query_ids)
        extra_queries = sorted(query_ids - qrel_query_ids)
        details: list[str] = []
        if missing_queries:
            details.append(f"missing query vectors for {missing_queries[:5]}")
        if extra_queries:
            details.append(f"query vectors without qrels: {extra_queries[:5]}")
        raise ValueError("query vector/qrels mismatch: " + "; ".join(details))

    corpus_ids = set(corpus_vectors.ids)
    missing_docs = sorted(
        {
            doc_id
            for docs in qrels.values()
            for doc_id, relevance in docs.items()
            if relevance > 0 and doc_id not in corpus_ids
        }
    )
    if missing_docs:
        raise ValueError(
            f"qrels reference corpus ids that are missing from the corpus vectors: {missing_docs[:5]}"
        )

    _, dense_ranked_doc_ids = brute_force_search(
        corpus_vectors=corpus_vectors,
        query_vectors=query_vectors,
        k=top_k,
    )
    return qrels, dense_ranked_doc_ids


def emit_rerank_requests(
    *,
    corpus_vectors_path: Path,
    query_vectors_path: Path,
    qrels_path: Path,
    queries_jsonl_path: Path,
    corpus_jsonl_path: Path,
    top_k: int,
    out_path: Path,
) -> None:
    query_texts = load_texts(queries_jsonl_path)
    corpus_texts = load_texts(corpus_jsonl_path)
    qrels, dense_ranked_doc_ids = load_dense_top_k(
        corpus_vectors_path=corpus_vectors_path,
        query_vectors_path=query_vectors_path,
        qrels_path=qrels_path,
        top_k=top_k,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in sorted(qrels):
            if query_id not in query_texts:
                raise ValueError(f"query text missing for rerank request {query_id!r}")
            document_ids = dense_ranked_doc_ids[query_id]
            try:
                documents = [corpus_texts[doc_id] for doc_id in document_ids]
            except KeyError as error:
                missing_doc_id = error.args[0]
                raise ValueError(f"corpus text missing for rerank request doc {missing_doc_id!r}") from error
            handle.write(
                json.dumps(
                    {
                        "id": query_id,
                        "query": query_texts[query_id],
                        "documents": documents,
                        "document_ids": document_ids,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def load_rerank_scores(path: Path) -> dict[str, list[float | None]]:
    rows_by_id: dict[str, list[float | None]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError(f"{path}:{line_no} must decode to an object")
            row_id = data.get("id")
            scores = data.get("scores")
            if not isinstance(row_id, str):
                raise TypeError(f"{path}:{line_no} id must be a string")
            if row_id in rows_by_id:
                raise ValueError(f"{path}:{line_no} duplicate rerank score id {row_id!r}")
            if not isinstance(scores, list) or not scores:
                raise TypeError(f"{path}:{line_no} scores must be a non-empty list")
            normalized_scores: list[float | None] = []
            for score_index, score in enumerate(scores):
                if score is None:
                    normalized_scores.append(None)
                elif isinstance(score, (int, float)):
                    normalized_scores.append(float(score))
                else:
                    raise TypeError(
                        f"{path}:{line_no} scores[{score_index}] must be a number or null"
                    )
            rows_by_id[row_id] = normalized_scores

    if not rows_by_id:
        raise ValueError(f"no rerank scores loaded from {path}")
    return rows_by_id


def rerank_doc_ids(dense_doc_ids: list[str], scores: list[float | None]) -> list[str]:
    if len(scores) != len(dense_doc_ids):
        raise ValueError(
            f"rerank scores length {len(scores)} does not match dense document count {len(dense_doc_ids)}"
        )
    ranked_indices = sorted(
        range(len(dense_doc_ids)),
        key=lambda index: (scores[index] is None, -(scores[index] or 0.0), index),
    )
    return [dense_doc_ids[index] for index in ranked_indices]


def run_from_ranked_doc_ids(ranked_doc_ids: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    return {
        query_id: {
            doc_id: float(len(doc_ids) - index)
            for index, doc_id in enumerate(doc_ids)
        }
        for query_id, doc_ids in ranked_doc_ids.items()
    }


def score_rerank_delta(
    *,
    corpus_vectors_path: Path,
    query_vectors_path: Path,
    qrels_path: Path,
    rerank_scores_path: Path,
    top_k: int,
) -> dict[str, float | int | str]:
    qrels, dense_ranked_doc_ids = load_dense_top_k(
        corpus_vectors_path=corpus_vectors_path,
        query_vectors_path=query_vectors_path,
        qrels_path=qrels_path,
        top_k=top_k,
    )
    rerank_scores = load_rerank_scores(rerank_scores_path)

    dense_query_ids = set(dense_ranked_doc_ids)
    rerank_query_ids = set(rerank_scores)
    if dense_query_ids != rerank_query_ids:
        missing_queries = sorted(dense_query_ids - rerank_query_ids)
        extra_queries = sorted(rerank_query_ids - dense_query_ids)
        details: list[str] = []
        if missing_queries:
            details.append(f"missing rerank scores for {missing_queries[:5]}")
        if extra_queries:
            details.append(f"rerank scores without dense candidates: {extra_queries[:5]}")
        raise ValueError("rerank score/query mismatch: " + "; ".join(details))

    reranked_doc_ids = {
        query_id: rerank_doc_ids(dense_ranked_doc_ids[query_id], rerank_scores[query_id])
        for query_id in sorted(qrels)
    }
    reranked_run = run_from_ranked_doc_ids(reranked_doc_ids)

    dense_report = score_task(
        corpus_vectors_path=corpus_vectors_path,
        query_vectors_path=query_vectors_path,
        qrels_path=qrels_path,
        k=METRIC_K,
    )
    reranked_metrics = evaluate_run(
        qrels=qrels,
        run=reranked_run,
        ranked_doc_ids=reranked_doc_ids,
        k=METRIC_K,
    )
    ndcg_dense = float(dense_report["ndcg_at_10"])
    ndcg_reranked = reranked_metrics[f"ndcg_at_{METRIC_K}"]
    return {
        "task": infer_task_label(qrels_path),
        "k": top_k,
        "ndcg_dense": ndcg_dense,
        "ndcg_reranked": ndcg_reranked,
        "delta": ndcg_reranked - ndcg_dense,
        "n_queries": len(qrels),
    }


def main() -> None:
    args = parse_args()
    if args.emit_rerank_requests is not None:
        emit_rerank_requests(
            corpus_vectors_path=args.corpus_vectors,
            query_vectors_path=args.query_vectors,
            qrels_path=args.qrels,
            queries_jsonl_path=args.queries_jsonl,
            corpus_jsonl_path=args.corpus_jsonl,
            top_k=args.top_k,
            out_path=args.emit_rerank_requests,
        )
    if args.rerank_scores is not None:
        report = score_rerank_delta(
            corpus_vectors_path=args.corpus_vectors,
            query_vectors_path=args.query_vectors,
            qrels_path=args.qrels,
            rerank_scores_path=args.rerank_scores,
            top_k=args.top_k,
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
