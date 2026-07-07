from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import numpy as np
import pytrec_eval  # pyright: ignore[reportMissingImports]
from datasets import load_dataset

COSQA_DATASET = "CoIR-Retrieval/cosqa"
DEFAULT_K = 10


@dataclass(frozen=True)
class TextRow:
    id: str
    text: str


@dataclass(frozen=True)
class VectorTable:
    ids: list[str]
    matrix: np.ndarray


def prepare_cosqa(out_dir: Path) -> dict[str, int]:
    corpus_ds = load_dataset(COSQA_DATASET, name="corpus", split="corpus")
    queries_ds = load_dataset(COSQA_DATASET, name="queries", split="queries")
    qrels_ds = load_dataset(COSQA_DATASET, split="test")

    corpus_rows = sorted(
        (text_row_from_record(record) for record in corpus_ds),
        key=lambda row: row.id,
    )
    query_rows = sorted(
        (
            text_row_from_record(record)
            for record in queries_ds
            if record_value(record, "partition") == "test"
        ),
        key=lambda row: row.id,
    )
    qrels_rows = sorted(
        (qrel_row_from_record(record) for record in qrels_ds),
        key=lambda row: (row[0], row[1], row[2]),
    )

    query_ids = {row.id for row in query_rows}
    qrel_query_ids = {query_id for query_id, _, _ in qrels_rows}
    if query_ids != qrel_query_ids:
        missing_queries = sorted(qrel_query_ids - query_ids)
        extra_queries = sorted(query_ids - qrel_query_ids)
        details: list[str] = []
        if missing_queries:
            details.append(f"missing queries for qrels: {missing_queries[:5]}")
        if extra_queries:
            details.append(f"queries without qrels: {extra_queries[:5]}")
        raise ValueError("query/qrels mismatch: " + "; ".join(details))

    corpus_ids = {row.id for row in corpus_rows}
    missing_docs = sorted({doc_id for _, doc_id, _ in qrels_rows} - corpus_ids)
    if missing_docs:
        raise ValueError(
            f"qrels reference corpus ids that are missing from the corpus split: {missing_docs[:5]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_text_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    write_text_jsonl(out_dir / "queries.jsonl", query_rows)
    write_qrels_tsv(out_dir / "qrels.tsv", qrels_rows)

    return {
        "n_corpus": len(corpus_rows),
        "n_queries": len(query_rows),
        "n_qrels": len(qrels_rows),
    }


def text_row_from_record(record: object) -> TextRow:
    row = as_mapping(record)
    row_id = record_value(row, "_id")
    text = record_value(row, "text")
    if not isinstance(row_id, str):
        raise TypeError(f"dataset _id must be a string, got {type(row_id).__name__}")
    if not isinstance(text, str):
        raise TypeError(f"dataset text must be a string, got {type(text).__name__}")
    return TextRow(id=row_id, text=text)


def qrel_row_from_record(record: object) -> tuple[str, str, int]:
    row = as_mapping(record)
    query_id = record_value(row, "query-id")
    doc_id = record_value(row, "corpus-id")
    score = record_value(row, "score")
    if not isinstance(query_id, str):
        raise TypeError(f"dataset query-id must be a string, got {type(query_id).__name__}")
    if not isinstance(doc_id, str):
        raise TypeError(f"dataset corpus-id must be a string, got {type(doc_id).__name__}")
    if not isinstance(score, int):
        raise TypeError(f"dataset score must be an integer, got {type(score).__name__}")
    return query_id, doc_id, score


def as_mapping(record: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], record)


def record_value(record: object, key: str) -> Any:
    return as_mapping(record)[key]


def write_text_jsonl(path: Path, rows: Iterable[TextRow]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row.id, "text": row.text}, ensure_ascii=False))
            handle.write("\n")


def write_qrels_tsv(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id, doc_id, relevance in rows:
            handle.write(f"{query_id}\t{doc_id}\t{relevance}\n")


def load_vectors(path: Path) -> VectorTable:
    rows_by_id: dict[str, np.ndarray] = {}
    dims: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError(f"{path}:{line_no} must decode to an object")
            row_id = data.get("id")
            vec = data.get("vec")
            if not isinstance(row_id, str):
                raise TypeError(f"{path}:{line_no} id must be a string")
            if row_id in rows_by_id:
                raise ValueError(f"{path}:{line_no} duplicate vector id {row_id!r}")
            if not isinstance(vec, list) or not vec:
                raise TypeError(f"{path}:{line_no} vec must be a non-empty list")
            array = np.asarray(vec, dtype=np.float32)
            if array.ndim != 1:
                raise TypeError(f"{path}:{line_no} vec must be one-dimensional")
            if dims is None:
                dims = int(array.shape[0])
            elif int(array.shape[0]) != dims:
                raise ValueError(
                    f"{path}:{line_no} vector dimension {int(array.shape[0])} does not match expected {dims}"
                )
            rows_by_id[row_id] = array

    if not rows_by_id:
        raise ValueError(f"no vectors loaded from {path}")

    ids = sorted(rows_by_id)
    matrix = np.vstack([rows_by_id[row_id] for row_id in ids])
    return VectorTable(ids=ids, matrix=l2_normalize(matrix, path))


def l2_normalize(matrix: np.ndarray, path: Path | str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        zero_rows = np.flatnonzero(norms.ravel() == 0)
        raise ValueError(f"{path} contains zero-length vectors at rows {zero_rows[:5].tolist()}")
    return matrix / norms


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no} must contain exactly three tab-separated fields")
            query_id, doc_id, relevance_text = parts
            try:
                relevance = int(relevance_text)
            except ValueError as error:
                raise ValueError(f"{path}:{line_no} relevance must be an integer") from error
            if doc_id in qrels[query_id]:
                raise ValueError(f"{path}:{line_no} duplicate qrel for query {query_id!r} and doc {doc_id!r}")
            qrels[query_id][doc_id] = relevance

    if not qrels:
        raise ValueError(f"no qrels loaded from {path}")

    return {query_id: docs for query_id, docs in sorted(qrels.items())}


def infer_task_label(qrels_path: Path) -> str:
    if qrels_path.name == "qrels.tsv" and qrels_path.parent.name:
        return qrels_path.parent.name
    return qrels_path.stem


def score_task(
    *,
    corpus_vectors_path: Path,
    query_vectors_path: Path,
    qrels_path: Path,
    k: int = DEFAULT_K,
) -> dict[str, float | int | str]:
    if k <= 0:
        raise ValueError("k must be positive")

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

    run, ranked_doc_ids = brute_force_search(corpus_vectors=corpus_vectors, query_vectors=query_vectors, k=k)
    metrics = evaluate_run(qrels=qrels, run=run, ranked_doc_ids=ranked_doc_ids, k=k)
    return {
        "task": infer_task_label(qrels_path),
        "ndcg_at_10": metrics[f"ndcg_at_{k}"],
        "recall_at_10": metrics[f"recall_at_{k}"],
        "mrr_at_10": metrics[f"mrr_at_{k}"],
        "n_queries": len(query_vectors.ids),
        "n_corpus": len(corpus_vectors.ids),
    }


def brute_force_search(
    *, corpus_vectors: VectorTable, query_vectors: VectorTable, k: int
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    score_matrix = query_vectors.matrix @ corpus_vectors.matrix.T
    run: dict[str, dict[str, float]] = {}
    ranked_doc_ids: dict[str, list[str]] = {}

    for row_index, query_id in enumerate(query_vectors.ids):
        row_scores = score_matrix[row_index]
        top_indices = np.argsort(-row_scores, kind="stable")[: min(k, len(corpus_vectors.ids))]
        ranked_ids = [corpus_vectors.ids[index] for index in top_indices]
        ranked_doc_ids[query_id] = ranked_ids
        run[query_id] = {corpus_vectors.ids[index]: float(row_scores[index]) for index in top_indices}

    return run, ranked_doc_ids


def evaluate_run(
    *,
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    ranked_doc_ids: dict[str, list[str]],
    k: int,
) -> dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{k}", f"recall_{k}"})
    per_query = evaluator.evaluate(run)
    ordered_queries = sorted(qrels)

    ndcg = float(np.mean([per_query[query_id][f"ndcg_cut_{k}"] for query_id in ordered_queries]))
    recall = float(np.mean([per_query[query_id][f"recall_{k}"] for query_id in ordered_queries]))
    mrr = float(
        np.mean([reciprocal_rank_at_k(qrels[query_id], ranked_doc_ids[query_id], k) for query_id in ordered_queries])
    )

    return {
        f"ndcg_at_{k}": ndcg,
        f"recall_at_{k}": recall,
        f"mrr_at_{k}": mrr,
    }


def reciprocal_rank_at_k(qrels_for_query: dict[str, int], ranked_doc_ids: list[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels_for_query.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_at_k(
    qrels: dict[str, dict[str, int]], ranked_doc_ids: dict[str, list[str]], k: int
) -> float:
    ordered_queries = sorted(qrels)
    if not ordered_queries:
        return 0.0
    return sum(
        reciprocal_rank_at_k(qrels[query_id], ranked_doc_ids[query_id], k) for query_id in ordered_queries
    ) / len(ordered_queries)


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(relevance / math.log2(rank + 1) for rank, relevance in enumerate(relevances[:k], start=1))


def ndcg_at_k_from_ranked_ids(qrels_for_query: dict[str, int], ranked_doc_ids: list[str], k: int) -> float:
    observed = [qrels_for_query.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted((relevance for relevance in qrels_for_query.values() if relevance > 0), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(observed, k) / ideal_dcg
