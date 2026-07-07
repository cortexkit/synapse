from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rerank_eval import load_rerank_scores

DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class RequestMetrics:
    query_id: str
    pair_count: int
    pearson: float | None
    spearman: float | None
    top_1_match: bool
    top_1_tie_aware_match: bool
    top_k_overlap: float
    failure: str | None = None


@dataclass(frozen=True)
class RequestMetadata:
    query_chars: int
    max_document_chars: int
    mean_document_chars: float
    n_documents: int



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rerank score files from two implementations on the same request set."
    )
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args



def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_array = np.asarray(xs, dtype=np.float64)
    y_array = np.asarray(ys, dtype=np.float64)
    if x_array.shape != y_array.shape:
        raise ValueError("pearson inputs must have matching shapes")
    if x_array.size < 2:
        raise ValueError("pearson correlation requires at least two points")
    x_centered = x_array - x_array.mean()
    y_centered = y_array - y_array.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        raise ValueError("pearson correlation is undefined for a constant score series")
    return float(np.dot(x_centered, y_centered) / denominator)



def average_ranks(values: Sequence[float]) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim != 1:
        raise ValueError("average_ranks expects a one-dimensional sequence")
    order = np.argsort(value_array, kind="stable")
    ranks = np.empty(value_array.shape[0], dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and value_array[order[end]] == value_array[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks



def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_ranks = average_ranks(xs)
    y_ranks = average_ranks(ys)
    return pearson_correlation(x_ranks.tolist(), y_ranks.tolist())



def ranked_indices(scores: Sequence[float | None]) -> list[int]:
    return sorted(
        range(len(scores)),
        key=lambda index: (scores[index] is None, -(scores[index] or 0.0), index),
    )



def top_score_tie_set(scores: Sequence[float | None]) -> set[int]:
    numeric_scores = [score for score in scores if score is not None]
    if not numeric_scores:
        raise ValueError("top score tie set requires at least one numeric score")
    top_score = max(numeric_scores)
    return {index for index, score in enumerate(scores) if score == top_score}



def top_k_overlap_ratio(reference_order: Sequence[int], candidate_order: Sequence[int], *, k: int) -> float:
    if len(reference_order) != len(candidate_order):
        raise ValueError("top-k overlap inputs must have matching lengths")
    cutoff = min(k, len(reference_order))
    if cutoff == 0:
        raise ValueError("top-k overlap requires at least one document")
    reference_top = set(reference_order[:cutoff])
    candidate_top = set(candidate_order[:cutoff])
    return len(reference_top & candidate_top) / cutoff



def aligned_score_pairs(
    reference_scores: Sequence[float | None],
    candidate_scores: Sequence[float | None],
) -> tuple[list[float], list[float]]:
    if len(reference_scores) != len(candidate_scores):
        raise ValueError("reference and candidate score lists must have the same length")
    paired_reference: list[float] = []
    paired_candidate: list[float] = []
    for reference_score, candidate_score in zip(reference_scores, candidate_scores, strict=True):
        if reference_score is None or candidate_score is None:
            continue
        paired_reference.append(reference_score)
        paired_candidate.append(candidate_score)
    return paired_reference, paired_candidate



def compare_request_scores(
    query_id: str,
    reference_scores: Sequence[float | None],
    candidate_scores: Sequence[float | None],
    *,
    top_k: int,
) -> RequestMetrics:
    paired_reference, paired_candidate = aligned_score_pairs(reference_scores, candidate_scores)
    reference_order = ranked_indices(reference_scores)
    candidate_order = ranked_indices(candidate_scores)
    reference_top_ties = top_score_tie_set(reference_scores)
    candidate_top_ties = top_score_tie_set(candidate_scores)
    failure: str | None = None
    pearson: float | None = None
    spearman: float | None = None
    try:
        pearson = pearson_correlation(paired_reference, paired_candidate)
        spearman = spearman_correlation(paired_reference, paired_candidate)
    except ValueError as error:
        failure = str(error)
    return RequestMetrics(
        query_id=query_id,
        pair_count=len(paired_reference),
        pearson=pearson,
        spearman=spearman,
        top_1_match=reference_order[0] == candidate_order[0],
        top_1_tie_aware_match=bool(reference_top_ties & candidate_top_ties),
        top_k_overlap=top_k_overlap_ratio(reference_order, candidate_order, k=top_k),
        failure=failure,
    )



def summarize_distribution(values: Sequence[float]) -> dict[str, float | int]:
    value_array = np.asarray(values, dtype=np.float64)
    if value_array.size == 0:
        raise ValueError("distribution summary requires at least one value")
    return {
        "count": int(value_array.size),
        "min": float(value_array.min()),
        "p05": float(np.percentile(value_array, 5)),
        "p50": float(np.percentile(value_array, 50)),
        "mean": float(value_array.mean()),
        "p95": float(np.percentile(value_array, 95)),
        "max": float(value_array.max()),
    }



def load_request_metadata(path: Path) -> dict[str, RequestMetadata]:
    metadata: dict[str, RequestMetadata] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no} must decode to an object")
            query_id = row.get("id")
            query = row.get("query")
            documents = row.get("documents")
            if not isinstance(query_id, str):
                raise TypeError(f"{path}:{line_no} id must be a string")
            if not isinstance(query, str):
                raise TypeError(f"{path}:{line_no} query must be a string")
            if not isinstance(documents, list) or not documents:
                raise TypeError(f"{path}:{line_no} documents must be a non-empty list")
            if not all(isinstance(document, str) for document in documents):
                raise TypeError(f"{path}:{line_no} documents entries must all be strings")
            document_lengths = [len(document) for document in documents]
            metadata[query_id] = RequestMetadata(
                query_chars=len(query),
                max_document_chars=max(document_lengths),
                mean_document_chars=float(sum(document_lengths) / len(document_lengths)),
                n_documents=len(documents),
            )
    return metadata



def compare_score_files(
    *,
    reference_scores_path: Path,
    candidate_scores_path: Path,
    requests_path: Path | None,
    top_k: int,
) -> dict[str, object]:
    reference_scores = load_rerank_scores(reference_scores_path)
    candidate_scores = load_rerank_scores(candidate_scores_path)
    if set(reference_scores) != set(candidate_scores):
        missing_queries = sorted(set(reference_scores) - set(candidate_scores))
        extra_queries = sorted(set(candidate_scores) - set(reference_scores))
        details: list[str] = []
        if missing_queries:
            details.append(f"candidate missing queries {missing_queries[:5]}")
        if extra_queries:
            details.append(f"candidate has extra queries {extra_queries[:5]}")
        raise ValueError("score file/query mismatch: " + "; ".join(details))

    request_metadata = load_request_metadata(requests_path) if requests_path is not None else {}
    compared_requests: list[RequestMetrics] = []
    for query_id in sorted(reference_scores):
        compared_requests.append(
            compare_request_scores(
                query_id,
                reference_scores[query_id],
                candidate_scores[query_id],
                top_k=top_k,
            )
        )

    pair_count = sum(metric.pair_count for metric in compared_requests)
    pearsons = [metric.pearson for metric in compared_requests if metric.pearson is not None]
    spearmans = [metric.spearman for metric in compared_requests if metric.spearman is not None]
    overlap_values = [metric.top_k_overlap for metric in compared_requests]
    failures = [metric for metric in compared_requests if metric.failure is not None]

    report: dict[str, object] = {
        "top_k_overlap_cutoff": top_k,
        "n_requests": len(compared_requests),
        "n_requests_with_valid_correlations": len(pearsons),
        "n_failed_requests": len(failures),
        "n_pairs": pair_count,
        "top_1_match_rate": float(np.mean([metric.top_1_match for metric in compared_requests])),
        "top_1_tie_aware_match_rate": float(np.mean([metric.top_1_tie_aware_match for metric in compared_requests])),
        "mean_top_k_overlap": float(np.mean(overlap_values)),
        "top_k_overlap_per_request": summarize_distribution(overlap_values),
    }
    if pair_count >= 2:
        overall_reference: list[float] = []
        overall_candidate: list[float] = []
        for query_id in sorted(reference_scores):
            paired_reference, paired_candidate = aligned_score_pairs(
                reference_scores[query_id],
                candidate_scores[query_id],
            )
            overall_reference.extend(paired_reference)
            overall_candidate.extend(paired_candidate)
        report["overall_pearson"] = pearson_correlation(overall_reference, overall_candidate)
        report["overall_spearman"] = spearman_correlation(overall_reference, overall_candidate)

    if pearsons:
        report["pearson_per_request"] = summarize_distribution(pearsons)
    if spearmans:
        report["spearman_per_request"] = summarize_distribution(spearmans)
    if failures:
        report["failed_requests"] = [
            {
                "id": metric.query_id,
                "pair_count": metric.pair_count,
                "failure": metric.failure,
            }
            for metric in failures[:10]
        ]

    worst_requests: list[dict[str, object]] = []
    sortable_requests = sorted(
        compared_requests,
        key=lambda metric: (
            metric.pearson if metric.pearson is not None else float("-inf"),
            metric.spearman if metric.spearman is not None else float("-inf"),
            metric.top_k_overlap,
        ),
    )
    for metric in sortable_requests[:5]:
        row: dict[str, object] = {
            "id": metric.query_id,
            "pair_count": metric.pair_count,
            "pearson": metric.pearson,
            "spearman": metric.spearman,
            "top_1_match": metric.top_1_match,
            "top_1_tie_aware_match": metric.top_1_tie_aware_match,
            "top_k_overlap": metric.top_k_overlap,
        }
        if metric.failure is not None:
            row["failure"] = metric.failure
        if metric.query_id in request_metadata:
            metadata = request_metadata[metric.query_id]
            row["query_chars"] = metadata.query_chars
            row["max_document_chars"] = metadata.max_document_chars
            row["mean_document_chars"] = metadata.mean_document_chars
            row["n_documents"] = metadata.n_documents
        worst_requests.append(row)
    report["worst_requests"] = worst_requests
    return report



def main() -> None:
    args = parse_args()
    report = compare_score_files(
        reference_scores_path=args.reference_scores,
        candidate_scores_path=args.candidate_scores,
        requests_path=args.requests,
        top_k=args.top_k,
    )
    encoded = json.dumps(report, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
