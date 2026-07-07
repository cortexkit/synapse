from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_rerank_scores import (  # noqa: E402
    aligned_score_pairs,
    compare_request_scores,
    pearson_correlation,
    spearman_correlation,
    top_k_overlap_ratio,
)



def test_correlations_match_expected_orderings() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [10.0, 20.0, 30.0, 40.0]
    reversed_ys = list(reversed(ys))

    assert math.isclose(pearson_correlation(xs, ys), 1.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(spearman_correlation(xs, ys), 1.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(pearson_correlation(xs, reversed_ys), -1.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(spearman_correlation(xs, reversed_ys), -1.0, rel_tol=0, abs_tol=1e-9)



def test_aligned_score_pairs_skip_nulls() -> None:
    paired_reference, paired_candidate = aligned_score_pairs(
        [0.9, None, 0.4, 0.2],
        [0.8, 0.7, None, 0.1],
    )

    assert paired_reference == [0.9, 0.2]
    assert paired_candidate == [0.8, 0.1]



def test_compare_request_scores_reports_rank_agreement() -> None:
    metrics = compare_request_scores(
        "q1",
        [0.8, 0.6, 0.2, 0.1],
        [0.79, 0.58, 0.19, 0.08],
        top_k=3,
    )

    assert metrics.query_id == "q1"
    assert metrics.pair_count == 4
    assert metrics.failure is None
    assert metrics.top_1_match is True
    assert metrics.top_1_tie_aware_match is True
    assert math.isclose(metrics.top_k_overlap, 1.0, rel_tol=0, abs_tol=1e-9)
    assert metrics.pearson is not None and metrics.pearson > 0.99
    assert metrics.spearman is not None and metrics.spearman > 0.99



def test_top_k_overlap_ratio_uses_set_overlap() -> None:
    overlap = top_k_overlap_ratio([0, 1, 2, 3], [2, 1, 0, 3], k=3)
    assert math.isclose(overlap, 1.0, rel_tol=0, abs_tol=1e-9)



def test_compare_request_scores_tie_aware_top_1_handles_near_ties() -> None:
    metrics = compare_request_scores(
        "q2",
        [0.5, 0.500001, 0.1],
        [0.5, 0.5, 0.0],
        top_k=2,
    )

    assert metrics.top_1_match is False
    assert metrics.top_1_tie_aware_match is True
