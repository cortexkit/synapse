#!/usr/bin/env python3
"""Neighbour-ranking quality of f16 Metal graphs against an f32 reference.

Usage: [QUALITY_CHECK_SEED=N] quality_check.py REFERENCE.f32 NAME=LANE.f32 [NAME=LANE.f32 ...]

Each file is a metal_fixed_cost `.vectors.f32` dump: little-endian f32, one
768-wide L2-normalized row per engram replay row, in input order. 500 query
rows are drawn with a fixed seed (20261010 unless QUALITY_CHECK_SEED is set); each is ranked against every other row
(itself excluded) by cosine similarity, which for unit vectors is the dot
product, computed in f64.

For each lane, against the reference:
- top-10 overlap: |lane top-10 ∩ reference top-10| / 10 per query; mean and p5.
- order swaps above a margin: for each query, take the union of both top-10
  sets; count item pairs (a, b) the reference orders a above b by more than the
  margin while the lane orders b above a. Reported for margins 1e-4 and 5e-4.
- mean |similarity error|: mean over all query x corpus pairs of
  |lane similarity - reference similarity|.
"""

import os
import sys

import numpy as np

SEED = int(os.environ.get("QUALITY_CHECK_SEED", "20261010"))
QUERIES = 500
K = 10
DIM = 768


def load(path):
    vectors = np.fromfile(path, dtype="<f4").reshape(-1, DIM).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def similarities(vectors, queries):
    with np.errstate(all="ignore"):
        # Some numpy builds raise spurious floating-point flags inside matmul.
        sims = vectors[queries] @ vectors.T
    sims[np.arange(len(queries)), queries] = -np.inf
    return sims


def top_k(sims):
    # Stable order: higher similarity first, then lower row index.
    order = np.lexsort((np.arange(sims.shape[1])[None, :].repeat(len(sims), 0), -sims), axis=1)
    return order[:, :K]


def measure(reference_sims, reference_top, lane_sims):
    lane_top = top_k(lane_sims)
    overlaps = np.array(
        [len(set(a) & set(b)) / K for a, b in zip(lane_top, reference_top)]
    )
    swaps = {1e-4: 0, 5e-4: 0}
    for row in range(len(lane_sims)):
        items = np.array(sorted(set(lane_top[row]) | set(reference_top[row])))
        ref = reference_sims[row, items]
        lane = lane_sims[row, items]
        ref_gap = ref[:, None] - ref[None, :]
        lane_gap = lane[:, None] - lane[None, :]
        for margin in swaps:
            swaps[margin] += int(np.sum((ref_gap > margin) & (lane_gap < 0)))
    finite = np.isfinite(reference_sims)
    error = np.abs(lane_sims[finite] - reference_sims[finite]).mean()
    return {
        "top10_mean": overlaps.mean(),
        "top10_p5": np.quantile(overlaps, 0.05),
        "swaps_gt_1e-4": swaps[1e-4],
        "swaps_gt_5e-4": swaps[5e-4],
        "mean_abs_sim_error": error,
    }


def main():
    reference = load(sys.argv[1])
    queries = np.random.default_rng(SEED).choice(len(reference), QUERIES, replace=False)
    reference_sims = similarities(reference, queries)
    reference_top = top_k(reference_sims)
    print(f"rows {len(reference)}, queries {QUERIES} (seed {SEED}), k {K}")
    print(
        f"{'lane':<28} {'top10 mean':>11} {'top10 p5':>9} {'swaps>1e-4':>11} "
        f"{'swaps>5e-4':>11} {'mean|dsim|':>11}"
    )
    for argument in sys.argv[2:]:
        name, path = argument.split("=", 1)
        lane = load(path)
        assert lane.shape == reference.shape, name
        result = measure(reference_sims, reference_top, similarities(lane, queries))
        print(
            f"{name:<28} {result['top10_mean']:>11.6f} {result['top10_p5']:>9.3f} "
            f"{result['swaps_gt_1e-4']:>11d} {result['swaps_gt_5e-4']:>11d} "
            f"{result['mean_abs_sim_error']:>11.3e}"
        )


if __name__ == "__main__":
    main()
