# Shape-recency experiment, M4 Mac mini, 2026-10-10

Three fresh processes of `crates/synapse-worker-ane-direct/examples/shape_recency.rs`
(commit a7a1c297), each timing 64 identical 100-token rows on the Qwen3 128-token
shape, with `TMPDIR` and `SYNAPSE_ANE_PROFILE_DIR` unset. Raw output: `run-1.json`
to `run-3.json`.

Per-row medians in ms (run 1, run 2, run 3), after one untimed first row per phase:

| Phase | Medians | p90s | First row |
| --- | --- | --- | --- |
| a: after compiling 128 | 21.20, 21.06, 20.94 | 22.46, 21.52, 21.10 | 40.5, 37.4, 38.6 |
| d: again, no compile | 20.98, 21.28, 20.93 | 21.18, 21.49, 21.06 | 21.4, 21.1, 21.0 |
| b: after compiling 512 and 1024, 128 resident | 21.60, 21.08, 21.00 | 23.77, 23.48, 22.09 | 946.6, 1088.2, 834.3 |
| c: after evicting and recompiling 128 | 20.59, 20.39, 21.15 | 21.32, 20.83, 21.63 | 86.2, 36.2, 33.9 |
| c_control: again, no compile | 21.03, 20.28, 20.93 | 21.54, 20.55, 21.15 | 21.3, 22.1, 20.5 |

Every output vector in every phase and run has the same SHA-256 (`bacc3cc8b50b…`).

Steady per-row time on the 128 shape does not depend on compile order or recency:
phase b matches a and d. What does change is the first use of the older shape after
the larger compiles: about one second (834–1088 ms) instead of about 40 ms, after
which rows are back to about 21 ms. A recompile (phase c) avoids that first-row cost
but takes 11–17 s itself.

## Paired A/B: shape-compile lane change against master

This experiment followed two single runs of the profile harness in which the 64-row
calls looked slower with the compile-lane change than without it. Four alternating
rounds of the same harness (`crates/synapse-module/examples/aft_embed_headtohead.rs`
with `SYNAPSE_ANE_PROFILE_CATALOG=1` and `SYNAPSE_ANE_PROFILE_SETTLE_MS=20000`), with
master and the change built from the same tree apart from the change, gave these
wall times in ms:

| Call | Master (4 rounds) | Median | With the change (4 rounds) | Median |
| --- | --- | --- | --- | --- |
| 64 rows, after the incident compiles | 2018, 1844, 1833, 4366 | 1931 | 1688, 1837, 4092, 2170 | 2004 |
| 64 rows, warm | 1532, 1501, 1412, 1724 | 1517 | 1511, 1411, 1362, 3063 | 1461 |
| two concurrent 64-row calls, first | 2873, 3261, 2483, 2948 | 2911 | 2613, 2602, 2633, 3098 | 2623 |
| two concurrent 64-row calls, second | 3364, 3528, 2942, 3149 | 3257 | 2784, 2771, 2801, 3291 | 2793 |

Both sides have occasional runs near twice the median, so the earlier single-run
slowdown was run-to-run variance on the Neural Engine, not the change. With the
change, concurrent calls finish about 10–14% sooner and single calls are unchanged.
