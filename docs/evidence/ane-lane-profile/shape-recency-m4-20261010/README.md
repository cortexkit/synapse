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
