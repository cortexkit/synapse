# Second NVIDIA progress-liveness rental

## Revisions and recipe

Candidate: `be6cd118a40cc8cc5e999fad9e1e880e43dcd644`. Baseline:
`5f6d7419fc42a5ebf6ed1b8eab2115b6cc84fe5a`. The candidate's
`crates/synapse-engine-cuda` tree is identical to the baseline (empty
`git diff 5f6d7419 be6cd118 -- crates/synapse-engine-cuda`).

The recipe comes from `db3b0176cf619f5f45c9e447dc74cea4ccc9b117`, under
`docs/evidence/worker-progress-liveness/cuda-rental/`. The retained tools are
copied from that revision with only the candidate SHA, rental instance ID,
previously required X/GLVND dependencies and Mesa ICD removal, and separate
Cargo targets per revision made explicit. The redundant rebuild script is
omitted. The timing probe and mirror are unchanged. No tolerance, request
budget, timeout, or certification configuration is changed.

One RTX 5090 rental used `nvidia/cuda:12.8.1-base-ubuntu24.04`, 100 GB
local disk, all NVIDIA driver capabilities, checksum-pinned CUDA 13.2.1
redistributables and Vulkan SDK 1.4.357.0. Eight candidate certification cells
and paired baseline/candidate CUDA timings used the original recipe and
byte-pinned mirror. The baseline was also built on the rental, although its
GPU and driver match the previous rental.

The hard spend cap is $3.00. A local watchdog was armed to destroy the instance at two
hours or a $2.00 observed account debit, leaving headroom for delayed billing
and bandwidth. Cleanup must confirm the account instance list contains none
created by this run. Raw logs and source bundles stay outside git under
`evidence/cuda-rental-2/`; their hashes and compact results will be committed
here. Initial account resources: no instances.

**Result: all eight candidate cells passed, including first-load self-check,
fingerprint pins, fixture parity, and both admission boundaries.** The four
same-machine master CUDA controls passed too. No candidate runtime failure,
tolerance change, timeout change, or bypass was needed. Instance `55712356`
was destroyed and the account instance list returned `[]`.

## Machine and certification results

RTX 5090, driver **595.84**, CUDA driver API **13020**, compute capability 12.0.
This matches the previous rental's GPU class and driver. The same-machine master
control additionally accounts for host/CPU differences. Rust and Cargo were
1.99.0; release builds used eight jobs. CUDA compiler V13.2.78 and Vulkan SDK
1.4.357.0 came from the recipe's unchanged checksum pins. Floor probes passed
for both builds and both backends; no software Vulkan device was certified.

| Candidate cell | Status | Fingerprint = catalog pin | Fixtures | 8192 tokens | 8193 tokens |
| --- | --- | --- | ---: | --- | --- |
| GTE embed / CUDA | passed | yes | 17/17 | processed, 1 worker request | sequence_too_long, 0 worker requests |
| GTE rerank / CUDA | passed | yes | 125/125 | processed, 1 worker request | sequence_too_long, 0 worker requests |
| Qwen3 embed / CUDA | passed | yes | 17/17 | processed, 1 worker request | sequence_too_long, 0 worker requests |
| Qwen3 rerank / CUDA | passed | yes | 125/125 | processed, 1 worker request | sequence_too_long, 0 worker requests |
| GTE embed / Vulkan | passed | yes | 17/17 | processed, 1 worker request | sequence_too_long, 0 worker requests |
| GTE rerank / Vulkan | passed | yes | 125/125 | processed, 1 worker request | sequence_too_long, 0 worker requests |
| Qwen3 embed / Vulkan | passed | yes | 17/17 | processed, 1 worker request | sequence_too_long, 0 worker requests |
| Qwen3 rerank / Vulkan | passed | yes | 125/125 | processed, 1 worker request | sequence_too_long, 0 worker requests |

All admission checks report `truncated: false` and `diverted: false`.
[records/](records/) retains unmodified schema-3 runner records, including the
four master CUDA controls. [pins.json](pins.json) snapshots independent expected
fingerprints from `crates/synapse-module/src/catalog/models.json`; it was identical
at the candidate, master and this evidence worktree's base. The runner's
fingerprint gates and the evidence validator check those values, not pins inferred
from a returned record. Full fingerprints, metrics and outcomes are also in
[summary.json](summary.json).

## End-to-end timings

Milliseconds per single-fixture RPC, using `boundary-512` and `long-8192`.
Every shape completed 6/6 samples. **Warm** means the median of repeats 1–5;
**first** means repeat 0 after the shortest fixture loaded and self-checked the
lane. First-shape time includes shape-specific capture/setup, not initial model
load. The same unchanged timing helper and generated certification config were
used throughout; no timing tolerance or performance threshold was introduced.

Each table entry below is **warm / first-shape ms**. The last column compares
warm medians against master on this rental. Negative is faster.

| CUDA model | Tokens | Previous rental master | Same-machine master | Corrected candidate | Warm change vs same-machine master |
| --- | ---: | ---: | ---: | ---: | ---: |
| GTE embed | 512 | 10.908 / 25.410 | 5.211 / 14.467 | 4.692 / 14.295 | -9.96% |
| GTE embed | 8192 | 191.512 / 812.188 | 170.497 / 621.304 | 170.976 / 622.685 | +0.28% |
| GTE rerank | 512 | 13.039 / 28.279 | 5.924 / 17.941 | 5.731 / 15.302 | -3.26% |
| GTE rerank | 8192 | 230.344 / 834.618 | 178.523 / 630.515 | 178.272 / 637.545 | -0.14% |
| Qwen3 embed | 512 | 13.572 / 34.750 | 8.158 / 24.931 | 9.895 / 29.742 | +21.28% |
| Qwen3 embed | 8192 | 307.430 / 890.480 | 272.304 / 819.361 | 272.070 / 829.655 | -0.09% |
| Qwen3 rerank | 512 | 14.680 / 36.831 | 8.500 / 27.526 | 8.640 / 26.283 | +1.65% |
| Qwen3 rerank | 8192 | 328.990 / 905.165 | 276.733 / 825.748 | 276.137 / 825.022 | -0.22% |

| Candidate Vulkan model | 512 warm / first ms | 8192 warm / first ms |
| --- | ---: | ---: |
| GTE embed | 34.131 / 33.027 | 306.554 / 307.041 |
| GTE rerank | 35.074 / 36.119 | 312.982 / 326.457 |
| Qwen3 embed | 74.560 / 74.935 | 1599.814 / 1601.191 |
| Qwen3 rerank | 80.343 / 75.385 | 1592.644 / 1604.279 |

The 8192-token paired CUDA warm medians differ by -0.22% to +0.28%. The
512-token Qwen3 embedding median is **21.28% slower** (+1.736 ms) than the
same-machine master control; that observation is retained, not dismissed or
retested away. Five warm samples per shape do not establish a stable overhead
bound. All candidate CUDA warm medians are lower than the previous rental's
master values, but that cross-host comparison is not an attribution of improvement.
[cuda-comparison.csv](cuda-comparison.csv) preserves both comparisons;
[previous-timings.csv](previous-timings.csv) is the exact prior rental CSV.
[timings.csv](timings.csv) and `summary.json` retain unrounded values, first-shape
and cold-load times, and every warm sample.

## Completed-work progress gaps

From host `completed work received` logs, within each `req_id`: take the largest
logged `gap_ms` after `completed=1`. The first event measures dispatch-to-progress,
not an interval between completed-work events. Idle gaps between requests and
LOAD (which has no completed-work events) are excluded. These are host-observed
intervals, not inferred GPU layer timings. Certification and timing sessions are
reported separately; the larger value is the per-cell maximum across both runs.

| Candidate cell | Certification events | Certification max ms | Timing events | Timing max ms | Overall max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| GTE embed / CUDA | 32 | 606.742 | 34 | 614.765 | 614.765 |
| GTE rerank / CUDA | 44 | 610.735 | 40 | 624.295 | 624.295 |
| Qwen3 embed / CUDA | 32 | 961.898 | 34 | 928.495 | 961.898 |
| Qwen3 rerank / CUDA | 44 | 887.288 | 40 | 918.251 | 918.251 |
| GTE embed / Vulkan | 504 | 24.210 | 408 | 25.286 | 25.286 |
| GTE rerank / Vulkan | 3168 | 24.347 | 480 | 26.071 | 26.071 |
| Qwen3 embed / Vulkan | 630 | 56.919 | 510 | 58.747 | 58.747 |
| Qwen3 rerank / Vulkan | 3960 | 55.695 | 600 | 58.754 | 58.754 |

CUDA emits setup and completed-forward progress only, not per-layer events.
Thus a long unchanged forward call remains one observable progress interval.
These finite successful runs do not establish a universal no-hang guarantee.
[progress.csv](progress.csv) retains 316 candidate exchanges, including LOAD
separately and the host's broader `max_gap_ms` (which also covers dispatch and
terminal-reply gaps). The reducer checked event sequences and equality with each
exchange's `progress_units`. Master has no such progress instrumentation.

## Execution and provisioning findings

The shipped release binaries and runners were built successfully from the exact
requested commits in distinct targets. [tools/](tools/) retains the recipe;
[provenance.json](provenance.json) records build commands, binary hashes, toolkit
versions, eight verified mirror file hashes, and all 24 captured config checks.
The generated config uses `certify_observation: true`, `deadline_ms: 3600000`,
`max_queue_ms: 3600000`, `max_items: 64`, and `max_tokens: 8192`, exactly as the
previous rental. It is the runner-generated configuration, not a claim about the
module's implicit defaults.

Two provisioning errors are preserved in raw logs; neither is a candidate
inference failure or a reason to alter candidate source or budgets:

- The initial source bundle contained only `refs/mason/*`; `git clone` created an
  empty checkout and setup reported `fatal: reference is not a tree:
  be6cd118a40cc8cc5e999fad9e1e880e43dcd644`. The upload was corrected to contain
  branch refs and pinned setup resumed, without modifying either source tree.
- The scratch timing helper's initial `--locked` fetch refused the copied workspace
  lock: `error: cannot update the lock file /root/timing-probe/Cargo.lock because
  --locked was passed to prevent this`. The original recipe's offline build
  generated the standalone helper lock, then reported `error: failed to download
  lock_api v0.4.14`. The previous rental's documented `cargo fetch --locked`
  followed by `cargo build --release --locked --offline` succeeded. `build.sh`
  now makes the standalone lock-generation step explicit; shipped lockfiles were
  never changed. The timing source and dependency manifest remain byte-for-byte
  copies of the previous rental.

Headless `vulkaninfo` reported the absent display/runtime-directory warnings;
its NVIDIA adapter and both Vulkan worker floor probes nevertheless passed.
No compatibility shim, CUDA graph-disable switch, native instrumentation,
software ICD fallback or candidate failure workaround was used.

Raw certification/timing output and structured host logs remain outside git in
`evidence/cuda-rental-2/` in this worktree (not a regenerable build directory).
[raw-files.sha256](raw-files.sha256) identifies the retained files. The host logs
are the structured `data/cortexkit/synapse/logs/synapse.*.log` files; the timing
helper's redirected stderr files are empty and are not progress evidence.

See [RENTAL.md](RENTAL.md) and [spend.json](spend.json) for final billing and
independently confirmed destruction. The evidence validator covers seven
invariants; its fingerprint guard was also checked by deliberately corrupting one
record, verifying only `test_fingerprint_pins` failed, then restoring it before
commit. This control validates the report checker, not a mutation of candidate
inference behavior.
