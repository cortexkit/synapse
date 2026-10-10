# CUDA and Vulkan catalog lanes on a real NVIDIA GPU

One rented Linux machine (vast.ai instance 55175162, see [RENTAL.md](RENTAL.md)):
RTX 5070 Ti 16 GB, compute capability 12.0, NVIDIA driver 610.57.04 (CUDA driver
API 13030), Ubuntu 24.04 container, kernel 6.8. `vulkaninfo --summary` listed one
device, the NVIDIA GPU (`DRIVER_ID_NVIDIA_PROPRIETARY`, Vulkan 1.4.341); no
software rasterizer was installed. Toolkits were the release workflow's pinned
CUDA 13.2.1 redistributables and Vulkan SDK 1.4.357.0, checked by
`scripts/check-release-candidate.py`.

## Summary

Commits named below:

- **Public master, `ff24e671`**: the `master` branch of
  `https://github.com/cortexkit/synapse` when the box was set up. It was the
  starting point for every run.
- **`76bf4cd1` and `dd7412b0`**: the two fix commits described next. They
  were committed on top of `ff24e671` on the branch that also adds this
  evidence (`alfonso/task/bg_b173cdb59264cfff-...`).

At `ff24e671`, every CUDA lane failed to load. Two bugs were fixed, each with
a test that fails without the fix:

1. `76bf4cd1`: owned-cuda catalog profiles were stored with `execution: explicit`.
   Loading them then failed with
   `config: owned-cuda requires supervised worker execution`. This affected both
   the catalog path and the certification runner.
2. `dd7412b0`: the CUDA engine refused a package path without a `.safetensors`
   extension. The catalog path then failed with
   `model path .../model-cache/blobs/<sha256> is neither a directory nor a safetensors file`.
   The runner was not affected, because it names its package `profile.safetensors`.

The final results come from candidate binaries and the runner, both built in
release mode from a clean checkout of `dd7412b0`, which contains both fixes:

- **Certification records:** 6 of 8 lanes produce a release certification
  record. Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B on Vulkan fail the gate,
  and for the same reason: their 8192-token fixture takes longer than the
  worker's fixed 30 s request timeout. Every other fixture case matches fp32
  closely. Details are below.
- **Catalog end to end:** all 8 lanes pass. `models.download` commits, the
  request by lane id loads and passes its self-check, and the batch is served.
  The fingerprint in `models.list` and in the served response equals the pin
  in `crates/synapse-module/src/catalog/models.json`.

All eight lanes also pass the catalog's own load-time self-check. That check
never sends an 8192-token input.

## Per lane, at dd7412b0

Certification runs `ckdev-synapse-certify run`, built from the same clean
commit. Its records are in
`docs/evidence/certification/dd7412b0409eb3c97d026c1bdd38797669813752/<row>/<model>.json`.
Parity is measured against the committed fp32 fixtures under
`bench/parity/fixtures`: 17 embed cases or 125 rerank cases, including one
8192-token case.

The serving latency column comes from `catalog-e2e`. It uses one `embed.batch`
or `rerank.score` request of 4 items on the warm lane, and reports the median
of 5 runs. "Cold" is the first request, including the load and self-check.

The catalog e2e column reports two checks:

- the fingerprint check: the listed and served fingerprints equal the pin;
- the self-check: the module's load-time comparison of the lane's outputs
  against bundled fp32 reference outputs. When it passes, `models.list` shows
  `self_check.state: passed` and `certified: true`.

| lane | certification | error vs fp32 fixtures | catalog e2e (fingerprint = pin, self-check) | warm 4-item batch | cold first request |
| --- | --- | --- | --- | --- | --- |
| gte-modernbert-base-cuda | passed | min cosine 0.9999751 | pass, passed | 4.4 ms | 2.8 s |
| gte-modernbert-base-vulkan | passed | min cosine 0.9999991 | pass, passed | 91.1 ms | 2.3 s |
| gte-reranker-modernbert-base-cuda | passed | max abs score error 4.94e-3, pool tau 1.0 | pass, passed | 5.7 ms | 3.7 s |
| gte-reranker-modernbert-base-vulkan | passed | max abs score error 2.82e-3, pool tau 1.0 | pass, passed | 96.6 ms | 3.4 s |
| qwen3-embedding-0.6b-cuda | passed | min cosine 0.9999967 | pass, passed | 4.9 ms | 11.5 s |
| qwen3-embedding-0.6b-vulkan | **failed** (`parity gate failed`, `incomplete_output`, 16/17) | min cosine 0.9999998 over the 16 completed cases | pass, passed | 225.4 ms | 9.4 s |
| qwen3-reranker-0.6b-cuda | passed | max abs score error 6.06e-5, pool tau 1.0 | pass, passed | 9.9 ms | 11.3 s |
| qwen3-reranker-0.6b-vulkan | **failed** (`parity gate failed`, `incomplete_output`, 124/125) | max abs score error 1.18e-5 over the 124 completed cases, pool tau 1.0 | pass, passed | 363.7 ms | 10.3 s |

Every passed record has all ten gates true, `tokens_8192` processed in one
worker request, and `tokens_8193` refused with `sequence_too_long` before
reaching the worker. The two GTE Vulkan lanes also passed at public master
`ff24e671`, without the supervised-execution fix or the extension fix. Their
records are in
`docs/evidence/certification/ff24e6715e1bb58ad6ba0c120b889ef7431ca6c3/`.

**Records are tied to the fix commit.** `dd7412b0` is a commit on this branch,
not on public master. The runner refuses to produce records unless the
candidate and the runner were both built from the same clean commit. These
records therefore stay valid only if the branch is merged without rewriting
that commit; a rebase or squash would orphan them.

## Vulkan Qwen3 failures

The runner writes no record on a failed gate, and it prints only
`certification_refused: parity gate failed`. To see per-case errors,
`tools/parity-probe.rs` repeats the runner's exact session: the same preload
config, the same request grouping, and the same evaluator. It then prints every
case. Outputs: `runs/dd7412b0/probe.vulkan-linux-nvidia.<model>.jsonl`, which
contains the full evaluator output (metrics, gates, failures) and one line per
case.

- **Only the 8192-token case fails, and it fails on time, not on numbers.**
  `long-8192` returns
  `engine_crashed at timeout: request exceeded 30000 ms` after 30.6 s, for
  both models. That limit is the fixed worker request timeout,
  `request_timeout: Duration::from_secs(30)` in
  `crates/synapse-module/src/worker_host/mod.rs`. The missing output makes the
  evaluator report `incomplete_output`, and every gate is false.
- **Errors do not grow with length.** For Qwen3-Embedding, every case from 4
  to 513 tokens, including the 6-item batch, has cosine ≥ 0.9999998 and
  a largest element error ≤ 9.8e-5. The 511-, 512- and 513-token cases are no
  worse than the 4-token one. For the Qwen3 reranker, the largest score error
  is 3.1e-6 for the 127–513-token boundary cases and 1.18e-5 overall. The
  10- and 100-candidate pools have tau 1.0 with no discordant pairs. On every
  case, the token ids the backend saw equal the fixture's `input_ids`.
- **Time grows with length.** On Vulkan, a single Qwen3-Embedding request
  takes about 0.10 s at 128 tokens and about 0.42 s at 512 tokens. At 8192
  tokens it does not finish within 30 s. The same model on CUDA completes
  the 8192 case and passes.

Fixing the Vulkan Qwen3 timeout needs a maintainer decision: either make
Qwen3 on Vulkan faster at 8192 tokens, or give long requests a longer worker
timeout. Nothing was changed for it here, and no tolerance was touched.

## Pin-match check, and whether a mismatched fingerprint is refused

Code, at `crates/synapse-module/src/lib.rs` (line numbers as of this branch):

- Before a load, catalog verification hashes the installed files. It then
  rebuilds the lane spec with `catalog_lane_spec(.., verify = true)`. That
  spec's fingerprint is computed from the verified artifact digest, the
  sanitized tokenizer digest and the numeric profile (`NumericProfile` in
  `build_stored_model_config`).
- If that computed fingerprint differs from the pin
  (`if spec.fingerprint.0 != backend.fingerprint`, line 27460), the load is
  **refused**. The error is `artifact_invalid` with `expected_fingerprint` and
  `actual_fingerprint`, and the lane slot is set to `Failed`. The engine is
  never invoked.
- Before a load, an unloaded catalog row only displays the pinned value
  (`spec.fingerprint = Fingerprint(backend.fingerprint.clone())`, line 26414).
  `models.list` proves the pin only after a load, because the load
  recomputes the fingerprint.
- Gap: no test exercises the mismatch branch. The e2e tests only check that
  the served fingerprint equals the pin.
- The `76bf4cd1` fix changes only the stored `execution` value for owned-cuda
  profiles, from `explicit` to `supervised`. `execution` is not part of the
  numeric profile that the fingerprint hashes, so the fix leaves every pinned
  fingerprint unchanged. All eight measured fingerprints
  equal their pins.

Measured with the release build of `dd7412b0`, after load and serve.
`models.list` `fingerprints` and
the served response's `fingerprint` both equal the pin:

| lane | pinned = listed = served |
| --- | --- |
| gte-modernbert-base-cuda | `5f07dca3b028ed3db106e4f9d05ad0452ea43199c0830e600a480b0c2eb81010` |
| gte-modernbert-base-vulkan | `885776108d3fd5be2e9e3f17cc02df97d8d3ec83fcbd7978913093e1fdee4bad` |
| gte-reranker-modernbert-base-cuda | `829c20af3757ecd5bb53ab89507ff64d4382225688f5888d6236ffc1670f2d4d` |
| gte-reranker-modernbert-base-vulkan | `084f55877211334b1a11354668afd243637d9897747a539c74281fa43f298205` |
| qwen3-embedding-0.6b-cuda | `9628e6a8b0f30e9eb459f09cfb985306bbbaf2440579d38660e9795c4353554b` |
| qwen3-embedding-0.6b-vulkan | `62e7abe5dacb0c8f8932a057f39c04e7b7a1f8994897b138cabe88ce9461eed5` |
| qwen3-reranker-0.6b-cuda | `28ac017d49cf3ec25f1a343a5316287cfeefb131ac6ed5bbf3c56ca392090688` |
| qwen3-reranker-0.6b-vulkan | `e892cb51a90e2ccb87a4a88d6fab5237ecb48959ee50da69cd7fee2a1111764a` |

## Not produced: catalog release evidence records

The catalog release verifier,
`crates/synapse-module/src/catalog/evidence.rs`, expects a different record
type: `docs/evidence/catalog-backends/<catalog_id>__<backend>.json`, with
`manifest_digest`, `engine_tree`, `corpus_id` and corpus metrics measured on
the named corpora in `crates/synapse-module/src/fixtures/`. Nothing in the
repository produces those records. The certification runner writes schema-2
records under `docs/evidence/certification/` instead. No `__` records were
hand-written here.

## Files

- `runs/ff24e671/`: public master, before either fix.
  - CUDA certification refusal (`requires supervised worker execution`) and the
    CUDA catalog e2e failure.
  - Vulkan certification logs.
  - The module test failing before its fix (`test-before-module-fix.log`).
- `runs/76bf4cd1/`: after the module fix.
  - All four CUDA certifications passed. These records were superseded by
    `dd7412b0` and are not committed; their logs are kept.
  - The CUDA catalog e2e failure (`... neither a directory nor a safetensors file`).
  - The first Vulkan probe runs (`probe-timed.*` has request timings).
- `runs/dd7412b0/`: the final build.
  - Certification stderr/stdout for all 8 rows.
  - `e2e.<model>.<backend>.jsonl` for all 8 lanes (catalog, download job,
    `models.list` before and after, first and warm serve, pin check).
  - Vulkan Qwen3 probes.
  - `nvidia-smi.txt`, `vulkaninfo-summary.txt`, `artifact-sha256.txt`,
    `build-and-test-after-engine-fix.log`, `test-before-engine-fix.txt`.
- `tools/`: the scripts run on the box and the two scratch drivers
  (`catalog-e2e.rs`, `parity-probe.rs`). The drivers were built outside the
  synapse checkout with the subc crates pinned by this repository's
  `Cargo.lock`. They talk to `ck-synapse` the way the certification runner
  does: an in-process subc daemon, then `route.open` on the synapse
  management surface.

On the box, the synapse source was never edited. The build used a fresh clone
at `ff24e671`. The two fix commits, `76bf4cd1` and `dd7412b0`, arrived as a git bundle of
the evidence branch and were checked out cleanly (`git status --porcelain` was empty before every
build). The before-fix test runs used a separate scratch worktree: `ff24e671`
plus only the new test hunk. Release build times on 32 vCPUs were 61 s for the
candidate binaries and 15 s for the runner.
