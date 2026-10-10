# Release evidence: catalog evidence vs certification records

Report only. No code or workflow was changed. Everything below was read at
`ff24e671` (master merged into `land/compile-lane`) unless a branch is named.
The CUDA/Vulkan rental evidence lives on `land/cuda-vulkan-proof` (commit
`6d295228`), which is not merged into this base.

## Summary

- **The question's premise is out of date. The release workflow no longer runs
  catalog evidence.** The `verify-catalog-evidence` job was added to
  `release.yml` on 2026-10-04 (`dd264d37`). One day later, `52c80e29` replaced it
  with the certification-record gate. Today
  `.github/workflows/release.yml:10-39` runs
  `crates/synapse-release-checks/check.py source|validate`, which ends by
  running the extracted candidate's `ck-synapse certify validate`
  (`check.py:224`). The only remaining consumer of the catalog verifier is an
  `#[ignore]` test that nothing in CI runs
  (`crates/synapse-module/src/catalog/evidence.rs:509-536`). The module doc
  comment at `evidence.rs:9-11` still says the release workflow runs it, and so
  does `docs/qwen3-ane-development.md:31-33`. Both are stale.
- **The missing catalog records are therefore not what blocks a release.** What
  blocks it is the certification matrix. `check.py validate` needs all 32
  `(row, model)` records for one candidate source commit `S`, each bound to the
  sha256 of the binaries in the GitHub `candidate-S` draft. It also needs a
  benchmark report, `RELEASE-NOTES.md`, and, where relevant, ANE stress and
  transition files (`check.py:169-203`). No candidate draft exists. The
  records that do exist cover 2 of the 8 rows and were made with binaries built
  locally on the rental box (see Findings).
- **Recommendation (high confidence):** keep certification records as the only
  release evidence system. Delete the catalog evidence verifier. Move three of
  its unique checks into the certification validator: re-derive "passed" from
  the gates, pin the fixture set, and pin the fingerprint. Drop the rest, for
  the reasons below.

## 1. What each mechanism binds, checks and refuses

### 1a. Catalog evidence (`crates/synapse-module/src/catalog/evidence.rs`)

**Location and unit.** There is one record per `(catalog_id, backend)` at
`docs/evidence/catalog-backends/<catalog_id>__<backend>.json` (`evidence.rs:22-23, 84-87`).
The compiled catalog (`catalog/models.json`, embedded at `catalog/mod.rs:33`)
declares 14 backends: gte-modernbert-base {metal, ane, cuda, vulkan},
gte-reranker-modernbert-base {metal, cuda, vulkan}, qwen3-embedding-0.6b {metal,
ane, cuda, vulkan} and qwen3-reranker-0.6b {ane, cuda, vulkan}. The unit is a
*backend*, not a hardware row. A single `vulkan` record would vouch for AMD and
NVIDIA, on both Linux and Windows.

**Record schema** (`evidence.rs:89-130`, all `deny_unknown_fields`):
`catalog_id`, `backend`, `manifest_digest`, `fixture_revision`, `fingerprint`,
`engine_tree`, `machine` (free-form map), `engine_build`, `dtype`, `corpus_id`,
`metrics {min_cosine | max_abs_sigmoid_deviation, order_violations}`,
`thresholds {min_cosine | rerank_abs_tolerance}`, `passed`.

**Binds and checks** (`verify_backend`, `evidence.rs:205-311`):

- The catalog must first pass release validation (`load_release_valid`), or the verifier fails as a whole (`evidence.rs:185-186`).
- Exact equality on seven keys (`evidence.rs:254-292`): `catalog_id`, `backend`,
  `manifest_digest` (the JCS manifest digest of the entry, `catalog/mod.rs:361`),
  `fixture_revision` (from `self_check`, or the default `"1"` when the entry has none, `evidence.rs:245-253`),
  `fingerprint` (the backend's pinned lane fingerprint in `models.json`),
  `engine_tree`, and `dtype`.
- `engine_tree` is the newline-joined `git rev-parse <rev>:<dir>` tree hashes of
  the backend's engine crates (`evidence.rs:25-51, 399-428`). Metal and ANE share
  engine-owned, worker-ane and worker-ane-direct; CUDA uses engine-cuda and worker-cuda; Vulkan uses worker-vulkan. A
  record therefore stays valid across commits that do not touch those crates.
- A named corpus per catalog entry (`evidence.rs:57-82, 294-305`):
  `probe_corpus_gte_modernbert_ort_fp32`, `probe_corpus_qwen3_embedding_fp32`,
  `catalog_rerank_gte_modernbert_fp32` (24 fp32 Transformers pairs, `evidence.rs:834-876`),
  and `qwen3-reranker-0.6b.ref-v1.transformers-5.16.1.seed-0` (the bench/parity fixture set).
- Metric thresholds are recomputed from the record's numbers, and the record's
  `passed` flag is never trusted on its own (`evidence.rs:313-375`). Embed needs
  `min_cosine >= 0.999` (`evidence.rs:53-55`). Rerank needs
  `max_abs_sigmoid_deviation <= backend.rerank_abs_tolerance` and
  `order_violations == 0`. The record must also state exactly those thresholds.
- `machine` must be non-empty and `engine_build` non-blank, but neither is compared to anything (`evidence.rs:238-243`).

**Refusals:** the `EvidenceFailure` variants `Catalog`, `MissingRecord`,
`MalformedRecord`, `KeyMismatch`, `NoNamedCorpus`, `WrongCorpus`,
`FailingMetric` and `NotPassed` (`evidence.rs:132-170`). All failures are
accumulated, not just the first. A missing directory counts as zero records, so
every backend then fails as missing (`evidence.rs:377-397`).

**Does not check:** binary bytes, source commit cleanliness, hardware row
coverage (one record per backend), 8192/8193 admission, any ANE-vs-Metal
comparison, or the machine's identity.

**Producer:** none. No Rust, Python, shell or workflow file on any local or
`origin/*` branch writes `EvidenceRecord`s or files into
`docs/evidence/catalog-backends/*.json`. (I ran `git grep` for
`catalog-backends|EVIDENCE_DIR|EvidenceRecord` over every ref. The only hits
are `evidence.rs` itself and an unrelated `SchedulerEvidenceRecord`.) The
commit that wired the gate said so outright: "No evidence producer exists yet:
tags intentionally remain blocked until fresh hardware records are generated"
(`dd264d37`). On `land/cuda-vulkan-proof`, `docs/evidence/catalog-backends/`
contains only `README.md`, `RENTAL.md`, `tools/` and `runs/` logs, and no
`*.json` record at the top level.

### 1b. Certification records (`crates/synapse-certify`, `crates/synapse-certify-runner`)

**Location and unit.** There is one record per `(row, model)` at
`docs/evidence/certification/<source_commit>/<row>/<model>.json`
(`crates/synapse-certify/src/lib.rs:224-230`). The matrix is fixed at 8 rows ×
4 models = 32 (`lib.rs:13-28`): rows vulkan-{windows,linux}-{amd,nvidia},
cuda-{linux,windows}-nvidia, metal-m5 and ane-m5.

**Record schema 2** (`lib.rs:200-219`): `schema`, `row_id`, `source_commit`,
`machine`, `executed_artifacts[{role,file,sha256}]`, `model`, `operation`,
`profile_id`, `fingerprint`, `parity` (the evaluator's output verbatim:
identity, `fixture_set_id`, completed/expected counts, `tolerance_class`,
`metrics`, ten named `gates`, `failures`; `lib.rs:121-146`), `admission`
(`tokens_8192`, `tokens_8193`; `lib.rs:148-172`), `status`, and optional
`drop_cause` and `raw_series`.

**Producer** (`ckdev-synapse-certify run --row --model --assets --checkout --weights`;
`crates/synapse-certify-runner/src/command.rs:6-124`, `live.rs`). The candidate
`ck-synapse` and the runner must report the same *clean* source commit
(`synapse-certify/src/command.rs:8-15`; runner `command.rs:79-95`). The runner
starts the candidate through hard links with a preload profile
`<model>.<lane>` from `bench/parity/models.json` (`live.rs:50-60, 486`). It
sends the full committed fp32 fixture set (`live.rs:451-477`; 17 embed or 125
rerank cases, including one 8192-token case) and grades the outputs with the
parity evaluator (`live.rs:557-561`). The fingerprint is whatever the candidate
reports, and the run refuses if it changes mid-run (`live.rs:489-520, 557`).
`produce` (`lib.rs:364-468`) refuses on any of: a failed floor probe (Metal
excepted), failed parity, failed admission (8192 processed untruncated, 8193
`sequence_too_long` with 0 worker requests; `lib.rs:162-172`), a failed ANE
placement inventory, or a missing executed artifact. It hashes the executed
candidate files (`lib.rs:435-445`). A refused run writes nothing
(`lib.rs:470-481`).

**Drop rule** (ANE vs Metal): only `ane-m5 × qwen3-*` may be `dropped`
(`lib.rs:315-317`), and only for a failing parity gate or an ANE/Metal median
latency ratio strictly above 3 over 3 warm-up + 20 measured samples from one
session (`lib.rs:174-198, 390-403`).

**Offline validator** (`validate`/`validate_checkout`, `lib.rs:483-554`;
`ck-synapse certify validate --assets <dir> <root>`, `synapse-certify/src/command.rs:23-58`):

- The candidate's own build stamp must be a clean 40-hex commit, or validation refuses (`command.rs:47-58`).
- It reads exactly the 32 canonical paths. Each must exist and match its path's row, model and commit (`lib.rs:527-554`).
- Per record: schema 2, a 40-hex commit, a known row and model, hashed (not raw)
  hardware UUIDs (`lib.rs:87-103`), and parity identity equal to the record
  identity (`lib.rs:347-362`).
- No duplicate combinations, and 32 in total (`lib.rs:491-493, 521-523`).
- `status` must be `passed`, or `dropped` on a droppable pair with failing
  parity or a recomputed ratio > 3 (`lib.rs:494-511`).
- Every `executed_artifacts[].sha256` must equal the sha256 of that file in the extracted candidate (`lib.rs:512-519`).
- **It does not recheck the gates of a `passed` record.** This is deliberate
  and pinned by the test `validator_does_not_add_producer_gates`
  (`crates/synapse-certify/tests/records.rs:419-426`), which accepts a `passed`
  record with `score: false` and a truncated 8192 admission.

**Tag-only checks in `crates/synapse-release-checks/check.py`:**

- The tag commit must have exactly one parent `S` and differ from it only under `docs/evidence/`, with certification changes only under `certification/S/` (`check.py:35-43`).
- `candidate-S` must be a draft release (`check.py:46-50`), and the downloads must match the sidecars and inventory (`check.py:53-76`).
- Each lane-bearing binary in `bench/parity/release-assets.json` must be bound to its rows, and every passed record of those rows must name the same extracted binary sha256 (`check.py:79-112`).
- Every model needs a passed CUDA record at exactly `cuda_min_driver_api` (`check.py:145-152, 185-186`).
- There must be no raw UUIDs (`check.py:178-182`).
- Dropped records must be named in `RELEASE-NOTES.md`, and Metal records must list the record fingerprint next to the preload fingerprint (`check.py:190-197`).
- There must be a complete benchmark report whose latency drops recompute to > 3 (`check.py:115-142, 198-199`), ANE stress evidence (`check.py:155-166, 187-189`), and an ANE CoreML→direct transition series (`check.py:200-203`).
- Finally it runs `ck-synapse certify validate` (`check.py:224`).

## 2. Which release workflow steps call each

| Step | Mechanism |
| --- | --- |
| `release-candidate.yml` `prepare` (`:28-53`) | Neither runs. It creates the `candidate-S` draft, and the draft's notes tell operators to run `ckdev-synapse-certify run` on release hardware and say `ck-synapse certify validate` "remains the offline tag gate" (`:53`). |
| `release-candidate.yml` `build`/`reconvert`/`inventory` (`:55-347`) | Neither. They build assets once, run protocol tests, reconvert digests, and check the inventory. |
| `release.yml` `validate` (`:11-39`) | Certification: `check.py source`, then `check.py validate`, which ends in `ck-synapse certify validate` (`check.py:224`). |
| `release.yml` `smoke`/`promote` (`:41-131`) | Neither (no-GPU refusal smoke, byte-preserving promotion). |
| `tests.yml` | Runs the synthetic tests of both crates (`SYNAPSE_CRATES` at `:43`). The ignored `release_catalog_evidence` test is not run anywhere. The only `--ignored` run in any workflow is `candidate_workers` (`release-candidate.yml:235`). |
| `scripts/` | No script references either mechanism. `scripts/check-release-candidate.py` only checks source, extraction, digests, SPIR-V, CUDA ELF and inventory. |

The catalog gate was in `release.yml` only from `dd264d37` (2026-10-04 08:49)
to `52c80e29` (2026-10-05 11:10). In that window the job ran
`cargo test -p synapse-module --lib catalog::evidence::tests::release_catalog_evidence -- --ignored --exact`
and grepped for `1 passed` before any build (visible in `git show 52c80e29 -- .github/workflows/release.yml`).

## 3. Overlap and unique checks

| Property | Catalog evidence | Certification |
| --- | --- | --- |
| Accuracy vs fp32 reference | A record states metrics on a named corpus, and the verifier re-applies thresholds (cos ≥ 0.999; sigmoid deviation ≤ 0.005 from the catalog; 0 order violations). | The producer measures accuracy on the full bench/parity ref-v1 fixture set with the evaluator's 10 gates (cos ≥ 0.999; rerank score error ≤ 0.005 for fp32 or ≤ 0.02 for fp16, `evaluator.rs:361, 392-396`). The validator only checks that parity identity matches. |
| Trusts the record's own pass flag | No (`evidence.rs:313-315`). | Yes for `passed` (`records.rs:419-426`). Drops are recomputed. |
| Corpus identity | Pinned per entry (`evidence.rs:60-74`). | Recorded as `parity.fixture_set_id`, but no validator compares it to anything. |
| Corpus size | gte-reranker: 24 pairs. Embed: probe corpora. | 17 embed / 125 rerank cases, including 8192 tokens. |
| Fingerprint | Must equal the catalog pin (`evidence.rs:267-271`). | Taken from the candidate. Must be stable mid-run and equal to `parity.fingerprint`. Not compared to any pin, except that Metal must be listed in the release notes next to the preload pin (`check.py:194-197`). |
| Engine identity | Git tree hashes of the engine crates (`evidence.rs:25-51`). | sha256 of the executed candidate binaries, plus the clean source commit and the sole-parent evidence commit. |
| Manifest digest / fixture revision / dtype | Pinned. | Not recorded. Implied by the binary, which embeds the catalog. |
| Hardware coverage | One record per backend. | 8 rows × 4 models. Inventory binds each binary to its rows. |
| Admission 8192/8193 | — | Producer only. |
| ANE placement inventory | — | Producer only (`lib.rs:381-389`). |
| ANE vs Metal drop rule | — | `lib.rs:315-317, 494-510`, plus release notes and benchmark (`check.py:125-131, 191-193`). |
| Machine identity | Non-empty map. | Hashed GPU or platform UUID required (`lib.rs:87-103`). |
| CUDA driver floor | — | `check.py:145-152, 185-186`. |
| Code path exercised | Not specified (no producer). | Preload profile path (`certify-candidate`), **not** catalog download plus lane load. |

On real data the fingerprints agree for CUDA and Vulkan. The six distinct
CUDA/Vulkan records on `land/cuda-vulkan-proof` carry the same fingerprints as
the `models.json` pins (`5f07dca3…`, `885776108d3f…`, `829c20af3757…`,
`084f55877211…`, `9628e6a8b0f3…`, `28ac017d49cf…`). They also measured
min cosine ≥ 0.99997 for embeds and max |Δscore| ≤ 4.94e-3 for rerankers
against fp32. For Metal they do not agree; see Finding F5.

## 4. Was one meant to replace the other?

Timeline (`git log`):

| Date | Commit | Event |
| --- | --- | --- |
| 2026-10-03 04:53 | `da256a58` | Catalog data and `evidence.rs` ("verify-catalog-evidence logic … metric checks that do not trust `passed`"). |
| 2026-10-04 08:49 | `dd264d37` | Catalog evidence gates `release.yml`. The commit says "No evidence producer exists yet". |
| 2026-10-04 14:50 | `b4147555` | Certification records and the 32-cell validator added (schema 1, live run refused). |
| 2026-10-05 09:57 | `e9a099a0` | `release-candidate.yml` (immutable candidate built once). |
| 2026-10-05 11:10 | `52c80e29` | `release.yml` rewritten. `verify-catalog-evidence` removed and replaced by `check.py validate`. The message says "Keep the candidate record validator as the authority for all 32 parity/drop records". |
| 2026-10-08 → 10-10 | `46a2071e`, `b64014d9` | `evidence.rs` still extended (ANE lanes; 14 lanes, "Bind evidence to each backend's ordered engine trees"). |
| 2026-10-10 | `876be280` | Live certification moved into the non-shipped `ckdev-synapse-certify`. |

So catalog evidence was designed first, about 34 hours before certification
records existed. The workflow change in `52c80e29` replaced it as the tag
gate, but no commit message, doc or code comment says the catalog verifier is
retired. Later commits kept extending it as if it were live, and
`docs/qwen3-ane-development.md:31-36` still treats "the catalog evidence gate"
and certification as two separate live obligations. The replacement happened
in the workflow, but it was never written down anywhere.

`.cortexkit/alfonso/drafts/` does not exist in this worktree, so I found no
draft spec to settle the intent (Finding F9).

## 5. What it would take

### 5a. Producing catalog evidence records today

1. A producer that, for each of the 14 `(catalog_id, backend)` pairs, downloads
   the catalog entry and loads it by catalog lane on reference hardware. The
   scratch `catalog-e2e.rs` in
   `land/cuda-vulkan-proof:docs/evidence/catalog-backends/tools/` already does
   download, lane load and the fingerprint-equals-pin check, but it computes
   no corpus metrics. The producer must run the named corpus and compute
   `min_cosine`, or for rerankers `max_abs_sigmoid_deviation` and
   `order_violations`. The certification evaluator reports
   `max_absolute_error` and ranking tau instead, so its numbers do not map
   directly onto the catalog metrics.
2. It must fill `manifest_digest` (`CatalogEntry::manifest_digest`),
   `fixture_revision`, `dtype` and `fingerprint` from the catalog, and
   `engine_tree` from `engine_tree_at` at the tag commit. It also needs a
   `machine` map and an `engine_build` string, for which no format is defined.
3. It must write the 14 records into `docs/evidence/catalog-backends/`, and
   `release.yml` would need the deleted job restored before `validate`.
4. Gaps that remain even then: one Vulkan record would vouch for four hardware
   rows; nothing ties a record to the shipped bytes; there is no 8192 or
   admission evidence; there is no drop rule. A qwen3 `ane` backend whose
   certification is dropped would block the release outright under catalog
   evidence, because it has no passing record, while certification allows the
   drop.

Rough cost: one new producer, about the size of the runner's `observe`, plus
a second hardware pass per backend on every engine-tree change. Every
property it would add is either already covered by certification or listed
in §6 as something to move into it.

### 5b. Having the release gate consume certification records

The gate already consumes them. To actually cut a release:

1. Dispatch `release-candidate.yml` with a clean `S` so that the
   `candidate-S` draft exists. None exists today (`gh release list` shows only
   `v0.1.0-alpha.1/2`).
2. Run `ckdev-synapse-certify run` **against the extracted draft assets**,
   with no local rebuild, on all 8 rows: Linux and Windows on AMD and NVIDIA
   for Vulkan, Linux and Windows NVIDIA for CUDA, and M5 for Metal and ANE.
   CUDA must include a machine at exactly `cuda_min_driver_api`.
3. Fix or drop the failing cells. Qwen3 on Vulkan currently fails the 8192 case
   on a 30 s worker request timeout (`land/cuda-vulkan-proof`
   `docs/evidence/catalog-backends/README.md`, table). Drops are permitted only
   for `ane-m5 × qwen3-*`, so Vulkan must be fixed.
4. Add `docs/evidence/benchmark/S/report.json`, the ANE stress and transition
   files, and `RELEASE-NOTES.md` under `certification/S/`. Tag a commit whose
   sole parent is `S` and which touches only `docs/evidence/`.

## 6. Recommendation

Ranked by confidence.

1. **Keep certification records as the single release evidence system.
   Confidence: high.** It is the only one wired into `release.yml`, the only
   one with a producer, and the only one with real records. It binds what
   actually ships: the clean commit, the exact binary bytes per hardware row,
   8192/8193 admission, and ANE placement. It is also the only one with a
   reasoned ANE drop rule. The catalog's engine-tree binding is strictly
   weaker than "the binaries with this sha256, built once from clean commit
   S". The trade-off is evidence reuse. Engine-tree records survive unrelated
   commits, while certification must re-run all 32 cells for every candidate
   `S`, which means renting GPUs again for each release. Accept that cost for
   now. If it becomes a problem, add reuse later inside certification (for
   example, accept a prior `S'` record when every bound artifact sha256 is
   unchanged) rather than keeping a second system.

2. **Move three catalog-only checks into the certification validator, and
   settle one threshold disagreement. Confidence: medium-high.**
   - *Do not trust `passed`.* Require `parity.passed()` and
     `admission.passed()` for `status == "passed"` in `validate`. Records
     are hand-editable JSON in git, and the catalog verifier recomputed
     thresholds for this reason. Doing this reverses the contract that the
     test `validator_does_not_add_producer_gates` (`records.rs:419-426`) pins,
     so it is a deliberate contract change for the owner to approve. It is
     not a test fix. Also correct
     `crates/synapse-release-checks/README.md:43`, which already claims the
     candidate CLI "validates numerical parity against reference outputs".
   - *Pin the corpus.* Require `parity.fixture_set_id ==
     synapse_parity::evaluator::fixture_set_id(manifest, model)` and
     `completed_fixtures == expected_fixtures`. This replaces the catalog's
     named-corpus check with a larger corpus that includes the 8192 case. The
     catalog's gte probe and 24-pair rerank corpora are then not release
     evidence. They stay useful for module tests and self-checks only.
   - *Reconcile the rerank tolerance (F10).* The catalog pins 0.005 and the
     evaluator allows 0.02 for `fp16`. Pick one number and make it the
     evaluator's, so there is only one place to read it. The owner should
     decide which value is correct; this report does not.
   - *Pin the fingerprint.* For rows whose lane the catalog declares, require
     the record fingerprint to equal the `models.json` backend pin, and the
     preload pin where one exists. Today `check.py` only requires Metal pins
     to appear in the release notes.

3. **Drop the rest of the catalog's unique checks. Confidence: medium.**
   - `manifest_digest`, `fixture_revision`, `dtype`: the catalog is
     `include_str!`'d into `ck-synapse` (`catalog/mod.rs:33`), so the
     binary sha256 already fixes them. Release validation of the catalog
     itself stays in ordinary unit tests.
   - `engine_tree`: subsumed by the binary sha256 and the source commit (see 1).
   - `machine` non-empty: subsumed by the hashed-UUID identity checks.

4. **Close the code-path gap with a catalog-lane probe. Confidence: medium.**
   Certification runs the preload profile (`certify-candidate`), not the
   catalog download and lane-load path. The rental found a bug that hit only
   the catalog path (`dd7412b0`: a package path without an extension).
   Either fold `catalog-e2e`'s checks (download commits, lane load,
   self-check passed, listed and served fingerprint equal the pin) into
   `ckdev-synapse-certify run` as an extra recorded field, or make it a
   per-row supplemental evidence file that `check.py` requires, in the
   style of stress and transition. This is the one property that only
   catalog evidence promised and certification lacks.

5. **Decide how a certification drop affects the shipped catalog.
   Confidence: medium; needs an owner decision.** The catalog declares `ane`
   backends for both qwen3 models. Certification may drop `ane-m5 × qwen3-*`,
   and the release would still ship a catalog that advertises those lanes.
   Nothing reads the validator's `eligible` list: `check.py:224` only checks
   the exit code. Options: the release notes are enough (today's behaviour), or
   `check.py` refuses a release whose catalog declares a lane with no
   `passed` record.

6. **Retire catalog evidence after 2 and 4 land. Confidence: high.** Delete
   `catalog/evidence.rs` (verifier, `EVIDENCE_DIR`, the ignored
   `release_catalog_evidence` test and its synthetic tests) and the
   `mod evidence` line (`catalog/mod.rs:23`). Keep
   `catalog_rerank_gte_modernbert_fp32.json` only if something else still
   uses it; today only `evidence.rs` reads it. Rename
   `docs/evidence/catalog-backends/` on `land/cuda-vulkan-proof` to
   something like `docs/evidence/cuda-vulkan-rental-2026-10/`, since it
   holds rental logs and not catalog records.

**Order:**

1. Fix the stale docs now, since this costs nothing: `evidence.rs:1-11`,
   `docs/qwen3-ane-development.md:31-33`, `docs/wire-contract-v1.md:979-981`
   (which says the producer "currently certifies only `metal-m5`", though
   CUDA and Vulkan now run), and `crates/synapse-release-checks/README.md:43`.
2. Harden the validator (recommendation 2), with owner sign-off on the
   `passed` contract change.
3. Add the catalog-lane probe (recommendation 4) and decide the drop
   semantics (recommendation 5).
4. Delete catalog evidence (recommendation 6).
5. Cut a candidate and certify all 8 rows from its draft assets (§5b).

Steps 1-4 do not need hardware. Step 5 does, and it is the real release
blocker whatever happens to catalog evidence.

## Findings (unresolved or noteworthy)

- **F1. No catalog evidence producer exists on any branch.** See §1a.
  `dd264d37` acknowledged this when it wired the gate.
- **F2. The catalog evidence gate is not in any workflow.** It was removed by
  `52c80e29`. Its docs (`evidence.rs:9-11`, `docs/qwen3-ane-development.md:32`)
  are stale. `evidence.rs` was still extended on 10-08 and 10-10, so whoever
  extended it may still believe it is live.
- **F3. The existing certification records cannot pass the tag gate as they
  are.** The records at `dd7412b0` and `ff24e671` on `land/cuda-vulkan-proof`
  were produced with binaries built on the rental box
  (`docs/evidence/catalog-backends/tools/build.sh` on that branch: `cargo
  build --release` of `ck-synapse` and the workers). `check.py:97-102` requires
  their `executed_artifacts` sha256 to equal the CI-built `candidate-S`
  binaries. This only works if the CI build reproduces the same bytes, which I
  could not verify. The records also cover 2 of 8 rows, and the Vulkan qwen3
  cells failed. Their evidence commit also bundles code fixes, while
  `check.py:35-43` requires the tag commit to change only `docs/evidence/`
  relative to `S`.
- **F4. No Metal M5 schema-2 record is checked in on any local or `origin/*`
  ref.** The only Metal record is
  `crates/synapse-certify/tests/evidence/metal-m5-gte-modernbert-base.development.json`,
  a schema-1 development record marked `release_gate_eligible: false`. If
  real M5 records exist, they are outside git (as `docs/qwen3-ane-development.md:34-36`
  instructs for development records).
- **F5. Metal fingerprints differ across three sources.** For gte-modernbert-base:
  catalog `metal` pin `b904dd7b9b8b…`, preload `bench/parity/preload/gte-modernbert-base-f16.json`
  `expected_fingerprint` `24cc5271f42d…`, and the development record
  `be6452a6cbcc…`. That is why `check.py` asks the release notes to list a
  fingerprint pair for Metal. I could not establish which identity a
  release-time Metal record will carry, or whether pinning it to the catalog
  (recommendation 2) is achievable for Metal without changing the
  preload/catalog identity.
- **F6. The certification matrix and the catalog backend set differ.**
  Certification requires `metal-m5 × qwen3-reranker-0.6b` and
  `ane-m5 × gte-reranker-modernbert-base` to pass (neither is droppable),
  but the catalog declares no such backends. Certification gates preload
  profiles the catalog does not ship, and a catalog-only lane could ship
  with no row of its own. Today every catalog backend does map to some row.
- **F7. The certification validator trusts `passed`.** This is by design
  (`records.rs:419-426`), but the release-checks README (`:43`) describes
  stronger behaviour than the code has.
- **F8. `fixture_set_id` and `completed_fixtures` are recorded but never
  validated offline.** A record graded on a different or partial fixture set
  passes `validate` if its status says `passed`.
- **F9. Drafts not available.** `.cortexkit/alfonso/drafts/` is absent from
  this worktree, and the worker contract forbids reading the parent
  checkout, so design intent comes only from commit messages and `docs/`.
  No document there says that one mechanism replaces the other.
- **F10. Rerank tolerances differ. Embed tolerances agree.** For embeds,
  both mechanisms require cosine ≥ 0.999: the catalog in `evidence.rs:55`,
  the parity evaluator in `bench/parity/src/evaluator.rs:361`. For rerankers,
  every catalog backend pins `rerank_abs_tolerance: 0.005`
  (`catalog/models.json:154,166,178,357,369,381`). The evaluator's score gate
  allows 0.005 for the `fp32` class but **0.02 for the `fp16` class**
  (`evaluator.rs:392-396`), and most profiles are `fp16`
  (`bench/parity/models.json:448-545`). Certification is therefore up to 4×
  looser on reranker scores than catalog evidence would have been. The
  CUDA gte-reranker record (max abs error 4.94e-3) passes both only just. The
  catalog also requires zero order violations, while certification uses the
  evaluator's per-pool ranking metric. I did not establish whether the two
  ranking criteria are equivalent.
