# GTE reranker preload logits vs the committed baseline (2026-10-10)

## Verdict

The production GTE reranker (`gte-reranker-modernbert-base-f32`, expected
fingerprint `2fa5f24c0208f30c6db4bf18eb66bc0b2c46f765882cb744fcc58cd080b2b92d`,
`bench/parity/preload/gte-reranker-modernbert-base-f32.json:20`) has **not**
drifted. No code change and no OS update altered its numerics. The committed
baseline is stale for a benign reason: **it was captured from a debug
(unoptimized) build, and an optimized (`--release`) build produces slightly
different but equally accurate logits.** The optimized result is the same
today at master as at the commit that recorded the baseline.

- A debug build reproduces the committed baseline byte for byte, at the
  baseline commit `180c30f2` and at master `e42617cd`.
- A release build gives the same 500 bytes at both commits (SHA-256
  `7d6eff7f4816a80e703df7ea37ff765d92a751c82246e1e096966cdcc178b4a2`). Those
  bytes differ from the baseline in exactly the reported way: 123 of 125 logits
  differ, with a maximum absolute difference of 2.21e-5.
- Cause: `rope_tables` (`crates/synapse-engine-owned/src/modernbert.rs:1370-1385`)
  calls `f32::sin_cos` on each RoPE angle (line 1377). Rust's `sin_cos` makes two
  separate calls, `sin` and `cos`. With optimization on, LLVM on Darwin merges
  those two calls into one call to Apple's `__sincosf_stret`. Without
  optimization, the debug build calls `sinf` and `cosf` separately. The two
  routes disagree in the last bit for about 4% of RoPE table entries. That
  changes the GPU inputs, and the change reaches the final logit as a difference
  of about 1e-5.
- Against the fp32 CPU reference, the release logits are slightly *closer*
  than the baseline. Ranking on both candidate pools is identical: the full
  order is the same, Kendall tau is 1.0, and the top-10 matches.

Production binaries are optimized builds. The macOS release-candidate job
builds `synapse-module`, which links `synapse-engine-owned`, with
`cargo build --release` (`.github/workflows/release-candidate.yml:76-77`, `:192`).
So production has always served the release numerics, never the bytes in the
baseline file.

## 1. The baseline

- File: `crates/synapse-engine-owned/tests/hardware/evidence/preload-baseline.f32le`
  holds 125 little-endian f32 logits (500 bytes, SHA-256 `34c95ae0…04c72`).
- Only one commit touches it: `180c30f2b3ce2f0456d506fc91f3be53d0acd7c5`,
  2026-10-05 10:15:44 +0200, "mason: record passing real-weight Metal parity
  and preload identity" (`git log -- …/preload-baseline.f32le`).
- Source revision: master `d49576a7664d333e4d173adfa8758d850745d317`, whose
  engine tree `d3a6c0a2…` is the same as that of `6c306520`
  (`crates/synapse-engine-owned/tests/hardware/evidence/README.md:16-25`). The
  same 180c30f2 run confirmed byte identity against that commit's own engine
  source.
- Machine: Mac17,6, macOS 27.0.1 **(26A434)**, Xcode 27.0 (27A266a), cargo
  1.99.0 (`evidence/README.md:4-5`). The OS install history on this Mac shows
  macOS 27.0.1 installed on 2026-10-01 22:35 UTC
  (`/Library/Receipts/InstallHistory.plist`, `softwareupdate --history`). That
  is four days before the capture, and nothing has been installed on the OS
  since. The baseline was therefore taken on the build this Mac runs now
  (`sw_vers`: 27.0.1 / 26A434), not on 26A428. Xcode 27.0 (27A266a) and cargo
  1.99.0 are also unchanged.
- Build profile: the documented command has no `--release`
  (`crates/synapse-engine-owned/tests/hardware/README.md:6-12`). The baseline
  is a debug-profile capture.
- Tolerance: none. The test compares the full byte buffer with
  `assert_eq!(bytes, baseline, "preload logits changed")`
  (`crates/synapse-engine-owned/tests/hardware/src/lib.rs:221-224`). Any
  last-bit change fails it.
- Test inputs have not changed: the test scores `reference.cases()` in chunks
  of 8 (`src/lib.rs:209-217`). The reranker fixture was resealed in `149ba5ea`
  (2026-10-06). Between 180c30f2 and master, all 125 cases keep the same id,
  order, query, document and `input_ids`. Only the reference `output` values
  moved, in 95 cases, when the reference was regenerated with eager attention on
  a single thread.

## 2. Is the test run anywhere?

No. It is a manual, ignored test.

- `#[ignore = "requires verified GTE reranker weights; …"]`
  (`crates/synapse-engine-owned/tests/hardware/src/lib.rs:205`).
- The hardware crate declares its own `[workspace]`
  (`crates/synapse-engine-owned/tests/hardware/Cargo.toml:7-8`), so
  `cargo test --workspace` never builds it.
- CI: the test job's package list does not include it
  (`.github/workflows/tests.yml:43`). Metal lanes are not run on hosted runners
  (`.github/workflows/tests.yml:55-56`). The only `--ignored` run in a workflow
  is `synapse-core`'s `candidate_workers` (`.github/workflows/release-candidate.yml:235`).
- `mutations.toml`: it has no control that references the hardware crate, this
  test, or the baseline.
- Release checks: `crates/synapse-release-checks/check.py:194-197` checks only
  that the release notes carry the preload fingerprints. It does not run
  inference.
- A repository-wide search for `preload_gte_raw_logits`, `tests/hardware`,
  `synapse-metal-hardware-parity`, `METAL_PRELOAD_LABEL` and `preload-baseline`
  finds only the hardware crate's own files.

## 3. When did the logits change?

They never changed. The build profile, not the commit or the OS, decides which
bytes come out. Every run below used Metal on this Mac (26A434), with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Each source tree is
a `git archive` of the named commit under `/tmp`. The manifest-verified weights
were copied from the pinned Hugging Face snapshot, and all four SHA-256 digests
matched. Test binaries were copied to `ckdev-gte-preload-*` names and run one at
a time with `--ignored --exact tests::preload_gte_raw_logits_match_baseline`.
No production state was touched.

| Commit | Profile | Package cache | Result | SHA-256 of 500 bytes |
|---|---|---|---|---|
| 180c30f2 (baseline) | debug | fresh | **pass** | `34c95ae0…04c72` |
| e42617cd (master) | debug | fresh | **pass** | `34c95ae0…04c72` |
| e42617cd (master) | debug | warm | **pass** | `34c95ae0…04c72` |
| e42617cd (master) | release | warm (from debug run) | fail: 123/125 differ, max 2.21e-5 | `7d6eff7f…b4a2` |
| e42617cd (master) | release | fresh | fail: same | `7d6eff7f…b4a2` |
| 180c30f2 (baseline) | release | fresh | fail: same | `7d6eff7f…b4a2` |
| e42617cd, release, Objective-C forced to `-O0` (`CFLAGS_aarch64_apple_darwin=-O0`) | release | fresh | fail: same | `7d6eff7f…b4a2` |
| e42617cd, release, `synapse-engine-owned` at `opt-level=0` | release (engine unoptimized) | fresh | **pass** | `34c95ae0…04c72` |
| e42617cd, release, `/tmp` copy with `sin` and `cos` called through `std::hint::black_box` in `rope_tables` | release | fresh | **pass** | `34c95ae0…04c72` |

How to read the table:

- Same commit, same OS, two results. The deciding variable is the build
  profile, so no bisect is needed. Debug and release are each stable across
  180c30f2 and master, and the engine source that changed between them is
  numerically inert. The `src/` diff in `crates/synapse-engine-owned` is limited
  to moved functions, `is_multiple_of`, and `cfg_attr` lint attributes. None of
  the `.m` sources or `build.rs` changed.
- The Metal package cache cannot explain it: fresh and warm caches agree. The
  cache key already includes the OS build
  (`crates/synapse-engine-owned/src/lib.rs:935-944`).
- The difference is in optimized Rust code inside `synapse-engine-owned`, not in
  the Objective-C graph builders. Forcing the Objective-C to `-O0` leaves the
  release bytes unchanged. Unoptimizing the engine crate restores the baseline.
  Stopping LLVM from merging `sin` and `cos` in `rope_tables` restores the
  baseline in an otherwise optimized build.
- Standalone check of the mechanism: the same `rope_tables` body was compiled
  outside the repo with `rustc -C opt-level=0` and with `-C opt-level=3`, for
  8192 positions, head_dim 64, and theta 160000 and 10000. 80,094 of
  2,097,152 table values differ, by at most 5.96e-8. `nm` shows `_sinf`/`_cosf`
  in the `-O0` binary and `___sincosf_stret` in the `-O3` binary.

The reported failure (123/125, max abs 2.2e-5) matches the release row exactly.
The failing run was most likely built with `--release`, or with any optimized
profile.

## 4. Does the drift matter?

No. The reference is the fp32 CPU fixture
`bench/parity/fixtures/gte-reranker-modernbert-base/gte-reranker-modernbert-base.ref-v1.transformers-5.16.1.seed-0.json`
(torch, CPU, fp32, one thread, eager attention). Its `output` is the sigmoid of
the classifier logit (`bench/parity/reference/generate_reference.py:296-298`).
The comparison below is in that sigmoid space, which is the space the evaluator
gates in (`bench/parity/src/evaluator.rs:370`, `:392-396`; the fp32 class allows
0.005). It also gives logit-space error against `logit(reference)`.

| Reference | Output | max abs (sigmoid) | mean abs (sigmoid) | max abs (logit) | mean abs (logit) |
|---|---|---|---|---|---|
| master fixture | baseline (debug) | 6.87e-6 | 5.03e-7 | 2.81e-5 | 2.82e-6 |
| master fixture | release (today) | 6.81e-6 | 4.91e-7 | 2.73e-5 | 2.64e-6 |
| 180c30f2 fixture | baseline (debug) | 5.86e-6 | 5.56e-7 | 2.44e-5 | 3.04e-6 |
| 180c30f2 fixture | release (today) | 5.60e-6 | 4.16e-7 | 2.34e-5 | 2.33e-6 |

- Per case against the master fixture, release is closer in 68 cases, the
  baseline in 55, and 2 are bit-identical (`boundary-129`, `pool100-048`).
  Against the 180c30f2 fixture the split is 71 / 52 / 2. Both are about three
  orders of magnitude inside the 0.005 gate.
- Debug and release differ from each other by at most 5.42e-6 in sigmoid space
  and 2.21e-5 in logits. That is the same size as each one's distance from the
  reference.
- Ranking, using the evaluator's definition (`bench/parity/src/evaluator.rs:216-263`):
  `pool-10` gives tau 1.0 (43 concordant, 0 discordant) with top-10 true for
  both outputs. `pool-100` gives tau 1.0 (4232 / 0) with top-10 true for both.
  The full sorted order of every candidate in both pools is identical between
  baseline and release, with no ties. The smallest adjacent score gap in
  `pool-100` (1.3e-5 in sigmoid space) is larger than any debug/release
  difference that would reorder it, and no reordering occurs.

## Which bytes production serves

Production runs a release build. The macOS release-candidate matrix entry builds
`synapse-module`, which links `synapse-engine-owned`, among other packages
(`.github/workflows/release-candidate.yml:76-77`; dependency at
`crates/synapse-module/Cargo.toml:40`). It builds them with
`cargo build --release --locked` (`.github/workflows/release-candidate.yml:192`).
The production GTE reranker therefore serves the release bytes, SHA-256
`7d6eff7f4816a80e703df7ea37ff765d92a751c82246e1e096966cdcc178b4a2`, not the
committed debug baseline `34c95ae0…04c72`. The release bytes were the same at
the baseline commit and at master.

## Proposed fix (not applied)

Recommendation: **recapture the baseline from a release build, and make the
test refuse to run in a debug build.** The second part only works together with
the first, so the two belong in one change.

- Why release: the README describes the test as the way "to compare production
  preload logits" (`crates/synapse-engine-owned/tests/hardware/README.md:25`),
  and production is a release build. A debug baseline tests numerics that no
  shipped binary produces. Today it would pass while the shipped bytes moved,
  and fail while they stayed the same.
- Why also refuse debug: the comparison is exact (`src/lib.rs:224`), and debug
  and release differ by design (the `sin`/`cos` merge above). A release baseline
  without a guard would fail every plain `cargo test` run in the documented
  debug invocation (`README.md:10-11`). That failure would look exactly like
  real drift, which is how this question arose. At the top of
  `preload_gte_raw_logits_match_baseline`, add
  `assert!(!cfg!(debug_assertions), "run with --release: the baseline is production (optimized) numerics")`,
  or mark the test `#[cfg_attr(debug_assertions, ignore = "…")]`. Either turns
  the confusing byte mismatch into a clear instruction.
- Accompanying edits: add `--release` to the README commands; record the build
  profile next to the machine and toolchain in `evidence/README.md`; regenerate
  `evidence/preload-baseline.f32le` with `METAL_PRELOAD_LABEL=baseline` from a
  release build, which should give `7d6eff7f…b4a2` on this Mac.
- The rejected alternative is to keep the debug baseline and require debug runs.
  It keeps the test green, but the test then guards bytes production never
  serves, so it would not catch a real change in the release numerics.

## Notes

- Nothing in the repository was changed apart from this report. The test, the
  baseline, and the tolerance are untouched, as asked.
- `sin_cos` also appears in `modernbert.rs:775` (`apply_rope`) and in
  `qwen3.rs:626`, `:832` and `:1069`. Any byte-exact baseline taken over those
  paths will show the same debug/release split. This was not measured for the
  Qwen models.
- `d49576a` was not rebuilt separately. The evidence README records that its
  engine bytes matched 180c30f2 under debug. The debug run at 180c30f2 above
  reproduces the committed file exactly.
