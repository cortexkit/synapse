# Whole-export planning and variable-length Neural Engine packing

This experiment extends [equal-slot Qwen3 row dispatch](../ane-multirow-dispatch/README.md)
with whole-export planning and variable-length runtime packing on the same setup.
Only the Apple M4 Mac mini runs Neural Engine compilation and inference. The
serving Mac and the mini's `~/synapse` checkout (including its local harness edit)
are untouched. The experiment uses a separate clone.

## Conclusions, ranked by confidence

1. **High: tight 256-column packing is the best correct measured layout.**
   With the constant-compatible runtime rotary encoding it embeds the entire
   export in **101.12 s / 62.7 rows/s** at 64-row planning, **1.54×** today's
   paired single-row baseline. With production's eight-row engine slices it is
   **116.24 s / 54.5 rows/s**, **1.34×** today and **1.17×** the best measured
   84-executable equal-slot planner. These are complete-export measurements,
   not a per-pass projection or first-batch extrapolation.
2. **High: the rotary conversion policy mattered, not the accuracy gate.**
   Nearest-rounded runtime rotary fails the standalone gate on row 39 at all
   three widths and both windows. Reusing the constant encoder eliminates the
   difference: **6,341/6,341 bit-identical** in each of the six repaired audits
   and every timed replay, max abs **0.0**, no neighbour leakage, first/last
   identical. No standalone code, vendor encoding, or 0.9999 gate was changed.
3. **High: the eight-row production split has a real cost.** For tight 256,
   passes rise **2,877 → 3,317**, fill falls **92.2% → 80.0%**, and wall time
   rises **15.12 s / 15.0%**. The 64-row result is not a production service claim.
   Both windows still beat the equal-slot planners. Wider 384/512 passes are
   correct but slower; **512 under eight-row planning is slower than today**.
4. **High: this runtime-input design has no large cost penalty on this M4.**
   Nine alternating control passes measure repaired runtime mask/rotary at
   **35.41 / 64.30 / 93.21 ms** for 256/384/512 columns. Comparable constant
   arms cost 37.50 / 64.69 / 97.55 ms. Encode/decode adds **0.14–0.18 ms/pass**
   on the CPU and is included in whole-export timing. The earlier runtime
   *matmul operand* penalty does not transfer to these mask/rotary inputs.
5. **Medium: exact config rankings and reload penalties are less portable.**
   Equal-slot 84-config differences at eight rows are below 2 s, and first
   exports after recompilation vary substantially. Only one M4/OS build was
   tested, in-process, with the mini's normal background load. Production
   batching, shared budgets, IPC and other loaded models still need wiring.

## Part 2: tight packing after the encoding repair

Source `d9b83baa044e39ed7f4bf35d5762f2b18899e596`, PID 2269, Mac mini M4,
macOS 27.0 build 26A428, release Rust 1.99.0. The ignored audit passed its own
exit status. The committed `raw/tight-audit.json` is a **435,001-byte compact
report**: all 12 summaries from **two rotary encodings (nearest and
constant-compatible) × three widths (256/384/512) × two windows (64/eight)**,
counts covering **76,092**
comparisons, worst 20 rows with positions per summary, exact per-pass fill
histograms, position bins, and every accepted timing/fixture/setup record.
`summary.json` regenerates byte-for-byte from this committed report and the
clean planner confirmation; full per-row records are archived outside git.
Rejected nearest-rounded arms have audit durations for diagnosis, **no accepted
benchmark timings**. Only fully qualified repaired arms appear in the wall table.

### Per-width parity, with the worst row and position

The gate is cosine ≥ 0.9999 against **the same row run alone on the existing
single-row path**, not just against a common fp32 reference. Each cell represents
all 6,341 rows. Position is the first token's zero-based packed-column offset.

| Width | Rotary encoding | Window | Result | Min cosine | Max abs | Bit-identical | Worst row (tokens), position |
| ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| 256 | nearest | 64 | **fail, 1 row** | 0.999877415 | 0.001886189 | 0/6,341 | 39 (25), 133 |
| 256 | nearest | 8 | **fail, 1 row** | 0.999877415 | 0.001886189 | 0/6,341 | 39 (25), 28 |
| 384 | nearest | 64 | **fail, 1 row** | 0.999877415 | 0.001886189 | 0/6,341 | 39 (25), 242 |
| 384 | nearest | 8 | **fail, 1 row** | 0.999877415 | 0.001886189 | 0/6,341 | 39 (25), 265 |
| 512 | nearest | 64 | **fail, 1 row** | 0.999877415 | 0.001886189 | 0/6,341 | 39 (25), 130 |
| 512 | nearest | 8 | **fail, 1 row** | 0.999877415 | 0.001886189 | 0/6,341 | 39 (25), 265 |
| 256 | constant-compatible | 64 | **pass** | 0.9999999999999998 | 0.0 | 6,341/6,341 | 25 (34), 215 |
| 256 | constant-compatible | 8 | **pass** | 0.9999999999999998 | 0.0 | 6,341/6,341 | 25 (34), 145 |
| 384 | constant-compatible | 64 | **pass** | 0.9999999999999998 | 0.0 | 6,341/6,341 | 25 (34), 179 |
| 384 | constant-compatible | 8 | **pass** | 0.9999999999999998 | 0.0 | 6,341/6,341 | 25 (34), 145 |
| 512 | constant-compatible | 64 | **pass** | 0.9999999999999998 | 0.0 | 6,341/6,341 | 25 (34), 179 |
| 512 | constant-compatible | 8 | **pass** | 0.9999999999999998 | 0.0 | 6,341/6,341 | 25 (34), 145 |

The repaired “worst” row is a cosine-summation rounding tie: its output bits
are identical and its difference is zero. All 13 fitting fixtures, alone and
mixed, also pass. All six short-row neighbour/first-last isolation audits pass
bit-for-bit. The unchanged single-row fp32 gate remains at min 0.99988196,
max abs 0.00170860 over the 13 fitting fixtures; token composition matches 17/17.

**Error does not show a monotonic increase with packed position.** Row 39 has
exactly the same cosine/max-abs at offsets 28, 130, 133, 242 and 265. The sole
failure therefore moves between the 0–127, 128–255 and 256–383 position bins as
packing changes. Those bins also contain different row lengths because FFD
sorts by length, so they are not a controlled position-causality experiment.
Matching the encoder removes every difference at every observed position.

### Per-pass cost and runtime-input cost

Same full mixed pass per width, one warm-up per control, nine repetitions with
alternating order, three control programs resident (84 executables), baseline
released. Timed runtime controls include CPU masks/tables, staging, dispatch,
readback and normalization. The mask-only implementation also builds/discards
base rotary arrays; its total cost is an implementation measurement, not a
pure device-only mask increment. The repair-only CPU column uses 100 explicit
encode/decode measurements, excludes initial table construction, and includes
its transient allocation. Whole-export timing includes all of that work.

| Columns | Real segment lengths | Fill | Constant mask+rotary | Runtime mask, constant rotary | Runtime mask+compatible rotary | Runtime/all vs constants | CPU encode/decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 104 + 103 + 49 | 100% | 37.504 ms | 35.725 ms | **35.413 ms** | 0.944× | 0.150 ms |
| 384 | 110 + 105 + 105 + 64 | 100% | 64.689 ms | 62.801 ms | **64.299 ms** | 0.994× | 0.136 ms |
| 512 | 138 + 136 + 136 + 102 | 100% | 97.554 ms | 94.670 ms | **93.207 ms** | 0.955× | 0.176 ms |

No large runtime-input penalty is observed. Sub-percent differences, especially
at 384, are not evidence of a special faster device path. The CPU repair uses
65,536 / 98,304 / 131,072 coefficients and allocates 128 / 192 / 256 KiB total
fp16 temporary bytes per pass, sequentially across the two tables.

### Fill ratio and whole-export wall time

Every repaired arm holds the single-row 128/256 baseline plus one packed
program: **84 executables**. All rows fit the packed width; no single fallbacks,
evictions or recompiles occur within a timed export. The paired baseline is
repeated for each width, so gains use that width's same-process baseline.

| Method | 64-window passes / fill | 64-window wall | Rows/s | Gain vs paired today | 8-window passes / fill | 8-window wall | Rows/s | Gain vs paired today |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Today paired with 256 | 6,341 / rung padding | 155.23 s | 40.8 | 1.00× | same | 155.23 s | 40.8 | 1.00× |
| **Tight 256, repaired** | **2,877 / 92.2%** | **101.12 s** | **62.7** | **1.54×** | **3,317 / 80.0%** | **116.24 s** | **54.5** | **1.34×** |
| Today paired with 384 | 6,341 / rung padding | 155.88 s | 40.7 | 1.00× | same | 155.88 s | 40.7 | 1.00× |
| Tight 384, repaired | 1,896 / 93.3% | 121.43 s | 52.2 | 1.28× | 2,289 / 77.3% | 145.74 s | 43.5 | 1.07× |
| Today paired with 512 | 6,341 / rung padding | 155.57 s | 40.8 | 1.00× | same | 155.57 s | 40.8 | 1.00× |
| Tight 512, repaired | 1,431 / 92.7% | 133.27 s | 47.6 | 1.17× | 1,717 / 77.2% | 159.58 s | 39.7 | **0.97×** |
| Part 1 best 84 | 3,780 / equal slots | 129.50 s | 49.0 | 1.20×¹ | 4,014 / equal slots | 136.28 s | 46.5 | 1.14×¹ |
| Part 1 ceiling 112 | 3,785 / equal slots | 131.03 s | 48.4 | 1.19×¹ | 4,086 / equal slots | 141.14 s | 44.9 | 1.10×¹ |

¹ Part 1 gains use its own 155.73 s paired baseline. Direct wall comparisons
against Part 1 are between experiments, not newly interleaved arms: tight 256
is **28.1% faster in rows/s than the best 84 set at 64, 17.2% at eight**.
All raw repetitions are retained. The old ~82 s filled-pass estimate is not
realized: actual 64-window fill is 92.2% and the correct fully runtime pass costs
about 35 ms rather than the estimated 31 ms. Wider passes improve neither fill
enough nor per-column cost; do not extrapolate their theoretical pass count.

### First-after-reload measurements for repaired layouts

One additional first export after reload per width (not three); compare to that
width's warm median. The baseline 128/256 programs are readmitted after the
control phase; the tight program is freshly compiled. Tight first exports use
64-row planning. Setup/compile times remain outside these inference durations.

| Reloaded shape(s) | First export | Warm median | Measured extra |
| --- | ---: | ---: | ---: |
| single 128/256, 256-width experiment | 182.58 s | 155.23 s | +27.35 s |
| tight 256, constant-compatible | 103.66 s | 101.12 s | +2.54 s |
| single 128/256, 384-width experiment | 183.32 s | 155.88 s | +27.44 s |
| tight 384, constant-compatible | 121.52 s | 121.43 s | +0.09 s |
| single 128/256, 512-width experiment | 175.84 s | 155.57 s | +20.27 s |
| tight 512, constant-compatible | 140.10 s | 133.27 s | +6.83 s |

These are measured observations, not a fixed surcharge or an explanation of
compiler/cache behavior. Normal load endpoints for accepted samples span
1.41–7.33; an eight-row 256 sample reaches 7.33 and takes 117.56 s, and remains
in the median. No concurrent Neural Engine client was reported during PID 2269's
exhaustive comparison/timing run; per-sample UTC bounds are retained.

## Part 1: clean whole-export results

All numbers below use `raw/planner-confirm-clean.json` (source
`a0f0da1b549032e18cbadd9bbfea25d1d50f4d8c`, PID 61406), not the quarantined
confirmation. They are medians of three alternating, complete-export samples.
The baseline is identical for either planning window because it runs the same
rows in the same order without grouping.

### Equal-slot planning: wall time and throughput

Every candidate includes a single-256 fallback for rows no packed slot fits
(including all 129–256-token rows). Shapes in this table are additional
width-folded programs, not a replacement single-row 128 rung. “Today” means
one row at a time on its smallest 128/256 rung; gain is the baseline median
wall time divided by the candidate median wall time.

| Method | Resident executables | 64-row window wall | Rows/s | Gain vs today | 8-row window wall | Rows/s | Gain vs today |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Today: single 128/256 | 56 | 155.73 s | 40.7 | 1.00× | 155.73 s | 40.7 | 1.00× |
| Full planner: 8×32 + 4×64 + 2×128 — **ceiling, does not fit production** | 112 | 131.03 s | 48.4 | 1.19× | 141.14 s | 44.9 | 1.10× |
| 8×32 + 2×128 — **deployable** | 84 | 132.39 s | 47.9 | 1.18× | **136.28 s** | **46.5** | **1.14×** |
| 4×64 + 2×128 — **deployable** | 84 | **129.50 s** | **49.0** | **1.20×** | 138.08 s | 45.9 | 1.13× |
| 2×64 + 2×128 — **deployable** | 84 | 131.46 s | 48.2 | 1.18× | 137.18 s | 46.2 | 1.14× |

**The full planner is not faster than the best 84-executable set.** All of its
packed tensors are still 256 columns wide. Separating rows into more narrow-slot
bins creates more partially occupied passes. With eight-row planning the full
planner needs 4,086 total passes versus 4,014 for 8×32 + 2×128; with 64-row
planning it needs 3,785 versus 3,780 for 4×64 + 2×128. More resident shapes are
therefore not an automatic throughput gain.

For the same full planner, the eight-row split costs **10.11 s / 7.7%** in warm
wall time. For 4×64 + 2×128 it costs **8.59 s / 6.6%**. Choosing the best measured
set separately for each window gives **129.50 → 136.28 s**, an extra **6.78 s /
5.2%**. The three 84-executable contenders differ by less than 2 s at eight rows;
this is a measured ranking, not a statistically established unique optimum.
The 8×32 + 4×64 pair from the initial sweep was plainly worse (median 241.75 s
at 64, 222.77 s at eight), because 4,185 rows of 65–128 tokens use the wider
single-row fallback. It was not included in the warm confirmation.

### Real pass counts and slot fragmentation

| Set | Window | Engine windows | 32-slot passes | 64-slot passes | 128-slot passes | Single-256 passes | Total passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ceiling | 64 | 127 | 41 | 167 | 2,174 | 1,403 | 3,785 |
| Ceiling | 8 | 826 | 61 | 291 | 2,331 | 1,403 | 4,086 |
| 8×32 + 2×128 | 64 | 127 | 41 | — | 2,421 | 1,403 | 3,865 |
| 8×32 + 2×128 | 8 | 826 | 61 | — | 2,550 | 1,403 | 4,014 |
| 4×64 + 2×128 | 64 | 127 | — | 203 | 2,174 | 1,403 | 3,780 |
| 4×64 + 2×128 | 8 | 826 | — | 335 | 2,331 | 1,403 | 4,069 |
| 2×64 + 2×128 | 64 | 127 | — | 349 | 2,174 | 1,403 | 3,926 |
| 2×64 + 2×128 | 8 | 826 | — | 436 | 2,331 | 1,403 | 4,170 |

### First export after reload, and compilation separately

The following penalties are **measured paired differences**, first full export
after program reload minus the warm sample for the same method, repeat and
window. In the three measurement cycles, candidate phases run window 64 first
on cycles 0/2 and window eight first on cycle 1. A
production eviction/reload can pay this penalty *in addition* to compilation;
warm wall time is not a cold-start service claim. Sample ranges are included
because the first-export effect is variable. Setup includes admission/compilation
after eviction, not eviction duration; none is charged to warm inference walls.

| Method | Shapes reloaded | Median admit/compile setup | Median first-export extra | Paired extras (s) |
| --- | --- | ---: | ---: | --- |
| Today | single 128 + single 256 | 33.88 s | +17.10 s | 46.57, 17.10, 3.04 |
| Ceiling | single 256 + 8×32 + 4×64 + 2×128 | 73.90 s | +24.89 s | 15.68, 53.01, 24.89 |
| Deployable 32/128 | single 256 + 8×32 + 2×128 | 69.25 s | +20.96 s | 20.96, 15.13, 30.69 |
| Deployable 64/128 | single 256 + 4×64 + 2×128 | 61.38 s | +14.90 s | 1.89, 17.65, 14.90 |
| Deployable small64/128 | single 256 + 2×64 + 2×128 | 57.67 s | +16.53 s | 5.73, 17.53, 16.53 |

### Parity and wider fallback

Every row in every warm and first-after-reload sample is bit-identical to the
same row run alone: **6,341/6,341**, minimum computed cosine
**0.9999999999999998**, maximum absolute difference **0.0**, 1,024 dimensions.
All 17 fixture token compositions match; the 13 fixtures fitting the resident
128/256 single-row rungs pass fp32 at minimum cosine **0.99988196** (gate 0.999),
maximum absolute difference **0.00170860**. Longer fixture rungs are outside
this export's residency set, not silently counted as checked.

The explicit-256 fallback passes ten real 65–128-token rows before timing:
all bit-identical to rung 128, cosine at least **0.9999999999999998**, max abs
**0.0**. Their measured median call cost is **20.91 ms at 128 versus 36.46 ms at
256**. The ~31 ms from the prior experiment describes a 256-column *folded*
pass, not this single-row 256 rung. The extra fallback cost is counted in the
short-only 84-configuration's wall. No row above 256 occurs in the export;
in-wall eviction/recompile counts are exactly zero for every configuration.

The one-minute system load average (dimensionless), read before and after each
clean warm sample, ranges 1.67–4.29. Raw JSON retains every
sample, clock boundary, parity result and setup time; no outlier is dropped.

## Measurement contract

- The exact same 6,341 composed export rows, 679,090 tokens, and 127 metadata
  batches run on every path. The original tokenizer/document/EOS composition is
  checked against all 17 fp32 fixture token-id lists. Raw reports contain hashes,
  counts, case names and numbers, not input text or token IDs.
- **64-row window:** preserve the export's metadata batches, not arbitrary groups
  over the flattened export. **8-row window:** split each of those batches into
  consecutive groups of at most eight, including its final short group. Never
  merge tails across metadata batches. This models the module's recommended
  eight-row engine dispatch; it does not time module IPC or tokenization.
- Whole-export inference walls include embedding gather, per-layer dispatch,
  mask/rotary construction and copies where needed, output readback, CPU final
  norm, output-order restoration, and planning. Package load, tokenization,
  compilation and parity comparison are outside the clock. The initial sweep
  warms each shape once. The confirmation measures the first whole export after
  each reload separately, then treats that complete pass as warm-up before its
  three steady-state repetitions. Method/window order reverses on odd repeats.
  Baseline reloads and first-after-reload penalties are measured the same way.
- For the full equal-slot planner, the baseline holds two single-row programs
  (128/256, 56 executables). Before candidate phases these are released; only
  the 256 fallback plus the selected packed programs remain. The full planner
  holds **112: a ceiling that does not fit production**, whose shared budget is
  100 across *all* loaded direct-ANE models. Three shapes (84 executables) are
  the maximum Qwen3 set below that budget, and only when other models leave room.
  Residency setup/reloads are reported separately, not charged to steady-state
  wall time. These are opt-in experiments, not a serving policy.
- All pairs of the three throughput-leading equal-slot programs (8×32, 4×64,
  2×128) are measured, plus 2×64 + 2×128 to check sparse tails. The planner
  assigns each row to its narrowest fitting slot. Prior passing shapes with
  larger total widths were slower; retaining every passing shape simultaneously
  would exceed even the process ceiling. Empty slots still cost a full pass.
- The pair without 2×128 requires an explicit **single-256 fallback** for real
  65–128-token rows. Ordinary `Model::run` still chooses the smallest rung;
  opt-in `run_at_rung` rejects undersized/non-ladder widths and uses the real
  token count for mask and last-token pooling. Ten real rows are checked against
  rung 128 before any candidate timing. The wider fallback's cost is included
  in candidate walls, not treated as a 128-rung call.
- There are **zero rows above 256**, hence zero in-wall evictions and recompiles
  for this export. A future 84-executable configuration would need to evict a
  packed shape to admit 512/1024, and then pay to restore it. Those costs are
  **not measured here**. The harness refuses an export needing those rungs
  rather than quietly reporting an unrealistically cheap steady-state wall.

## Run isolation and quarantined data

The first confirmation that warmed the complete export after each reload
(`raw/planner-confirm-contaminated.json`,
source commit `5d531056`) ran in PID 42652 from 00:29Z for 7,398 seconds. The
operator reported another Synapse Neural Engine reproduction from **2026-10-10
01:01Z to approximately 01:08Z**, overlapping that test. It passed output gates,
but **none of its timings are used for conclusions or configuration selection**.
Its reports lack per-sample UTC boundaries, so the entire confirmation is
quarantined and redone rather than guessing which samples are unaffected.
The earlier `planner-exploratory.json` sweep finished before that overlap.

Subsequent reports include process ID, source commit, and each timed sample's
UTC start/end in milliseconds to identify overlap. Monotonic `Instant` measures
inference duration independently of any UTC clock adjustment.
The busy check matches the Rust test executable's underscore spelling
(`ck_synapse_worker_ane_direct-*`) as well as production worker names; matching
only a Synapse application name can miss an ignored hardware test.

## Rotary conversion mismatch and qualified repair

The first tight experiment stopped before any whole-export benchmark samples:
row 39 failed at cosine **0.999877415** against that same row run alone (required
0.9999). Its 256-column `[104, 103, 49]` control had bit-identical constant and
runtime-mask-only output, but nearest-rounded runtime rotary differed by up to
**0.000789616**, at minimum cosine **0.999985629**. Short neighbour/first-last
checks still passed exactly. `raw/tight-initial.json` preserves this failed
qualification attempt, not an accepted speed-up.

The binding has **different fp32→fp16 policies**: `Graph.constant` uses
`WeightBlob::from_f32` and the truncating `ops/weights.rs::f32_to_f16`; runtime
`TensorData::copy_from_f32` uses NEON `fcvtn`/`fcvtn2` or nearest-rounded
`half::f16::from_f32`. Both inputs are fp16. They are not necessarily the same
fp16 bits. The unit control identifies a real coefficient difference: cosine
coefficient 1 at width 256 stages as bits **14419** under nearest rounding,
versus **14418** in the actual graph-constant MIL blob.

The repair leaves tables computed in fp32 on the CPU, then reuses
`ane::f32_to_fp16_bytes` (the same encoder core as graph constants), decodes those
bits into exactly representable fp32 values, and uses the existing runtime
staging path. It adds no inputs, executables, or vendor changes. The encode/decode
occurs inside each `TightProgram::run`, so accepted timing includes it. Pure CPU
unit tests compare the staged bits with the actual `Graph.constant` payload for
both tables at all three widths, not a duplicate copy of the encoding algorithm.

`multirow::replay::audit::tight_packing_audit` records every row under original
and constant-compatible runtime rotary: 76,092 comparisons across three widths,
two planning windows and two encodings. Records carry hashes, lengths, the
window/pass, row ordinal, start/last-token positions, cosine and max abs. No failed arm is admitted to whole-export timing. The public tight entry point
now accepts only the fastest qualified 256-column compatible layout on exact
pinned Qwen3 metadata. Other widths/encodings remain diagnostic-only; no
unqualified profile or nearest-rounded layout is silently enabled.

The submitted graph has ordinary fp16 multiplies; private-compiler fusion of
constant operands is not visible here. Matching coefficient bits isolates the
known conversion difference without guessing about that compiler behavior or
changing the standalone path. The binding's older IOSurface fp32 comment also
conflicts with the active two-byte `TensorData` allocation and fp16 MIL I/O;
this experiment follows the active implementation and does not change the
vendor's encoding or documentation.

## Tight layout and opt-in prototype

`crates/synapse-worker-ane-direct/src/tight_packing.rs` is a child of `multirow`.
The diagnostic constructor compiles 28 layer executables at fixed width 256,
384 or 512. The public `Model::compile_tight_packing(256)` accepts only the
pinned Qwen3 profile, rejects other widths/metadata, and uses constant-compatible
runtime rotary. `RECOMMENDED_TIGHT_WIDTH` is 256. `TightProgram::run(model, rows)` places actual tokens back to
back, fills only the unused suffix with padding, and returns vectors in supplied
row order. `plan_tight(lengths, width)` is stable first-fit-decreasing within one
engine call and carries original indices for output restoration. Over-width
rows remain single-row fallbacks.

Each Qwen3 decoder layer has four runtime inputs, in declaration order:

| Input | NCHW shape | CPU construction per pass |
| --- | --- | --- |
| Activations | `[1, 1024, 1, width]` | token embedding gather |
| Mask | `[1, 1, width, width]` | block-causal bias, zero only for same-row past/current keys |
| Cosine | `[1, 1, 128, width]` | each segment's positions restart at zero |
| Sine | `[1, 1, 128, width]` | same segment positions |

Padding queries attend only themselves to keep softmax finite; real queries
never attend padding. Query tiles still compute cross-row scores and then mask
those scores away. Unlike width-folded slots, tight packing does **not** eliminate
that attention arithmetic. Runtime `TensorData`/IOSurface objects are persistent
and only their contents change, as required by the binding's cached requests.

To measure runtime cost rather than assume it, the same high-fill real mixed
pass runs with (1) mask/rotary constants, (2) runtime mask and constant rotary,
and (3) qualified runtime mask/constant-compatible rotary. These three programs occupy 84 executables
with the baseline released; nine timed passes each alternate mode order after
warm-up and parity. Constants fix segment lengths only in these comparison
arms, never in the whole-export candidate. CPU operand-building time is also
reported separately (100 encode/decode runs). The baseline 128/256 programs are then
restored; each fully runtime packed width runs at 84 total executables.

The request loop and `synapse-module` are not wired to any new API. The ordinary
single-row path's graph digest, fp32 gate, mask and last-token readout remain
covered. Backend factoring adds only the opt-in wider-rung entry; ordinary
`run` selects the same rung as before.

## Verification and non-vacuity

The CPU-only unit suite does not require a Neural Engine; all hardware
experiments are ignored by default. New tests cover first-fit-decreasing ties,
coverage and over-width fallback, independent causal/cross-row mask expectations,
rotary restarts at variable boundaries, real segment readout positions, four
actual compiled-graph inputs, explicit-rung refusals and metadata-window tails.
The graph-input test uses the same graph builder as compilation, not a proxy
with placeholders manufactured only by the test.

`raw/non-vacuity.json` records safe staged/restore controls. Replacing the
runtime mask with a constant turns only
`graph_has_four_runtime_inputs_and_no_constant_segment_geometry` red; removing
the undersized-rung refusal turns only
`explicit_fallback_refuses_narrow_nonladder_empty_and_nonresident_rungs` red.
The input-text fence is also exercised by inserting a synthetic `text` key in
the actual clean report: only `raw_reports_do_not_export_inputs` fails, while
JSON parsing and digest validation stay green. Neutralizing the compatible
coefficient encoder turns only
`runtime_rope_coefficients_match_actual_graph_constant_bits` red; the other
32 unit tests remain green. All mutations are restored before rebuilding the
hardware binary. `verify-raw.py` checks published JSON for payload fields and
canonical SHA-256 digests without displaying values. The optional external-archive
check also has a staged/restore control: a syntactically valid wrong expected
digest turns only `full_raw_matches_compact` red, while parsing, privacy, digest
syntax and compact aggregate checks stay green.

## Unresolved and deployment limits

- One M4/OS build only; other chips or OS/compiler releases can choose different
  kernels and must repeat qualification. Dataset-wide bit identity here is not
  a universal proof for all possible token sequences.
- No serving wiring, IPC/tokenization timing, multi-client contention, model
  eviction policy, or shared-budget scheduling is implemented. The 84 budget
  assumes no other model consumes the remaining capacity. The prototype owns
  28 extra executables; its caller must budget and serialize access.
- This export has no rows over 256. Production traffic needing 512/1024 single
  rungs would require admission/eviction/recompile work not represented by this
  export; the 84 equal-slot replay deliberately refuses such an export.
- The binding's truncation/underflow policy is retained for compatibility,
  **not** proposed as a generally more accurate converter. Changing it globally
  would change existing standalone vectors and is outside this experiment.
- Private compiler fusion/cache details and the first-export reload penalty's
  cause are not measured. Matching public-source encoder bits is sufficient for
  the observed correction; it does not expose internal ANE arithmetic.
- FFD within each eight-row call is not a global optimum. The 15% split penalty
  for tight 256 is real on this data; buffering across calls could improve fill
  but would change latency/serving semantics and is not prototyped here.
- Equal-slot best-config differences are small; only the tested shape sets are
  ranked. Non-power-of-two slot widths and tail-specialized dispatch are not
  explored. Wider tight shapes pass but are not enabled by the public prototype.

## Files and retained evidence

- `raw/planner-exploratory.json`: initial three-repeat, per-shape-warmed sweep.
- `raw/planner-confirm-contaminated.json`: excluded because a separate Synapse
  Neural Engine reproduction overlapped the measurements.
- `raw/planner-confirm-clean.json`: isolated redo with UTC/sample/source fields.
- `raw/tight-initial.json`: nearest-rounded rotary's initial failed qualification.
- `raw/tight-audit.json`: compact counts/extrema, worst 20 rows, fill histograms
  and position bins for all 12 encodings/width/window combinations; all accepted
  timing repetitions, fixtures and setup are retained.
- `raw/non-vacuity.json`: safe mutation controls and exact red/green test names.
- `summary.json`, `summarize.py`: derived tables and the generator.
- `verify-raw.py`: JSON/input-payload/digest fence; `run.sh`: mini busy-check and
  exact ignored-test invocation.

The **full raw audit is outside git**, at
`~/.local/share/cortexkit/synapse/evidence-raw/ane-tight-packing/tight-audit.json`:

- Byte size: **38,967,456**.
- SHA-256: **`336928bd24e969afbfa1c1dccb8a8140b53b159e4a16ab1734c176b8f65343b3`**.

The compact report also records this fingerprint. `verify-raw.py --full-raw PATH`
optionally checks the archive's payload/digests, exact size/hash, and whether its
records regenerate the committed compact report. Default checks cover JSON
parsing, forbidden input-payload fields, digest syntax, and compact aggregate/
size consistency. Those checks and all tables need only committed files, not
the full archive. `summarize.py
--compact-full PATH --out FILE` produces the compact form, including per-pass
occupancy histograms, from the immutable full audit.

The experiment's commit history was rewritten to remove the 39 MB raw blob,
not merely delete it in a later commit. The range
`80a055510ae0b2cec13b82799651343886f9a362..HEAD` covers every experiment commit
after its original baseline. Original `source_commit` values
remain execution provenance rather than checkout refs in the rewritten task
history; the delivered harness and numerical implementation are retained.

Committed evidence is kept under `docs/evidence`, never in a regenerable Cargo target.
The mini scratch is retained at **`~/mason-ane-tight-bg10a30` (2.5G from
`du -sh`)** for reproduction: own clone, build directory, linked converted
package and raw runs. No experiment process remains running. The mini's
working `~/synapse` checkout and the previous experiment directory were not
edited or cleaned up.

## Reproduce

Never run the ignored hardware tests on a machine serving from the Neural
Engine. Before **each** hardware invocation `run.sh` refuses if another
`ckdev-*`, worker, Synapse, cargo or rustc process is active; wait for its owner
instead of stopping it. Do not use the mini's working `~/synapse` checkout as a
scratch directory.

From the task checkout, transfer a committed branch using a bundle, then on the
mini use your own clone (or a fresh Git worktree), not hand edits:

```sh
git bundle create /tmp/ane-packing.bundle HEAD
scp /tmp/ane-packing.bundle macmini:ane-packing.bundle
ssh macmini
root="$HOME/mason-ane-tight-bg10a30"
git clone --shared "$HOME/synapse" "$root/synapse"
git -C "$root/synapse" fetch "$HOME/ane-packing.bundle" HEAD
git -C "$root/synapse" checkout -b ane-packing FETCH_HEAD
source ~/.cargo/env
export CARGO_BUILD_JOBS=6 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR="$root/target"
cd "$root/synapse"
env -u TMPDIR cargo test --release --locked -p synapse-worker-ane-direct --no-run
# Set this to the unit-test executable printed by that command, not an old binary:
export ANE_PACKING_BINARY="$root/target/release/deps/ck_synapse_worker_ane_direct-<hash>"
mkdir -p "$root/packages" "$root/runs"
# Reuse only a converted package matching the complete pinned SHA-256:
cp "$HOME/mason-ane-multirow/packages/qwen3-embedding-0.6b.safetensors" "$root/packages/"
shasum -a 256 "$root/packages/qwen3-embedding-0.6b.safetensors"
# Expected: 1af6d091d7d5f21a998af0f438333d9af051c82af8b27bd2439c60efa09bd5db
sh docs/evidence/ane-tight-packing/run.sh "$root/runs/planner-exploratory.json" \
  multirow::replay::whole_export_planner
ANE_MULTIROW_CONFIRM=1 sh docs/evidence/ane-tight-packing/run.sh \
  "$root/runs/planner-confirm.json" multirow::replay::whole_export_planner
full="$HOME/.local/share/cortexkit/synapse/evidence-raw/ane-tight-packing/tight-audit.json"
mkdir -p "$(dirname "$full")"
sh docs/evidence/ane-tight-packing/run.sh "$full" \
  multirow::replay::audit::tight_packing_audit
python3 docs/evidence/ane-tight-packing/summarize.py \
  --compact-full "$full" --out docs/evidence/ane-tight-packing/raw/tight-audit.json
python3 docs/evidence/ane-tight-packing/summarize.py \
  "$root/runs/planner-confirm.json" docs/evidence/ane-tight-packing/raw/tight-audit.json \
  --out summary.json
python3 docs/evidence/ane-tight-packing/verify-raw.py \
  docs/evidence/ane-tight-packing/raw summary.json --full-raw "$full"
```

`run.sh` defaults `ANE_PACKING_ROOT` to the scratch root above; set that variable
if using another clone. It reuses the specified pinned tokenizer/checkpoint and
the mini's export, unsets `TMPDIR`, and runs exactly one ignored test with one
test thread. The scratch build/cache is separate from production.

## Module-serving measurement driver

`crates/synapse-module/examples/ane_packing_serving.rs` measures serving rather
than the earlier worker-only replay. It installs the checksum-pinned Qwen
catalog lane in a private module/store, with a loopback checkpoint server and
no daemon discovery. Only `ane_direct.tight_packing` differs between arms;
the default eight-row engine quantum and 3,072-token quantum budget remain
unchanged. The export's original 127 calls (up to 64 rows each) run with two
calls in flight, covering all 6,341 rows exactly once per repetition.

The default order is serial, packed, serial, packed, serial, packed. Each
fresh private module is warmed before its timed export. A second stratum
makes 1,000 warm `embed.query` calls while two document-batch callers cycle
through the export continuously. Query text is deterministically export row
zero, not chosen to improve a result. All document vectors, including bulk
background results, and query vectors are compared by `f32::to_bits()` against
the first serial reference. Reports contain indices, counts, hashes and
timings, never export text or vector payloads.

Both arms use the same Unix-socket frame relay. It preserves the worker's
HELLO identity/capabilities and raw frames, records operation names without
tensor payloads, and passes the inherited global ANE lane-lock descriptor as
the real worker's stdin. The module still acquires the normal global lock;
neither `HOME` nor the lock location is overridden. A normal module request
must send `EMBED_PACKED` after the module's standalone-versus-packed fixture
checks; successful embeddings alone are not evidence that packing was used.
The relay also records actual worker call sizes to enforce the eight-row cap.

Packed compilation is reported as the worker-side duration of the first
`admit-packed` request, from the existing worker profiling log. This includes
admission/request handling and response writing, not just compiler internals;
it excludes catalog download, model load, standalone reference generation
and packed qualification. Later admissions are retained separately so a
recompile cannot be mistaken for a one-time cost. Profiling and the relay are
enabled identically in both arms and their overhead is part of serving time.

### Reproduce the module comparison

Obtain approval for the target device before **each** invocation. If another
Neural Engine client is present, ask its owner rather than stopping it. The
driver checks both hyphenated and underscored Synapse names and `ckdev` names
before the run and before each fresh arm. Use an isolated copy of the committed
source, never a serving checkout or production store. Verify the transfer's
SHA-256 at the destination. Keep raw JSON, profiles, module logs, private
configs and stores outside the repository and outside regenerable build trees.

In the isolated source copy, build native macOS binaries (serialized Cargo
invocations) with Xcode selected:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo build --release --locked -p synapse-module --features test-support \
  --bin ck-synapse --example ane_packing_serving
cargo build --release --locked -p synapse-worker-ane-direct
```

Copy the example to a `ckdev-ane-packing-serving` alias and ad-hoc sign it:

```sh
cp "$BUILD_ASSETS/examples/ane_packing_serving" "$RUN_ASSETS/ckdev-ane-packing-serving"
codesign --force --sign - "$RUN_ASSETS/ckdev-ane-packing-serving"
codesign --verify --strict "$RUN_ASSETS/ckdev-ane-packing-serving"
env -u TMPDIR SYNAPSE_PACKING_APPROVED=1 \
  SYNAPSE_COMPARE_ASSETS="$BUILD_ASSETS" \
  SYNAPSE_QWEN_WEIGHTS="$PINNED_CHECKPOINT_DIRECTORY" \
  SYNAPSE_PACKING_INPUT="$FULL_EXPORT_JSONL" \
  SYNAPSE_PACKING_OUT="$RAW_OUTPUT_JSON" \
  "$RUN_ASSETS/ckdev-ane-packing-serving"
```

The driver copies and ad-hoc signs its private module, real worker and relay
aliases before executing them. The output must not already exist. At least
three repetitions and 1,000 queries per arm are enforced even when overriding
`SYNAPSE_PACKING_REPEATS` or `SYNAPSE_PACKING_QUERY_SAMPLES`. A failed acceptance
gate exits nonzero **after** preserving the measurements; do not change the
workload or thresholds to make it pass.

Run the driver's hardware-free unit tests and strict lint check:

```sh
cargo test -p synapse-module --example ane_packing_serving --features test-support
cargo clippy -p synapse-module --example ane_packing_serving --features test-support -- -D warnings
```

The tests cover export coverage, ID-based vector restoration, signed-zero bit
identity, nearest-rank percentiles, median/tolerance gates, refusal of absent
packed dispatch or an active serial switch, and payload-free socket relaying.
Forcing `assess`'s `packed_dispatch_and_off_switch` check to return true
regardless of worker exchanges fails only
`gate_tests::acceptance_requires_real_packed_dispatch_and_serial_off_switch`;
the seven other tests in that mutation run remained green.

## First module-serving measurement: failed, incomplete

Measured on Apple M4 on 2026-10-11 at source
`b755101caef11f00dff9c3dd916fad93c01560fc`. Owner approval covered the device
and all invocations. Busy checks passed before invocation and each arm. All
ANE clients were ad-hoc signed `ckdev-*` aliases, with private configuration,
lease and store roots. A retained `lsof` snapshot shows the module holding the
global lane lock and both relay and real worker retaining it on descriptor
zero. No M5 inference or production-store access occurred.

**Acceptance is not passed.** The order was serial/packed/serial/packed/serial/
packed, but the final packed export failed with `model_loading`. Three serial
and only two packed arms completed; no extra packed arm was stitched in, no
failed sample was discarded, and thresholds/workload were not adjusted.
The completed-arm medians below are descriptive, not a rollout approval.

| Arm | Full export (s) | Outer-call p50 / p99 (ms) | Query-under-bulk p50 / p99 (ms) | First packed admission/compile (ms) |
| --- | ---: | ---: | ---: | ---: |
| Serial 1 | 225.412 | 3,616.166 / 7,312.260 | 377.943 / 696.911 | — |
| Packed 1 | 156.940 | 2,559.450 / 4,516.341 | 306.779 / 592.834 | 22,561.678 |
| Serial 2 | 218.120 | 3,283.177 / 9,960.611 | 378.574 / 897.053 | — |
| Packed 2 | 164.756 | 2,526.325 / 6,059.445 | 304.478 / 579.562 | 26,497.272 |
| Serial 3 | 208.078 | 3,305.101 / 6,750.708 | 397.443 / 921.245 | — |
| Packed 3 | **Failed mid-export** | Not a completed measurement | Not reached | Retained in failed-arm profile |

Each completed export returned 6,341 bit-identical vectors: 31,705 completed
row comparisons, plus 5,000 identical query vectors and 68,723 identical
background-bulk vectors, with zero mismatches. Every complete export has the
same ordered f32-bit digest:
`412b933a40a22143a06325e8df8950ce419e7f7e71bb7369c02720f4067b5811`.
The pinned fingerprint remains
`5f7ac7cdbf63e71f04405b2130b2bde8eb7b307e5d929419bcd747b129806a6b`.
The completed packed exports each sent 825 `EMBED_PACKED` exchanges and 81
single exchanges; serial exports sent 931 single exchanges and no packed
exchanges. The largest observed engine call was eight rows. Each complete
packed arm compiled the packed shape once, with no later packed re-admission;
median first compile-request wall time was 24,529.475 ms.

Among completed arms only, median export times were 218.120 s serial and
160.848 s packed (**0.73743×**, below 0.80×). Pooled warm query p50/p99 were
382.198/844.604 ms serial (3,000 samples) and 305.183/587.734 ms packed
(2,000 samples), below the respective 420.418/929.064 ms limits. Missing the
third packed repetition prevents these descriptive successes from clearing
acceptance.

### Initial incident diagnosis from retained evidence

The final packed arm started at 14:26:12 UTC. Its private store retains one
passed self-check, generation 1, checked at 14:26:49.403 UTC, with unchanged
identity. There is one worker profile PID, one `LOAD`, one `PING`, one packed
admission, and one single-shape eviction during preparation. The 103 completed
wire exchanges include 65 packed executions and no `ERR` responses. No packed
eviction, packed re-admission, second load or second worker-profile PID is
recorded. The retained module log has no resource-exhaustion or recovery
warning. These observations are not a substitute for a durable crash ledger;
the private store has no general worker crash/recovery event table.

The slowdown is explicit in module log lines (UTC):

```text
2026-10-11T14:27:51.281Z INFO  synapse.perf: job done model_id=qwen3-embedding-0.6b-ane job_id=inline-1-1 lane=ane-direct tokens=2489 wall_ms=844
2026-10-11T14:28:14.659Z INFO  synapse.perf: job done model_id=qwen3-embedding-0.6b-ane job_id=inline-1-7 lane=ane-direct tokens=7112 wall_ms=18452
2026-10-11T14:28:32.325Z INFO  synapse.perf: job done model_id=qwen3-embedding-0.6b-ane job_id=inline-1-10 lane=ane-direct tokens=1997 wall_ms=13900
```

The worker profile ends with successful vector-producing exchanges at
829.583, 1,100.528, 1,582.756 and 1,872.109 ms. The last is `embed-101`,
completed at Unix microsecond `1791728925474414` (14:28:45.474 UTC). The
retained module profile records a 4,661.794 ms direct-ANE roundtrip and an
11,069.915 ms handler interval near the failure. The terminal response was:

```text
embed.batch: model_loading: catalog lane 'qwen3-embedding-0.6b-ane' is loading
```

`resolve_serving_model` and `catalog_resolution_guard` in
`crates/synapse-module/src/lib.rs` cap catalog lifecycle/readiness waits at
five seconds even with the private 600-second inline deadline. Their timeout
response uses that exact `model_loading` message. It therefore does **not**
establish a worker crash or lane reload. The worker exchange timeout is
separately 30 seconds, not five. The relay's retained `Broken pipe (os error
32)` can result from aborting the remaining in-flight call when the harness
exits on the first refusal; it does not establish the initiating cause.

The data supports a catalog-resolution/readiness timeout during the slowdown,
but does not settle why execution/readiness work slowed, or whether the
five-second timer expired waiting for the lifecycle lock or the detached
readiness task. Worker stderr and module stderr are empty. No RSS or system
memory-pressure samples were collected in this first run, so memory pressure
on the 16-GB M4 cannot be ruled out. An owner-approved second **full** matched
comparison adds memory sampling and retained stderr, without changing module
code, workload, private module configuration, or thresholds.

### Retained first-run evidence and reduction

`module-serving-first.json` is the compact failed-run summary, containing
hashes, counters and independently recomputed statistics. Full call samples,
profiles, private logs and selected private-store event rows remain outside
Git. Input identities:

- Export SHA-256: `1db27b7fd308b1e2d805f034b9f9f990cab5e321d04484f9d6b97980bd5ded4c`.
- Batch metadata SHA-256: `bfcef3e0035a9854b5bcc92bab3c500b566067885e9b33f02d1411199a83cfa4`.
- Checkpoint SHA-256: `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`.
- Tokenizer SHA-256: `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`.
- Full first report: 359,744 bytes, SHA-256 `d9cbd980a0682c6a9148217d6f361f47f27718158ad72dd6f8eaaac4cd8b8b8a`.
- Global-lock observation SHA-256: `7e1bb3c84b429487465a2f8e3b5568c2caca028555ca9f1024d89485581e80ca`.

The reducer refuses incomplete evidence by default. Explicitly preserving an
incomplete run never marks its acceptance gate as passed:

```sh
python3 docs/evidence/ane-tight-packing/summarize-serving.py "$FIRST_RAW_REPORT" \
  --allow-incomplete --out docs/evidence/ane-tight-packing/module-serving-first.json
python3 docs/evidence/ane-tight-packing/summarize-serving.py --self-test
```

The independent reducer recomputes medians and nearest-rank percentiles from
raw samples, checks identity/call counts and real dispatch, and refuses a
contradictory driver gate claim. Forcing its dispatch check to return true
fails only `Tests.test_complete_run_and_failed_dispatch_are_computed_from_raw_counters`;
the four other reducer tests in that mutation run remain green.

For the instrumented comparison, run
`monitor-serving-memory.py --out "$MEMORY_JSONL" --stop-file "$MONITOR_STOP"`
beside the driver, and create the stop file only after the driver exits. It
samples module/worker/relay RSS every three seconds, captures `vm_stat`, and
uses **only** `memory_pressure -Q` (query mode, never synthetic pressure).
Every sample has a Unix timestamp for correlation with module/worker profiles.
The monitor output and full stderr stay outside Git.

## Instrumented full comparison: failed, incomplete

A second full alternating comparison used source
`78a611b4131cf114842ddf9a80a1683f31d028a8`, the **same** export/checkpoint,
module configuration, thresholds and 1,000-query strata, with query-only
memory sampling and append-only worker stderr. It again completed three
serial and two packed arms. This time the third packed arm aborted during
warm preparation, before any packed admission, with a harness JSON-reader
error rather than a recorded module refusal:

```text
EOF while parsing a string at line 1 column 85
```

| Arm | Full export (s) | Outer-call p50 / p99 (ms) | Query-under-bulk p50 / p99 (ms) | First packed admission/compile (ms) |
| --- | ---: | ---: | ---: | ---: |
| Serial 1 | 204.736 | 3,353.987 / 5,479.778 | 369.554 / 709.426 | — |
| Packed 1 | 170.882 | 2,695.075 / 5,423.585 | 305.530 / 585.851 | 27,489.785 |
| Serial 2 | 189.574 | 3,114.464 / 6,233.985 | 366.616 / 704.831 | — |
| Packed 2 | 170.523 | 2,650.436 / 6,215.023 | 308.307 / 591.749 | 51,595.191 |
| Serial 3 | 226.662 | 3,497.436 / 6,951.202 | 382.970 / 896.262 | — |
| Packed 3 | **Aborted during warm preparation** | Not reached | Not reached | Not admitted |

Among these completed arms, the serial median was 204.736 s and the packed
median 170.703 s (**0.83377×**, above 0.80×). Even an arbitrarily fast third
packed result would leave a three-sample packed median at least 170.523 s,
above the 163.788 s threshold for the three completed serial arms. No additional
arm or third comparison was selected to improve these numbers.

All 31,705 completed export rows again compare bit-identical and retain the
same ordered vector digest as the first run. The 5,000 query vectors and
68,886 background vectors also compare identically, with zero mismatches.
Pooled query p50/p99 were 370.421/829.333 ms serial (3,000 samples) and
307.026/588.260 ms packed (2,000 samples), below the 407.463/912.266 ms limits.
Each complete packed export again recorded 825 packed worker exchanges and
81 single exchanges, with an eight-row maximum. Each complete packed arm
admitted the packed program once; median compile-request cost was 39,542.488 ms.

### Memory observations and remaining diagnosis

The monitor retained **1,147 samples, zero sampling errors**, with no sample
containing clients from two different arms. The worker and module from each
arm therefore did not remain as overlapping clients of the next arm in the
sampled intervals. These process observations do not measure memory retained
inside the system Neural Engine service or kernel.

| Arm | Peak real-worker RSS (MiB) | Peak module RSS (MiB) |
| --- | ---: | ---: |
| Serial 1 | 1,114.438 | 160.719 |
| Packed 1 | 1,438.047 | 297.828 |
| Serial 2 | 1,078.047 | 207.141 |
| Packed 2 | 863.906 | 219.891 |
| Serial 3 | 1,234.188 | 150.922 |
| Packed 3 warm-up only | 543.719 | 113.562 |

There is no monotonic growth of sampled per-process RSS across arms; these
samples do not establish a module/worker process leak. They do establish
substantial **system-wide paging** on the 16-GiB machine:

- `memory_pressure -Q` free percentage ranged from 14% to 43%. This is Apple's
  system-wide metric, not simply the physical free-page count.
- At 16,384 bytes per page, physical free pages reached 1,103 (17.234 MiB), and
  occupied compressor pages peaked at 523,806 (7.993 GiB).
- Over the captured interval, `vm_stat` counters increased by 7,153,671 swap-in
  pages (109.156 GiB) and 8,432,130 swap-out pages (128.664 GiB), alongside
  63,594,082 page compressions and 59,859,657 decompressions. These are global
  counter deltas, not attribution of all paging to the embedding worker.

The final arm's initial 64-row warm-up took 79.005 s, compared with 39.382 s
for the same initial batch in the failed final arm of the first comparison.
Neither cold warm-up is included in the export medians. Its private module
log records that warm-up interval:

```text
2026-10-11T15:45:59.827Z INFO  synapse.admission: job admitted model_id=qwen3-embedding-0.6b-ane job_id=inline-1-0
2026-10-11T15:47:18.834Z INFO  synapse.perf: job done model_id=qwen3-embedding-0.6b-ane job_id=inline-1-0 lane=ane-direct tokens=2489 wall_ms=79005
```

That arm's completed wire log contains one `PING`, one `LOAD`, three single
shape admissions and 24 standalone embeddings, with no packed admission and
no `ERR` response. Its last worker response (`embed-27`) finished successfully
in 276.097 ms at Unix microsecond `1791733640563969`. Full retained worker
stderr and module stderr are empty. Thus the second termination is not
evidence of a packed inference crash or numeric mismatch.

A separate harness defect was found: `wire_snapshot` could parse the final
NDJSON record while the worker relay was still appending it. The writer used
piecewise formatted writes, and the reader treated an unterminated last line
as complete JSON. A portable regression reproduces the same EOF error class.
The final driver serializes each record before writing it and reads only
newline-terminated observations; malformed **complete** records still fail.
Removing that boundary check fails only
`wire_snapshot_tests::concurrent_wire_reader_ignores_only_uncommitted_trailing_records`;
the ten other driver tests in that mutation run remain green. This repair was
not used to retcon either measurement or to select a replacement arm.

**The first run's initiating slowdown remains undiagnosed.** System memory
pressure is now demonstrated in the second comparison and is consistent with
slow execution/readiness work, but cannot retrospectively prove the cause of
the first run's catalog timeout. The retained evidence does not show packed
recompilation, resource-exhaustion responses or a worker reload as its trigger.
The five-second catalog-resolution/readiness budget and the unexplained
slowdown require investigation before enabling packing on any device. No
module or worker implementation was changed to force a pass.

### Final gate disposition and retained second-run evidence

**Rollout gate: NOT PASSED. Keep packing disabled outside private runs.**
Neither attempt completed three clean repetitions per arm; the second
completed-arm timing also misses the 0.80× limit. Bit identity and the measured
query-latency limits passed only for the completed strata. The first failed
attempt is retained alongside the second, not replaced or pooled with it.

- `module-serving-instrumented.json`: compact second-run statistics; full raw
  report is 360,243 bytes, SHA-256 `ab97720b0e4d84083a49ffd505ef7694738a4a46897c88a215f08ac68db21748`.
- `module-serving-memory.json`: compact RSS/paging observations; full memory
  trace is 2,247,377 bytes, SHA-256 `7b9342eac9c049331ac3932116605fecf543c2102b2b608cc20040c16bcbd38f`.
- Full call timings, all private-arm profiles/logs, complete stderr and
  privacy-filtered first-failure store events remain outside Git and outside
  regenerable build directories. No input text, vectors or machine paths are
  committed in the new summaries. All test/measurement clients exited and the
  global lane lock has no remaining measurement client.

Recompute the second attempt's compact summaries with its missing third packed
repetition explicitly marked incomplete and its acceptance gate kept failed:

```sh
python3 docs/evidence/ane-tight-packing/summarize-serving.py "$SECOND_RAW_REPORT" \
  --allow-incomplete --out docs/evidence/ane-tight-packing/module-serving-instrumented.json
python3 docs/evidence/ane-tight-packing/monitor-serving-memory.py \
  --summarize "$MEMORY_JSONL" --summary-out docs/evidence/ane-tight-packing/module-serving-memory.json
python3 docs/evidence/ane-tight-packing/monitor-serving-memory.py --self-test
```

## Third module-serving rerun: incomplete, stopped for paging

A third invocation used the current `origin/master` source at
`c00feeb728ec4dd6b95d0a0c4cd6d6ca94614870` on the M4 mini (16 GiB, Apple M4).
The committed driver and direct-ANE worker were built in an isolated source clone;
the transferred bundle SHA-256 `bfe41fd5ebeeff6aa7037cde6795279ad32f0449f841fb1a029707ba6ce7c4b5`
was verified on the mini, and native builds used
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` (rustc 1.99.0,
Cargo 1.99.0). The 6,341-row export and pinned checkpoint match the prior
attempts by SHA-256.

The sequence remained serial then packed, with the driver's defaults (three
repetitions and 1,000 query samples per arm); no workload, configuration,
acceptance threshold or module code was changed. The run used signed `ckdev-*`
aliases and per-arm private config, store, lease and data roots. The module
acquired the normal global ANE lock: `lsof` showed the module holding it and the
relay and real worker inheriting it on descriptor zero. The lock location and
`HOME` were not overridden. The `ck-synapse`, `ck_synapse`, and `ckdev` busy
scan was clear before launch and before each fresh candidate; no measurement
client or global-lock holder remained afterward. No M5 ANE
or production store was used.

At the initial mini check (00:05 CEST), `memory_pressure` reported **26% free**
on the 16-GiB machine. The largest observed processes were the Virtualization
VM (about 424 MiB RSS), Spark Mail renderer (280 MiB), three WebKit content
processes (264, 261 and 234 MiB), `whisper-server` (169 MiB), ClickUp renderer
(126 MiB), Claude (101 MiB) and Finder (83 MiB). These were the mini owner's
apps and were left running untouched. Immediately before the actual invocation
(00:20 CEST), the busy scan was clear and `memory_pressure -Q` reported 28% free.
The beside-driver monitor started at 26% free and retained 603 samples with no
sampling errors or overlapping arms. Across the capture, query-only memory
pressure ranged from 11% to 37%; free pages reached 1,018 (15.9 MiB), compressor
occupancy reached 502,380 pages (7.666 GiB), swap-ins increased by 6,496,149
pages (99.123 GiB), and swap-outs by 7,163,645 pages (109.308 GiB). Those are
global system counters, not attribution to the embedding process. Because swap
was growing sharply, the run was stopped after its current complete alternating
pair rather than continuing into more paging.

| Arm | Full export (s) | Outer-call p50 / p99 (ms) | Query-under-bulk p50 / p99 (ms) | First packed admission/compile (ms) |
| --- | ---: | ---: | ---: | ---: |
| Serial 1 | 300.168 | 3,964.181 / 16,558.979 | 513.768 / 1,953.525 | — |
| Packed 1 | 183.529 | 2,817.610 / 6,174.728 | 342.131 / 737.338 | 49,643.687 |

Both recorded exports covered all 6,341 rows with zero vector mismatches and the
same ordered vector digest as the earlier attempts. Both had 1,000 identical
query vectors and zero mismatches among the background document vectors
(13,220 serial and 13,847 packed). The packed export observed 825 packed worker
calls, 81 single calls, and an eight-row maximum. The partial one-pair export
ratio is 0.61142; its query-under-bulk p50/p99 are below the limits calculated
from the one recorded serial arm. These are descriptive partial results, not an
acceptance result: only one serial/packed pair was recorded, so three complete
repetitions per arm were not obtained.

**Acceptance gate: NOT PASSED.** The raw report and independent reducer both
mark the run incomplete and `passed: false`. After completing repetition 0, the
runner began setting up `serial-1` but was interrupted before that candidate
started serving; it has no report arm or wire observations. No sample was
stitched, selected or discarded. The two completed arms' module stderr is empty
and their wire logs contain no refusal response; in particular, no
`model_loading` refusal occurred.

The compact record is `module-serving-rerun.json`. The full raw report and
memory trace, private stores/configs, profiles, wire logs and stderr remain
outside Git and outside regenerable build directories. The raw report is
150,696 bytes, SHA-256
`47dbc7af9e2f02758c631d918792614ac756196f49eb58e6fe3ddab7f51913c3`; the
1,182,694-byte memory trace has SHA-256
`81dab7ed49de4a165a51cf33ea75402b856adb1a4f51f8d9bc8ccc99a510ece7`.
This rerun does not change either earlier attempt or the rollout disposition:
packing remains disabled outside private runs.
