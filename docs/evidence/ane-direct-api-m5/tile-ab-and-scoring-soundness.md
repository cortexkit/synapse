# Tile 128 vs 256 on the direct-API tree, and what the control run says about scoring

2026-09-20, M5 Max, macOS 26A428. Written because a campaign readiness panel
found that the committed direct-API graph tiles queries at 128 while our own
steering called tile 128 a closed +62% negative, and neither the panel nor the
steering could say which tree that +62% was measured on.

## The contradiction, resolved from the evidence record

It is not a contradiction. The +62% arm lives in
`docs/evidence/ane-modernbert-8192-latency/README.md`, whose table reads
`Control, tile 256 | 32.677 ms` against `Tile 128 | 52.950 ms | +62.0%`. That
investigation measured the **Core ML tiled export** at sequence 1024. The
direct-API probe is a different graph on a different execution path, and tile
size had **never** been measured on it.

So the steering was not wrong about its own subject. It was applied to a tree it
never described — the same error class as quoting a measured baseline as a
threshold, which the same brief also did.

## The A/B, and it does not settle the question

Two binaries differing by one line (`query_tile = 128` vs `256`), the same rows,
the same model snapshot, the same pinned ANE binding, alternating in one session.
Aggregate throughput computed exactly as `validate_report` does:
`sum(active_tokens) * 1000 / sum(warm_wall_ms_median)`.

| pass | control (tile 128) | candidate (tile 256) | ratio |
|---|---:|---:|---:|
| 1 | 7,618.8 | 6,473.0 | 0.850 |
| 2 | 9,049.0 | 8,477.0 | 0.937 |
| 3 | 7,586.5 | 8,176.4 | 1.078 |

Paired median ratio 0.937; pooled medians point the other way (control 7,618.8,
candidate 8,176.4). **When the paired and pooled statistics disagree in
direction, the effect is smaller than the noise.** Three ratios spanning 0.85 to
1.08 do not distinguish the two trees.

Verdict: **inconclusive**. Tile 256 is not the free win it looked like, and tile
128 is not proven optimal here either. The question stays open and needs a quiet
box, not another opinion.

## CORRECTION (2026-09-20, same day): what this says about the ENGINE

The section below was published claiming the campaign gate would reject the
unmodified tree 3 of 3. **That is wrong and the error is mine.** No surface
reported it; I computed it by hand against the pinned constant and asserted it
about a system I had not read.

The campaign engine never scores an absolute against a cross-session constant.
It runs an interleaved order of control and candidate samples inside one block,
folds `paired_ratios(candidate, control)` per position, takes the median, and
promotes on a bootstrapped `interval.lower >= min_gain`. The 19.3% swing below
cancels in that ratio by construction and the pinned baseline never enters the
verdict.

What my run *would* have triggered is the engine's own drift check: a control at
7.6k against a 9.2k epoch reference is `control_drifted`, which aborts the block
`InvalidControlDrift` with `RebaselineRequired` and re-measures before scoring
anything. So the engine detects exactly the condition I claimed it was blind to.

The numbers below are sound. Their subject was wrong: they are evidence about
THE BOX, and the empirical case for why paired ratios and drift detection are
necessary — not evidence that they are missing. The load-ceiling fix still
stands on its own, because drift detection catches drift and does not catch a
ceiling nobody validated.

## The control's own variance, as evidence about the machine

The three control passes are the *same unmodified tree*:

    7,619 / 9,049 / 7,586 tok/s        spread 19.3%, zero code change

Ambient load during the run was 27-30. The pinned campaign baseline of
9,222.45 tok/s was admitted at load 8.13.

Consequences, corrected:

1. **A bare harness invocation compared to a cross-session constant is not a
   score on this box.** The unmodified tree lands 17-18% below its own pinned
   baseline in 3 of 3 passes. That is a statement about running the harness
   DIRECTLY, which is what I did; inside a block the engine pairs against a
   co-measured control and would call this drift, not a failure.
2. **The load ceiling admits that.** The registration sets
   `SYNAPSE_CAMPAIGN_MAX_LOAD_1M=24` while the harness documents 16 and its own
   emitted note asserts "a threshold of 16 admits this shared workstation's
   normal background load". `configured_constants()` validates the baseline and
   three digests but **not** the load threshold, so 24 wins silently and the
   result note contradicts itself.
3. **The fix is the sibling lane's discipline, not a tighter ceiling.** The
   Metal embed lane schedules a current-best control beside every candidate in
   the same session and re-baselines on >3% drift. Under that protocol the
   19.3% swing above cancels; under absolute scoring it is the result.

## What this changes for the campaign

Before re-firing:

- Nothing about scoring. The engine already pairs within a block; that item was
  my error and is struck. What remains true is that the harness must keep being
  invoked per side by the block rather than compared to a constant by hand,
  which is how it is registered.
- Reconcile the load ceiling with the baseline's admission load, and make
  `configured_constants()` validate it so the two cannot silently diverge again.
- Tell proposal authors the tile question is **open on this tree**, not closed.
  The closed negatives (tile 128/512, head grouping, shared masks) are closed for
  the Core ML export only.
- Correct the gate in the brief: the enforced minimum cosine is `0.999`, pinned
  structurally by a digest over `const GATE: f32 = 0.999;`. The figure 0.9990908
  is the measured baseline minimum at sequence 2048 and is not a threshold.

## Method note

The first extraction of these runs reported "12 tok/s" six times. Six identical
values across six runs is not a result, it is a parser reading `active_tokens`
off the first row. Caught by the distribution rather than by the value — the same
check that a constant `act_probability` of 1.0 across 2,464 predictions failed.
Print the spread before believing the mean.
