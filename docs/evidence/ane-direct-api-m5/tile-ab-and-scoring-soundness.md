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

## The control's own variance is the finding worth keeping

The three control passes are the *same unmodified tree*:

    7,619 / 9,049 / 7,586 tok/s        spread 19.3%, zero code change

Ambient load during the run was 27-30. The pinned campaign baseline of
9,222.45 tok/s was admitted at load 8.13.

Consequences for the harness as registered:

1. **Absolute-against-a-pinned-constant scoring is unsound on this box.** The
   unmodified tree failed a 3%-win threshold against its own baseline in 3 of 3
   passes. A candidate is therefore scored mostly on the ambient load it drew.
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

- Score candidate/control ratios measured in one session. The harness currently
  measures one tree per invocation and compares to a constant from another
  session; that has to change or the campaign measures load.
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
