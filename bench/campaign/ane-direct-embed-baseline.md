# ANE direct-API embed baseline

Target `gte-modernbert-ane-direct-embed`, machine `m5-local`.

## What is measured

`bench/spikes/ane-direct-probe/src/bin/modernbert_full.rs` runs GTE
ModernBERT through Apple's private `_ANEInMemoryModel` interface rather than
Core ML, one transformer layer per compiled executable, and embeds the eight
protected rows in `bench/spikes/ane-direct-probe/rows.jsonl`.

The objective is **aggregate throughput at sequence 512**: every active token
across every row divided by the summed warm wall time. Sequences 1024 and 2048
run as reported controls and do not gate.

Aggregate rather than a median of per-row rates, because the row set is
deliberately half short rows (12 to 38 tokens) and half near-512-token rows. A
median of per-row rates sits between those two regimes and rewards shaving the
per-row fixed cost that dominates the short rows, which is not what the lane
exists to do. Weighting rows by the work they carry is.

## Gates

Every one of these is refused by the harness, not merely reported:

| gate | threshold | why |
|---|---|---|
| min cosine vs in-process fp32 reference | >= 0.999 | same gate every ANE result in this repository has used |
| byte-identical repeat | required per row | a candidate that is fast and nondeterministic is not a candidate |
| reported min/mean cosine vs row metrics | must agree | a candidate cannot report a number its own rows contradict |
| row identity and active-token accounting | unchanged | the comparator and row set are protected paths |

The margin at 512 is thin by construction: the pinned baseline sits at
min cosine 0.9990908 against a 0.999 gate, with a transient layer-16 checkpoint
minimum of 0.9989. The tanh GELU approximation and the fp16 IOSurface boundary
after every layer have already spent most of the headroom, so any candidate
that buys speed with precision fails rather than scores.

## Pinned baseline

Measured with the external upstream binding at ec54af9, on this machine, under
the objective above. These historical numbers remain pinned for reference, not
as a re-measurement of the vendored binding now used by the harness; re-measure
before the next campaign (see the [current binding contract and pins](README.md#ane-direct-api-embedding-campaign)).

```
aggregate_tok_s @ 512   9222.454033809034
sequence_1024_tok_s     7630.882525180985
sequence_2048_tok_s     5710.282009150915
min_cosine              0.9990908
deterministic           true
load1m at admission     8.13
```

Re-pin by running the harness against an unmodified tree and taking the
reported `aggregate_tok_s`. Do not carry a number forward from an earlier
objective definition: the pre-2026-09-15 figure of 8941.75 was a median of
per-row rates and is not comparable.

## Load

`maxLoad1m` is 16, which is this machine's ambient rather than an idle floor —
it is a shared workstation running agents and builds continuously. The gate
therefore reads as "no spike above ambient", and every sample records the
one-minute load at admission, at report, and at completion so a contaminated
result carries its own evidence.
