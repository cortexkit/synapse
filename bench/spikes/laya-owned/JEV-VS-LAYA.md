# Jev vs Laya on our own classification task

## Verdict

**The System One model class handles this task well. The 421M open checkpoint does not.** Jev scores **94.9%** on the balanced 999-row set where laya scores 46.7%, against a 26.6% floor. So my earlier report was right to close the lane and wrong about why: I framed laya's 19.7% as evidence that a typed-decision model cannot separate our six classes, and the frontier instance of the same class, same API, same criteria, separates them nearly perfectly. **What failed was the instance, not the idea.**

Three things follow. Laya's failure is capability rather than context — clipping Jev to laya's window costs it 3 points, not 32. Our 233-row real set is a weak instrument and roughly a third of its labels are routing conventions rather than properties of the text. And a laya-class lane only becomes interesting if it is trained, which is now a measurement with a known ceiling rather than a hope.

## The numbers

Identical rows, identical `choice` criteria (PLACEHOLDER-CRITERIA-V0 for laya, reused verbatim for Jev), identical scoring.

| | real 233 | real text-derivable 154 | synthetic 999 |
|---|---:|---:|---:|
| constant-predictor floor | 0.524 | 0.792 | 0.266 |
| laya V1, english-512 | 0.197 | 0.299 | 0.378 |
| laya V1, typed-decisions-1024 | 0.197 | 0.286 | 0.467 |
| **Jev (jev-1.13.0), full prose** | **0.549** | **0.805** | **0.949** |
| Jev, clipped to laya's window | 0.519 | — | — |

Jev per-class recall on the synthetic set: EXPLAIN 1.00, DIAGNOSE 0.99, AUDIT 0.98, SPEC 0.93, EVALUATE 0.92, PLAN 0.87. Laya's collapse onto PLAN (38-55% of all predictions) has no counterpart here.

Cost of the whole comparison: **$0.045** for 2,464 requests, 1.08M input tokens at their published $0.042/MTok.

## Context was not laya's problem

The obvious objection to the original result was that laya's English checkpoint truncated 85.8% of real rows, keeping roughly 320 tokens of state. Jev's 32k budget makes the control cheap: run it twice, once on full prose and once clipped to the same ~1300 characters laya saw.

    Jev, full prose      0.549
    Jev, clipped         0.519      204 of 233 rows actually clipped
    laya, that same clip 0.197

**Context costs Jev 3.0 points. Laya sits 32.2 points below Jev on identical bytes.** So truncation explains almost none of the gap, and the 1024-token checkpoint moving laya from 0.378 to 0.467 on the synthetic set is a capability effect, not a context one.

## Our real-set labels are one-third convention

Jev at 94.9% on synthetic and 54.9% on real is too large a gap to be noise, and the confusion matrix names the cause. Jev recalls AUDIT at 0.99 on the real set but EVALUATE at 0.03 — and per ALF, Athena's prompt says *"for route=campaign use EVALUATE"*:

    EVALUATE gold rows          91
      of which route=campaign   73   (80%)
    SPEC gold rows               6   (module-authored, never classified)

So **79 of 233 real rows (33.9%) carry a label that is a routing convention rather than a property of the request text.** A campaign request reads like an optimization brief ("raise the owned runtime's Qwen3-0.6B f16 single-stream decode throughput, baseline 40.55 tok/s"), and nothing in that text says EVALUATE except a convention the model was never told. On the 154 text-derivable rows Jev reaches 0.805 — though the floor there is 0.792, because those rows are 79% AUDIT.

**Consequence for anyone using this set: the 233-row real corpus cannot rank models well.** Its floor is 52.4%, a third of its labels are unstated conventions, and two of six classes never appear. The 999-row synthetic set is balanced, labelled from the text by Opus, and separates models by 48 points where the real set separates them by 35. Quote the real set for production realism, rank on the synthetic one.

## What this changes

The original QUALITY.md verdict stands on its own terms — laya zero-shot is unusable for this task and no Metal decision head is justified — but its reasoning needs the correction above: the class works, the checkpoint does not.

- **A trained laya head is now a measurement with a known ceiling.** Jev at 94.9% says the task is learnable by this architecture. Laya ships a fine-tuning recipe, our 999 synthetic rows are already in typed-decision shape, and ALF's class-conditioned panel addenda (`council.rs:124` onward) are the natural source for training-time class semantics. That is the experiment worth running if anyone wants a local lane.
- **`act_probability` remains the disqualifier for gating, and it is laya-specific.** It was 1.0 on all 2,464 laya predictions. Jev's `confidence` varies normally (0.35 on the smoke row, spread across the corpus), so a constant confidence is a property of the open checkpoint rather than of the class.
- **Jev itself is a viable hosted option at this price.** $0.045 for 2,464 classifications is far below what the trained-2B lane costs to serve, though it is a remote dependency on request prose, which is the tradeoff our owned lanes exist to avoid.

## Method

- `jev.py`, model `jev-latest` resolving to `jev-1.13.0`, endpoint `POST https://api.typesafe.ai/v1/systemone`, key read from `~/.config/jev.key` and never committed.
- Same corpora and digests as QUALITY.md: `real-gold.jsonl` sha256 `c90b052c…`, `classify-gold-v1.jsonl` sha256 `bbfe2575…`, both gitignored.
- Per-row output in `rows-jev-*.jsonl` (gitignored): gold, predicted, full probability distribution, confidence, sent vs original character counts, clipped flag. Runs are resumable by row id, so a rate-limit stop does not re-spend.
- 429 honours `retry-after`; 5xx retries with exponential backoff; neither fired during these runs.
- **This sent 233 real Athena consult requests and 999 synthetic ones to a third-party API.** The real prose contains internal project and file names. TypeSafe's published policy states customer requests are not used for training, but the exposure is real and is recorded here rather than assumed away.
