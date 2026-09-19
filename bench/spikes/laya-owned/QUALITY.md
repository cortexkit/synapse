# Is Laya any good at OUR classification task? — no

## Verdict

**No, and the lane closes for this task. Do not build a Metal head on this evidence.** Zero-shot on 233 real Athena consult requests, laya scores **19.7%** class-exact where a **constant "AUDIT" predictor scores 52.4%** and a trained Qwen3.5-2B student scores 92.0%. It is not merely worse than the trained student — it is worse than a single hardcoded string, on both checkpoints, under two independently authored criteria sets. On the balanced 999-row synthetic set it does show real signal (46.7% against a 16.7% random baseline), so the model is not broken; it simply cannot separate these six classes on this text. The cost finding from the feasibility spike (CPU head 2.08x the Metal encoder block at 512 tokens) is now moot: there is nothing here worth making faster.

Two further findings hold independently of the verdict, and both would have to be fixed before any future attempt: **every one of the 2,464 predictions returned `act_probability` = 1.0**, so the act head — the entire "should software act on this" claim of the model class — is unusable as shipped; and calibration is poor (ECE 0.101 to 0.304), so the confidence cannot carry a gate either.

## The numbers

Comparators on the identical 233 rows, from the classify-distill work: **trained Qwen3.5-2B 92.0%**, **stock Qwen3.5-2B 87.7%**. Floor on that set: a constant AUDIT predictor scores **52.4% (122/233)**. Random over six classes is 16.7%.

| arm | n | accuracy | fitting rows | truncated rows | truncated |
|---|---:|---:|---:|---:|---:|
| english-512 / real | 233 | **0.189** | 0.303 (33) | 0.170 (200) | 85.8% |
| typed-decisions-1024 / real | 233 | **0.245** | 0.191 (162) | 0.366 (71) | 30.5% |
| english-512 / synthetic | 999 | 0.389 | 0.389 (999) | — (0) | 0.0% |
| typed-decisions-1024 / synthetic | 999 | 0.458 | 0.458 (999) | — (0) | 0.0% |

Every real-set number is below 52.4%. The best real-set arm (24.5%) is less than half the floor.

## The criteria are not the cause, which is the part that makes this decisive

The obvious objection to a bad zero-shot number is that the option descriptions were wrong. So two criteria sets were authored independently, each fixed before any accuracy from it was seen:

- **PLACEHOLDER-CRITERIA-V0** — one neutral sentence per class derived from the class name alone. Authored by a worker whose run died on an unrelated tool failure before it computed anything, and reused here verbatim, so the wording provably cannot have been tuned against labels.
- **SOURCE-GROUNDED-V1** — derived from the athena tool's own published description of the six shapes ("design review, audit, diagnosis, evaluation, explanation, or an optimization campaign..."), quoted in `criteria_v1.py`. A real source describing the task rather than the labels.

| arm | V0 | V1 |
|---|---:|---:|
| english-512 / real | 0.189 | 0.197 |
| typed-decisions-1024 / real | 0.245 | 0.197 |
| english-512 / synthetic | 0.389 | 0.378 |
| typed-decisions-1024 / synthetic | 0.458 | 0.467 |

The two sets agree within ±0.05 everywhere, and neither approaches the floor. **Criteria wording is not the lever.** Both criteria sets are quoted in full in `quality.py` and `criteria_v1.py`; a third set from Athena's actual classify prompt (requested from ALF, not yet received) would run as V2, but on this evidence it would have to move accuracy by more than 30 points to change the verdict.

## What it actually does, which explains the sub-floor result

The failure is not noise, it is a systematic collapse onto two classes:

    V1, synthetic 1024    predicts PLAN 55%, AUDIT 18%, EXPLAIN 11%, EVALUATE 9%
                          recall   PLAN 0.98, EXPLAIN 0.82, AUDIT 0.33, EVALUATE 0.30, DIAGNOSE 0.33, SPEC 0.02
    V1, real 512          predicts SPEC 40%, PLAN 38%, AUDIT 21%
                          recall   AUDIT 0.38, EVALUATE 0.00, DIAGNOSE 0.00, SPEC 0.00

PLAN absorbs 38-55% of all predictions. On the real set PLAN is **0% of gold** — Athena never routed any of those 233 requests to PLAN — so every PLAN prediction there is automatically wrong, which alone accounts for most of the gap to the floor. Recall on EVALUATE (39% of real gold) is 0.00 to 0.30. The model reads almost any engineering request as a plan-shaped one, and no wording tested moved that.

Length is a confound worth naming rather than a finding: on the 1024 checkpoint the *truncated* rows scored higher (0.366) than the fitting rows (0.191). Length correlates with class here, so truncation is entangled with what the row is about; it is not evidence that truncation helps.

## Calibration and the act head

On the balanced synthetic set (the only place all six classes appear):

- **ECE 0.101** (english-512) and **0.304** (typed-decisions-1024) over the library's normalized-entropy confidence.
- **`act_probability` = 1.0 on 2,464 of 2,464 rows**, every arm, every corpus. The feasibility spike saw this on 24 rows and flagged it as possibly fixture-specific; it is not. The act head as shipped emits a constant.

A confidence gate is the whole product claim of a System One model, and neither output supports one here.

## Method

- Instrument: the Python reference (`laya.load` / `Agent.predict`), not the owned lane — parity between them is already proven in `REPORT.md`, and Python iterates faster. Budgets read from each checkpoint's own `rl_agent_config.json` (english 512/192, typed-decisions 1024/256) rather than inheriting the English constants.
- Corpora: `data/real-gold.jsonl` sha256 `c90b052c6ae0b12ab18db53c65e8bfb012cb69da7a2208eebc8fe0963a73664f` (233 rows), `data/classify-gold-v1.jsonl` sha256 `bbfe2575d8d8f8dd159f64f8e6c224355aed12045807abce9f89f6278e0c7044` (999 rows). Both gitignored, never committed.
- Case normalised: 14 real rows carry lowercase `diagnose`; case is not a prediction error.
- Per-row output retained in `rows-*.jsonl` and `rows-v1-*.jsonl` (id, gold, predicted, probabilities, confidence, act_probability, token counts, truncation flag, surviving fraction). Aggregates in `quality-v0.json` / `quality-v1.json`.
- Latencies (43-90 ms median per question) are ambient-load context, not a result; the box was under concurrent load throughout.

## What would change the answer

Not a Metal head, and not better criteria. Only training: laya ships a fine-tuning recipe, and the Athena data (999 synthetic + 233 real) is already in typed-decision shape. A trained head on this encoder is a different experiment with a different cost, and the trained-2B comparator at 92.0% is the bar it would have to approach to be worth the lane. Zero-shot, on this task, the answer is no.
