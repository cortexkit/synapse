# Zero-shot gatherer bake-off

## Methodology

- Evaluation: the same 40 SHA-pinned CK-repository jobs in `data/eval-jobs.jsonl`, scored mechanically against the 40 Opus gold rows in `data/eval-gold-rows.jsonl`.
- Serving: llama.cpp `llama-server -ngl 99 --jinja -fa on`, OpenAI-compatible gather lane, concurrency 2, and the production 40-step gather contract.
- **Context correction:** the first 32k runs were discarded. Their tool-result-heavy trajectories hit HTTP 500 context-window errors on roughly 35–70% of jobs, which unevenly penalized more thorough tool users. Every scored candidate below was re-run at 128k (`-c 131072`) with flash attention; a model trained below 128k was clamped to its trained maximum and its served context is recorded.
- `file F1` and `line Jaccard` below are means over naturally completed trajectories only; this avoids treating forced budget finalization and rejected rows as quality results. Contract validity, API errors, tool calls, thinking tokens, budgets, and wall time remain whole-run measures across all 40 jobs. `N`, `F`, `A`, and `I` mean natural, budget-finalize, API error, and invalid final.

## Leaderboard

| model | natural file F1 | natural line Jaccard | contract-valid rate | API-error rate | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| qwen36-27b | 0.818 | 0.677 | 80.0% | 0.0% | 15.68 | 724 | 26/40 | N 26/F 10/A 0/I 4 | 128k | 96.1s |
| ornith-35b | 0.745 | 0.628 | 67.5% | 0.0% | 14.62 | 670 | 20/40 | N 20/F 7/A 0/I 13 | 128k | 37.0s |
| nemotron3-nano-30b | 0.700 | 0.442 | 80.0% | 0.0% | 10.45 | 3853 | 23/40 | N 23/F 11/A 0/I 6 | 128k | 51.1s |
| gemma4-26b-a4b | 0.683 | 0.399 | 97.5% | 2.5% | 9.62 | 1305 | 25/40 | N 25/F 14/A 1/I 0 | 128k | 51.6s |
| qwen36-35b-a3b | 0.662 | 0.476 | 85.0% | 0.0% | 12.35 | 458 | 29/40 | N 29/F 8/A 0/I 3 | 128k | 34.9s |
| nemotron-cascade2-30b | 0.652 | 0.443 | 62.5% | 0.0% | 12.47 | 1297 | 24/40 | N 24/F 13/A 0/I 3 | 128k | 35.9s |
| gemma4-31b-it | 0.610 | 0.398 | 100.0% | 0.0% | 8.70 | 880 | 32/40 | N 32/F 8/A 0/I 0 | 128k | 112.6s |
| ornith-9b | 0.563 | 0.381 | 65.0% | 5.0% | 14.00 | 701 | 22/40 | N 22/F 8/A 2/I 8 | 128k | 36.5s |
| gemma4-e4b-it | 0.460 | 0.252 | 95.0% | 0.0% | 6.53 | 1167 | 38/40 | N 38/F 1/A 0/I 1 | 128k | 31.6s |
| minicpm5-1b | 0.000 | 0.000 | 0.0% | 12.5% | 14.25 | 1027 | 1/40 | N 1/F 3/A 5/I 31 | 128k | 39.1s |
| lfm25-12b | 0.000 | 0.000 | 0.0% | 0.0% | 1.35 | 0 | 0/40 | N 0/F 0/A 0/I 40 | 128k | 2.0s |
| lfm2-8b-a1b (UNSERVABLE-STAGING) | n/a | n/a | n/a | n/a | n/a | n/a | 0/40 | n/a | n/a | n/a |

## Per-model serving notes

- **qwen36-27b** — Q8_0; `-c 131072` (trained maximum 262,144); `-ngl 99 --jinja -fa on`; anomalies: 4/40 invalid finals.
- **ornith-35b** — Q8_0; `-c 131072` (trained maximum 262,144); `-ngl 99 --jinja -fa on`; anomalies: 13/40 invalid finals.
- **nemotron3-nano-30b** — Q8_0; `-c 131072` (trained maximum 1,048,576); `-ngl 99 --jinja -fa on`; anomalies: 6/40 invalid finals.
- **gemma4-26b-a4b** — Q8_0; `-c 131072` (trained maximum 262,144); `-ngl 99 --jinja -fa on`; anomalies: 2.5% API errors; 14/40 budget-finalized.
- **qwen36-35b-a3b** — Q8_0; `-c 131072` (trained maximum 262,144); `-ngl 99 --jinja -fa on`; anomalies: 3/40 invalid finals.
- **nemotron-cascade2-30b** — Q8_0; `-c 131072` (trained maximum 1,048,576); `-ngl 99 --jinja -fa on`; anomalies: 3/40 invalid finals.
- **gemma4-31b-it** — Q8_0; `-c 131072` (trained maximum 262,144); `-ngl 99 --jinja -fa on`; anomalies: 8/40 budget-finalized.
- **ornith-9b** — Q8_0; `-c 131072` (trained maximum 262,144); `-ngl 99 --jinja -fa on`; anomalies: 2/40 context-window errors; 5.0% API errors; 8/40 invalid finals.
- **gemma4-e4b-it** — BF16; `-c 131072` (trained maximum 128k); `-ngl 99 --jinja -fa on`; anomalies: 1/40 invalid finals.
- **minicpm5-1b** — F16; `-c 131072` (trained maximum 128k); `-ngl 99 --jinja -fa on`; anomalies: 12.5% API errors; no contract-valid final packages.
- **lfm25-12b** — BF16; `-c 128000` (trained maximum 128k); `-ngl 99 --jinja -fa on`; anomalies: no contract-valid final packages.
- **lfm2-8b-a1b** — F16 / Q8 unavailable; F16 and Q8 artifacts failed staging download after repeated resumptions; never served or scored.

## Small-tier contract finding

**MiniCPM5-1B and LFM2.5-1.2B score 0.000 because their zero-shot finals are malformed, not because the scorer is lenient:** they call repository tools, but production rejects their final JSON (`scope` is not an array and snippets omit `startLine`). This is a genuine production-contract result. The same validator rejects those packages in production, which is the empirical justification for the SFT distillation program: a bridge-tier target needs fine-tuning rather than stock 1–1.2B zero-shot prompting.

No small model is viable as a zero-shot bridge-tier deployment. Ornith-9B and Gemma E4B provide exploratory small-model baselines, but neither closes the hosted-tier natural-overlap gap.

## Recommendation

- **Top-3 hosted-tier candidates by natural file F1:** qwen36-27b (0.818), ornith-35b (0.745), nemotron3-nano-30b (0.700).
- **Small-tier ranking is exploratory only:** ornith-9b (0.563), gemma4-e4b-it (0.460), minicpm5-1b (0.000); the 1–1.2B bridge-tier candidates are not contract-valid zero-shot.

Scorer JSONs, rows, ledgers, raw probes, and status files are retained under ignored `data/bakeoff/` for audit and reproducibility.
