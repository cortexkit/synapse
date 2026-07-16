# Utility judge evaluation

The judge used the operator's OpenCode OpenAI OAuth subscription account through the Codex/ChatGPT Responses route. No platform API key was used, and no OAuth token material was written to logs, verdicts, or this report.

## Wire verification

The implementation was source-checked against `~/Work/OSS/opencode`:

- `packages/opencode/src/plugin/openai/codex.ts:12` defines `https://chatgpt.com/backend-api/codex/responses`.
- `packages/opencode/src/plugin/openai/codex.ts:405-408` adds bearer access and `ChatGPT-Account-Id`; the harness derives the account ID from JWT claims without refreshing.
- `packages/opencode/src/plugin/openai/codex.ts:415-425` routes Responses requests to Codex, while `:549-553` adds `originator: opencode`, the OpenCode user agent, and `session-id`.
- `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:253-305` and `:372-389` establish the Responses body and function-tool shape. The harness maps all nine AFT tools and round-trips `function_call`/`function_call_output` items.
- `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:311-320` removes temperature for reasoning models. The judge requested temperature `0`, but the OAuth wire pins it as omitted (`null` in verdict rows).
- `packages/opencode/src/plugin/openai/codex.ts:559-563` removes `maxOutputTokens`; the live endpoint rejected `max_output_tokens`, so the adapter omits it. Streaming follows `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:777-792`.
- OpenCode excludes bare `gpt-5.6` at `packages/opencode/src/plugin/openai/codex.ts:289-291`; the selected non-mini route was `gpt-5.6-luna`.

One `read` tool-call round-trip returned `PROBE_OK`, and one gold-package phase-1 smoke call returned `answerable_fully`. The initial non-streaming probe was rejected with `Stream must be set to true`; the source-verified SSE adapter then succeeded.

## Phase-1 calibration semantics

The gate measures package utility from `phase1_sufficiency` while preserving candidate final-sufficiency semantics; exploration of mismatches remains enabled. Gold must have zero `not_answerable` phase-1 results, mean top-up at most 4, and mean top-up at most half the empty mean. Empty packages must be `not_answerable` in phase 1. Mismatched packages must be `not_answerable` in phase 1 on at least 60% of jobs, regardless of their final post-exploration sufficiency. The absolute thresholds are operator-owned; the load-bearing property is gold-versus-empty separation, not forcing gold browsing to zero.

Final calibration rerun: prompt iteration 1, SHA `2c27195db8c4…`, model `gpt-5.6-luna`, concurrency 2.

| control | rows | phase-1 distribution | final full / partial / none | top-up calls mean | gate |
| --- | ---: | --- | --- | ---: | --- |
| gold | 5 | full 5 / partial 0 / not_answerable 0 | 5 / 0 / 0 | 3.00 | PASS |
| empty | 5 | full 0 / partial 0 / not_answerable 5 | 4 / 1 / 0 | 9.80 | PASS |
| mismatched | 5 | full 0 / partial 2 / not_answerable 3 | 4 / 1 / 0 | 10.00 | PASS |

Gold/empty mean ratio was `0.3061`, satisfying the separation criterion. Mismatched packages were allowed to explore after phase 1; their post-exploration success did not grant package credit.

## Cost projection

The calibration command printed this projection before the matrix launch:

- sample rows: 15
- calibration projected packages: 40
- mean input tokens: 29,431.4
- mean output tokens: 1,098.5
- projected input tokens: 1,177,256
- projected output tokens: 43,939
- projected USD: unpriced on the subscription route

The available full matrix contained 40 gold-control packages plus 40 available-candidate packages (80 total); the full command printed an unpriced 80-package projection before execution.

## Utility versus F1

The full matrix ran all 40 gold controls and the only candidate present on disk. The requested 4B, 9B, and DeepSeek artifacts were absent from `/Users/[owner]/Work/Projects/CortexKit/synapse/tools/gather-distill/data/students/` at run time:

- `qwen35-4b-lora-v1-rows.jsonl` and `qwen35-4b-lora-v1-scores.json`
- `qwen35-9b-lora-v1-rows.jsonl` and `qwen35-9b-lora-v1-scores.json`
- `deepseek-v4-flash-zeroshot*-rows.jsonl` and matching scores
- `deepseek-v4-flash-nothink*-rows.jsonl` and matching scores

The available `qwen35-2b-sft-v1` artifacts were copied read-only (40 row lines; 586 score lines). All 40 candidate rows failed pinned-package validation and were recorded as skipped invalid, so they did not consume judge calls and are not evidence of a cheap sufficient package.

| system | full / partial / none | top-up calls mean | top-up calls median | top-up tokens mean | score mean | F1 | skipped invalid | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gold-control | 36 / 4 / 0 | 3.48 | 0.00 | 25,152 | 8.90 | 1.00 | 0 | 0 |
| qwen35-2b-sft-v1 | 0 / 0 / 40 | 0.00 | 0.00 | 0 | 1.00 | 0.00 | 40 | 0 |

## Ranking conclusion

No valid student candidate was available, so a meaningful utility-versus-F1 ranking agreement verdict is **not assessable**. The available 2B row's zero-call result is entirely skipped-invalid and must not be ranked as a successful package. No pairwise divergence examples are reported because there are no valid candidate verdicts; the ignored per-job verdict files remain the source for future comparisons.
