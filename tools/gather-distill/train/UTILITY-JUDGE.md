# Utility judge evaluation

This run used the operator's OpenCode OpenAI OAuth subscription account through the Codex/ChatGPT Responses route. No platform API key was used, and the OAuth access or refresh token was never written to a log, verdict, or report. The full matrix was not started because the phase-1 calibration gate remained red after the authorized rerun.

## Wire verification

The implementation was source-checked against `~/Work/OSS/opencode`:

- `packages/opencode/src/plugin/openai/codex.ts:12` defines `https://chatgpt.com/backend-api/codex/responses`.
- `packages/opencode/src/plugin/openai/codex.ts:405-408` adds bearer access and `ChatGPT-Account-Id`; the harness derives the account ID from JWT claims without refreshing.
- `packages/opencode/src/plugin/openai/codex.ts:415-425` routes Responses requests to Codex, while `:549-553` adds `originator: opencode`, the OpenCode user agent, and `session-id`.
- `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:253-305` and `:372-389` establish the Responses body and function-tool shape. The harness maps all nine AFT tools and round-trips `function_call`/`function_call_output` items.
- `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:311-320` removes temperature for reasoning models. The judge requested temperature `0`, but the OAuth wire pins it as omitted (`null` in verdict rows).
- `packages/opencode/src/plugin/openai/codex.ts:559-563` removes `maxOutputTokens`; the live endpoint rejected `max_output_tokens`, so the adapter omits it. Streaming follows `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:777-792`.
- OpenCode excludes bare `gpt-5.6` at `packages/opencode/src/plugin/openai/codex.ts:289-291`; the selected non-mini route was `gpt-5.6-luna`.

The live smoke checks passed: one `read` tool-call round-trip returned `PROBE_OK`, and one gold-package phase-1 call returned `answerable_fully`. The initial non-streaming probe was rejected with `Stream must be set to true`; the source-verified SSE adapter then succeeded.

## Phase-1 calibration semantics

The gate now measures package utility at phase 1, before repository top-ups. Gold must be phase-1 answerable (full or partial) with mean top-up below two calls. Empty packages must be `not_answerable` in phase 1. Mismatched packages must be `not_answerable` in phase 1 on at least 60% of jobs, regardless of their final post-exploration sufficiency. Candidate rows retain the original headline metric: final sufficiency plus top-up cost.

The authorized rerun used the clean base prompt, model `gpt-5.6-luna`, concurrency 2, and prompt SHA `2c27195db8c4…`:

| control | rows | phase-1 distribution | top-up calls mean | gate |
| --- | ---: | --- | ---: | --- |
| gold | 5 | full 3 / partial 2 / not_answerable 0 | 2.4 | FAIL |
| empty | 5 | full 0 / partial 0 / not_answerable 5 | 9.6 | PASS |
| mismatched | 5 | full 0 / partial 2 / not_answerable 3 | 9.6 | PASS |

The only failing condition was gold mean top-up: `2.4`, above the `<2` threshold. The phase-1 mismatch metric itself passed at `3/5`; several mismatched packages were correctly explored after phase 1, and their final sufficiency was intentionally not used for package credit.

## Cost projection

The calibration command printed the projection before any full-matrix launch:

- sample rows: 15
- projected packages: 40
- mean input tokens: 29,376.7
- mean output tokens: 1,110.3
- projected input tokens: 1,175,069
- projected output tokens: 44,411
- projected USD: unpriced on the subscription route

## Utility versus F1

The full matrix was not run after the failed gate. The requested 4B, 9B, and DeepSeek row files were absent from the specified parent artifact directory. The available 2B row and score artifacts were copied read-only for staging (40 row lines; 586 score lines), but were not judged because the gate blocked the full phase.

| system | full / partial / none | top-up calls mean | top-up calls median | top-up tokens mean | score mean | F1 | skipped invalid | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gold-control | not run | — | — | — | — | — | — | — |
| qwen35-2b-sft-v1 (available, not run) | — | — | — | — | — | existing score not compared | — | — |
| qwen35-4b-lora-v1 (rows absent) | — | — | — | — | — | unavailable | — | — |
| qwen35-9b-lora-v1 (rows absent) | — | — | — | — | — | unavailable | — | — |
| deepseek-v4-flash-zeroshot (rows absent) | — | — | — | — | — | unavailable | — | — |

**Ranking agreement:** not assessable because the gate blocked the full phase. **Divergence examples:** none are reported; producing examples without full per-job verdicts would fabricate evidence.
