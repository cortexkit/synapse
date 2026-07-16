# Utility judge evaluation

This run used the operator's OpenCode OpenAI OAuth subscription account through the Codex/ChatGPT Responses route. No platform API key was used, and the OAuth access or refresh token was never written to a log, verdict, or report. The full matrix was not started because the calibration gate remained red after the allowed three prompt iterations.

## Wire verification

The implementation was source-checked against the OpenCode checkout at `~/Work/OSS/opencode`:

- `packages/opencode/src/plugin/openai/codex.ts:12` defines the subscription endpoint as `https://chatgpt.com/backend-api/codex/responses`.
- `packages/opencode/src/plugin/openai/codex.ts:405-408` adds the bearer access token and `ChatGPT-Account-Id`; the harness derives the account ID from the JWT claims without refreshing the token.
- `packages/opencode/src/plugin/openai/codex.ts:415-425` routes Responses requests to the Codex endpoint, and `:549-553` adds `originator: opencode`, the OpenCode user agent, and a stable `session-id`. The harness sends the same headers and uses a per-run session ID.
- `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:253-305` and `:372-389` establish the Responses body and function-tool shape. The harness maps the nine AFT declarations to `{type:"function",name,description,parameters,strict:false}` and maps `function_call`/`function_call_output` input items round-trip.
- `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:311-320` removes temperature for reasoning models. The judge requested temperature `0`, but the OAuth Responses wire pins it as **omitted** (`null` in verdict rows), not as `0`.
- `packages/opencode/src/plugin/openai/codex.ts:559-563` removes `maxOutputTokens` for the Codex route. The first live probe confirmed that the subscription endpoint rejects `max_output_tokens`; the adapter therefore omits it. `packages/core/src/github-copilot/responses/openai-responses-language-model.ts:777-792` uses streaming Responses, so the adapter sends `stream:true` and reconstructs the terminal response from SSE events.
- OpenCode explicitly excludes the bare `gpt-5.6` catalog ID at `packages/opencode/src/plugin/openai/codex.ts:289-291`; the selected allowed class route was **`gpt-5.6-luna`**, which matches the same `gpt-5.6-*` route family and has no mini/nano suffix.

The subscription smoke checks passed: one `read` tool-call round-trip returned `PROBE_OK`, and one gold-package phase-1 call returned `answerable_fully`. The first non-streaming probe was rejected with `Stream must be set to true`; this was a protocol requirement, not a custom-tool refusal. After switching to the source-verified SSE shape, the custom AFT tool round-trip succeeded.

## Run protocol

The judge receives the original question and hydrated snippet bytes. Phase 1 runs without tools; a non-full result may open the nine read-only AFT tools with a hard cap of 15 calls. OAuth reads `~/.local/share/opencode/auth.json` fresh for every request and never calls the refresh endpoint. If the access entry is expired, it waits and re-reads with capped retries, then stops without attempting a refresh.

The calibration command used concurrency 2, model `gpt-5.6-luna`, and no token-rate inputs because the subscription route is not priced per token:

```sh
GATHER_DISTILL_AFT_BINARY=/Users/[owner]/Work/Projects/CortexKit/synapse/tools/gather-distill/bin/aft-v0.46.0 \
bun run src/cli.ts judge --phase calibration \
  --jobs /Users/[owner]/Work/Projects/CortexKit/synapse/tools/gather-distill/data/eval-jobs.jsonl \
  --gold /Users/[owner]/Work/Projects/CortexKit/synapse/tools/gather-distill/data/eval-gold-rows.jsonl \
  --oauth opencode --judge-model gpt-5.6-luna --concurrency 2
```

## Calibration evidence

The first prompt iteration exposed a Responses bookkeeping bug: when the model emitted more tool calls than remained in the budget, the harness sent some `function_call` items without matching outputs. The adapter was fixed to omit unexecuted calls before continuing. That first result is therefore retained only as a debugging record.

| iteration | prompt SHA | gold mean calls | empty mean calls | mismatched none | gate |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 (pre-fix) | `2c27195db8c4…` | 0.0 | 5.0 | 0/2 (errors) | FAIL |
| 2 | `a3281195e603…` | 6.0 | 9.8 | 0/5 | FAIL |
| 3 (final) | `5b8d15bfc7…` | 0.0 | 10.4 | 0/5 | FAIL |

The final gate failed only on the mismatched-control requirement: gold mean top-up was `0.0` (pass), empty mean was `10.4` (pass), but `0/5` mismatched packages were classified `none` (required at least `3/5`). The prompt was tightened through the allowed three iterations, including an explicit rule not to replace an unrelated non-empty package with repository browsing. GPT-5.6 still used top-up tools to repair those packages, so the calibration gate was not bypassed.

Calibration projection for the 40-package full matrix (gold plus the candidate lanes requested by the command) was:

- sample rows: 15
- mean input tokens: 31,935.2
- mean output tokens: 1,060.5
- projected input tokens: 1,277,408
- projected output tokens: 42,421
- projected USD: unpriced on the subscription route

## Utility versus F1

The full matrix was intentionally not run after the failed gate. Existing rows contained only the older 2B student; the requested `qwen35-4b-lora-v1`, `qwen35-9b-lora-v1`, and `deepseek-v4-flash-zeroshot` row files were absent from `data/students/` at run time. No system received judge calls in the blocked full phase, so no utility/F1 ranking or pairwise divergence can be claimed.

| system | full / partial / none | top-up calls mean | top-up calls median | top-up tokens mean | score mean | F1 | skipped invalid | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gold-control | not run | — | — | — | — | — | — | — |
| qwen35-2b-sft-v1 (available, not run) | — | — | — | — | — | existing score not compared | — | — |
| qwen35-4b-lora-v1 (rows absent) | — | — | — | — | — | unavailable | — | — |
| qwen35-9b-lora-v1 (rows absent) | — | — | — | — | — | unavailable | — | — |
| deepseek-v4-flash-zeroshot (rows absent) | — | — | — | — | — | unavailable | — | — |

**Ranking agreement:** not assessable because the gate blocked the full phase. **Divergence examples:** none are reported; producing examples without full per-job verdicts would be fabricated evidence.
