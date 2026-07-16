# DeepSeek v4 Flash vs the 4B student

Date: 2026-07-16

## Executive result

`deepseek-v4-flash` produced the strongest *conditional* gather packages in this run, but it did not reliably finish the production contract within the fixed budget. Its natural-only file F1 was **0.872** and line Jaccard **0.743** on 16/40 jobs. The trained local `qwen35-4b-lora-v1` was less accurate on natural completions (**0.637 / 0.639**) but reached a natural completion on 37/40 jobs. For production gather work, the 4B is the more dependable default; DeepSeek is a useful higher-quality hosted fallback only when its extra budget and quota cost are acceptable.

## Protocol and smoke probe

The run replayed the same 40 jobs from `data/eval-jobs.jsonl` against the 40 Opus gold rows, with the same read-only corpora, AFT binary `aft-dev-7cabfdd0`, production gather prompt, 40-step budget, inline validation, and concurrency 2. The verified pins were:

- eval jobs SHA-256: `ca25a1fc77b001fc1b582ab0ff9112eb59938139a9e66037341000a6d09ecf9c`
- gold rows SHA-256: `c469e507ed900913e553c1aa63ad59d216729903ac71501b863ad89273600483`
- AFT binary SHA-256: `25cafa202e726a6b2d363fef4efac6e60ee6128105e7dbc42da7119e82b9a294`

The runtime key was read from `~/.local/share/opencode/auth.json` and kept in the shell environment only. The unchanged OpenAI lane does not add authorization headers, so a loopback forwarding proxy added the runtime bearer header before forwarding the exact harness payload to `https://ollama.com/v1`; no source, scorer, row, or key changes were committed.

The one raw smoke request used all nine production AFT tool schemas and asked the model to call `search` with `{"query":"hello"}`. The endpoint returned HTTP 200 and a standard structured tool call:

```json
{
  "content": "",
  "reasoning": "The user wants me to use the search tool with the query \"hello\". Let me dothat.",
  "tool_calls": [
    {
      "type": "function",
      "function": {"name": "search", "arguments": "{\"query\":\"hello\"}"}
    }
  ]
}
```

The response used `message.reasoning`, not `reasoning_content`; `content` was empty on the pure tool-call turn and `tool_calls` populated correctly. Usage was `prompt_tokens=3,528`, `completion_tokens=64`, `total_tokens=3,592`. The full run had no API errors, no observed 429 responses, and no context-window refusal; the largest accumulated prompt usage was 318,194 tokens, so the endpoint did not cap context near 40k.

## Head-to-head

F1 and line Jaccard are means over **natural completions only**. Contract validity, natural-job count, tool calls, thinking tokens, and wall time are whole-run measures over all 40 jobs. `N/F/A/I` means natural, budget-finalized, API-error, and invalid-final outcomes.

| model | natural file F1 | natural line Jaccard | contract-valid | naturals | avg tool calls | thinking/reasoning tokens / traj | wall / traj | rough package economics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `deepseek-v4-flash` zero-shot | **0.872** | **0.743** | 85.0% | 16/40 | 23.40 | 1,209 | **36.4s** | Ollama Cloud subscription/quota; no published per-token rate. 5.635M input + 164.906k output tokens consumed |
| `qwen35-4b-lora-v1` | 0.637 | 0.639 | **87.5%** | **37/40** | 12.38 | 0 | 72.5s | Local Metal, effectively $0 marginal API cost; 4.48 GB Q8 artifact |
| `qwen36-27b` zero-shot | 0.818 | 0.677 | 80.0% | 26/40 | 15.68 | 724 | 96.1s | H100 bake-off reference; benchmark infrastructure rate not retained |

DeepSeek's whole-run diagnostic means were file F1 **0.633** and line Jaccard **0.524** once the 24 budget-finalized jobs are included. That is why the natural-only 0.872 must not be read as a 40-job production score: the model often found useful evidence but kept exploring until the harness forced finalization. Its outcome split was `N 16/F 24/A 0/I 0`; 18 of the forced finals were schema-valid, but they are excluded from the natural quality columns by the fixed scorer convention.

The local 4B has the better quality-per-dollar story because its inference has no hosted token bill after the one-time local artifact/training cost. It also has the better dependable quality-per-job: 37 natural completions versus 16. DeepSeek has the better quality-per-second **conditional on finishing naturally** (0.872 F1 in 36.4 seconds versus 0.637 in 72.5 seconds), but the whole-run completion rate reverses the operational conclusion. The 27B reference remains the quality ceiling among the three on the natural-only metric, while being slower and less contract-reliable than the trained 4B.

## Token and cost accounting

The 40 DeepSeek trajectories recorded these API usage totals:

| usage field | total | mean / trajectory | min–max / trajectory |
| --- | ---: | ---: | ---: |
| input tokens | 5,634,775 | 140,869.4 | 14,440–318,194 |
| output tokens | 164,906 | 4,122.7 | 809–14,567 |
| reasoning/thinking tokens | 48,370 | 1,209.3 | 111–5,996 |

The endpoint did not expose an explicit `reasoning_tokens` usage field. The harness therefore counted whitespace-delimited tokens in the returned `message.reasoning` field; these are a deterministic reasoning estimate, while `completion_tokens` is the API-reported output total.

Ollama Cloud currently describes usage as subscription/quota consumption measured by GPU time rather than a published token price. The model page labels `deepseek-v4-flash:cloud` as medium usage. Therefore this run cannot honestly assign an Ollama per-package dollar amount from token counts. As a non-Ollama reference only, the public DeepSeek API rates linked below would price this usage at approximately **$0.835** if every input token were a cache miss (`5.635M × $0.14/M + 0.165M × $0.28/M`), or approximately **$0.062** if every input token were a cache hit; neither number is an Ollama Cloud invoice.

Sources: [Ollama pricing](https://ollama.com/pricing), [DeepSeek v4 Flash model page](https://ollama.com/library/deepseek-v4-flash:cloud), and [DeepSeek API pricing](https://api-docs.deepseek.com/quick_start/pricing).

## Where the behavior diverged

On the cross-toolchain trace in `cortexkit/aft` — “trace a value defined in `.cortexkit/aft.jsonc` into Rust through `.cargo/config.toml`” — DeepSeek spent all 32 tool calls on broad reconnaissance: it read the config and package manifests, walked `packages/aft-bridge`, inspected `crates/aft`, ran three increasingly specific searches, and then was still issuing searches when forced to finalize. The gold path needed the bridge resolver plus `crates/aft-tokenizer/build.rs`, `crates/aft/src/subc_config.rs`, and `crates/aft/src/config_resolve.rs`. This is the clearest example of frontier-style breadth turning into a budget failure; the trained 4B's 37/40 natural rate and 12.38 average calls make it the safer bounded-exploration choice even when its citations are less complete.

On the ANE/CoreML-versus-CPU/Vulkan trace in `cortexkit/synapse`, DeepSeek began with `.alfonso/spikes/coreml_spike.rs`, then outlined `synapse-core`, both engine crates, and runtime files, repeatedly reread `crates/synapse-engine-ort/src/lib.rs` and `crates/synapse-engine-owned/src/lib.rs`, and ended after 28 calls without a final package. The gold answer spans the spike, `synapse-core/src/engine.rs`, ORT, worker protocol, ANE worker, Swift worker, and module bindings, so the model's broad exploration was directionally relevant but too expansive for the fixed finalization budget. The 4B's advantage here is not deeper search; it is terminating with a valid package far more often and at zero marginal token cost.

The qwen4B per-job rows are ignored run artifacts and were not retained beside the committed ladder report, so these two paragraphs do not invent paired qwen citation lists. They identify the two clearest DeepSeek budget-failure traces and contrast their observed exploration style with the 4B's retained aggregate behavior.

## Verdict

The local 4B is not frontier-equivalent on conditional citation overlap, but it is the better gather worker for a fixed production contract: it is contract-valid at nearly the same rate as DeepSeek, finishes 21 more of 40 jobs naturally, consumes no hosted token budget, and has a much smaller reasoning/tool-call tail. DeepSeek v4 Flash is attractive as a hosted cheap-frontier fallback when the caller values the quality of the subset that completes and can tolerate budget-finalized answers; it is not a drop-in replacement for the trained 4B under this 40-step gather policy without a tighter exploration/finalization behavior.
