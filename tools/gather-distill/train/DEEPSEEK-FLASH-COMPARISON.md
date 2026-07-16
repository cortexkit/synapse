# DeepSeek v4 Flash vs the 4B student

Date: 2026-07-16

## Executive result

`deepseek-v4-flash` still produced the strongest *conditional* gather packages in this comparison, but disabling reasoning did not remove its fixed-budget completion problem. The thinking arm reached a natural-only file F1 of **0.872** and line Jaccard **0.743** on 16/40 jobs; the nothink arm reached **0.862 / 0.677** on 18/40. The trained local `qwen35-4b-lora-v1` was less accurate on natural completions (**0.637 / 0.639**) but reached a natural completion on 37/40 jobs. For production gather work, the 4B remains the more dependable default; DeepSeek is a useful higher-quality hosted fallback only when its extra quota cost and budget-finalized tail are acceptable.

## Protocol and smoke probe

The run replayed the same 40 jobs from `data/eval-jobs.jsonl` against the 40 Opus gold rows, with the same read-only corpora, AFT binary `aft-dev-7cabfdd0`, production gather prompt, 40-step budget, inline validation, and concurrency 2. The verified pins were:

- eval jobs SHA-256: `ca25a1fc77b001fc1b582ab0ff9112eb59938139a9e66037341000a6d09ecf9c`
- gold rows SHA-256: `c469e507ed900913e553c1aa63ad59d216729903ac71501b863ad89273600483`
- AFT binary SHA-256: `25cafa202e726a6b2d363fef4efac6e60ee6128105e7dbc42da7119e82b9a294`

The runtime key was read from `~/.local/share/opencode/auth.json` and kept in the shell environment only. The unchanged OpenAI lane does not add authorization headers, so a loopback forwarding proxy added the runtime bearer header before forwarding the exact harness payload to `https://ollama.com/v1`; no source, scorer, row, or key changes were committed. The nothink replay used the same proxy, with one additional request-body transformation documented below.

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

The response used `message.reasoning`, not `reasoning_content`; `content` was empty on the pure tool-call turn and `tool_calls` populated correctly. Usage was `prompt_tokens=3,528`, `completion_tokens=64`, `total_tokens=3,592`. The thinking run had no API errors, no observed 429 responses, and no context-window refusal; the largest accumulated prompt usage was 318,194 tokens, so the endpoint did not cap context near 40k. The nothink replay likewise had 0 API errors, 0 observed 429 responses, and no context-window refusal; its largest accumulated prompt usage was 307,719 tokens.

## Reasoning-disable knob discovery

The raw probe tried each candidate against `POST https://ollama.com/v1/chat/completions` with the nine production AFT tool schemas and the same structured `search` smoke request. All four probes returned HTTP 200 and a structured `search` tool call, but only one disabled reasoning:

| request change | result |
| --- | --- |
| no extra field | `message.reasoning` was nonempty; `tool_calls` was structured |
| `"reasoning":{"enabled":false}` | `message.reasoning` was still nonempty; `tool_calls` was structured |
| `"think":false` | `message.reasoning` was still nonempty; `tool_calls` was structured |
| `"reasoning_effort":"none"` | `message.reasoning` and `message.reasoning_content` were absent; `content` was empty and `tool_calls` was structured |

The working probe reported `prompt_tokens=3,527`, `completion_tokens=43`, and `total_tokens=3,570`, with `tool_calls[0].function` equal to `{"name":"search","arguments":"{\"query\":\"hello\"}"}`. For the replay, the loopback proxy parsed each harness JSON body and appended exactly one top-level member, `"reasoning_effort":"none"`, before forwarding it. It did not change `messages`, the nine tool schemas, or `tool_choice`; this is the only model-request variable between the arms. The endpoint honored the OpenAI-style field directly; no native `/api/chat` translation or model-tag variant was needed.

## Head-to-head

F1 and line Jaccard are means over **natural completions only**. Contract validity, natural-job count, tool calls, thinking tokens, and wall time are whole-run measures over all 40 jobs. `N/F/A/I` means natural, budget-finalized, API-error, and invalid-final outcomes.

| model | natural file F1 | natural line Jaccard | contract-valid | naturals | avg tool calls | thinking/reasoning tokens / traj | wall / traj | rough package economics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `deepseek-v4-flash` zero-shot (thinking) | **0.872** | **0.743** | 85.0% | 16/40 | 23.40 | 1,209 | **36.4s** | Ollama Cloud subscription/quota; no published per-token rate. 5.635M input + 164.906k output tokens consumed |
| `deepseek-v4-flash-nothink` zero-shot | 0.862 | 0.677 | **97.5%** | 18/40 | 19.40 | 0 | **31.2s** | Same Ollama Cloud subscription/quota; 5.799M input + 91.805k output tokens consumed |
| `qwen35-4b-lora-v1` | 0.637 | 0.639 | **87.5%** | **37/40** | 12.38 | 0 | 72.5s | Local Metal, effectively $0 marginal API cost; 4.48 GB Q8 artifact |
| `qwen36-27b` zero-shot | 0.818 | 0.677 | 80.0% | 26/40 | 15.68 | 724 | 96.1s | H100 bake-off reference; benchmark infrastructure rate not retained |

DeepSeek's thinking-arm whole-run diagnostic means were file F1 **0.633** and line Jaccard **0.524** once the 24 budget-finalized jobs were included. That is why the natural-only 0.872 must not be read as a 40-job production score: the model often found useful evidence but kept exploring until the harness forced finalization. Its outcome split was `N 16/F 24/A 0/I 0`; 18 of the forced finals were schema-valid, but they are excluded from the natural quality columns by the fixed scorer convention.

The nothink arm's whole-run means were file F1 **0.695** and line Jaccard **0.575**. It finished naturally on only two additional jobs (`N 18/F 22/A 0/I 0`), although 21 of the forced finals were schema-valid, giving 39/40 contract-valid rows overall. Reasoning removal reduced average tool calls from 23.40 to 19.40 and output tokens from 4,122.7 to 2,295.1 per trajectory, but it did not turn the hosted model into a reliable bounded worker. Its natural-only F1 fell by **0.010** and line Jaccard by **0.066**, so the original 0.872 conditional advantage was not purely budget burned by visible thinking.

The local 4B has the better quality-per-dollar story because its inference has no hosted token bill after the one-time local artifact/training cost. It also has the better dependable quality-per-job: 37 natural completions versus 16 for thinking DeepSeek and 18 for nothink DeepSeek. Thinking DeepSeek has the better conditional citation quality (0.872 F1 in 36.4 seconds versus 0.862 in 31.2 seconds for nothink and 0.637 in 72.5 seconds for the 4B); nothink is faster and cheaper in output tokens, but the whole-run completion rate reverses the operational conclusion for both hosted arms. The 27B reference remains the quality ceiling among the three original comparison points on the natural-only metric, while being slower and less contract-reliable than the trained 4B.

## Token and cost accounting

The 40 DeepSeek trajectories recorded these API usage totals:

| arm / usage field | total | mean / trajectory | min–max / trajectory |
| --- | ---: | ---: | ---: |
| thinking / input tokens | 5,634,775 | 140,869.4 | 14,440–318,194 |
| thinking / output tokens | 164,906 | 4,122.7 | 809–14,567 |
| thinking / reasoning tokens | 48,370 | 1,209.3 | 111–5,996 |
| nothink / input tokens | 5,799,303 | 144,982.6 | 15,026–307,719 |
| nothink / output tokens | 91,805 | 2,295.1 | 696–9,670 |
| nothink / reasoning tokens | 0 | 0 | 0–0 |

The endpoint did not expose an explicit `reasoning_tokens` usage field. For the thinking arm, the harness therefore counted whitespace-delimited tokens in the returned `message.reasoning` field; these are a deterministic reasoning estimate, while `completion_tokens` is the API-reported output total. In the nothink arm, the working knob removed both reasoning fields from every response, so the harness recorded zero reasoning tokens and the API-reported output total fell by 44.3%.

Ollama Cloud currently describes usage as subscription/quota consumption measured by GPU time rather than a published token price. The model page labels `deepseek-v4-flash:cloud` as medium usage. Therefore this run cannot honestly assign an Ollama per-package dollar amount from token counts. As a non-Ollama reference only, the public DeepSeek API rates linked below would price this usage at approximately **$0.835** if every input token were a cache miss (`5.635M × $0.14/M + 0.165M × $0.28/M`), or approximately **$0.062** if every input token were a cache hit; neither number is an Ollama Cloud invoice.

Sources: [Ollama pricing](https://ollama.com/pricing), [DeepSeek v4 Flash model page](https://ollama.com/library/deepseek-v4-flash:cloud), and [DeepSeek API pricing](https://api-docs.deepseek.com/quick_start/pricing).

## Where the behavior diverged

On the cross-toolchain trace in `cortexkit/aft` — “trace a value defined in `.cortexkit/aft.jsonc` into Rust through `.cargo/config.toml`” — DeepSeek spent all 32 tool calls on broad reconnaissance: it read the config and package manifests, walked `packages/aft-bridge`, inspected `crates/aft`, ran three increasingly specific searches, and then was still issuing searches when forced to finalize. The gold path needed the bridge resolver plus `crates/aft-tokenizer/build.rs`, `crates/aft/src/subc_config.rs`, and `crates/aft/src/config_resolve.rs`. This is the clearest example of frontier-style breadth turning into a budget failure; the trained 4B's 37/40 natural rate and 12.38 average calls make it the safer bounded-exploration choice even when its citations are less complete.

On the ANE/CoreML-versus-CPU/Vulkan trace in `cortexkit/synapse`, DeepSeek began with `.alfonso/spikes/coreml_spike.rs`, then outlined `synapse-core`, both engine crates, and runtime files, repeatedly reread `crates/synapse-engine-ort/src/lib.rs` and `crates/synapse-engine-owned/src/lib.rs`, and ended after 28 calls without a final package. The gold answer spans the spike, `synapse-core/src/engine.rs`, ORT, worker protocol, ANE worker, Swift worker, and module bindings, so the model's broad exploration was directionally relevant but too expansive for the fixed finalization budget. The 4B's advantage here is not deeper search; it is terminating with a valid package far more often and at zero marginal token cost.

The nothink replay reduced the amount of visible reasoning and tool exploration but retained the same broad-search failure mode: 22 of 40 jobs still reached forced finalization. The small natural-count gain therefore looks like a modest budget relief, not a recovery of the production completion rate. The qwen4B per-job rows are ignored run artifacts and were not retained beside the committed ladder report, so these paragraphs do not invent paired qwen citation lists. They identify the two clearest DeepSeek budget-failure traces and contrast their observed exploration style with the 4B's retained aggregate behavior.

## Nothink-arm verdict

Disabling reasoning did **not** materially recover natural completion: the rate rose only from 16/40 (40.0%) to 18/40 (45.0%), while the local 4B remained at 37/40 (92.5%). Conditional file F1 moved from **0.872** to **0.862**, and conditional line Jaccard fell from **0.743** to **0.677**. The trade was real budget relief — zero reasoning tokens, 19.40 rather than 23.40 tool calls per trajectory, and 44.3% fewer output tokens — but the model still spent enough of the fixed trajectory budget exploring to force 22 finalizations. Thinking was therefore contributing to the budget failure, but it was not the whole explanation for the frontier-breadth failure, and it was buying a small amount of conditional citation quality.

## Verdict

The local 4B is not frontier-equivalent on conditional citation overlap, but it is the better gather worker for a fixed production contract: it is contract-valid at **97.5%** in nothink mode and **85.0%** in thinking mode, yet finishes naturally on only 18 or 16 of 40 jobs versus 37 for the 4B, while consuming hosted quota in both arms. DeepSeek v4 Flash remains attractive as a hosted cheap-frontier fallback when the caller values the quality of the subset that completes and can tolerate budget-finalized answers. The operational conclusion does not change: reasoning-off is a useful hosted cost/latency setting, not a drop-in replacement for the trained 4B under this 40-step gather policy; the fallback still needs a tighter exploration/finalization behavior or a larger budget.
