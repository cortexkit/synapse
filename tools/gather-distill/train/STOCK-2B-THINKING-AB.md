# Stock Qwen3.5-2B thinking A/B

Date: 2026-07-15

Labels: `qwen35-2b-stock-think` and `qwen35-2b-stock-nothink`

Verdict: **thinking did not improve stock 2B gather quality. It reduced contract validity from 22.5% to 0.0% and the diagnostic all-job file F1 from 0.049 to 0.000. The full SFT checkpoint, served with the same thinking-disabled setup, recovered 82.5% contract validity and a 0.561 natural file F1.**

## Artifact and control pins

| item | value |
| --- | --- |
| Training base | `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Stock GGUF | `unsloth/Qwen3.5-2B-GGUF` at `f6d5376be1edb4d416d56da11e5397a961aca8ae`, file `Qwen3.5-2B-Q8_0.gguf` |
| GGUF bytes / SHA-256 | 2,012,012,800 / `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1` |
| llama.cpp | `b4e3dc613baa92a3884d4151e3d631395c81934a`, build 9580 |
| Eval jobs SHA-256 | `ca25a1fc77b001fc1b582ab0ff9112eb59938139a9e66037341000a6d09ecf9c` |
| Gold rows SHA-256 | `c469e507ed900913e553c1aa63ad59d216729903ac71501b863ad89273600483` |
| AFT binary SHA-256 | `25cafa202e726a6b2d363fef4efac6e60ee6128105e7dbc42da7119e82b9a294` |
| Serving | local Metal, Q8_0, 131,072 context, concurrency 2 |

The Qwen Hub head was still the exact training-base revision when this run was prepared. The Unsloth repository identifies both `base_model:Qwen/Qwen3.5-2B` and `base_model:quantized:Qwen/Qwen3.5-2B`; its GGUF reports the same `qwen35` text architecture, 1,881,825,088 parameters, 262,144 trained context, and stock chat template. The GGUF repository does not publish a separate source-commit field, so this establishes the requested revision family rather than bit-level conversion provenance. No family delta from the training base was found.

Both arms used the same 40 jobs, gold rows, corpus checkouts, AFT executable, concurrency, model bytes, and server arguments. The only arm difference was `EVAL_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'` for `stock-nothink`; `stock-think` omitted the override. The one-minute load averages immediately before launch were 6.98 and 11.96 respectively, both below 12. Neither arm had an API error.

## Raw serving probes

The probes used the same tool declaration and request: “Use the lookup tool with query hello.” Raw responses remain in the ignored `data/students/` run evidence.

### Thinking disabled

`/apply-template` ended with the required suffix:

```text
<|im_start|>assistant
<think>

</think>


```

The raw chat completion was:

```json
{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{\"query\":\"hello\"}"},"id":"YXUSCIF9Yw1DrtA1TZ9EdWttdBam8iUS"}]}}],"model":"Qwen3.5-2B-Q8_0.gguf","system_fingerprint":"b9580-b4e3dc613","usage":{"completion_tokens":25,"prompt_tokens":287,"total_tokens":312}}
```

The pure tool-call turn correctly left `content` empty and populated the structured `tool_calls` field.

### Thinking enabled

With no template kwargs, `/apply-template` ended in an open thinking span:

```text
<|im_start|>assistant
<think>

```

The required pre-eval raw completion showed that stock Qwen closed and surfaced its reasoning and produced a structured tool call:

```json
{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"role":"assistant","content":"","reasoning_content":"The user is asking me to use the lookup tool with the query \"hello\". I need to call the lookup function with this query.\n","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{\"query\":\"hello\"}"},"id":"kicEGWB14VkTP1XFM5VH54bpFS8smw5P"}]}}],"model":"Qwen3.5-2B-Q8_0.gguf","system_fingerprint":"b9580-b4e3dc613","usage":{"completion_tokens":55,"prompt_tokens":285,"total_tokens":340}}
```

This rules out the earlier fine-tuned model's unclosed-span serving failure as the explanation for the thinking-arm result. The raw thinking probe took 416 ms total versus 227 ms without thinking (1.84×, +189 ms), and generation took 338 ms versus 152 ms (2.22×) for this otherwise matched call.

## Forty-job results

| model | natural file F1 | natural line Jaccard | contract-valid rate | API-error rate | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `qwen35-2b-stock-think` | n/a | n/a | 0.0% | 0.0% | 6.28 | 304 | 0/40 | N 0/F 2/A 0/I 38 | 131072 | 20.1s |
| `qwen35-2b-stock-nothink` | n/a | n/a | 22.5% | 0.0% | 14.97 | 0 | 0/40 | N 0/F 23/A 0/I 17 | 131072 | 42.8s |
| `qwen35-2b-sft-v1-fixed` | 0.561 | 0.542 | 82.5% | 0.0% | 13.75 | 0 | 37/40 | N 37/F 3/A 0/I 0 | 131072 | 35.9s |
| `qwen35-2b-half-v1` | 0.600 | 0.535 | 87.5% | 0.0% | 12.78 | 0 | 37/40 | N 37/F 1/A 0/I 2 | 131072 | 26.7s |

Natural file F1 is undefined for both stock arms because neither produced a natural completion. For the direct stock A/B, the scorer's all-job diagnostic file F1 was 0.000 with thinking and 0.049 without thinking; all-job line overlap was 0.000 and 0.042. These diagnostics include forced-finalization and invalid trajectories and are therefore not substituted into the natural-only ladder column. The ignored `data/students/LADDER.md` has both new ladder rows appended, and each arm's rows, ledger, status, server log, and score JSON remain ignored.

## Verdicts

**What thinking buys at 2B zero-shot.** Nothing on this gather workload: thinking changed diagnostic all-job file F1 by **-0.049**, contract validity by **-22.5 percentage points**, and natural completions by 0. It consumed 304 thinking tokens per trajectory but made only 6.28 tool calls instead of 14.97 and failed early in 38/40 jobs. Consequently its observed end-to-end wall time was 20.1 s/traj rather than 42.8 s/traj: a 22.7-second reduction caused by shorter failed trajectories, not a latency win at equal work. The matched raw probe isolates the intrinsic reasoning cost—+189 ms and 1.84× total latency—while the gather run shows no quality return that would justify generating a future 2B reasoning-trace tranche.

**Isolated SFT lift.** With thinking disabled on identical serving, SFT raised contract validity from **22.5% to 82.5% (+60.0 points)** and natural completions from **0/40 to 37/40**, while slightly reducing tool calls from 14.97 to 13.75 and wall time from 42.8 to 35.9 s/traj. Stock-nothink has no natural-only F1 from which a valid numeric F1 delta can be subtracted; treating its undefined ladder value as zero would be misleading. The isolated quality result is therefore that SFT creates a measurable natural regime at **0.561 file F1 / 0.542 line Jaccard**, whereas stock-nothink only reaches 0.049 all-job diagnostic F1 through forced-finalization outputs. This is a large contract-following and task-behavior lift from SFT, not evidence that 2B reasoning traces are the missing ingredient.
