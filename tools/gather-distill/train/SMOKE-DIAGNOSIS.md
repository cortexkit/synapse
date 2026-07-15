# Qwen3.5-2B SFT 0/40 diagnosis

Date: 2026-07-15

## Verdict

**Root cause: serving enabled Qwen3.5 thinking even though the SFT examples were rendered in non-thinking mode.** The default generation prompt ended with an open `<think>` tag. The student emitted its ordinary prose and Qwen-native `<tool_call>` XML inside that reasoning span, then emitted EOS without `</think>`. llama-server therefore returned everything as `reasoning_content`, with empty `content` and no OpenAI `tool_calls`. The gather harness correctly saw an empty final answer and reported `JSON Parse error: Unexpected EOF`.

Setting `enable_thinking:false` through llama-server's pinned-build `--chat-template-kwargs` support changes the prompt suffix to a closed empty thinking span. The same checkpoint then emits parseable OpenAI tool calls and final JSON. The full fixed 40-job run scored 33/40 contract-valid (82.5%), with 39/40 syntactically parsed final packages and no API errors. This is a serving-mode failure, not failed SFT.

## Reproduction environment

- GGUF: `data/students/models/qwen35-2b-sft-v1-q8_0.gguf`
- SHA-256: `d63ed1ca210afbd84c5507ae1a312e7520a3b7344042655ccce3562764ef849a` (matches `qwen35-2b-gguf.json`)
- Server: llama.cpp build 9580, revision `b4e3dc613`, AppleClang arm64
- Device: Apple M5 Max / Metal, `-ngl 99 --jinja -fa on`
- Manual probe context: 32,768 (the request used 6,025 prompt tokens)
- Full eval context: 131,072, matching the original smoke
- Before the probes, one-minute load was below 12 (10.88, later 9.78); no GPU rental or parent-checkout writes were used.

The manual request was a raw `curl` to `/v1/chat/completions`, not the gather harness. It used the first fixed eval request, all nine OpenAI tool declarations, and `max_tokens:8000`.

## Hypothesis 1: thinking-mode mismatch — confirmed

### Default server

Command shape:

```sh
llama-server -m qwen35-2b-sft-v1-q8_0.gguf \
  --host 127.0.0.1 --port 8091 -ngl 99 --jinja -fa on -c 32768
```

The server detected `thinking = 1`. `/apply-template` ended verbatim with:

```text
<|im_end|>
<|im_start|>assistant
<think>
```

The raw completion returned:

```json
{
  "finish_reason": "stop",
  "message": {
    "role": "assistant",
    "content": "",
    "reasoning_content": "I'll explore the two files the request asks about.\n\n<tool_call>\n<function=outline>\n<parameter=target>\nclients/store/src/index.ts\n</parameter>\n<parameter=files>\nTrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=outline>\n<parameter=target>\nclients/store/src/derivation.ts\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=outline>\n<parameter=target>\nclients/store/src/descriptor.ts\n</parameter>\n</function>\n</tool_call>"
  }
}
```

llama-server's verbose response also reported:

```text
generation_prompt: "<|im_start|>assistant\n<think>\n"
stop_type: "eos"
tokens_predicted: 115
max_tokens: 8000
```

This exactly explains the harness phenotype: the model did useful work, but the parser classified it as reasoning because the model never closed the server-injected thinking span. Since the XML remained in `reasoning_content`, llama-server exposed zero `tool_calls`; the gatherer then attempted to parse empty `content` as final JSON.

### `enable_thinking:false`

The pinned build advertises both `--chat-template-kwargs STRING` and `--reasoning [on|off|auto]`. The tested request added:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

`/apply-template` then ended verbatim with the non-thinking prefix used by training:

```text
<|im_start|>assistant
<think>

</think>

```

The raw completion changed to:

```json
{
  "finish_reason": "tool_calls",
  "message": {
    "role": "assistant",
    "content": "I'll explore the two files to understand the re-exports.\n\n",
    "tool_calls": [
      {"type":"function","function":{"name":"read","arguments":"{\"filePath\":\"clients/store/src/index.ts\"}"}},
      {"type":"function","function":{"name":"read","arguments":"{\"filePath\":\"clients/store/src/derivation.ts\"}"}},
      {"type":"function","function":{"name":"read","arguments":"{\"filePath\":\"clients/store/src/descriptor.ts\"}"}}
    ]
  }
}
```

The generation still ended on the normal message EOS after 105 tokens, but the calls were now outside reasoning and were translated to the OpenAI `tool_calls` field.

### `/no_think` soft switch

Appending `/no_think` to the user message did **not** alter the template on build 9580. The raw response again had empty `content`, no `tool_calls`, all XML inside `reasoning_content`, `finish_reason:"stop"`, and this generation prompt:

```text
<|im_start|>assistant
<think>
```

Therefore `/no_think` is not a usable soft switch for this pinned server/template combination.

## Hypothesis 2: EOS/termination artifact — disproven as an independent cause

The default probe did stop on EOS after 115 of 8,000 allowed tokens, but the model stopped after a complete native tool-call turn, not at an arbitrary token boundary. With thinking disabled, it again stopped on EOS after a complete 105-token tool-call turn and llama-server correctly returned `finish_reason:"tool_calls"`. EOS is the trained turn delimiter; it becomes destructive only because the still-open reasoning span hides the turn.

The loss-mask artifact proves the training/serving prefix mismatch. Before the first supervised assistant token, the training render records these exact boundary tokens:

```text
<|im_start|> assistant \n <think> \n\n </think> \n\n I'll explore ...
```

Its first assistant boundary has `<think>` and `</think>` masked, assistant prose/tool XML trained, and `<|im_end|>` trained under `train_on_eos: turn`. The final-answer window likewise trains the complete fenced JSON followed by `<|im_end|>`. `full_render_byte_equal_to_tokenizer_template` is true for all three audited samples.

Thus the training render agrees with the **disabled-thinking** `/apply-template` output, while default serving supplies only `<think>\n`. No stop-token adjustment, `ignore_eos`, or `train_on_eos` change is warranted.

The original 40 rows also do not resemble a fixed max-token truncation: output ranged from 50 to 559 tokens (median 116.5), all were far below the 8,000 response limit, and every failure was EOF after only one to three assistant turns. Nineteen jobs made at least one parsed tool call because some sampled turns happened to close thinking before producing XML; none ever produced the four-key final package shape.

## Hypothesis 3: tool-call format mismatch — disproven

The GGUF template declares Qwen-native XML:

```text
<tool_call>
<function=read>
<parameter=filePath>
clients/store/src/index.ts
</parameter>
</function>
</tool_call>
```

The default raw probe emitted that exact format, and llama-server's parser reported `tool_mode: TAG_WITH_TAGGED`. With thinking disabled, the same native XML was correctly surfaced as OpenAI `message.tool_calls`. In the fixed full run all 40 jobs used tools (13.75 calls/job average). The format and parser agree; placement inside an unclosed reasoning span was the problem.

## Full fixed eval

The harness now has an additive serving knob:

```sh
EVAL_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
  scripts/eval-student.sh \
  data/students/models/qwen35-2b-sft-v1-q8_0.gguf \
  qwen35-2b-sft-v1-fixed
```

This passes the JSON as one argument to llama-server's `--chat-template-kwargs`. The local run used the pinned AFT binary, concurrency 2, the shared read-only eval corpora, and 131,072 served context.

The first fixed trajectory completed two exploration rounds and then emitted the complete package. Its final raw assistant text began:

````text
The full picture is clear. The index.ts only re-exports from descriptor.ts and derivation.ts ...

```json
{
  "interpretation": "What public symbols/types index.ts re-exports from derivation.ts and descriptor.ts, and whether any internal symbols leak into the public API.",
  "scope": ["clients/store/src/index.ts", "clients/store/src/derivation.ts", "clients/store/src/descriptor.ts"],
  "snippets": [
````

Results:

- 40/40 requests completed; 0 API errors.
- 39/40 produced a syntactically parseable final JSON object.
- 33/40 passed contract and on-disk snippet validation (82.5%).
- The one parse failure supplied a non-integer `endLine`; the other six failures referenced a missing path or a directory. These are answer-quality errors, not truncation.
- 37 natural completions, 3 budget finalizations, 0 invalid-final outcomes.
- 13.75 tool calls/job, 1,568.85 output tokens/job, and 0 thinking tokens/job.
- Natural-job file F1 0.561 and line Jaccard 0.542.

The tracked `data/students/LADDER.md` retains both rows; raw rows and score JSON remain ignored:

| model | natural file F1 | natural line Jaccard | contract-valid rate | API-error rate | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| qwen35-2b-sft-v1 | n/a | n/a | 0.0% | 0.0% | 1.02 | 57 | 0/40 | N 0/F 0/A 0/I 40 | 131072 | 1.3s |
| qwen35-2b-sft-v1-fixed | 0.561 | 0.542 | 82.5% | 0.0% | 13.75 | 0 | 37/40 | N 37/F 3/A 0/I 0 | 131072 | 35.9s |

## Bulk-ladder recommendation

**GO, with non-thinking serving made mandatory before spending on larger lanes.** The fixed 2B score is sufficient evidence that the SFT objective learned tool use and final packaging. Do not retrain this smoke, disable EOS, add final-answer epochs, or change `train_on_eos: turn` in `qwen35-2b-full.yaml`; those would address the wrong layer.

For every Qwen3.5 eval, set:

```sh
EVAL_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
```

For the planned Qwen3.5-9B training config, make the training template mode explicit instead of leaving line 5 as `chat_template: tokenizer_default`: use the same audited `chat_template: jinja` plus `chat_template_jinja: train/axolotl/templates/qwen35-aft.jinja` pairing as the 2B config, retain `roles_to_train: [assistant]` and `train_on_eos: turn`, and rerun the existing loss-mask/render-equality audit before rental. This keeps the supervised prefix and production generation prefix aligned. The larger-model go/no-go gate should reject any smoke whose `/apply-template` suffix differs from `<think>\n\n</think>\n\n` when `add_generation_prompt` is true.
