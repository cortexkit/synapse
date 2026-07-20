# LFM2 tool-calling verification — owned bridge gate

**Measured:** 2026-07-20 on the local M5 only. **Owned checkpoint:**
`LiquidAI/LFM2-1.2B`, HF snapshot
`933cee00d754fb3bfe06c644c0cb95453f2d8bb2`. **Owned mode:**
`spike-unified-rt`, CPU/Accelerate, fp32, greedy (`--decode-top-k 1`).
**Cross-check:** `llama-cli` build `b9580-b4e3dc613` (llama.cpp 9580), official
local `LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-F16.gguf`, `--temp 0`,
`--top-k 1`, `--top-p 1`.

This is a format gate, not a claim that the LFM2-1.2B base checkpoint is a
reliable tool router. The local cache contained the LFM2-1.2B HF snapshot and
an official LFM2.5-230M GGUF, but not an LFM2.5-1.2B HF snapshot or an
LFM2-1.2B GGUF. Therefore the llama.cpp comparison uses the same five
rendered prompts, but a smaller LFM2.5 checkpoint.

## Format contract

The captured Leap/LFM2 data contract in
[`docs/leap-finetune-assessment.md`](../../../docs/leap-finetune-assessment.md#tool-calling-data-contract)
is:

- Definitions are in the system turn. The canonical content includes
  `List of tools: <|tool_list_start|>[{...}]<|tool_list_end|>`.
- A legacy LFM2 assistant call is tool-call-first and has the form
  `<|tool_call_start|>[func(arg="value")]<|tool_call_end|>`. Prose before the
  call is not accepted by the documented LFM2 validator.
- The call body is a Pythonic list of calls, not JSON. The harness parses it
  with Python `ast.parse` plus `ast.literal_eval` for each argument; it does
  not accept a marker-only or regex-only match.
- A `role: "tool"` message contains only the response payload. The LFM2
  template supplies `<|tool_response_start|>` and
  `<|tool_response_end|>` around it. Those response markers must not be put
  in the tool content by the caller.
- Structured OpenAI tool fields are converted to the bracket form by the
  training/tokenization path; foreign XML and Mistral-style markers are not
  the contract.

The HF snapshot's `chat_template.jinja` independently confirms the contract.
With `apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True)`,
it appends the tool list to the system message, emits the `im_start`/`im_end`
turn framing, and ends at `<|im_start|>assistant\n`. The harness uses that
rendering path, rather than hand-rendering. It writes the exact rendered text
and token IDs to `target/lfm2-toolcall-verification/rendered-prompts.jsonl`.

### Token evidence

The model tokenizer metadata and the template probe produced these IDs:

| token | ID |
|---|---:|
| `<|startoftext|>` (BOS) | 1 |
| `<|im_start|>` | 6 |
| `<|im_end|>` (EOS) | 7 |
| `<|tool_list_start|>` | 8 |
| `<|tool_list_end|>` | 9 |
| `<|tool_call_start|>` | 10 |
| `<|tool_call_end|>` | 11 |
| `<|tool_response_start|>` | 12 |
| `<|tool_response_end|>` | 13 |

The five template-tokenized prompts were 328, 333, 344, 330, and 342 tokens.
The owned runner calls `Tokenizer::encode(text, true)` after receiving the
rendered string, so it prepended another BOS to each input; its actual prompt
lengths were 329, 334, 345, 331, and 343. This existing loader behavior is
preserved and is recorded in the fixture as `template_input_ids` versus
`input_ids`; both engines otherwise received the same rendered prompt text.

## Harness

The committed harness is
[`verify_lfm2_toolcall.py`](verify_lfm2_toolcall.py). It:

1. renders five prompts with the HF tokenizer and three tool definitions;
2. writes the prompt/token-ID fixture;
3. runs the owned binary on CPU fp32 and decodes its generated IDs with the
   same tokenizer;
4. mechanically validates markers, Python call syntax, expected tool name,
   required arguments, scalar literal arguments, and declared argument names;
5. runs each same rendered prompt through `llama-cli --single-turn` against the
   official GGUF at temperature zero; and
6. runs one `--decode-json` probe on the first prompt and records the JSON
   result.

Reproduction (after building the owned binary):

```sh
cargo build --release -p spike-unified-rt
python bench/spikes/unified-rt/verify_lfm2_toolcall.py \
  --model "$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2-1.2B/snapshots/933cee00d754fb3bfe06c644c0cb95453f2d8bb2" \
  --gguf "$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M-GGUF/snapshots/fa224d4cb60cffe61eb58726712ef255bb64d0b7/LFM2.5-230M-F16.gguf" \
  --artifact-dir target/lfm2-toolcall-verification
```

The harness completed green with five prompts, three schema-valid owned calls,
five valid llama.cpp calls, and three structural agreements. For structural
agreement, the gate compares the selected tool, positional arity, and the
required schema arguments. Optional-argument differences remain visible in the
transcripts and are not silently called token-exact.

## Owned decode transcripts

The generated text below is decoded from the owned token IDs. `PASS` requires
both the marker pair and a schema-plausible AST call.

| prompt | result | decoded assistant completion |
|---|---|---|
| `weather-istanbul` | **PASS** | `<|tool_call_start|>[get_weather(location="Istanbul", units="celsius")]<|tool_call_end|><|im_end|>` |
| `weather-ankara` | **PASS** | `<|tool_call_start|>[get_weather(location="Ankara", units="fahrenheit")]<|tool_call_end|><|im_end|>` |
| `reminder-call-mom` | **PASS** | `<|tool_call_start|>[create_reminder(title="Call mom", time="2026-07-21T09:00:00", timezone="Europe/Istanbul")]<|tool_call_end|><|im_end|>` |
| `currency-usd-try` | **FAIL: no call attempted** | `I'm sorry, I don't have access to a tool to convert currencies directly. However, I can help you with other tasks such as getting the current weather, creating reminders, or converting currencies to different units like Celsius or Fahrenheit. Let me know if you need assistance with any of these!<|im_end|>` |
| `reminder-groceries` | **FAIL: schema argument** | `<|tool_call_start|>[create_reminder(title="Buy groceries", time="2026-07-21T18:30:00", reminder_type="event")]<|tool_call_end|><|im_end|>` — `reminder_type` is not declared by the tool schema. |

The owned path therefore preserved the exact LFM2 marker IDs and generated three
fully valid calls out of five. The two failures are model adherence failures,
not parser false positives: one omitted the call and one selected an undeclared
argument while retaining otherwise correct bracket syntax.

## llama.cpp cross-check

`llama-cli --single-turn --temp 0 --top-k 1 --top-p 1 --n-predict 96` produced
the following generated call portions. The llama.cpp conversation frontend
returned the bracket body without the LFM marker tokens, so the harness accepts
that frontend-normalized form for the cross-engine structural comparison; the
owned gate above still requires both markers.

| prompt | llama.cpp generated call | structural result |
|---|---|---|
| `weather-istanbul` | `[get_weather(location="Istanbul")]` | agree on `get_weather` and required `location` |
| `weather-ankara` | `[get_weather(location="Ankara", units="fahrenheit")]` | agree |
| `reminder-call-mom` | `[create_reminder(title="Call mom", time="2026-07-21T09:00:00", timezone="Europe/Istanbul")]` | agree |
| `currency-usd-try` | `[convert_currency(amount=100, from_currency="USD", to_currency="TRY")]` | llama valid; owned did not call |
| `reminder-groceries` | `[create_reminder(title="Buy groceries", time="2026-07-21T18:30:00", timezone="America/New_York")]` | not counted: llama selected the same required fields, but owned is schema-invalid (`reminder_type`) |

This is structural agreement only. The GGUF run is not a token-exact or
same-checkpoint parity claim, and llama.cpp's frontend marker normalization is
an integration detail the bridge must account for when it consumes the result.

## Constrained-decoding interaction probe

The harness reran the first rendered prompt through the owned runtime with
`--decode-json`, which activates the existing `JsonConstraint`/`TokenMask` at
the pre-commit tap. The probe reported `constraint: "json"`, vocabulary size
64,400, `constraint_valid_prompts: 1`, and a valid JSON completion:

```json
{
"weather": "Sunny, with a high of 28°C (82°F) and a low of 22°C (72°F). Light breeze from the north."
}
```

The exact weather prose is model output and is not the gate. The decoded
constrained text contained neither `<|tool_call_start|>` nor `<|tool_call_end|>`.

This is the interaction hazard: the JSON recognizer starts at JSON root state,
while LFM2 bracket notation starts with a special marker and then a Python call
list. The marker IDs are special and are excluded from ordinary JSON byte-trie
transitions, so JSON masking cannot produce a bracket call and cannot be
blindly left enabled for an LFM2 tool-call request. The pre-commit tap itself
is clean—the probe completed and revalidated JSON—but the constraint must be
disabled for bracket-format generation or replaced by a marker-aware constraint
scoped only to an explicitly selected payload span. Do not try to constrain the
whole assistant completion as JSON and then expect LFM markers to survive.

## Verdict

**Bridge tool-calling path: viable on owned today for format emission, not yet
product-reliable for tool selection/schema adherence.** The owned LFM2-1.2B
CPU path emitted exact bracket markers and three mechanically/schema-valid
calls in five demanding prompts, while llama.cpp independently agreed on the
required call structure for those three. Before shipping, the bridge still
needs a model/tool-adherence policy (or retry/fallback) for no-call and
undeclared-argument outputs, an output parser that accepts the owned markers
and llama.cpp's normalized bracket body, and explicit constraint routing that
keeps JSON `TokenMask` off bracket-format spans or adds a marker-aware grammar.
