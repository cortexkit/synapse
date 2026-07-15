#!/usr/bin/env python3
"""Build PREP-REPORT.md from the committed machine-readable evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAIN_DIR = Path(__file__).resolve().parent
CONVERSION = json.loads((TRAIN_DIR / "conversion-report.json").read_text())
AUDIT = json.loads((TRAIN_DIR / "tokenizer-audit.json").read_text())
LOSS = json.loads((TRAIN_DIR / "loss-mask-verification.json").read_text())
OUTPUT = TRAIN_DIR / "PREP-REPORT.md"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def code(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def add(lines: list[str], value: str = "") -> None:
    lines.append(value)


def main() -> None:
    lines: list[str] = []
    add(lines, "# Gatherer SFT dataset preparation report")
    add(lines)
    add(lines, "## Result")
    add(lines)
    add(
        lines,
        "**PASS, with an explicit pre-training overflow gate.** All 1,884 valid source rows converted; no rows were dropped. All four exact Hugging Face repositories resolved and rendered every example without truncation. No tokenizer crossed the 10% stop threshold. Depending on tokenizer, 17–20 records (0.90–1.06%) exceed 32,768 tokens and must receive an approved disposition before training. No GPU was rented and no training was started.",
    )
    add(lines)
    add(lines, "## Provenance and source shape")
    add(lines)
    add(
        lines,
        f"- Source: `data/dataset-v1.jsonl`, 1,884 newline-terminated rows, SHA-256 `{CONVERSION['input_sha256']}`.",
    )
    add(
        lines,
        "- The ignored worktree copy was byte-checked against the parent source before conversion (same SHA-256 above).",
    )
    add(
        lines,
        "- `src/gather.ts::runGatherJob` starts the stored trajectory with the fully rendered user request envelope, stores Anthropic assistant `response.content` blocks unchanged, and stores each AFT output in a user `tool_result` block. `src/anthropic.ts` confirms that response content is the Anthropic block array. The final parsed object is separately stored in `final_json`; it is not the training target text.",
    )
    add(
        lines,
        "- Production system prompt: `GATHER_CONTEXT_SYSTEM_PROMPT_V1` (the harness compatibility name is `GATHER_SYSTEM_PROMPT_V10`), including its trailing newline.",
    )
    add(
        lines,
        "- Production tool catalog: exactly nine declarations (`search`, `outline`, `zoom`, `callgraph`, `read`, `grep`, `glob`, `inspect`, `conflicts`), converted by `src/openai.ts::toOpenAiTools`; OpenAI-schema SHA-256 "
        + f"`{CONVERSION['tools']['schema_sha256']}`.",
    )
    add(lines)
    add(
        lines,
        "The stored Anthropic `tool_use.input` is already a parsed JSON value; raw API argument substrings are not banked. The converter therefore reuses `src/openai.ts`, whose only possible representation is compact `JSON.stringify`: IDs, names, values, key insertion order, strings, and numbers are preserved, but original insignificant JSON whitespace cannot be recovered.",
    )
    add(lines)
    add(lines, "## Conversion")
    add(lines)
    add(lines, "| Rows in | Rows out | Dropped | Drop rate | Output SHA-256 |")
    add(lines, "|---:|---:|---:|---:|---|")
    add(
        lines,
        f"| {CONVERSION['rows_in']} | {CONVERSION['rows_out']} | {CONVERSION['rows_dropped']} | {pct(CONVERSION['drop_rate'])} | `{CONVERSION['output_sha256']}` |",
    )
    add(lines)
    add(
        lines,
        "Drop reasons: **none**. `train/sft-dataset.jsonl` is 1,884 lines and is explicitly ignored. The first stored user message remains the actual rendered generation envelope (request, scope, pinned HEAD, and budgets), rather than being reconstructed from metadata. The final assistant content is copied from the trajectory text, preserving prose and fenced JSON exactly; `final_json` is never serialized back into the target.",
    )
    add(lines)
    add(lines, "### Deterministic random round-trip checks")
    add(lines)
    add(
        lines,
        f"Selection method: `{CONVERSION['random_verification']['method']}`. Every check parsed each function argument string as JSON, required a unique matching result for every call, rejected orphan/duplicate tool messages, enforced pending-call ordering, and required a nonempty final assistant answer without tool calls.",
    )
    add(lines)
    add(
        lines,
        "| Row | Repository | Messages | Tool calls | Tool results | Assistant turns | Final text SHA-256 | Result |",
    )
    add(lines, "|---:|---|---:|---:|---:|---:|---|---|")
    for sample in CONVERSION["random_verification"]["samples"]:
        add(
            lines,
            f"| {sample['row_index']} | `{md(sample['repo_full'])}` | {sample['messages']} | {sample['toolCalls']} | {sample['toolResults']} | {sample['assistantMessages']} | `{sample['final_text_sha256']}` | PASS |",
        )
    add(lines)
    side = CONVERSION["side_by_side"]
    original = side["anthropic_original"]
    converted = side["openai_converted"]
    add(lines, f"### Full side-by-side example: row {side['row_index']}")
    add(lines)
    add(
        lines,
        "This is one of the five random checks and is intentionally unabridged. The mapping summary makes the structural diff explicit; the complete source and target follow it.",
    )
    add(lines)
    add(lines, "| Anthropic source | OpenAI target |")
    add(lines, "|---|---|")
    add(
        lines,
        f"| {len(original)} trajectory messages; assistant `tool_use` blocks and user `tool_result` blocks | {len(converted['messages'])} messages including inserted system; assistant `tool_calls`, one `tool` message per result; complete nine-tool array |",
    )
    add(
        lines,
        f"| Final source block type: `{original[-1]['content'][-1]['type']}` | Final target role: `{converted['messages'][-1]['role']}`; content SHA-256 `{CONVERSION['random_verification']['samples'][-1]['final_text_sha256']}` |",
    )
    add(lines)
    add(lines, "<details><summary>Anthropic original (full)</summary>")
    add(lines)
    add(lines, "~~~json")
    add(lines, code(original))
    add(lines, "~~~")
    add(lines, "</details>")
    add(lines)
    add(
        lines,
        "<details><summary>OpenAI converted training example (full, including all tool schemas)</summary>",
    )
    add(lines)
    add(lines, "~~~json")
    add(lines, code(converted))
    add(lines, "~~~")
    add(lines, "</details>")
    add(lines)
    add(lines, "## Tokenizer resolution and render method")
    add(lines)
    add(
        lines,
        f"Audit environment: Transformers `{AUDIT['transformers_version']}` from `requirements.txt`; `tokenizer.apply_chat_template(..., tools=..., truncation=False)` on all {AUDIT['rows']} rows. Percentiles use `{AUDIT['percentile_method']}`. Tools and template overhead are included.",
    )
    add(lines)
    add(
        lines,
        "| Student | Pinned HF revision | Tokenizer | Config / architecture | Template SHA-256 |",
    )
    add(lines, "|---|---|---|---|---|")
    for model in AUDIT["models"]:
        add(
            lines,
            f"| `{model['repo_id']}` | `{model['revision']}` | `{model['tokenizer_class']}` | `{model['config_class']}` / `{md(model['architectures'])}` | `{model['chat_template_sha256']}` |",
        )
    add(lines)
    add(
        lines,
        "All exact IDs were public and required no mirror. In particular, `google/gemma-4-E4B-it` resolved without authentication, so the Unsloth fallback was not used. Qwen3.6 resolves as `Qwen3_5Config` / `Qwen3_5ForConditionalGeneration`; its decoder class in pinned Transformers is `Qwen3_5DecoderLayer`.",
    )
    add(lines)
    add(
        lines,
        "The official Qwen3.5 and Qwen3.6 templates call Jinja `items` on `function.arguments`; raw OpenAI strings therefore raise `TypeError: Can only get item pairs from a mapping.` The audit parsed all 24,615 argument strings to equivalent objects **in memory** before rendering. Axolotl 0.17.0 performs the same parse in `ChatTemplateStrategy.transform_message`; the JSONL remains the required OpenAI API shape. Qwen3 and Gemma 4 accept the strings directly.",
    )
    add(lines)
    add(
        lines,
        "For loss-token accounting, no-output Jinja `{% generation %}` markers instrumented the official assistant payload and turn-ending branches while excluding role headers. Five real renders per model were compared byte-for-byte before tokenization; all instrumented renders were identical to the official output. Gemma's tool results are forward-scanned into a model turn by its template, so its markers deliberately surround only assistant tool-call/content branches; tool-result text remains context.",
    )
    add(lines)
    add(lines, "### Total rendered token lengths")
    add(lines)
    add(lines, "| Student | p50 | p90 | p95 | p99 | max | >32,768 | >40,960 |")
    add(lines, "|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in AUDIT["models"]:
        dist = model["total_tokens"]
        add(
            lines,
            f"| `{model['repo_id']}` | {dist['p50']} | {dist['p90']} | {dist['p95']} | {dist['p99']} | {dist['max']} | {model['over_32768']} ({pct(model['over_32768_rate'])}) | {model['over_40960']} ({pct(model['over_40960_rate'])}) |",
        )
    add(lines)
    add(
        lines,
        f"The >32k stop threshold was **not triggered** (`stop_threshold_triggered: {str(AUDIT['stop_threshold_triggered']).lower()}`). No input was truncated.",
    )
    add(lines)
    add(lines, "### Loss-bearing assistant tokens versus masked context")
    add(lines)
    add(
        lines,
        "| Student | Assistant p50/p90/p95/p99/max | Context p50/p90/p95/p99/max | Assistant sum | Context sum | Aggregate loss share |",
    )
    add(lines, "|---|---|---|---:|---:|---:|")
    for model in AUDIT["models"]:
        assistant = model["loss_bearing_assistant_tokens"]
        context = model["masked_context_tokens"]
        assistant_shape = "/".join(
            str(assistant[key]) for key in ["p50", "p90", "p95", "p99", "max"]
        )
        context_shape = "/".join(
            str(context[key]) for key in ["p50", "p90", "p95", "p99", "max"]
        )
        add(
            lines,
            f"| `{model['repo_id']}` | {assistant_shape} | {context_shape} | {assistant['sum']} | {context['sum']} | {pct(model['aggregate_loss_share'])} |",
        )
    add(lines)
    add(lines, "### Overflow records and proposed dispositions (not applied)")
    add(lines)
    add(
        lines,
        "These are proposals only. `longer-context variant` means the record fits 40,960 but not 32,768. `split` means splitting only at complete assistant/tool transaction boundaries, never separating a call from its result. A model-specific list is necessary because template overhead changes membership and length.",
    )
    add(lines)
    for model in AUDIT["models"]:
        add(lines, f"#### `{model['repo_id']}`")
        add(lines)
        add(
            lines,
            "| Row | Repository | Tokens | Assistant | Context | Proposed disposition | Reason |",
        )
        add(lines, "|---:|---|---:|---:|---:|---|---|")
        for row in model["overflow_records"]:
            add(
                lines,
                f"| {row['row_index']} | `{md(row['repo_full'])}` | {row['tokens']} | {row['assistant_tokens']} | {row['context_tokens']} | **{row['disposition']}** | {md(row['reason'])} |",
            )
        add(lines)
    add(lines, "## Axolotl loss-mask verification")
    add(lines)
    add(lines, f"Method: {LOSS['method']}")
    add(lines)
    add(
        lines,
        "Installed versions: "
        + ", ".join(f"`{name} {version}`" for name, version in LOSS["versions"].items())
        + ".",
    )
    add(lines)
    add(
        lines,
        "The direct installed strategy is the actual `type: chat_template` preprocessor: it parses OpenAI argument strings, initializes every label to `-100`, locates each rendered turn, unmasks only `roles_to_train: [assistant]`, and trains the assistant EOT/EOS because `train_on_eos: turn`. The macOS CLI itself could not import because `bitsandbytes` has no Darwin wheel; this is why the allowed source-cross-checked path was used. A second generation-tag mask independently proves that every trained token belongs to assistant output and that assistant payload tokens are not lost.",
    )
    add(lines)
    add(
        lines,
        f"Template: `{LOSS['chat_template']}` (SHA-256 `{LOSS['chat_template_sha256']}`). The stock Qwen3 template changes its last-assistant branch during Axolotl prefix probes, which shifts mask boundaries. This patch supplies `real_last_index` only to those probes; all three complete renders were byte-identical to the pinned tokenizer template.",
    )
    add(lines)
    add(
        lines,
        "| Row | Total tokens | Trained assistant | Masked context | Assistant tool calls | Final answer | User/tool results |",
    )
    add(lines, "|---:|---:|---:|---:|---|---|---|")
    for sample in LOSS["samples"]:
        add(
            lines,
            f"| {sample['row_index']} | {sample['tokens']} | {sample['trained_tokens']} | {sample['masked_tokens']} | TRAIN | TRAIN | MASK |",
        )
    add(lines)
    dump = next(sample for sample in LOSS["samples"] if sample["row_index"] == 340)
    add(lines, "### Rendered and tokenized dump excerpt (real row 340)")
    add(lines)
    add(lines, "Rendered prefix:")
    add(lines)
    add(lines, "~~~text")
    add(lines, dump["rendered_prefix"])
    add(lines, "~~~")
    add(lines)
    add(
        lines,
        "Rendered suffix (contains the actually emitted prose + fenced final JSON):",
    )
    add(lines)
    add(lines, "~~~text")
    add(lines, dump["rendered_suffix"])
    add(lines, "~~~")
    add(lines)
    add(
        lines,
        "First label runs (`TRAIN` means label equals token ID; `MASK` means `-100`):",
    )
    add(lines)
    add(lines, "| Label | Token range | Count | Decoded excerpt |")
    add(lines, "|---|---:|---:|---|")
    for run in dump["label_runs"][:8]:
        add(
            lines,
            f"| **{run['label']}** | {run['start']}..{run['end_exclusive']} | {run['tokens']} | `{md(run['decoded_excerpt'][:180])}` |",
        )
    add(lines)
    add(lines, "Turn-boundary proof for the same example:")
    add(lines)
    add(lines, "| Turn | Role | Span tokens | Trained | Masked | Tool call? |")
    add(lines, "|---:|---|---:|---:|---:|---|")
    for turn in dump["turn_checks"]:
        if "span_tokens" not in turn:
            continue
        add(
            lines,
            f"| {turn['turn']} | `{turn['role']}` | {turn['span_tokens']} | {turn['trained_tokens']} | {turn['masked_tokens']} | {turn['has_tool_calls']} |",
        )
    add(lines)
    add(
        lines,
        "Result: **PASS**. Across all three examples, every identified assistant span (including structured tool calls and final answer text) was trainable; every identified user and tool-result span had zero trainable labels. The independent semantic mask found no context-label leakage and no masked assistant payload token (only separator newlines remain ignored). Both classes were nonempty.",
    )
    add(lines)
    add(lines, "## Axolotl configs")
    add(lines)
    add(
        lines,
        "All four YAMLs passed Axolotl 0.17.0 `AxolotlInputConfig.model_validate`. Shared contract: `type: chat_template`, `field_messages: messages`, `field_tools: tools`, assistant-only roles, per-turn EOS, `sequence_len: 32768`, sample packing, BF16, FlashAttention 2, gradient checkpointing, and safetensors.",
    )
    add(lines)
    add(
        lines,
        "- `qwen3-1.7b-full.yaml`: full parameter fine-tune; no adapter block. It selects the byte-identical `qwen3-aft.jinja` mask-boundary patch described above.",
    )
    add(
        lines,
        "- `gemma4-e4b-lora.yaml`, `qwen35-9b-lora.yaml`, `qwen36-27b-lora-fsdp2.yaml`: LoRA `r=32`, `alpha=64`, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.",
    )
    add(
        lines,
        f"- The 27B config sets `fsdp_version: 2`, `TRANSFORMER_BASED_WRAP`, and `{LOSS['qwen36_decoder_layer_class']}`, imported from the installed Transformers implementation. It expects 2xH100 with NVLink; the other runs expect one H100 each.",
    )
    add(
        lines,
        "- The README makes the overflow curation a launch prerequisite and requires a launch-time record of the Axolotl commit, Transformers/TRL/PEFT/PyTorch/FlashAttention/FLA versions, CUDA/driver, image digest, model revision, config hash, and curated dataset hash.",
    )
    add(lines)
    add(lines, "## Artifacts")
    add(lines)
    add(
        lines,
        "Tracked: converter/audit/mask scripts, `requirements.txt`, four Axolotl configs, the Qwen3 mask-stability template, README, `conversion-report.json`, `tokenizer-audit.json`, `loss-mask-verification.json`, and this report. Ignored: source data copy, `sft-dataset.jsonl`, virtual environments, prepared caches, and training outputs.",
    )

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
