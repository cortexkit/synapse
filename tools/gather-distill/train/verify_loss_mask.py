#!/usr/bin/env python3
"""Exercise Axolotl's chat-template strategy on three real Qwen3 examples."""

from __future__ import annotations

import importlib
import importlib.metadata
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from audit_tokenizers import instrument_qwen

chat_template_strategy = importlib.import_module(
    "axolotl.prompt_strategies.chat_template"
)
ChatTemplatePrompter = chat_template_strategy.ChatTemplatePrompter
ChatTemplateStrategy = chat_template_strategy.ChatTemplateStrategy

TRAIN_DIR = Path(__file__).resolve().parent
DATASET = TRAIN_DIR / "sft-dataset.jsonl"
OUTPUT = TRAIN_DIR / "loss-mask-verification.json"
CHAT_TEMPLATE_PATH = TRAIN_DIR / "axolotl" / "templates" / "qwen3-aft.jinja"
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SAMPLE_INDICES = [155, 340, 677]
IGNORE_TOKEN_ID = -100


def load_samples() -> dict[int, dict[str, Any]]:
    wanted = set(SAMPLE_INDICES)
    samples: dict[int, dict[str, Any]] = {}
    with DATASET.open() as dataset:
        for index, line in enumerate(dataset):
            if index in wanted:
                samples[index] = json.loads(line)
    if set(samples) != wanted:
        raise ValueError(f"missing requested samples: {sorted(wanted - set(samples))}")
    return samples


def label_runs(
    tokenizer, input_ids: list[int], labels: list[int]
) -> list[dict[str, Any]]:
    runs = []
    start = 0
    current = labels[0] != IGNORE_TOKEN_ID
    for index in range(1, len(labels) + 1):
        next_value = (
            labels[index] != IGNORE_TOKEN_ID if index < len(labels) else not current
        )
        if next_value == current:
            continue
        text = tokenizer.decode(input_ids[start:index], skip_special_tokens=False)
        runs.append(
            {
                "start": start,
                "end_exclusive": index,
                "tokens": index - start,
                "label": "TRAIN" if current else "MASK",
                "decoded_excerpt": text[:500],
                "excerpt_truncated": len(text) > 500,
            }
        )
        start = index
        current = next_value
    return runs


def token_window(
    tokenizer, input_ids: list[int], labels: list[int], start: int, end: int
) -> dict[str, Any]:
    def tokens(indices: range) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "token_id": input_ids[index],
                "token": tokenizer.convert_ids_to_tokens(input_ids[index]),
                "label": labels[index],
                "trained": labels[index] != IGNORE_TOKEN_ID,
            }
            for index in indices
        ]

    start_indices = range(max(0, start - 6), min(len(input_ids), start + 12))
    end_indices = range(max(start + 12, end - 12), min(len(input_ids), end + 6))
    return {
        "span_start": start,
        "span_end_exclusive": end,
        "start_boundary": tokens(start_indices),
        "end_boundary": tokens(end_indices),
        "omitted_interior_tokens": max(0, end - start - 24),
    }


def verify_example(
    strategy,
    tokenizer,
    chat_template: str,
    example: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    tokenized = strategy._tokenize_single_prompt(example)
    input_ids = list(tokenized["input_ids"])
    labels = list(tokenized["labels"])
    if len(input_ids) != len(labels):
        raise ValueError(f"row {row_index}: input/label length mismatch")

    turns = strategy.get_conversation_thread(example)
    tools = strategy._get_tools(example)
    official_render = tokenizer.apply_chat_template(
        turns, tools=tools, chat_template=tokenizer.chat_template, tokenize=False
    )
    patched_render = tokenizer.apply_chat_template(
        turns, tools=tools, chat_template=chat_template, tokenize=False
    )
    if official_render != patched_render:
        raise ValueError(
            f"row {row_index}: mask-stability template changed full rendering"
        )

    semantic = tokenizer.apply_chat_template(
        turns,
        tools=tools,
        chat_template=instrument_qwen(chat_template),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=False,
    )
    if list(semantic["input_ids"]) != input_ids:
        raise ValueError(f"row {row_index}: semantic-mask rendering changed token IDs")
    semantic_mask = [bool(value) for value in semantic["assistant_masks"]]
    actual_mask = [label != IGNORE_TOKEN_ID for label in labels]
    leaked_context = [
        index
        for index, (actual, expected) in enumerate(zip(actual_mask, semantic_mask))
        if actual and not expected
    ]
    if leaked_context:
        raise ValueError(
            f"row {row_index}: Axolotl trained {len(leaked_context)} context tokens"
        )
    masked_assistant_payload = [
        index
        for index, (actual, expected) in enumerate(zip(actual_mask, semantic_mask))
        if expected
        and not actual
        and tokenizer.decode([input_ids[index]], skip_special_tokens=False) != "\n"
    ]
    if masked_assistant_payload:
        raise ValueError(
            f"row {row_index}: Axolotl masked {len(masked_assistant_payload)} assistant payload tokens"
        )

    turn_checks = []
    first_assistant_bounds: tuple[int, int] | None = None
    first_tool_bounds: tuple[int, int] | None = None
    final_assistant_bounds: tuple[int, int] | None = None

    for index, turn in enumerate(turns):
        start, end = strategy.find_turn(turns, index, tools=tools)
        if start == -1 or end == -1:
            turn_checks.append(
                {
                    "turn": index,
                    "role": turn.get("role"),
                    "bounds": None,
                    "note": "Axolotl did not identify a content span (expected for some empty/control turns).",
                }
            )
            continue
        trained = sum(label != IGNORE_TOKEN_ID for label in labels[start:end])
        role = turn.get("role")
        expected_trained = role == "assistant"
        if expected_trained and trained != end - start:
            raise ValueError(
                f"row {row_index} assistant turn {index} is not fully trainable"
            )
        if not expected_trained and trained != 0:
            raise ValueError(
                f"row {row_index} {role} turn {index} leaks {trained} trainable tokens"
            )
        turn_checks.append(
            {
                "turn": index,
                "role": role,
                "start": start,
                "end_exclusive": end,
                "span_tokens": end - start,
                "trained_tokens": trained,
                "masked_tokens": end - start - trained,
                "content_excerpt": str(turn.get("content", ""))[:180],
                "has_tool_calls": bool(turn.get("tool_calls")),
            }
        )
        if role == "assistant":
            first_assistant_bounds = first_assistant_bounds or (start, end)
            final_assistant_bounds = (start, end)
        elif role == "tool":
            first_tool_bounds = first_tool_bounds or (start, end)

    if (
        first_assistant_bounds is None
        or first_tool_bounds is None
        or final_assistant_bounds is None
    ):
        raise ValueError(
            f"row {row_index} lacks assistant/tool spans needed for non-vacuous verification"
        )
    if not any(
        turn.get("tool_calls") for turn in turns if turn.get("role") == "assistant"
    ):
        raise ValueError(f"row {row_index} has no assistant tool call")

    assistant_tokens = sum(label != IGNORE_TOKEN_ID for label in labels)
    context_tokens = len(labels) - assistant_tokens
    if assistant_tokens == 0 or context_tokens == 0:
        raise ValueError(f"row {row_index} produced a vacuous label mask")

    rendered = tokenizer.decode(input_ids, skip_special_tokens=False)
    return {
        "row_index": row_index,
        "tokens": len(input_ids),
        "trained_tokens": assistant_tokens,
        "masked_tokens": context_tokens,
        "assistant_tool_calls_present": True,
        "final_assistant_trained": True,
        "user_and_tool_turns_masked": True,
        "full_render_byte_equal_to_tokenizer_template": True,
        "semantic_assistant_tokens": sum(semantic_mask),
        "masked_separator_newlines": sum(
            expected and not actual
            for actual, expected in zip(actual_mask, semantic_mask)
        ),
        "turn_checks": turn_checks,
        "rendered_prefix": rendered[:500],
        "rendered_suffix": rendered[-500:],
        "label_runs": label_runs(tokenizer, input_ids, labels),
        "token_windows": {
            "first_assistant_tool_call": token_window(
                tokenizer, input_ids, labels, *first_assistant_bounds
            ),
            "first_tool_result": token_window(
                tokenizer, input_ids, labels, *first_tool_bounds
            ),
            "final_assistant_answer": token_window(
                tokenizer, input_ids, labels, *final_assistant_bounds
            ),
        },
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    chat_template = CHAT_TEMPLATE_PATH.read_text()
    prompter = ChatTemplatePrompter(
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=32_769,
        field_messages="messages",
        field_tools="tools",
    )
    strategy = ChatTemplateStrategy(
        prompter,
        tokenizer=tokenizer,
        train_on_inputs=False,
        sequence_len=32_768,
        roles_to_train=["assistant"],
        train_on_eos="turn",
    )
    decoder_layer_class = getattr(
        importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5"),
        "Qwen3_5DecoderLayer",
    ).__name__
    samples = load_samples()
    results = [
        verify_example(strategy, tokenizer, chat_template, samples[index], index)
        for index in SAMPLE_INDICES
    ]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "Direct invocation of the installed Axolotl 0.17.0 ChatTemplateStrategy used by type: chat_template datasets. The CLI import was unavailable on macOS because bitsandbytes has no Darwin package; no training was attempted.",
        "cross_check": "Axolotl transform_message parses OpenAI function.arguments JSON strings before apply_chat_template. Its _tokenize_single_prompt initializes every label to -100, locates each turn with find_turn, unmasks only roles_to_train, and trains the assistant turn EOS/EOT when train_on_eos is turn. A generation-tag semantic mask independently rejects context-label leakage and masked assistant payload tokens.",
        "chat_template": str(CHAT_TEMPLATE_PATH.relative_to(TRAIN_DIR.parent)),
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "template_patch": "The official Qwen3 template with a real_last_index override used only during Axolotl prefix probes. Full-conversation rendering is byte-identical to the pinned tokenizer template.",
        "model": MODEL,
        "revision": REVISION,
        "qwen36_decoder_layer_class": decoder_layer_class,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["axolotl", "transformers", "trl", "peft", "torch"]
        },
        "config_contract": {
            "dataset_type": "chat_template",
            "field_messages": "messages",
            "field_tools": "tools",
            "roles_to_train": ["assistant"],
            "train_on_eos": "turn",
            "sequence_len": 32_768,
            "chat_template": "jinja",
            "chat_template_jinja": "train/axolotl/templates/qwen3-aft.jinja",
        },
        "samples": results,
        "passed": True,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "samples": [
                    {
                        "row_index": result["row_index"],
                        "tokens": result["tokens"],
                        "trained": result["trained_tokens"],
                        "masked": result["masked_tokens"],
                    }
                    for result in results
                ],
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
