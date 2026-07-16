#!/usr/bin/env python3
"""Render the complete SFT dataset with pinned student tokenizers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoConfig, AutoTokenizer

TRAIN_DIR = Path(__file__).resolve().parent
DATASET = TRAIN_DIR / "sft-dataset.jsonl"
SOURCE_DATASET = TRAIN_DIR.parent / "data" / "dataset-v1.jsonl"
CONVERSION_REPORT = TRAIN_DIR / "conversion-report.json"
OUTPUT = TRAIN_DIR / "tokenizer-audit.json"

MODELS = [
    {
        "label": "Qwen3 1.7B",
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "argument_mode": "openai_string",
        "instrumentation": "qwen",
    },
    {
        "label": "Gemma 4 E4B IT",
        "repo_id": "google/gemma-4-E4B-it",
        "revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
        "argument_mode": "openai_string",
        "instrumentation": "gemma4",
    },
    {
        "label": "Qwen3.5 4B",
        "repo_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "argument_mode": "hf_object",
        "instrumentation": "qwen",
    },
    {
        "label": "Qwen3.5 9B",
        "repo_id": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "argument_mode": "hf_object",
        "instrumentation": "qwen",
    },
    {
        "label": "Qwen3.6 27B",
        "repo_id": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "argument_mode": "hf_object",
        "instrumentation": "qwen",
    },
]

QWEN3_CONTENT_EXPRESSIONS = {
    "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}": (
        "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' }}"
        "{%- generation -%}{{- content.lstrip('\\n') }}{%- endgeneration -%}"
    ),
    "                {{- '<|im_start|>' + message.role + '\\n' + content }}": (
        "                {{- '<|im_start|>' + message.role + '\\n' }}"
        "{%- generation -%}{{- content }}{%- endgeneration -%}"
    ),
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}": (
        "            {{- '<|im_start|>' + message.role + '\\n' }}"
        "{%- generation -%}{{- content }}{%- endgeneration -%}"
    ),
}
QWEN35_CONTENT_EXPRESSIONS = {
    "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}": (
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}"
        "{%- generation -%}{{- content }}{%- endgeneration -%}"
    ),
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}": (
        "            {{- '<|im_start|>' + message.role + '\\n' }}"
        "{%- generation -%}{{- content }}{%- endgeneration -%}"
    ),
}
QWEN_EOT = "        {{- '<|im_end|>\\n' }}"

GEMMA_TOOL_CALL_BLOCK = """            {%- if message['tool_calls'] -%}
                {%- for tool_call in message['tool_calls'] -%}
                    {%- set function = tool_call['function'] -%}
                    {{- '<|tool_call>call:' + function['name'] + '{' -}}
                    {%- if function['arguments'] is mapping -%}
                        {%- set ns_args = namespace(found_first=false) -%}
                        {%- for key, value in function['arguments'] | dictsort -%}
                            {%- if ns_args.found_first %},{% endif -%}
                            {%- set ns_args.found_first = true -%}
                            {{- key -}}:{{- format_argument(value, escape_keys=False) -}}
                        {%- endfor -%}
                    {%- elif function['arguments'] is string -%}
                        {{- function['arguments'] -}}
                    {%- endif -%}
                    {{- '}<tool_call|>' -}}
                {%- endfor -%}
                {%- set ns.prev_message_type = 'tool_call' -%}
            {%- endif -%}"""

GEMMA_OUTPUT_BLOCK = """            {{- captured_content -}}
            {%- set has_content = captured_content | trim | length > 0 -%}

        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}
            {{- '<|tool_response>' -}}
        {%- elif not (ns_tr_out.flag and not has_content) -%}
            {{- '<turn|>\\n' -}}
        {%- endif -%}"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def instrument_qwen(template: str) -> str:
    expressions = (
        QWEN35_CONTENT_EXPRESSIONS
        if "<function=example_function_name>" in template
        else QWEN3_CONTENT_EXPRESSIONS
    )
    instrumented = template
    for original, marked in expressions.items():
        if instrumented.count(original) != 1:
            raise ValueError(
                "Qwen chat template no longer has an audited content expression"
            )
        instrumented = instrumented.replace(original, marked)

    tool_start = next(
        (
            marker
            for marker in [
                "        {%- if message.tool_calls %}",
                "        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}",
            ]
            if marker in instrumented
        ),
        None,
    )
    if tool_start is None:
        raise ValueError(
            "Qwen chat template no longer has the audited tool-call branch"
        )
    tool_start_index = instrumented.index(tool_start)
    eot_index = instrumented.index(QWEN_EOT, tool_start_index)
    tool_block = instrumented[tool_start_index:eot_index]
    outer_end = "        {%- endif %}"
    outer_end_index = tool_block.rfind(outer_end)
    if outer_end_index == -1:
        raise ValueError("Qwen tool-call branch has no closing block")
    marked_tool_block = (
        tool_block[: len(tool_start)]
        + "\n            {%- generation -%}"
        + tool_block[len(tool_start) : outer_end_index]
        + "            {%- endgeneration -%}\n"
        + tool_block[outer_end_index:]
    )
    instrumented = (
        instrumented[:tool_start_index] + marked_tool_block + instrumented[eot_index:]
    )
    eot_index = instrumented.index(QWEN_EOT, tool_start_index)
    marked_eot = (
        "        {%- generation -%}" + QWEN_EOT.strip() + "{%- endgeneration -%}"
    )
    return (
        instrumented[:eot_index]
        + marked_eot
        + instrumented[eot_index + len(QWEN_EOT) :]
    )


def instrument_gemma4(template: str) -> str:
    if (
        template.count(GEMMA_TOOL_CALL_BLOCK) != 1
        or template.count(GEMMA_OUTPUT_BLOCK) != 1
    ):
        raise ValueError(
            "Gemma 4 chat template no longer has the audited tool/content branches"
        )
    marked_calls = GEMMA_TOOL_CALL_BLOCK.replace(
        "            {%- if message['tool_calls'] -%}",
        "            {%- if message['tool_calls'] -%}\n                {%- generation -%}",
    ).replace(
        "                {%- set ns.prev_message_type = 'tool_call' -%}\n            {%- endif -%}",
        "                {%- set ns.prev_message_type = 'tool_call' -%}\n"
        "                {%- endgeneration -%}\n"
        "            {%- endif -%}",
    )
    marked_output = (
        "            {%- if role == 'model' -%}\n"
        "                {%- generation -%}\n"
        f"{GEMMA_OUTPUT_BLOCK}\n"
        "                {%- endgeneration -%}\n"
        "            {%- else -%}\n"
        f"{GEMMA_OUTPUT_BLOCK}\n"
        "            {%- endif -%}"
    )
    return template.replace(GEMMA_TOOL_CALL_BLOCK, marked_calls).replace(
        GEMMA_OUTPUT_BLOCK, marked_output
    )


def normalize_messages(
    messages: list[dict[str, Any]], argument_mode: str
) -> tuple[list[dict[str, Any]], int]:
    normalized = copy.deepcopy(messages)
    parsed = 0
    if argument_mode == "openai_string":
        return normalized, parsed
    for message in normalized:
        for tool_call in message.get("tool_calls", []):
            arguments = tool_call["function"]["arguments"]
            if not isinstance(arguments, str):
                raise TypeError(
                    "OpenAI function.arguments must be a JSON string before HF normalization"
                )
            value = json.loads(arguments)
            if not isinstance(value, dict):
                raise TypeError(
                    "HF tool templates require function.arguments to decode to an object"
                )
            tool_call["function"]["arguments"] = value
            parsed += 1
    return normalized, parsed


def encoded_values(encoded: Any, key: str) -> list[int]:
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        values = encoded[key]
    else:
        raise TypeError(
            f"chat template returned unsupported token container {type(encoded).__name__}"
        )
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("unexpected batched chat template output")
        values = values[0]
    return list(values)


def percentile(values: list[int], probability: float) -> int:
    """Nearest-rank percentile, with p50 selecting the upper middle value."""
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def distribution(values: list[int]) -> dict[str, int]:
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "sum": sum(values),
    }


def proposed_disposition(length: int, assistant_messages: int) -> dict[str, str]:
    if length <= 40_960:
        return {
            "disposition": "longer-context variant",
            "reason": "Fits 40,960 tokens without truncation but not the 32,768-token ladder configs.",
        }
    if assistant_messages >= 4:
        return {
            "disposition": "split",
            "reason": "Split only at complete assistant-tool transaction boundaries; never cut a tool call from its results.",
        }
    return {
        "disposition": "drop",
        "reason": "Too long for 40,960 tokens and too few complete assistant turns for a useful transaction-boundary split.",
    }


def line_pairs(
    dataset: Path, source_dataset: Path | None
) -> Iterable[tuple[int, dict[str, Any], dict[str, Any]]]:
    if source_dataset is None:
        with dataset.open() as converted_file:
            for index, converted_line in enumerate(converted_file):
                yield index, json.loads(converted_line), {}
        return

    with dataset.open() as converted_file, source_dataset.open() as source_file:
        converted_lines = iter(converted_file)
        source_lines = iter(source_file)
        index = 0
        while True:
            converted_line = next(converted_lines, None)
            source_line = next(source_lines, None)
            if converted_line is None and source_line is None:
                return
            if converted_line is None or source_line is None:
                raise ValueError(
                    "converted and source datasets have different row counts"
                )
            yield index, json.loads(converted_line), json.loads(source_line)
            index += 1


def audit_model(
    model: dict[str, str],
    verification_indices: set[int],
    dataset: Path,
    source_dataset: Path | None,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model["repo_id"], revision=model["revision"]
    )
    config = AutoConfig.from_pretrained(model["repo_id"], revision=model["revision"])
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError(f"{model['repo_id']} has no chat template")
    instrumented = (
        instrument_qwen(template)
        if model["instrumentation"] == "qwen"
        else instrument_gemma4(template)
    )

    total_lengths: list[int] = []
    assistant_lengths: list[int] = []
    context_lengths: list[int] = []
    overflows: list[dict[str, Any]] = []
    render_checks: list[dict[str, Any]] = []
    normalized_argument_count = 0
    raw_argument_probe: dict[str, Any] | None = None

    for index, example, source in line_pairs(dataset, source_dataset):
        messages, parsed = normalize_messages(
            example["messages"], model["argument_mode"]
        )
        normalized_argument_count += parsed
        if index == 0:
            try:
                tokenizer.apply_chat_template(
                    example["messages"],
                    tools=example["tools"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                raw_argument_probe = {"supported": True, "error": None}
            except Exception as error:  # noqa: BLE001 - the report must capture tokenizer/Jinja errors verbatim.
                raw_argument_probe = {
                    "supported": False,
                    "error": f"{type(error).__name__}: {error}",
                }

        encoded = tokenizer.apply_chat_template(
            messages,
            tools=example["tools"],
            chat_template=instrumented,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            truncation=False,
        )
        input_ids = encoded_values(encoded, "input_ids")
        assistant_mask = encoded_values(encoded, "assistant_masks")
        if len(input_ids) != len(assistant_mask):
            raise ValueError(f"row {index} token/mask lengths differ")
        total = len(input_ids)
        assistant = sum(assistant_mask)
        context = total - assistant
        if assistant <= 0 or context <= 0:
            raise ValueError(f"row {index} has a vacuous assistant/context mask")
        total_lengths.append(total)
        assistant_lengths.append(assistant)
        context_lengths.append(context)

        if index in verification_indices:
            clean_render = tokenizer.apply_chat_template(
                messages,
                tools=example["tools"],
                chat_template=template,
                tokenize=False,
                add_generation_prompt=False,
            )
            instrumented_render = tokenizer.apply_chat_template(
                messages,
                tools=example["tools"],
                chat_template=instrumented,
                tokenize=False,
                add_generation_prompt=False,
            )
            if clean_render != instrumented_render:
                raise ValueError(
                    f"row {index} mask instrumentation changed rendered bytes"
                )
            render_checks.append(
                {
                    "row_index": index,
                    "render_sha256": sha256_text(clean_render),
                    "characters": len(clean_render),
                    "tokens": total,
                    "assistant_tokens": assistant,
                    "context_tokens": context,
                    "render_prefix": clean_render[:240],
                    "render_suffix": clean_render[-240:],
                    "instrumented_render_byte_identical": True,
                }
            )

        if total > 32_768:
            assistant_messages = sum(
                message["role"] == "assistant" for message in messages
            )
            overflow = {
                "row_index": index,
                "repo_full": source.get("repo_full"),
                "request": source.get("request"),
                "tokens": total,
                "assistant_tokens": assistant,
                "context_tokens": context,
                "assistant_messages": assistant_messages,
            }
            overflow.update(proposed_disposition(total, assistant_messages))
            overflows.append(overflow)

    if raw_argument_probe is None:
        raise ValueError("dataset is empty")
    count = len(total_lengths)
    over_40k = [row for row in overflows if row["tokens"] > 40_960]
    assistant_sum = sum(assistant_lengths)
    context_sum = sum(context_lengths)
    return {
        **model,
        "resolved": True,
        "tokenizer_class": type(tokenizer).__name__,
        "config_class": type(config).__name__,
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "chat_template_sha256": sha256_text(template),
        "rows": count,
        "raw_openai_argument_strings": raw_argument_probe,
        "hf_argument_objects_parsed": normalized_argument_count,
        "argument_adapter_note": (
            "No adapter needed; the official template accepts OpenAI JSON-string arguments."
            if model["argument_mode"] == "openai_string"
            else "The official template calls Jinja |items on function.arguments. Each OpenAI JSON string was parsed to an equivalent object in memory before rendering; the committed JSONL remains OpenAI-shaped."
        ),
        "total_tokens": distribution(total_lengths),
        "loss_bearing_assistant_tokens": distribution(assistant_lengths),
        "masked_context_tokens": distribution(context_lengths),
        "aggregate_loss_share": assistant_sum / (assistant_sum + context_sum),
        "over_32768": len(overflows),
        "over_32768_rate": len(overflows) / count,
        "over_40960": len(over_40k),
        "over_40960_rate": len(over_40k) / count,
        "overflow_records": overflows,
        "render_checks": sorted(render_checks, key=lambda check: check["row_index"]),
        "mask_method": (
            "No-output Jinja generation tags instrument the official template. The script proves byte-identical rendered text on the five conversion samples, then uses Transformers assistant_masks. Assistant payload and turn-ending branches are marked while role headers remain context; Gemma marks assistant tool-call/content branches separately so forward-scanned tool results remain context."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        dest="model_selectors",
        help="Audit only this model label or Hugging Face repo ID; repeat for multiple models.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        help="Optional source rows used to annotate overflow records.",
    )
    parser.add_argument("--conversion-report", type=Path, default=CONVERSION_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--verification-indices",
        type=int,
        nargs="+",
        help="Zero-based rows whose instrumented render must match the official template.",
    )
    args = parser.parse_args()

    if args.model_selectors:
        selected = [
            model
            for model in MODELS
            if model["label"] in args.model_selectors
            or model["repo_id"] in args.model_selectors
        ]
        matched = {model["label"] for model in selected} | {
            model["repo_id"] for model in selected
        }
        unknown = set(args.model_selectors) - matched
        if unknown:
            parser.error(f"unknown model selector(s): {', '.join(sorted(unknown))}")
    else:
        selected = MODELS

    dataset = args.dataset.resolve()
    source_dataset = args.source_dataset
    if source_dataset is None and dataset == DATASET.resolve():
        source_dataset = SOURCE_DATASET
    if source_dataset is not None:
        source_dataset = source_dataset.resolve()

    conversion: dict[str, Any] | None = None
    if args.conversion_report.exists():
        conversion = json.loads(args.conversion_report.read_text())
    if args.verification_indices is not None:
        verification_indices = set(args.verification_indices)
    elif conversion is not None and dataset == DATASET.resolve():
        verification_indices = {
            sample["row_index"]
            for sample in conversion["random_verification"]["samples"]
        }
    else:
        verification_indices = {0, 155, 340, 677}

    models = []
    for model in selected:
        print(f"Auditing {model['repo_id']} at {model['revision']}...", flush=True)
        result = audit_model(model, verification_indices, dataset, source_dataset)
        models.append(result)
        print(
            f"  rows={result['rows']} p95={result['total_tokens']['p95']} "
            f"max={result['total_tokens']['max']} over32k={result['over_32768']}",
            flush=True,
        )

    stopped = [model for model in models if model["over_32768_rate"] > 0.10]
    digest = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        dataset_label = str(dataset.relative_to(TRAIN_DIR.parent))
    except ValueError:
        dataset_label = str(dataset)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_label,
        "dataset_sha256": digest.hexdigest(),
        "rows": models[0]["rows"] if models else 0,
        "transformers_version": transformers.__version__,
        "percentile_method": "nearest-rank; rank=max(1, ceil(p*N))",
        "truncation": "disabled",
        "tools_included": True,
        "models": models,
        "stop_threshold_triggered": bool(stopped),
        "stop_threshold_models": [model["repo_id"] for model in stopped],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if stopped:
        print(
            "STOP: at least one tokenizer has more than 10% of rows above 32,768 tokens; "
            f"inspect {args.output} before proposing record drops.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
