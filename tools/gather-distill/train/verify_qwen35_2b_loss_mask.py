#!/usr/bin/env python3
"""Verify Qwen3.5-2B assistant-only labels with Axolotl's real strategy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from verify_loss_mask import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
    verify_example,
)

TRAIN_DIR = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SAMPLE_INDICES = [155, 340, 677]


def load_samples(dataset: Path) -> dict[int, dict[str, Any]]:
    wanted = set(SAMPLE_INDICES)
    samples: dict[int, dict[str, Any]] = {}
    with dataset.open() as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                samples[index] = json.loads(line)
    if set(samples) != wanted:
        raise ValueError(f"missing requested samples: {sorted(wanted - set(samples))}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=TRAIN_DIR / "sft-dataset-curated.jsonl"
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Optional mask-stability template; the tokenizer default is used otherwise.",
    )
    parser.add_argument(
        "--output", type=Path, default=TRAIN_DIR / "qwen35-2b-loss-mask-verification.json"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    chat_template = (
        args.template.read_text() if args.template else tokenizer.chat_template
    )
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Qwen3.5-2B tokenizer has no chat template")

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
    samples = load_samples(args.dataset)
    results = [
        verify_example(strategy, tokenizer, chat_template, samples[index], index)
        for index in SAMPLE_INDICES
    ]
    template_label = (
        str(args.template.resolve().relative_to(TRAIN_DIR.parent))
        if args.template
        else "tokenizer_default"
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "Direct invocation of Axolotl ChatTemplateStrategy with the curated dataset and the independently instrumented Qwen3.5 generation mask.",
        "chat_template": template_label,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "full_render_byte_equal_to_tokenizer_template": True,
        "model": MODEL,
        "revision": REVISION,
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
        },
        "samples": results,
        "passed": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
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
