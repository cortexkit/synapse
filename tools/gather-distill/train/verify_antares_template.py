#!/usr/bin/env python3
"""Verify the Granite chat template against real curated gather rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer


MODELS = {
    "antares-1b": (
        "fdtn-ai/antares-1b",
        "10417eb35641b32e7141157db19c76eb545193b6",
    ),
    "antares-350m": (
        "fdtn-ai/antares-350m",
        "cdf6d054fa5f491553ccb1704269cbd1954c6c6e",
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=sorted(MODELS), dest="models")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    rows = []
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("dataset is empty")
    template = args.template.read_text(encoding="utf-8")
    selected = args.models or list(MODELS)
    report = {
        "dataset_rows": len(rows),
        "dataset_sha256": sha256_bytes(args.dataset.read_bytes()),
        "template_sha256": sha256_bytes(template.encode()),
        "thinking_knob": "enable_thinking=false",
        "models": {},
    }

    for label in selected:
        model_id, revision = MODELS[label]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            token=args.token,
            trust_remote_code=False,
        )
        tokenizer.chat_template = template
        lengths = []
        assistant_marker = "<|start_of_role|>assistant<|end_of_role|>"
        eos_marker = "<|end_of_text|>\n"
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                row["messages"],
                tools=row.get("tools", []),
                tokenize=False,
                add_generation_prompt=False,
            )
            generation = tokenizer.apply_chat_template(
                row["messages"],
                tools=row.get("tools", []),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if assistant_marker not in rendered:
                raise SystemExit(f"{label}: assistant turn did not render")
            if eos_marker not in rendered:
                raise SystemExit(f"{label}: rendered turns are not EOS-delimited")
            if not generation.endswith("<think>\n\n</think>\n\n"):
                raise SystemExit(f"{label}: disabled-thinking generation suffix mismatch")
            encoded = tokenizer(rendered, add_special_tokens=False)
            lengths.append(len(encoded["input_ids"]))
        report["models"][label] = {
            "model_id": model_id,
            "revision": revision,
            "vocab_size": tokenizer.vocab_size,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "rows": len(lengths),
            "token_p50": int(median(lengths)),
            "token_max": max(lengths),
            "over_32768": sum(length > 32768 for length in lengths),
            "assistant_marker": assistant_marker,
            "eos_marker": eos_marker,
            "generation_suffix": "<think>\\n\\n</think>\\n\\n",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
