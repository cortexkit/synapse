#!/usr/bin/env python3
"""Create pinned CPU-fp32 token and hidden-state references for LFM2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PINNED_MODEL_REVISION = "933cee00d754fb3bfe06c644c0cb95453f2d8bb2"
PINNED_TRANSFORMERS = "5.12.1"
PINNED_TORCH = "2.12.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local snapshot or LiquidAI/LFM2-1.2B")
    parser.add_argument("--prompts", required=True, type=Path, help="JSONL {id,prompt}")
    parser.add_argument("--tokens-out", required=True, type=Path)
    parser.add_argument("--hidden-out", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k-logits", type=int, default=5)
    parser.add_argument(
        "--model-revision",
        default=PINNED_MODEL_REVISION,
        help="Hub commit to validate; defaults to the pinned LFM2-1.2B reference",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="Allow a Transformers version other than the parity-oracle pin",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("prompt set is empty")
    for row in rows:
        if not isinstance(row.get("id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError("each prompt row must contain string id and prompt fields")
    return rows


def top_logits(scores: torch.Tensor, count: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(scores.float(), k=min(count, scores.shape[-1]))
    return [
        {"token_id": int(token), "logit": float(value)}
        for token, value in zip(indices.tolist(), values.tolist(), strict=True)
    ]


def checked_revision(model_ref: str, expected_revision: str) -> str:
    path = Path(model_ref).expanduser()
    if path.exists():
        if path.name != expected_revision:
            raise RuntimeError(
                f"local model path must be snapshot {expected_revision}, "
                f"found leaf {path.name!r}"
            )
        return str(path)
    return model_ref


def main() -> None:
    args = parse_args()
    if transformers.__version__ != PINNED_TRANSFORMERS and not args.allow_version_drift:
        raise RuntimeError(
            f"transformers {PINNED_TRANSFORMERS} is required, "
            f"found {transformers.__version__}"
        )
    if torch.__version__ != PINNED_TORCH and not args.allow_version_drift:
        raise RuntimeError(f"torch {PINNED_TORCH} is required, found {torch.__version__}")
    if args.max_new_tokens <= 0 or args.top_k_logits <= 0:
        raise ValueError("generation and top-k sizes must be positive")

    model_ref = checked_revision(args.model, args.model_revision)
    local_only = not args.allow_download
    common = {
        "revision": args.model_revision,
        "local_files_only": local_only,
    }
    # A local snapshot path is already revision-addressed; passing a revision for
    # it makes older Hub clients attempt an unnecessary repository lookup.
    if Path(model_ref).exists():
        common.pop("revision")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, **common)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=torch.float32,
        **common,
    )
    model.eval()
    prompts = load_prompts(args.prompts)
    args.tokens_out.parent.mkdir(parents=True, exist_ok=True)
    args.hidden_out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_revision": args.model_revision,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_dtype": "float32",
        "checkpoint_dtype": "bfloat16",
        "device": "cpu",
        "do_sample": False,
        "use_cache": True,
        "add_special_tokens": True,
        "chat_template": False,
    }

    with (
        args.tokens_out.open("w") as token_output,
        args.hidden_out.open("w") as hidden_output,
        torch.inference_mode(),
    ):
        for row in prompts:
            encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True)
            forward = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            final_hidden = forward.hidden_states[-1][0].float().cpu().tolist()
            hidden_output.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "hidden_states": final_hidden,
                        "reference": metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=model.generation_config.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            prompt_length = encoded["input_ids"].shape[1]
            tokens = generated.sequences[0, prompt_length:].tolist()
            token_output.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "tokens": tokens,
                        "top_logits": [
                            top_logits(step_scores[0], args.top_k_logits)
                            for step_scores in generated.scores
                        ],
                        "reference": {
                            **metadata,
                            "max_new_tokens": args.max_new_tokens,
                            "eos_token_id": model.generation_config.eos_token_id,
                            "pad_token_id": model.generation_config.pad_token_id,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
