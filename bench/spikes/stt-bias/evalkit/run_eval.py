#!/usr/bin/env python3
"""Run one owned LFM2-Audio bias arm and score it against an audio manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ARMS = ("baseline", "prompt", "trie", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--decode-cache-bucket", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "metal"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kit = Path(__file__).resolve().parent
    repo = kit.parents[3]
    label = args.arm if args.arm not in {"trie", "combined"} else f"{args.arm}-delta-{args.delta:g}"
    manifest = args.out_dir / f"{label}-inputs.jsonl"
    raw_output = args.out_dir / f"{label}-owned.json"
    score_output = args.out_dir / f"{label}-score.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(kit / "prepare_arm_manifest.py"),
            "--audio-manifest",
            str(args.audio_manifest),
            "--arm",
            args.arm,
            "--out",
            str(manifest),
        ],
        check=True,
        cwd=repo,
    )
    command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "spike-unified-rt",
        "--bin",
        "spike-unified-rt",
        "--",
        "--model",
        str(args.model),
        "--tokenizer",
        str(args.model / "tokenizer.json"),
        "--asr-audio",
        str(manifest),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--decode-cache-bucket",
        str(args.decode_cache_bucket),
        "--device",
        args.device,
        "--dtype",
        "f32",
        "--out",
        str(raw_output),
    ]
    if args.arm in {"prompt", "combined"}:
        command.append("--asr-prompt-bias")
    if args.arm in {"trie", "combined"}:
        command.extend(["--asr-trie-delta", str(args.delta), "--asr-trie-window", str(args.window)])
    subprocess.run(command, check=True, cwd=repo)
    subprocess.run(
        [
            sys.executable,
            str(kit / "score.py"),
            "--arm",
            label,
            "--audio-manifest",
            str(args.audio_manifest),
            "--asr-output",
            str(raw_output),
            "--out",
            str(score_output),
        ],
        check=True,
        cwd=repo,
    )


if __name__ == "__main__":
    main()
