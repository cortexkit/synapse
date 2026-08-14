#!/usr/bin/env python3
"""Capture CPU_AND_NE/CPU_ONLY CoreML outputs and compare both with Metal."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import subprocess
import time
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            hasher.update(child.relative_to(path).as_posix().encode())
            hasher.update(b"\0")
            with child.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
    else:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def load_fixture(root: Path, case_id: str) -> dict[str, Any]:
    namespace = runpy.run_path(str(root / "tests/ane-prefill-certification/certify.py"))
    cases = namespace["fixture_cases"]()
    try:
        return next(case for case in cases if case["case_id"] == case_id)
    except StopIteration as error:
        raise ValueError(f"unknown certification fixture {case_id}") from error


def run_coreml(
    args: argparse.Namespace,
    fixture_path: Path,
    output_dir: Path,
    compute_units: str,
) -> None:
    subprocess.run(
        [
            str(args.runner),
            "run",
            "--model",
            str(args.compiled),
            "--input",
            str(fixture_path),
            "--stats",
            str(output_dir / f"{compute_units}-stats.json"),
            "--cache-out",
            str(output_dir / f"{compute_units}-cache.bin"),
            "--logits-out",
            str(output_dir / f"{compute_units}-logits.bin"),
            "--model-window",
            str(args.window),
            "--chunks",
            "1",
            "--cache-bucket",
            str(args.cache_bucket),
            "--calls",
            "1",
            "--warmup",
            "0",
            "--compute-units",
            compute_units,
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--window", type=int, default=128, choices=(128, 256, 512))
    parser.add_argument("--case-id", default="w128-width-01")
    parser.add_argument("--cache-bucket", type=int, default=512, choices=(512, 1024, 2048))
    parser.add_argument(
        "--decode-config", choices=("f16-step", "q8-step"), default="f16-step"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--forced-prefix-json",
        type=Path,
        help="JSON array of tokens shared by the production paths before they diverge",
    )
    parser.add_argument(
        "--skip-cpu-control",
        action="store_true",
        help="capture only CPU_AND_NE (useful for the 20-fixture flip-density battery)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.root = args.root.resolve()
    args.model = args.model.resolve()
    args.compiled = args.compiled.resolve()
    args.runner = args.runner.resolve()
    args.analyzer = args.analyzer.resolve()
    args.output_dir = args.output_dir.resolve()
    for path in (args.model, args.compiled, args.runner, args.analyzer):
        if not path.exists():
            raise FileNotFoundError(path)
    case = load_fixture(args.root, args.case_id)
    if case["bucket"] != args.window or len(case["prompt_token_ids"]) != args.window:
        raise ValueError("diagnostic fixtures must be width-exact for the selected graph window")
    if args.window + args.max_new_tokens > args.cache_bucket:
        raise ValueError("selected cache bucket cannot hold the prompt and continuation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output_dir / f"{args.case_id}.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "id": case["case_id"],
                "input_ids": case["prompt_token_ids"],
                "attention_mask": [1] * len(case["prompt_token_ids"]),
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    load_context = {
        "captured_at_epoch_seconds": time.time(),
        "hardware": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "machine": platform.machine(),
        "macos": command_output(["sw_vers", "-productVersion"]),
        "build": command_output(["sw_vers", "-buildVersion"]),
        "load_average_before": command_output(["sysctl", "-n", "vm.loadavg"]),
        "power": command_output(["pmset", "-g", "batt"]),
        "source_checkpoint_sha256": sha256_path(args.model),
        "compiled_package_sha256": sha256_path(args.compiled),
        "decode_config": args.decode_config,
        "fixture_sha256": sha256_path(fixture_path),
    }
    (args.output_dir / "load-context.json").write_text(
        json.dumps(load_context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_coreml(args, fixture_path, args.output_dir, "cpu-and-ne")
    analyzer_command = [
        str(args.analyzer),
        "--model",
        str(args.model),
        "--input",
        str(fixture_path),
        "--ane-cache",
        str(args.output_dir / "cpu-and-ne-cache.bin"),
        "--ane-logits",
        str(args.output_dir / "cpu-and-ne-logits.bin"),
        "--cache-bucket",
        str(args.cache_bucket),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--decode-config",
        args.decode_config,
        "--out",
        str(args.output_dir / "analysis.json"),
    ]
    if args.forced_prefix_json is not None:
        analyzer_command.extend(
            ["--forced-prefix-json", str(args.forced_prefix_json.resolve())]
        )
    if not args.skip_cpu_control:
        run_coreml(args, fixture_path, args.output_dir, "cpu-only")
        analyzer_command.extend(
            [
                "--cpu-cache",
                str(args.output_dir / "cpu-only-cache.bin"),
                "--cpu-logits",
                str(args.output_dir / "cpu-only-logits.bin"),
            ]
        )
    subprocess.run(analyzer_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
