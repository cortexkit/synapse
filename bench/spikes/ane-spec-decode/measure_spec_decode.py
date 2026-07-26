#!/usr/bin/env python3
"""Measure stateless Qwen3 draft windows on the local Apple Silicon host.

This script drives the Swift Core ML runner for warm p50/p95 latency, placement,
and an optional 30-second macmon power window.  It also compares greedy argmax
choices against a Transformers CPU-fp32 reference over 20 prompts and 8 steps.
It records failures as explicit holes in the raw JSON instead of turning a
conversion or runtime blocker into a fabricated zero.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPTS = (
    "Explain why fixed-shape neural network graphs can be easier to deploy.",
    "What is the difference between a compiler and an interpreter?",
    "Give a concise definition of speculative decoding.",
    "Why does a left-padded attention mask preserve causal order?",
    "Name one advantage of measuring p95 latency instead of only the mean.",
    "Write a short sentence about Apple Silicon hardware.",
    "How does greedy decoding choose the next token?",
    "What does a language model logits tensor contain?",
    "Describe bandwidth pressure in a large neural network.",
    "Why can dispatch overhead dominate a tiny inference call?",
    "Give an example of a systems programming language.",
    "What does CPU_ONLY mean in a Core ML configuration?",
    "Why should a verifier guarantee exact output in speculative decoding?",
    "Explain the purpose of a fixed window in this experiment.",
    "What is an attention mask used for?",
    "Describe a useful property of a good draft model.",
    "Why are warmup calls excluded from latency percentiles?",
    "What is an argmax operation?",
    "State one reason fp16 conversion can change model outputs.",
    "Summarize the role of an lm_head in a causal language model.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF repo id or local snapshot")
    parser.add_argument("--runner", type=Path, help="Built ane-spec-decode Swift runner")
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--out", type=Path, default=Path("results/phase-a-raw.json"))
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 128], choices=(32, 64, 128))
    parser.add_argument("--last-k", type=int, nargs="+", default=[1, 4, 8], choices=(1, 4, 8))
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--power-seconds", type=float, default=30.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="Require existing .mlmodelc bundles")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cached_model_snapshot(model_id: str) -> Path | None:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
    )
    if not cache_root.exists():
        return None
    snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def resolve_model_ref(requested: str) -> str:
    path = Path(requested).expanduser()
    if path.exists():
        return str(path.resolve())
    cached = cached_model_snapshot(requested)
    return str(cached.resolve()) if cached is not None else requested


def run_command(command: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    return completed.returncode, completed.stdout + completed.stderr


def parse_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_row(tokenizer: Any, text: str, window: int) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=window,
        padding=False,
        return_attention_mask=False,
    )
    ids = [int(value) for value in encoded["input_ids"]][-window:]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    pad_count = window - len(ids)
    return {
        "id": "timing-prompt",
        "input_ids": [int(pad_id)] * pad_count + ids,
        "attention_mask": [0] * pad_count + [1] * len(ids),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def placement_share(placement: dict[str, Any] | None) -> float | None:
    if placement is None:
        return None
    summary = placement.get("summary", {})
    shares = summary.get("dispatchable_device_share", {})
    value = shares.get("neuralEngine")
    return float(value) * 100.0 if value is not None else None


def compact_placement_summary(placement: dict[str, Any] | None) -> dict[str, Any] | None:
    if placement is None:
        return None
    summary = placement.get("summary", {})
    return {
        key: summary.get(key)
        for key in (
            "total_ops",
            "dispatchable_ops",
            "preferred_device_counts",
            "preferred_device_share",
            "dispatchable_device_counts",
            "dispatchable_device_share",
        )
        if key in summary
    }


def load_power_samples(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "ane_power" in value:
                rows.append(value)
    if not rows:
        return {"status": "no_samples", "sample_count": 0, "raw_path": str(path)}
    metrics: dict[str, Any] = {"sample_count": len(rows), "raw_path": str(path)}
    for name in ("ane_power", "cpu_power", "gpu_power", "all_power"):
        values = [float(row[name]) for row in rows if name in row]
        if values:
            metrics[name] = {
                "mean_w": fmean(values),
                "p50_w": median(values),
                "p95_w": percentile(values, 0.95),
                "max_w": max(values),
            }
    metrics["status"] = "complete"
    return metrics


def run_power_window(
    runner: Path,
    model_path: Path,
    input_path: Path,
    compute_units: str,
    seconds: float,
    warmup: int,
    stats_path: Path,
    raw_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    macmon = shutil.which("macmon")
    if macmon is None:
        return {"status": "skipped", "reason": "macmon is not installed"}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = raw_path.with_suffix(".stderr.log")
    with raw_path.open("w", encoding="utf-8") as raw_handle, error_path.open("w", encoding="utf-8") as error_handle:
        sampler = subprocess.Popen(
            [macmon, "pipe", "-i", "100"], stdout=raw_handle, stderr=error_handle, text=True
        )
        try:
            ready = False
            for _ in range(100):
                if raw_path.stat().st_size > 0:
                    ready = True
                    break
                time.sleep(0.1)
            if not ready:
                return {"status": "skipped", "reason": "macmon produced no sample", "raw_path": str(raw_path)}
            command = [
                str(runner),
                "run",
                "--model",
                str(model_path),
                "--input",
                str(input_path),
                "--stats",
                str(stats_path),
                "--compute-units",
                compute_units,
                "--warmup",
                str(warmup),
                "--duration-s",
                str(seconds),
            ]
            code, _ = run_command(command, log_path)
            result = load_power_samples(raw_path)
            result["runner_exit_code"] = code
            if code != 0:
                result["status"] = "runner_failed"
            return result
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()


def measurement_variant(
    args: argparse.Namespace,
    runner: Path,
    tokenizer: Any,
    window: int,
    last_k: int,
    compute_units: str,
    work_dir: Path,
) -> dict[str, Any]:
    package = args.models_dir / f"qwen3-w{window}-k{last_k}.mlpackage"
    compiled = args.models_dir / f"qwen3-w{window}-k{last_k}.mlmodelc"
    input_path = work_dir / f"input-w{window}-k{last_k}.jsonl"
    stats_path = work_dir / f"stats-w{window}-k{last_k}-{compute_units}.json"
    placement_path = work_dir / f"placement-w{window}-k{last_k}-{compute_units}.json"
    log_path = work_dir / f"run-w{window}-k{last_k}-{compute_units}.log"
    write_rows(input_path, [prepare_row(tokenizer, PROMPTS[0], window)])
    result: dict[str, Any] = {
        "window": window,
        "last_k": last_k,
        "compute_unit": compute_units,
        "package_path": str(package),
        "compiled_path": str(compiled),
        "status": "blocked",
    }

    if not compiled.exists():
        if args.no_compile:
            result["error"] = f"compiled model is missing: {compiled}"
            return result
        if not package.exists():
            result["error"] = f"Core ML package is missing: {package}"
            return result
        compile_stats = work_dir / f"compile-w{window}-k{last_k}.json"
        code, output = run_command(
            [
                str(runner),
                "compile",
                "--model",
                str(package),
                "--out",
                str(compiled),
                "--stats",
                str(compile_stats),
            ],
            work_dir / f"compile-w{window}-k{last_k}.log",
        )
        result["compile_exit_code"] = code
        if code != 0:
            result["error"] = "Core ML compilation failed"
            result["compile_output_tail"] = output[-2000:]
            return result
        if compile_stats.exists():
            result["compile"] = parse_json_file(compile_stats)

    command = [
        str(runner),
        "run",
        "--model",
        str(compiled),
        "--input",
        str(input_path),
        "--stats",
        str(stats_path),
        "--placement",
        str(placement_path),
        "--compute-units",
        compute_units,
        "--calls",
        str(args.calls),
        "--warmup",
        str(args.warmup),
    ]
    code, output = run_command(command, log_path)
    result["exit_code"] = code
    if code != 0:
        result["error"] = "Core ML timing run failed"
        result["output_tail"] = output[-3000:]
        return result
    stats = parse_json_file(stats_path)
    placement = parse_json_file(placement_path) if placement_path.exists() else None
    p50 = float(stats["request_latency_p50_ms"])
    result.update(
        {
            "status": "complete",
            "stats": stats,
            "placement_path": str(placement_path),
            "placement_summary": compact_placement_summary(placement),
            "ane_share_pct": placement_share(placement),
            "draft_tokens_per_s": 1000.0 / p50 if p50 > 0 else None,
            "effective_draft_tokens_per_s": last_k * 1000.0 / p50 if p50 > 0 else None,
        }
    )
    if not args.skip_power:
        power_stats_path = work_dir / f"power-stats-w{window}-k{last_k}-{compute_units}.json"
        result["power"] = run_power_window(
            runner,
            compiled,
            input_path,
            compute_units,
            args.power_seconds,
            args.warmup,
            power_stats_path,
            work_dir / f"macmon-w{window}-k{last_k}-{compute_units}.jsonl",
            work_dir / f"power-w{window}-k{last_k}-{compute_units}.log",
        )
    else:
        result["power"] = {"status": "skipped", "reason": "--skip-power"}
    return result


def parity_rows(
    model: Any, tokenizer: Any, window: int, pad_id: int
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    expected: list[int] = []
    with model_framework_inference():
        for prompt_index, prompt in enumerate(PROMPTS):
            encoded = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max(1, window - 8),
                padding=False,
                return_attention_mask=False,
            )
            context = [int(value) for value in encoded["input_ids"]]
            for step in range(8):
                current = context[-window:]
                pad_count = window - len(current)
                row_ids = [pad_id] * pad_count + current
                row_mask = [0] * pad_count + [1] * len(current)
                input_ids = model_framework_tensor(row_ids)
                attention_mask = model_framework_tensor(row_mask)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits
                token = int(output[0, -1, :].argmax().item())
                row_id = f"prompt-{prompt_index:02d}-step-{step:02d}"
                rows.append({"id": row_id, "input_ids": row_ids, "attention_mask": row_mask})
                expected.append(token)
                context.append(token)
    return rows, expected


class model_framework_inference:
    def __enter__(self):
        import torch

        self.torch = torch
        self.previous = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.torch.set_grad_enabled(self.previous)
        return False


def model_framework_tensor(values: list[int]) -> Any:
    import torch

    return torch.tensor([values], dtype=torch.long)


def run_parity(
    args: argparse.Namespace,
    runner: Path,
    tokenizer: Any,
    model_ref: str,
    windows: list[int],
    work_dir: Path,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=not args.allow_download,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        ).eval()
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad or eos token")
    except Exception as exc:
        return {"status": "blocked", "error": f"could not load Transformers reference: {exc}"}

    reports: list[dict[str, Any]] = []
    for window in windows:
        try:
            rows, expected = parity_rows(hf_model, tokenizer, window, int(pad_id))
            input_path = work_dir / f"parity-w{window}.jsonl"
            output_path = work_dir / f"parity-w{window}-coreml.jsonl"
            write_rows(input_path, rows)
            package = args.models_dir / f"qwen3-w{window}-k1.mlpackage"
            compiled = args.models_dir / f"qwen3-w{window}-k1.mlmodelc"
            if not compiled.exists():
                if args.no_compile or not package.exists():
                    raise FileNotFoundError(f"compiled parity model is missing: {compiled}")
                code, output = run_command(
                    [str(runner), "compile", "--model", str(package), "--out", str(compiled), "--stats", str(work_dir / f"parity-compile-w{window}.json")],
                    work_dir / f"parity-compile-w{window}.log",
                )
                if code != 0:
                    raise RuntimeError(output[-2000:])
            code, output = run_command(
                [
                    str(runner),
                    "predict",
                    "--model",
                    str(compiled),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--compute-units",
                    "CPU_AND_NE",
                ],
                work_dir / f"parity-run-w{window}.log",
            )
            if code != 0:
                raise RuntimeError(output[-2000:])
            observed = [int(json.loads(line)["argmax"]) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            agreements = sum(actual == want for actual, want in zip(observed, expected, strict=True))
            reports.append(
                {
                    "window": window,
                    "steps": len(expected),
                    "agreements": agreements,
                    "agreement_rate": agreements / len(expected),
                    "status": "complete",
                }
            )
        except Exception as exc:
            reports.append({"window": window, "status": "blocked", "error": str(exc)})
    complete = [row for row in reports if row["status"] == "complete"]
    total_steps = sum(int(row["steps"]) for row in complete)
    total_agreements = sum(int(row["agreements"]) for row in complete)
    return {
        "status": "complete" if complete and len(complete) == len(windows) else "partial",
        "prompts": 20,
        "steps_per_prompt": 8,
        "total_steps": total_steps,
        "total_agreements": total_agreements,
        "agreement_rate": total_agreements / total_steps if total_steps else None,
        "per_window": reports,
        "reference": "Transformers CPU fp32, greedy argmax, fixed-window stateless inputs",
    }


def toolchain_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "coremltools": None,
        "torch": None,
        "transformers": None,
        "macmon": shutil.which("macmon"),
    }
    for name in ("coremltools", "torch", "transformers"):
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            report[f"{name}_error"] = str(exc)
    return report


def main() -> int:
    args = parse_args()
    if args.calls <= 0 or args.warmup < 0:
        raise ValueError("--calls must be positive and --warmup must be nonnegative")
    if args.power_seconds < 0:
        raise ValueError("--power-seconds must be nonnegative")
    if args.last_k and any(k > max(args.windows) for k in args.last_k):
        raise ValueError("a --last-k value cannot exceed the largest selected window")
    runner = args.runner or Path(__file__).resolve().parent / ".build" / "ane-spec-decode"
    model_ref = resolve_model_ref(args.model)
    args.models_dir = args.models_dir.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    work_dir = args.out.parent / "phase-a-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite")

    report: dict[str, Any] = {
        "schema": "ane-spec-decode.phase-a.v1",
        "started_at": datetime.now().astimezone().isoformat(),
        "host_scope": "local M5 only; no M1 or production surfaces",
        "toolchain": toolchain_report(),
        "model": {"requested": args.model, "resolved": model_ref},
        "parameters": {
            "windows": args.windows,
            "last_k": args.last_k,
            "calls": args.calls,
            "warmup": args.warmup,
            "power_seconds": args.power_seconds,
        },
        "variants": [],
        "parity": {"status": "skipped", "reason": "not started"},
    }
    if not runner.exists():
        report["status"] = "blocked"
        report["error"] = f"runner is missing: {runner}; run build_runner.sh first"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 2

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_ref, local_files_only=not args.allow_download
        )
    except Exception as exc:
        report["status"] = "blocked"
        report["error"] = f"could not load tokenizer: {exc}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 2

    for window in args.windows:
        for last_k in args.last_k:
            if last_k > window:
                continue
            for compute_units in ("CPU_AND_NE", "CPU_ONLY"):
                report["variants"].append(
                    measurement_variant(args, runner, tokenizer, window, last_k, compute_units, work_dir)
                )

    if args.skip_parity:
        report["parity"] = {"status": "skipped", "reason": "--skip-parity"}
    else:
        report["parity"] = run_parity(args, runner, tokenizer, model_ref, args.windows, work_dir)
    complete_variants = [row for row in report["variants"] if row.get("status") == "complete"]
    report["status"] = "complete" if complete_variants else "blocked"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["outcome"] = {
        "complete_variants": len(complete_variants),
        "requested_variants": len(report["variants"]),
        "conversion_or_runtime_holes": [
            {"window": row["window"], "last_k": row["last_k"], "compute_unit": row["compute_unit"], "error": row.get("error")}
            for row in report["variants"]
            if row.get("status") != "complete"
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
