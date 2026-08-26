#!/usr/bin/env python3
"""Run the locked-M1 ANE-prefill/Metal-decode measurement and parity battery."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer


WINDOW = 128
VARIANTS = {
    "w128": {"model_window": 128, "chunks": 1},
    "w32x4": {"model_window": 32, "chunks": 4},
}


def bench_root() -> Path:
    configured = os.environ.get("SYNAPSE_BENCH_ROOT")
    return Path(configured) if configured else Path.home() / "synapse-bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--macmon", type=Path, default=bench_root() / "bench-tools/bin/macmon")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--battery-limit", type=int, default=20)
    parser.add_argument("--stage-calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def run_command(command: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + " ".join(command) + "\n" + completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout[-4000:]}"
        )
    return completed


def environment_report() -> dict[str, Any]:
    def capture(command: list[str]) -> str:
        return run_command(command).stdout.strip()

    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "hardware_model": capture(["sysctl", "-n", "hw.model"]),
        "loadavg": capture(["sysctl", "-n", "vm.loadavg"]),
        "power_source": capture(["pmset", "-g", "batt"]),
        "runner_worker_active": subprocess.run(
            ["pgrep", "-f", "Runner.Worker"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0,
    }


def acquire_lock(enabled: bool) -> Path | None:
    if not enabled:
        return None
    lock = bench_root() / "bench.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"benchmark lock is busy: {lock}") from error
    worker = subprocess.run(
        ["pgrep", "-f", "Runner.Worker"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if worker.returncode == 0:
        lock.rmdir()
        raise RuntimeError("Runner.Worker is active; released benchmark lock")
    battery = run_command(["pmset", "-g", "batt"]).stdout
    if "AC Power" not in battery:
        lock.rmdir()
        raise RuntimeError("locked measurement requires AC power")
    one_minute_load = os.getloadavg()[0]
    if one_minute_load >= 3.0:
        lock.rmdir()
        raise RuntimeError(
            f"one-minute load average is {one_minute_load:.2f}; locked measurement requires < 3.0"
        )
    return lock


def prepare_rows(model: Path, prompts_path: Path, work_dir: Path, limit: int) -> list[Path]:
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    prompts = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]
    if len(prompts) != limit:
        raise RuntimeError(f"requested {limit} prompts but found {len(prompts)}")
    rows_dir = work_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, prompt in enumerate(prompts):
        text = str(prompt["prompt"])
        repeated = (text + " ") * WINDOW
        encoded = tokenizer(
            repeated,
            add_special_tokens=True,
            truncation=True,
            max_length=WINDOW,
            padding=False,
            return_attention_mask=False,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        if len(ids) != WINDOW:
            raise RuntimeError(f"prompt {prompt['id']} produced {len(ids)} tokens, expected {WINDOW}")
        row = {
            "id": str(prompt["id"]),
            "source_prompt": text,
            "battery_index": index,
            "input_ids": ids,
            "attention_mask": [1] * WINDOW,
        }
        path = rows_dir / f"{index:02d}-{prompt['id']}.json"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def runner_command(
    args: argparse.Namespace,
    variant: str,
    row: Path,
    stats: Path,
    cache: Path,
    logits: Path,
    calls: int,
    warmup: int,
    placement: Path | None = None,
) -> list[str]:
    shape = VARIANTS[variant]
    command = [
        str(args.runner),
        "run",
        "--model",
        str(args.models_dir / f"qwen3-prefill-{variant.replace('x4', '')}.mlmodelc"),
        "--input",
        str(row),
        "--stats",
        str(stats),
        "--cache-out",
        str(cache),
        "--logits-out",
        str(logits),
        "--model-window",
        str(shape["model_window"]),
        "--chunks",
        str(shape["chunks"]),
        "--cache-bucket",
        "512",
        "--calls",
        str(calls),
        "--warmup",
        str(warmup),
        "--compute-units",
        "cpu-and-ne",
    ]
    if placement is not None:
        command.extend(["--placement", str(placement)])
    return command


def harness_command(
    args: argparse.Namespace,
    row: Path,
    cache: Path,
    logits: Path,
    out: Path,
    baseline_calls: int,
    upload_calls: int,
    extra: list[str] | None = None,
) -> list[str]:
    command = [
        str(args.harness),
        "--model",
        str(args.model),
        "--input",
        str(row),
        "--cache",
        str(cache),
        "--logits",
        str(logits),
        "--out",
        str(out),
        "--cache-bucket",
        "512",
        "--max-new-tokens",
        "64",
        "--baseline-prefill-calls",
        str(baseline_calls),
        "--upload-calls",
        str(upload_calls),
    ]
    if extra:
        command.extend(extra)
    return command


def run_variant(
    args: argparse.Namespace, variant: str, rows: list[Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variant_dir = args.work_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    stage_stats = variant_dir / "stage.json"
    placement = variant_dir / "placement.json"
    stage_cache = variant_dir / "cache.bin"
    stage_logits = variant_dir / "logits.bin"
    run_command(
        runner_command(
            args,
            variant,
            first,
            stage_stats,
            stage_cache,
            stage_logits,
            args.stage_calls,
            args.warmup,
            placement,
        ),
        variant_dir / "stage.log",
    )
    run_command(
        harness_command(
            args,
            first,
            stage_cache,
            stage_logits,
            variant_dir / "comparison.json",
            args.stage_calls,
            args.stage_calls,
        ),
        variant_dir / "comparison.log",
    )

    battery: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt_dir = variant_dir / "battery" / f"{index:02d}"
        stats = prompt_dir / "stage.json"
        cache = prompt_dir / "cache.bin"
        logits = prompt_dir / "logits.bin"
        comparison = prompt_dir / "comparison.json"
        run_command(
            runner_command(args, variant, row, stats, cache, logits, 1, 0),
            prompt_dir / "runner.log",
        )
        run_command(
            harness_command(args, row, cache, logits, comparison, 1, 1),
            prompt_dir / "harness.log",
        )
        battery.append(json.loads(comparison.read_text(encoding="utf-8")))

    stage = json.loads(stage_stats.read_text(encoding="utf-8"))
    stage.pop("stages", None)
    placement_summary = json.loads(placement.read_text(encoding="utf-8"))["summary"]
    placement_summary.pop("unknown_operations", None)
    measurement = {
        "stage": stage,
        "placement": placement_summary,
        "comparison": json.loads((variant_dir / "comparison.json").read_text(encoding="utf-8")),
    }
    return measurement, battery


def load_power(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "all_power" in row:
                rows.append(row)
    result: dict[str, Any] = {"sample_count": len(rows), "raw_path": str(path)}
    for name in ("ane_power", "cpu_power", "gpu_power", "all_power"):
        values = [float(row[name]) for row in rows if name in row]
        if values:
            result[name] = {
                "mean_w": statistics.fmean(values),
                "median_w": statistics.median(values),
                "max_w": max(values),
            }
    return result


def run_power(
    command: list[str], macmon: Path, raw_path: Path, log_path: Path
) -> dict[str, Any]:
    if not macmon.is_file():
        return {"status": "skipped", "reason": f"macmon is missing: {macmon}"}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = raw_path.with_suffix(".stderr.log")
    with raw_path.open("w", encoding="utf-8") as raw, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        sampler = subprocess.Popen(
            [str(macmon), "pipe", "-i", "100"], stdout=raw, stderr=stderr, text=True
        )
        try:
            for _ in range(100):
                if raw_path.stat().st_size > 0:
                    break
                time.sleep(0.1)
            started = time.monotonic()
            run_command(command, log_path)
            wall = time.monotonic() - started
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
    result = load_power(raw_path)
    result.update({"status": "complete", "command_wall_s": wall})
    return result


def power_measurements(args: argparse.Namespace, first: Path) -> dict[str, Any]:
    if args.skip_power:
        return {"status": "skipped", "reason": "--skip-power"}
    power_dir = args.work_dir / "power"
    power_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for variant, calls in (("w128", 300), ("w32x4", 100)):
        results[f"ane_{variant}"] = run_power(
            runner_command(
                args,
                variant,
                first,
                power_dir / f"ane-{variant}-stage.json",
                power_dir / f"ane-{variant}-cache.bin",
                power_dir / f"ane-{variant}-logits.bin",
                calls,
                args.warmup,
            ),
            args.macmon,
            power_dir / f"ane-{variant}-macmon.jsonl",
            power_dir / f"ane-{variant}.log",
        )
    cache = args.work_dir / "w128" / "cache.bin"
    logits = args.work_dir / "w128" / "logits.bin"
    results["gpu_prefill"] = run_power(
        harness_command(
            args,
            first,
            cache,
            logits,
            power_dir / "gpu-prefill-stage.json",
            10,
            1,
            ["--baseline-only"],
        ),
        args.macmon,
        power_dir / "gpu-prefill-macmon.jsonl",
        power_dir / "gpu-prefill.log",
    )
    results["gpu_upload"] = run_power(
        harness_command(
            args,
            first,
            cache,
            logits,
            power_dir / "gpu-upload-stage.json",
            1,
            1_000,
            ["--upload-only"],
        ),
        args.macmon,
        power_dir / "gpu-upload-macmon.jsonl",
        power_dir / "gpu-upload.log",
    )
    results["gpu_decode"] = run_power(
        harness_command(
            args,
            first,
            cache,
            logits,
            power_dir / "gpu-decode-stage.json",
            1,
            1,
            ["--decode-only", "--decode-calls", "5"],
        ),
        args.macmon,
        power_dir / "gpu-decode-macmon.jsonl",
        power_dir / "gpu-decode.log",
    )
    for name in ("gpu_prefill", "gpu_upload", "gpu_decode"):
        stage_path = power_dir / f"{name.replace('_', '-')}-stage.json"
        if stage_path.exists():
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage.pop("samples_ms", None)
            results[name]["stage"] = stage
    for variant in ("w128", "w32x4"):
        stage_path = power_dir / f"ane-{variant}-stage.json"
        if stage_path.exists():
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage.pop("stages", None)
            results[f"ane_{variant}"]["stage"] = stage
    return results


def battery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(bool(row["token_exact"]) for row in rows)
    depths = [int(row["match_depth"]) for row in rows]
    divergences = [
        {
            "id": row["id"],
            "depth": row["divergence_depth"],
            "baseline_gap": row["baseline_gap_at_divergence"],
            "ane_gap": row["ane_gap_at_divergence"],
            "baseline_token": row["baseline"]["generated_tokens"][row["divergence_depth"]],
            "ane_token": row["ane_split"]["generated_tokens"][row["divergence_depth"]],
        }
        for row in rows
        if row["divergence_depth"] is not None
    ]
    return {
        "prompts": len(rows),
        "exact_prompts": exact,
        "token_exact_pct": exact / len(rows) * 100.0,
        "mean_match_depth": statistics.fmean(depths),
        "median_match_depth": statistics.median(depths),
        "min_match_depth": min(depths),
        "max_match_depth": max(depths),
        "divergences": divergences,
    }


def energy_summary(
    variants: dict[str, dict[str, Any]], power: dict[str, Any]
) -> dict[str, Any] | None:
    if power.get("status") == "skipped":
        return None
    gpu_prefill_w = power["gpu_prefill"]["all_power"]["mean_w"]
    gpu_upload_w = power["gpu_upload"]["all_power"]["mean_w"]
    gpu_decode_w = power["gpu_decode"]["all_power"]["mean_w"]
    gpu_prefill_ms = power["gpu_prefill"]["stage"]["p50_ms"]
    gpu_upload_ms = power["gpu_upload"]["stage"]["p50_ms"]
    gpu_decode_ms = power["gpu_decode"]["stage"]["p50_ms"]
    result: dict[str, Any] = {
        "gpu_prefill_j": gpu_prefill_w * gpu_prefill_ms / 1000.0,
        "gpu_decode_j": gpu_decode_w * gpu_decode_ms / 1000.0,
        "baseline_request_j": gpu_prefill_w * gpu_prefill_ms / 1000.0
        + gpu_decode_w * gpu_decode_ms / 1000.0,
    }
    for variant in VARIANTS:
        ane_w = power[f"ane_{variant}"]["all_power"]["mean_w"]
        ane_ms = power[f"ane_{variant}"]["stage"]["compute_and_copy_p50_ms"]
        split_j = (
            ane_w * ane_ms / 1000.0
            + gpu_upload_w * gpu_upload_ms / 1000.0
            + gpu_decode_w * gpu_decode_ms / 1000.0
        )
        result[variant] = {
            "ane_compute_copy_j": ane_w * ane_ms / 1000.0,
            "gpu_upload_j": gpu_upload_w * gpu_upload_ms / 1000.0,
            "gpu_decode_j": gpu_decode_w * gpu_decode_ms / 1000.0,
            "split_request_j": split_j,
            "energy_win_x": result["baseline_request_j"] / split_j,
        }
    return result


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.locked)
    try:
        environment = environment_report()
        if args.locked and environment["runner_worker_active"]:
            raise RuntimeError("Runner.Worker appeared after lock acquisition")
        rows = prepare_rows(args.model, args.prompts, args.work_dir, args.battery_limit)
        variants: dict[str, dict[str, Any]] = {}
        batteries: dict[str, list[dict[str, Any]]] = {}
        for variant in VARIANTS:
            variants[variant], batteries[variant] = run_variant(args, variant, rows)
        power = power_measurements(args, rows[0])
        result = {
            "environment": environment,
            "protocol": {
                "locked": args.locked,
                "lock_path": str(bench_root() / "bench.lock") if args.locked else None,
                "prompt_tokens": WINDOW,
                "battery_prompts": len(rows),
                "generated_tokens": 64,
                "gpu_prefill_chunk": 16,
                "coreml_compute_units": "CPU_AND_NE",
            },
            "variants": variants,
            "battery": {name: battery_summary(rows_) for name, rows_ in batteries.items()},
            "power": power,
        }
        result["energy"] = energy_summary(variants, power)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    finally:
        if lock is not None and lock.exists():
            lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
