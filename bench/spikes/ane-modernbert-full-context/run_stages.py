#!/usr/bin/env python3
"""Run one guarded ModernBERT stage at a time in isolated child processes."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

STAGES = (1024, 2048, 4096, 8192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=STAGES, required=True)
    parser.add_argument("--query-tile", type=int, default=256)
    parser.add_argument("--key-tile", type=int, default=256)
    parser.add_argument("--warm-repetitions", type=int, default=3)
    parser.add_argument(
        "--measurement-slot-authorized",
        required=True,
        help="Human-readable authorization recorded in the run report; hardware work refuses to start without it.",
    )
    parser.add_argument("--minimum-available-gib", type=float, default=32.0)
    parser.add_argument("--abort-available-gib", type=float, default=16.0)
    parser.add_argument(
        "--abort-free-gib", type=float, default=0.0,
        help="Optional raw free-page floor; disabled by default because macOS aggressively repurposes free pages.",
    )
    parser.add_argument("--abort-wired-fraction", type=float, default=0.50)
    parser.add_argument("--abort-owned-rss-gib", type=float, default=32.0)
    parser.add_argument("--child-timeout-minutes", type=float, default=60.0)
    return parser.parse_args()


def vm_stat() -> dict[str, int]:
    output = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True).stdout
    page_match = re.search(r"page size of (\d+) bytes", output)
    if not page_match:
        raise RuntimeError("could not parse vm_stat page size")
    page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for label, value in re.findall(r"^Pages ([^:]+):\s+([0-9]+)\.$", output, re.MULTILINE):
        pages[label] = int(value)
    return {
        "page_size": page_size,
        "free_bytes": pages.get("free", 0) * page_size,
        "wired_bytes": pages.get("wired down", 0) * page_size,
        "inactive_bytes": pages.get("inactive", 0) * page_size,
        "compressor_bytes": pages.get("occupied by compressor", 0) * page_size,
    }


def memory_snapshot() -> dict[str, Any]:
    physical = int(subprocess.run(
        ["sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True
    ).stdout.strip())
    pressure = subprocess.run(
        ["memory_pressure", "-Q"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        **vm_stat(),
        "physical_bytes": physical,
        "available_bytes": int(psutil.virtual_memory().available),
        "memory_pressure": pressure,
        "captured_unix_s": time.time(),
    }


def owned_rss(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        descendants = process.children(recursive=True)
        return process.memory_info().rss + sum(
            child.memory_info().rss for child in descendants if child.is_running()
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def terminate_owned_child(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_guarded(
    command: list[str],
    log_path: Path,
    *,
    abort_available_bytes: int,
    abort_free_bytes: int,
    abort_wired_fraction: float,
    abort_owned_rss_bytes: int,
    timeout_s: float,
) -> dict[str, Any]:
    before = memory_snapshot()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_rss = 0
    minimum_available = before["available_bytes"]
    minimum_free = before["free_bytes"]
    maximum_wired = before["wired_bytes"]
    abort_reason: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            sample = vm_stat()
            available = int(psutil.virtual_memory().available)
            rss = owned_rss(process.pid)
            peak_rss = max(peak_rss, rss)
            minimum_available = min(minimum_available, available)
            minimum_free = min(minimum_free, sample["free_bytes"])
            maximum_wired = max(maximum_wired, sample["wired_bytes"])
            if available < abort_available_bytes:
                abort_reason = f"system available memory fell below {abort_available_bytes} bytes"
            elif abort_free_bytes > 0 and sample["free_bytes"] < abort_free_bytes:
                abort_reason = f"system free memory fell below {abort_free_bytes} bytes"
            elif sample["wired_bytes"] > before["physical_bytes"] * abort_wired_fraction:
                abort_reason = f"system wired memory exceeded fraction {abort_wired_fraction}"
            elif rss > abort_owned_rss_bytes:
                abort_reason = f"owned child RSS exceeded {abort_owned_rss_bytes} bytes"
            elif time.perf_counter() - started > timeout_s:
                abort_reason = f"owned child exceeded timeout of {timeout_s} seconds"
            if abort_reason:
                terminate_owned_child(process)
                break
            time.sleep(0.5)
        return_code = process.wait()
    after = memory_snapshot()
    result = {
        "command": command,
        "return_code": return_code,
        "elapsed_s": time.perf_counter() - started,
        "peak_owned_rss_bytes": peak_rss,
        "minimum_system_available_bytes": minimum_available,
        "minimum_system_free_bytes": minimum_free,
        "maximum_system_wired_bytes": maximum_wired,
        "before": before,
        "after": after,
        "abort_reason": abort_reason,
        "safeguard_limitations": (
            "RSS tracks only owned children. System wired memory is observed separately; these guards "
            "cannot guarantee an ANE allocation is bounded or prevent workstation pressure."
        ),
    }
    if return_code != 0 or abort_reason:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_can_continue(status: str | None) -> bool:
    return status == "passed"


def placement_classification(reload_report: dict[str, Any], placement: dict[str, Any]) -> str:
    if reload_report.get("status") == "failed":
        return "prediction_failed"
    parity_suffix = "_parity_failed" if reload_report.get("status") == "parity_failed" else ""
    expensive = placement.get("expensive_operator_device_counts", {})
    attention = {
        name: counts
        for name, counts in expensive.items()
        if name in {"einsum", "matmul", "softmax"}
    }
    if not attention:
        return "ane_residency_unproven_no_attention_ops_in_plan" + parity_suffix
    if any(counts.get("cpu", 0) or counts.get("gpu", 0) for counts in attention.values()):
        return "cpu_fallback_or_mixed_attention_placement" + parity_suffix
    if all(counts.get("neuralEngine", 0) > 0 for counts in attention.values()):
        return "cpu_and_ne_prediction_passed_with_attention_preferred_on_ane" + parity_suffix
    return "ane_residency_unproven_unknown_attention_placement" + parity_suffix


def main() -> int:
    args = parse_args()
    if not args.measurement_slot_authorized.strip():
        raise ValueError("measurement slot authorization must not be empty")
    initial = memory_snapshot()
    minimum_available = int(args.minimum_available_gib * 1024**3)
    if initial["available_bytes"] < minimum_available:
        raise RuntimeError(
            f"preflight available memory {initial['available_bytes']} is below {minimum_available}"
        )
    root = args.artifacts.resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().with_name("spike.py")
    # Keep the virtual-environment entry path; resolving its symlink drops that environment's site-packages.
    python = Path(sys.executable)
    placement_binary = Path(__file__).resolve().parent / ".build" / "modernbert-placement"
    if not placement_binary.exists():
        raise FileNotFoundError(f"build placement helper first: {placement_binary}")
    guard = {
        "abort_available_bytes": int(args.abort_available_gib * 1024**3),
        "abort_free_bytes": int(args.abort_free_gib * 1024**3),
        "abort_wired_fraction": args.abort_wired_fraction,
        "abort_owned_rss_bytes": int(args.abort_owned_rss_gib * 1024**3),
        "timeout_s": args.child_timeout_minutes * 60.0,
    }
    run_report: dict[str, Any] = {
        "status": "running",
        "measurement_slot_authorized": args.measurement_slot_authorized,
        "stages_requested": [stage for stage in STAGES if stage <= args.through],
        "initial_memory": initial,
        "guard": guard,
        "stage_reports": [],
    }
    report_path = root / "run.json"
    try:
        cpu_report = root / "cpu-check.json"
        run_report["cpu_check_process"] = run_guarded(
            [
                str(python), str(script), "--model", str(args.model), "cpu-check",
                "--report", str(cpu_report),
            ],
            root / "cpu-check.log",
            **guard,
        )
        if read_json(cpu_report).get("status") != "passed":
            raise RuntimeError("CPU check did not pass")

        for sequence_length in run_report["stages_requested"]:
            stage_root = root / f"seq{sequence_length}"
            stage_root.mkdir(parents=True, exist_ok=True)
            input_path = stage_root / "input.jsonl"
            reference_path = stage_root / "reference.jsonl"
            package_path = stage_root / "gte-modernbert.mlpackage"
            vectors_path = stage_root / "coreml-vectors.jsonl"
            commands = (
                (
                    "prepare",
                    [str(python), str(script), "--model", str(args.model), "prepare-input",
                     "--seq-len", str(sequence_length), "--query-tile", str(args.query_tile),
                     "--key-tile", str(args.key_tile), "--out", str(input_path),
                     "--report", str(stage_root / "prepare.json")],
                ),
                (
                    "reference",
                    [str(python), str(script), "--model", str(args.model), "reference",
                     "--seq-len", str(sequence_length), "--query-tile", str(args.query_tile),
                     "--key-tile", str(args.key_tile), "--input", str(input_path),
                     "--out", str(reference_path), "--report", str(stage_root / "reference.json")],
                ),
                (
                    "export",
                    [str(python), str(script), "--model", str(args.model), "export",
                     "--seq-len", str(sequence_length), "--query-tile", str(args.query_tile),
                     "--key-tile", str(args.key_tile), "--input", str(input_path),
                     "--reference", str(reference_path), "--out", str(package_path),
                     "--report", str(stage_root / "export.json"), "--overwrite"],
                ),
                (
                    "reload",
                    [str(python), str(script), "--model", str(args.model), "reload",
                     "--package", str(package_path), "--input", str(input_path),
                     "--reference", str(reference_path), "--warm-repetitions", str(args.warm_repetitions),
                     "--vectors-out", str(vectors_path), "--report", str(stage_root / "reload.json")],
                ),
            )
            stage_processes = {}
            for name, command in commands:
                stage_processes[name] = run_guarded(
                    command, stage_root / f"{name}.log", **guard
                )
            reload_report = read_json(stage_root / "reload.json")
            if not stage_can_continue(reload_report.get("status")):
                cpu_command = [
                    str(python), str(script), "--model", str(args.model), "reload",
                    "--package", str(package_path), "--input", str(input_path),
                    "--reference", str(reference_path), "--warm-repetitions", "1",
                    "--compute-units", "cpu-only",
                    "--vectors-out", str(stage_root / "coreml-cpu-vectors.jsonl"),
                    "--report", str(stage_root / "reload-cpu.json"),
                ]
                stage_processes["reload_cpu_diagnostic"] = run_guarded(
                    cpu_command, stage_root / "reload-cpu.log", **guard
                )
            stage_processes["placement"] = run_guarded(
                [str(placement_binary), str(package_path), str(stage_root / "placement.json")],
                stage_root / "placement.log",
                **guard,
            )
            placement_report = read_json(stage_root / "placement.json")
            summary = {
                "sequence_length": sequence_length,
                "status": reload_report.get("status", "failed"),
                "processes": stage_processes,
                "ane_classification": placement_classification(reload_report, placement_report),
                "classification_basis": (
                    "Successful CPU_AND_NE prediction plus MLComputePlan preferred placement. "
                    "MLComputePlan is not a runtime dispatch trace."
                ),
            }
            run_report["stage_reports"].append(summary)
            write_json(stage_root / "summary.json", summary)
            write_json(report_path, run_report)
            if not stage_can_continue(summary["status"]):
                raise RuntimeError(
                    f"stage {sequence_length} stopped after {summary['status']}; see compact reports"
                )
        run_report["status"] = "passed"
    except BaseException as error:
        run_report["status"] = "failed"
        run_report["error_type"] = type(error).__name__
        run_report["error"] = str(error)
        write_json(report_path, run_report)
        raise
    write_json(report_path, run_report)
    return 0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
