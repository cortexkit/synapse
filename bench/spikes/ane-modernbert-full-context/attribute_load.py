#!/usr/bin/env python3
"""Measure Core ML package compilation and repeat model-load behavior."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import psutil

from run_stages import memory_snapshot, run_guarded

STAGES = (1024, 2048, 4096, 8192)
ANE_WORKER_MARKER = "ck-synapse-worker-ane"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run guarded attribution for one or more packages")
    run.add_argument(
        "--stage",
        action="append",
        required=True,
        metavar="TOKENS=PACKAGE",
        help="Sequence length and source .mlpackage path; may be repeated.",
    )
    run.add_argument("--artifacts", type=Path, required=True)
    run.add_argument("--placement-binary", type=Path, required=True)
    run.add_argument(
        "--measurement-slot-authorized",
        required=True,
        help="Human-readable authorization recorded in the report.",
    )
    run.add_argument("--minimum-available-gib", type=float, default=32.0)
    run.add_argument("--abort-available-gib", type=float, default=16.0)
    run.add_argument("--abort-free-gib", type=float, default=0.0)
    run.add_argument("--abort-wired-fraction", type=float, default=0.50)
    run.add_argument("--abort-owned-rss-gib", type=float, default=32.0)
    run.add_argument("--child-timeout-minutes", type=float, default=60.0)

    compile_parser = subparsers.add_parser("_compile")
    compile_parser.add_argument("--package", type=Path, required=True)
    compile_parser.add_argument("--compiled", type=Path, required=True)
    compile_parser.add_argument("--report", type=Path, required=True)

    load = subparsers.add_parser("_load")
    load_model = load.add_mutually_exclusive_group(required=True)
    load_model.add_argument("--compiled", type=Path)
    load_model.add_argument("--package", type=Path)
    load.add_argument("--compute-units", choices=("cpu-and-ne", "all"), required=True)
    load.add_argument("--repetitions", type=int, required=True)
    load.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def directory_stats(path: Path) -> dict[str, int]:
    logical_bytes = 0
    allocated_bytes = 0
    file_count = 0
    for root, _, files in os.walk(path):
        for name in files:
            stat = (Path(root) / name).stat()
            logical_bytes += stat.st_size
            allocated_bytes += stat.st_blocks * 512
            file_count += 1
    return {
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "file_count": file_count,
    }


def parse_stages(values: list[str]) -> list[tuple[int, Path]]:
    stages: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for value in values:
        sequence_text, separator, package_text = value.partition("=")
        if not separator:
            raise ValueError(f"stage must have TOKENS=PACKAGE form: {value}")
        sequence_length = int(sequence_text)
        if sequence_length not in STAGES:
            raise ValueError(f"unsupported sequence length: {sequence_length}")
        if sequence_length in seen:
            raise ValueError(f"duplicate sequence length: {sequence_length}")
        package = Path(package_text).expanduser().resolve()
        if not package.is_dir() or package.suffix != ".mlpackage":
            raise FileNotFoundError(f"Core ML package does not exist: {package}")
        seen.add(sequence_length)
        stages.append((sequence_length, package))
    return sorted(stages)


def ane_worker_processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            info = process.info
            command_line = info.get("cmdline") or []
            command = " ".join(command_line)
            executable = Path(command_line[0]).name if command_line else ""
            name = info.get("name") or ""
            if ANE_WORKER_MARKER in executable or ANE_WORKER_MARKER in name:
                matches.append({"pid": info["pid"], "name": name, "command": command})
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return matches


def require_clear_slot(where: str) -> None:
    matches = ane_worker_processes()
    if matches:
        raise RuntimeError(f"unexpected ANE worker {where}: {matches}")


def command_compile(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct

    require_clear_slot("before package compilation")
    if args.compiled.exists():
        shutil.rmtree(args.compiled) if args.compiled.is_dir() else args.compiled.unlink()
    args.compiled.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    compiled_path = Path(
        ct.utils.compile_model(str(args.package), destination_path=str(args.compiled))
    ).resolve()
    elapsed = time.perf_counter() - started
    require_clear_slot("after package compilation")
    return {
        "status": "passed",
        "operation": "MLModel.compileModel package-to-mlmodelc",
        "package": str(args.package.resolve()),
        "compiled_model": str(compiled_path),
        "timing_s": {"package_compilation_wall": elapsed},
        "package_size": directory_stats(args.package),
        "compiled_size": directory_stats(compiled_path),
    }


def coreml_compute_unit(name: str, ct: Any) -> Any:
    if name == "cpu-and-ne":
        return ct.ComputeUnit.CPU_AND_NE
    if name == "all":
        return ct.ComputeUnit.ALL
    raise ValueError(f"unsupported compute unit: {name}")


def measure_loads(
    model_path: Path,
    model_kind: str,
    compute_units: str,
    repetitions: int,
    *,
    model_factory: Callable[[str, Any], Any] | None = None,
) -> dict[str, Any]:
    import coremltools as ct

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    unit = coreml_compute_unit(compute_units, ct)
    if model_kind not in {"compiled", "package"}:
        raise ValueError(f"unsupported model kind: {model_kind}")
    if model_factory is None:
        if model_kind == "compiled":
            model_factory = lambda path, selected: ct.models.CompiledMLModel(
                path, compute_units=selected
            )
        else:
            model_factory = lambda path, selected: ct.models.MLModel(
                path, compute_units=selected
            )
    process = psutil.Process()
    loads = []
    for index in range(repetitions):
        require_clear_slot(f"before timed load {index + 1}")
        memory_before = memory_snapshot()
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model = model_factory(str(model_path), unit)
        wall = time.perf_counter() - started
        framework_ns = getattr(model, "load_duration_in_nano_seconds", None)
        if framework_ns is None:
            proxy = getattr(model, "_proxy", None)
            framework_ns = (
                proxy.get_load_duration_in_nano_seconds() if proxy is not None else None
            )
        rss_loaded = process.memory_info().rss
        memory_loaded = memory_snapshot()
        require_clear_slot(f"after timed load {index + 1}")
        framework_compiled_path = (
            getattr(model, "get_compiled_model_path")()
            if model_kind == "package"
            else str(model_path)
        )
        loads.append(
            {
                "ordinal": index + 1,
                "wall_s": wall,
                "framework_load_s": None if framework_ns is None else framework_ns / 1_000_000_000,
                "python_wrapper_overhead_s": (
                    None if framework_ns is None else wall - framework_ns / 1_000_000_000
                ),
                "process_rss_before_bytes": rss_before,
                "process_rss_loaded_bytes": rss_loaded,
                "system_memory_before": memory_before,
                "system_memory_loaded": memory_loaded,
                "framework_compiled_model_path": framework_compiled_path,
            }
        )
        del model
        gc.collect()
        time.sleep(1.0)
    return {
        "status": "passed",
        "source_model": str(model_path.resolve()),
        "model_kind": model_kind,
        "compute_units": compute_units,
        "same_process_repetitions": repetitions,
        "loads": loads,
        "release_between_loads": "Delete the MLModel object, run Python GC, then wait one second.",
    }


def command_load(args: argparse.Namespace) -> dict[str, Any]:
    if args.compiled is not None:
        return measure_loads(
            args.compiled, "compiled", args.compute_units, args.repetitions
        )
    if args.package is not None:
        return measure_loads(
            args.package, "package", args.compute_units, args.repetitions
        )
    raise AssertionError("a compiled model or package is required")


def child_command(script: Path, *arguments: str) -> list[str]:
    # Preserve the virtual-environment entry path; resolving its symlink loses site-packages.
    return [str(Path(sys.executable)), str(script), *arguments]


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.measurement_slot_authorized.strip():
        raise ValueError("measurement slot authorization must not be empty")
    stages = parse_stages(args.stage)
    if not args.placement_binary.is_file():
        raise FileNotFoundError(f"placement helper does not exist: {args.placement_binary}")
    require_clear_slot("at attribution preflight")
    initial_memory = memory_snapshot()
    minimum_available = int(args.minimum_available_gib * 1024**3)
    if initial_memory["available_bytes"] < minimum_available:
        raise RuntimeError(
            f"preflight available memory {initial_memory['available_bytes']} is below {minimum_available}"
        )

    root = args.artifacts.resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    guard = {
        "abort_available_bytes": int(args.abort_available_gib * 1024**3),
        "abort_free_bytes": int(args.abort_free_gib * 1024**3),
        "abort_wired_fraction": args.abort_wired_fraction,
        "abort_owned_rss_bytes": int(args.abort_owned_rss_gib * 1024**3),
        "timeout_s": args.child_timeout_minutes * 60.0,
    }
    report: dict[str, Any] = {
        "status": "running",
        "measurement_slot_authorized": args.measurement_slot_authorized,
        "initial_memory": initial_memory,
        "guard": guard,
        "stage_order": [sequence_length for sequence_length, _ in stages],
        "stages": [],
    }
    report_path = root / "run.json"
    write_json(report_path, report)
    try:
        for sequence_length, package in stages:
            require_clear_slot(f"before stage {sequence_length}")
            stage_root = root / f"seq{sequence_length}"
            stage_root.mkdir(parents=True, exist_ok=True)
            compiled = stage_root / "gte-modernbert.mlmodelc"
            direct_same_process_report = stage_root / "direct-package-same-process.json"
            direct_fresh_process_report = stage_root / "direct-package-fresh-process.json"
            compile_report = stage_root / "compile.json"
            same_process_report = stage_root / "same-process-cpu-and-ne.json"
            fresh_process_report = stage_root / "fresh-process-cpu-and-ne.json"
            all_report = stage_root / "fresh-process-all.json"
            invalidated_compiled = stage_root / "gte-modernbert-versioned.mlmodelc"
            invalidated_compile_report = stage_root / "versioned-compile.json"
            invalidated_load_report = stage_root / "versioned-fresh-process-cpu-and-ne.json"
            processes: dict[str, Any] = {}

            processes["direct_package_same_process_cpu_and_ne"] = run_guarded(
                child_command(
                    script,
                    "_load",
                    "--package",
                    str(package),
                    "--compute-units",
                    "cpu-and-ne",
                    "--repetitions",
                    "2",
                    "--report",
                    str(direct_same_process_report),
                ),
                stage_root / "direct-package-same-process.log",
                **guard,
            )
            processes["direct_package_fresh_process_cpu_and_ne"] = run_guarded(
                child_command(
                    script,
                    "_load",
                    "--package",
                    str(package),
                    "--compute-units",
                    "cpu-and-ne",
                    "--repetitions",
                    "1",
                    "--report",
                    str(direct_fresh_process_report),
                ),
                stage_root / "direct-package-fresh-process.log",
                **guard,
            )
            processes["compile"] = run_guarded(
                child_command(
                    script,
                    "_compile",
                    "--package",
                    str(package),
                    "--compiled",
                    str(compiled),
                    "--report",
                    str(compile_report),
                ),
                stage_root / "compile.log",
                **guard,
            )
            processes["same_process_cpu_and_ne"] = run_guarded(
                child_command(
                    script,
                    "_load",
                    "--compiled",
                    str(compiled),
                    "--compute-units",
                    "cpu-and-ne",
                    "--repetitions",
                    "2",
                    "--report",
                    str(same_process_report),
                ),
                stage_root / "same-process-cpu-and-ne.log",
                **guard,
            )
            processes["fresh_process_cpu_and_ne"] = run_guarded(
                child_command(
                    script,
                    "_load",
                    "--compiled",
                    str(compiled),
                    "--compute-units",
                    "cpu-and-ne",
                    "--repetitions",
                    "1",
                    "--report",
                    str(fresh_process_report),
                ),
                stage_root / "fresh-process-cpu-and-ne.log",
                **guard,
            )
            if sequence_length == 1024:
                processes["versioned_compile"] = run_guarded(
                    child_command(
                        script,
                        "_compile",
                        "--package",
                        str(package),
                        "--compiled",
                        str(invalidated_compiled),
                        "--report",
                        str(invalidated_compile_report),
                    ),
                    stage_root / "versioned-compile.log",
                    **guard,
                )
                processes["versioned_fresh_process_cpu_and_ne"] = run_guarded(
                    child_command(
                        script,
                        "_load",
                        "--compiled",
                        str(invalidated_compiled),
                        "--compute-units",
                        "cpu-and-ne",
                        "--repetitions",
                        "1",
                        "--report",
                        str(invalidated_load_report),
                    ),
                    stage_root / "versioned-fresh-process-cpu-and-ne.log",
                    **guard,
                )
            processes["fresh_process_all"] = run_guarded(
                child_command(
                    script,
                    "_load",
                    "--compiled",
                    str(compiled),
                    "--compute-units",
                    "all",
                    "--repetitions",
                    "1",
                    "--report",
                    str(all_report),
                ),
                stage_root / "fresh-process-all.log",
                **guard,
            )
            placement_reports = {}
            for compute_units in ("cpu-and-ne", "all"):
                placement_path = stage_root / f"placement-{compute_units}.json"
                processes[f"placement_{compute_units}"] = run_guarded(
                    [str(args.placement_binary), str(package), str(placement_path), compute_units],
                    stage_root / f"placement-{compute_units}.log",
                    **guard,
                )
                placement_reports[compute_units] = json.loads(
                    placement_path.read_text(encoding="utf-8")
                )

            stage_report = {
                "sequence_length": sequence_length,
                "package": str(package),
                "package_size": directory_stats(package),
                "direct_package_same_process_cpu_and_ne": json.loads(
                    direct_same_process_report.read_text(encoding="utf-8")
                ),
                "direct_package_fresh_process_cpu_and_ne": json.loads(
                    direct_fresh_process_report.read_text(encoding="utf-8")
                ),
                "compilation": json.loads(compile_report.read_text(encoding="utf-8")),
                "same_process_cpu_and_ne": json.loads(
                    same_process_report.read_text(encoding="utf-8")
                ),
                "fresh_process_cpu_and_ne": json.loads(
                    fresh_process_report.read_text(encoding="utf-8")
                ),
                "fresh_process_all": json.loads(all_report.read_text(encoding="utf-8")),
                "versioned_path_invalidation": (
                    {
                        "method": (
                            "Compile the unchanged package to a new app-owned .mlmodelc path; "
                            "do not touch private OS cache directories."
                        ),
                        "compilation": json.loads(
                            invalidated_compile_report.read_text(encoding="utf-8")
                        ),
                        "fresh_process_cpu_and_ne": json.loads(
                            invalidated_load_report.read_text(encoding="utf-8")
                        ),
                    }
                    if sequence_length == 1024
                    else None
                ),
                "placement": placement_reports,
                "processes": processes,
            }
            report["stages"].append(stage_report)
            write_json(stage_root / "summary.json", stage_report)
            write_json(report_path, report)
            require_clear_slot(f"after stage {sequence_length}")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    return report


def main() -> int:
    args = parse_args()
    if args.command == "run":
        result = command_run(args)
        report_path = args.artifacts.resolve() / "run.json"
    elif args.command == "_compile":
        result = command_compile(args)
        report_path = args.report
    elif args.command == "_load":
        result = command_load(args)
        report_path = args.report
    else:
        raise AssertionError(args.command)
    write_json(report_path, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
