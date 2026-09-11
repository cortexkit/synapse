#!/usr/bin/env python3
"""Run guarded, matched-boundary long-row measurements on ANE and Metal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from attribute_load import ANE_WORKER_MARKER, ane_worker_processes, require_clear_slot
from run_stages import memory_snapshot, run_guarded
from spike import read_jsonl, sha256_file, sha256_tree, write_json

STAGE_LENGTHS = (1024, 4096, 8192)
CASE_LENGTHS = (128, 1024, 4096, 8192)
EXPECTED_ROTATION_SHA256 = "5d2b2b0a351b6f0526314fb27f50e6d4830bf7efb6c1d60fe6964b24ea1c8f45"
EXPECTED_MODEL_SHA256 = "3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba"
EXPECTED_BUCKET_POLICY = 2
DEFAULT_BENCH_LOCK = Path("/tmp/synapse-benchmark.lock")
DEFAULT_MEASURE_LOCK = Path("/tmp/aft-measure.lock")
PROFILE_SHAPE = re.compile(r"bucket_select .*?shape=(\d+)x(\d+)")


@dataclass(frozen=True)
class StageSpec:
    sequence_length: int
    package: Path
    compiled: Path
    input_path: Path
    export_report: Path
    compile_report: Path
    package_sha256: str
    input_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--metal-binary", type=Path, required=True)
    run.add_argument("--metal-cache", type=Path, required=True)
    run.add_argument(
        "--ane-stage",
        action="append",
        required=True,
        metavar="TOKENS=PACKAGE|COMPILED|INPUT|EXPORT|COMPILE",
    )
    run.add_argument("--artifacts", type=Path, required=True)
    run.add_argument("--portable-report", type=Path, required=True)
    run.add_argument("--macmon", type=Path, default=Path("macmon"))
    run.add_argument("--measurement-slot-authorized", required=True)
    run.add_argument("--minimum-repetitions", type=int, default=9)
    run.add_argument("--minimum-duration-s", type=float, default=12.0)
    run.add_argument("--maximum-repetitions", type=int, default=5000)
    run.add_argument("--minimum-available-gib", type=float, default=32.0)
    run.add_argument("--abort-available-gib", type=float, default=16.0)
    run.add_argument("--abort-free-gib", type=float, default=0.0)
    run.add_argument("--abort-wired-fraction", type=float, default=0.50)
    run.add_argument("--abort-owned-rss-gib", type=float, default=32.0)
    run.add_argument("--child-timeout-minutes", type=float, default=30.0)
    run.add_argument("--max-idle-cpu-percent", type=float, default=15.0)
    run.add_argument("--max-idle-gpu-percent", type=float, default=5.0)
    run.add_argument(
        "--ambient-load",
        action="store_true",
        help="Use interleaved paired arms and record host load instead of claiming idle power.",
    )

    ane = subparsers.add_parser("_ane")
    ane.add_argument("--compiled", type=Path, required=True)
    ane.add_argument("--case", type=Path, required=True)
    ane.add_argument("--report", type=Path, required=True)
    ane.add_argument("--minimum-repetitions", type=int, required=True)
    ane.add_argument("--minimum-duration-s", type=float, required=True)
    ane.add_argument("--maximum-repetitions", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_stage_specs(values: list[str]) -> dict[int, StageSpec]:
    specs: dict[int, StageSpec] = {}
    for value in values:
        sequence_text, separator, paths_text = value.partition("=")
        if not separator:
            raise ValueError(f"stage must start with TOKENS=: {value}")
        sequence_length = int(sequence_text)
        if sequence_length not in STAGE_LENGTHS:
            raise ValueError(f"unsupported ANE stage: {sequence_length}")
        if sequence_length in specs:
            raise ValueError(f"duplicate ANE stage: {sequence_length}")
        path_fields = paths_text.split("|")
        if len(path_fields) != 5:
            raise ValueError(f"stage must contain five pipe-separated paths: {value}")
        package, compiled, input_path, export_path, compile_path = (
            Path(field).expanduser().resolve() for field in path_fields
        )
        for path, suffix, label in (
            (package, ".mlpackage", "source package"),
            (compiled, ".mlmodelc", "compiled model"),
        ):
            if not path.is_dir() or path.suffix != suffix:
                raise FileNotFoundError(f"{label} is missing: {path}")
        for path, label in (
            (input_path, "input JSONL"),
            (export_path, "export report"),
            (compile_path, "compile report"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} is missing: {path}")

        export = load_json(export_path)
        compile_report = load_json(compile_path)
        rotation = export.get("rotation")
        model = export.get("model")
        if export.get("sequence_length") != sequence_length:
            raise ValueError(f"export stage mismatch for {sequence_length}")
        if not isinstance(rotation, dict) or rotation.get("kind") != "hadamard":
            raise ValueError(f"stage {sequence_length} is not Hadamard-rotated")
        if rotation.get("seed") != 0 or rotation.get("matrix_sha256") != EXPECTED_ROTATION_SHA256:
            raise ValueError(f"stage {sequence_length} rotation identity differs")
        if not isinstance(model, dict) or model.get("model.safetensors") != EXPECTED_MODEL_SHA256:
            raise ValueError(f"stage {sequence_length} model identity differs")
        package_digest = sha256_tree(package)
        input_digest = sha256_file(input_path)
        if export.get("package_sha256") != package_digest:
            raise ValueError(f"stage {sequence_length} package digest differs")
        if export.get("input_sha256") != input_digest:
            raise ValueError(f"stage {sequence_length} input digest differs")
        if Path(str(compile_report.get("package", ""))).resolve() != package:
            raise ValueError(f"stage {sequence_length} compile report names another package")
        if Path(str(compile_report.get("compiled_model", ""))).resolve() != compiled:
            raise ValueError(f"stage {sequence_length} compile report names another compiled model")
        specs[sequence_length] = StageSpec(
            sequence_length=sequence_length,
            package=package,
            compiled=compiled,
            input_path=input_path,
            export_report=export_path,
            compile_report=compile_path,
            package_sha256=package_digest,
            input_sha256=input_digest,
        )
    if set(specs) != set(STAGE_LENGTHS):
        raise ValueError(f"ANE stages must be exactly {STAGE_LENGTHS}")
    return specs


def active_full_context_row(spec: StageSpec) -> list[int]:
    rows = read_jsonl(spec.input_path)
    matches = [row for row in rows if row.get("id") == "full-context"]
    if len(matches) != 1:
        raise ValueError(f"stage {spec.sequence_length} must contain one full-context row")
    row = matches[0]
    ids = row.get("input_ids")
    mask = row.get("attention_mask")
    if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask):
        raise ValueError(f"stage {spec.sequence_length} input arrays are malformed")
    active = sum(int(value) for value in mask)
    if mask != [1] * active + [0] * (len(mask) - active):
        raise ValueError(f"stage {spec.sequence_length} mask is not prefix-active")
    if active != spec.sequence_length or len(ids) != spec.sequence_length:
        raise ValueError(f"stage {spec.sequence_length} full row is not full-context")
    result = [int(value) for value in ids[:active]]
    if len(set(result)) < 16:
        raise ValueError(f"stage {spec.sequence_length} row lacks token diversity")
    return result


def build_cases(specs: dict[int, StageSpec], destination: Path) -> dict[int, dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    stage_rows = {length: active_full_context_row(spec) for length, spec in specs.items()}
    short_source = stage_rows[1024]
    rows = {
        128: short_source[:127] + [short_source[-1]],
        1024: stage_rows[1024],
        4096: stage_rows[4096],
        8192: stage_rows[8192],
    }
    cases: dict[int, dict[str, Any]] = {}
    for length, ids in rows.items():
        if len(ids) != length:
            raise ValueError(f"case {length} has {len(ids)} tokens")
        stage = 1024 if length == 128 else length
        value = {
            "id": f"full-context-prefix-{length}",
            "real_tokens": length,
            "ane_sequence_length": stage,
            "input_ids": ids,
            "input_ids_sha256": canonical_sha256(ids),
        }
        path = destination / f"case-{length}.json"
        write_json(path, value)
        value["path"] = path
        cases[length] = value
    return cases


def runner_worker_processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            info = process.info
            command_line = info.get("cmdline") or []
            command = " ".join(command_line)
            if "Runner.Worker" in command:
                matches.append({"pid": info["pid"], "name": info.get("name") or ""})
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return matches


def acquire_bench_lock() -> Path:
    measure_lock = Path(os.environ.get("SYNAPSE_CAMPAIGN_MEASURE_LOCK", str(DEFAULT_MEASURE_LOCK)))
    if measure_lock.exists():
        raise RuntimeError(f"measurement lock is already present: {measure_lock}")
    lock = Path(os.environ.get("SYNAPSE_CAMPAIGN_BENCH_LOCK", str(DEFAULT_BENCH_LOCK)))
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"benchmark lock is already present: {lock}") from error
    workers = runner_worker_processes()
    if workers:
        lock.rmdir()
        raise RuntimeError(f"Runner.Worker is active: {workers}")
    return lock


def idle_preflight(macmon: Path, max_cpu: float, max_gpu: float) -> dict[str, Any]:
    completed = subprocess.run(
        [str(macmon), "pipe", "-s", "12", "-i", "500"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"macmon idle preflight exited {completed.returncode}: {completed.stderr}")
    samples = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(samples) < 6:
        raise RuntimeError("macmon idle preflight returned too few samples")
    measured = samples[1:]
    cpu = statistics.fmean(float(row.get("cpu_usage_pct", 0.0)) * 100.0 for row in measured)
    gpu = statistics.fmean(float(row.get("gpu_usage", [0.0, 0.0])[1]) * 100.0 for row in measured)
    if cpu > max_cpu or gpu > max_gpu:
        raise RuntimeError(
            f"machine not idle: CPU {cpu:.2f}% > {max_cpu:.2f}% or GPU {gpu:.2f}% > {max_gpu:.2f}%"
        )
    return {"sample_count": len(measured), "cpu_mean_percent": cpu, "gpu_mean_percent": gpu}


def ambient_snapshot() -> dict[str, Any]:
    cpu = psutil.cpu_times_percent(interval=1.0)
    return {
        "captured_unix_s": time.time(),
        "load_average_1m": os.getloadavg()[0],
        "cpu_idle_percent": float(cpu.idle),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_power(
    path: Path, measurement_window: list[float], *, reject_gpu_activity: bool
) -> dict[str, Any]:
    start, end = measurement_window
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        timestamp = datetime.fromisoformat(str(row["timestamp"])).timestamp()
        if start <= timestamp <= end:
            rows.append(row)
    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} macmon samples fell inside the timed window")
    metrics: dict[str, dict[str, float]] = {}
    for name in ("ane_power", "gpu_power", "cpu_power", "all_power", "ram_power", "sys_power"):
        values = [float(row.get(name, 0.0)) for row in rows]
        metrics[name] = {
            "mean_w": statistics.fmean(values),
            "p50_w": statistics.median(values),
            "p95_w": percentile(values, 0.95),
            "max_w": max(values),
        }
    gpu_usage = statistics.fmean(float(row.get("gpu_usage", [0.0, 0.0])[1]) for row in rows)
    if reject_gpu_activity and gpu_usage > 0.05:
        raise RuntimeError(
            f"ANE sample is contended: mean GPU utilization was {gpu_usage * 100.0:.2f}%"
        )
    return {
        "source": "macmon 10 Hz host PMU telemetry",
        "sample_count": len(rows),
        "duration_s": end - start,
        "gpu_usage_mean_percent": gpu_usage * 100.0,
        "power": metrics,
    }


def run_powered_guarded(
    command: list[str],
    report_path: Path,
    log_path: Path,
    macmon_path: Path,
    macmon_output: Path,
    guard: dict[str, float | int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    macmon_output.parent.mkdir(parents=True, exist_ok=True)
    with macmon_output.open("w", encoding="utf-8") as output:
        macmon = subprocess.Popen(
            [str(macmon_path), "-i", "100", "pipe"],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            deadline = time.monotonic() + 5.0
            while macmon_output.stat().st_size == 0 and time.monotonic() < deadline:
                time.sleep(0.1)
            if macmon_output.stat().st_size == 0:
                raise RuntimeError("macmon did not produce a startup sample")
            process = run_guarded(
                command,
                log_path,
                abort_available_bytes=int(guard["abort_available_bytes"]),
                abort_free_bytes=int(guard["abort_free_bytes"]),
                abort_wired_fraction=float(guard["abort_wired_fraction"]),
                abort_owned_rss_bytes=int(guard["abort_owned_rss_bytes"]),
                timeout_s=float(guard["timeout_s"]),
            )
        finally:
            macmon.terminate()
            try:
                macmon.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                macmon.kill()
                macmon.wait()
    return process, load_json(report_path)


def monitor_ane_worker(stop: threading.Event, found: list[dict[str, Any]]) -> None:
    while not stop.wait(0.1):
        matches = ane_worker_processes()
        if matches:
            found.extend(matches)
            return


def command_ane(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct

    case = load_json(args.case)
    ids = case.get("input_ids")
    real_tokens = case.get("real_tokens")
    sequence_length = case.get("ane_sequence_length")
    if not isinstance(ids, list) or not isinstance(real_tokens, int) or not isinstance(sequence_length, int):
        raise ValueError("ANE case fields are malformed")
    if len(ids) != real_tokens or sequence_length < real_tokens:
        raise ValueError("ANE case token accounting is inconsistent")
    pad_token_id = 50283
    input_dict = {
        "input_ids": np.asarray([ids + [pad_token_id] * (sequence_length - real_tokens)], dtype=np.int32),
        "attention_mask": np.asarray(
            [[1] * real_tokens + [0] * (sequence_length - real_tokens)], dtype=np.int32
        ),
    }

    require_clear_slot("before compiled ANE model load")
    load_started = time.perf_counter()
    model = ct.models.CompiledMLModel(str(args.compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_wall_s = time.perf_counter() - load_started
    require_clear_slot("after compiled ANE model load")

    first_started = time.perf_counter()
    first = np.asarray(model.predict(input_dict)["embedding"], dtype=np.float32).reshape(-1)
    first_use_wall_s = time.perf_counter() - first_started
    warmup_started = time.perf_counter()
    warmup = np.asarray(model.predict(input_dict)["embedding"], dtype=np.float32).reshape(-1)
    warmup_wall_s = time.perf_counter() - warmup_started
    if not np.array_equal(first, warmup):
        raise RuntimeError("ANE first-use and warmup vectors differ")

    require_clear_slot("immediately before timed ANE prediction block")
    stop = threading.Event()
    unexpected_workers: list[dict[str, Any]] = []
    monitor = threading.Thread(target=monitor_ane_worker, args=(stop, unexpected_workers), daemon=True)
    monitor.start()
    measurement_start_unix_s = time.time()
    measurement_started = time.perf_counter()
    walls: list[float] = []
    determinism_max_abs = 0.0
    try:
        while len(walls) < args.maximum_repetitions:
            started = time.perf_counter()
            vector = np.asarray(model.predict(input_dict)["embedding"], dtype=np.float32).reshape(-1)
            walls.append(time.perf_counter() - started)
            determinism_max_abs = max(determinism_max_abs, float(np.max(np.abs(first - vector))))
            if unexpected_workers:
                raise RuntimeError(f"unexpected ANE worker during timed block: {unexpected_workers}")
            if len(walls) >= args.minimum_repetitions and time.perf_counter() - measurement_started >= args.minimum_duration_s:
                break
    finally:
        measurement_end_unix_s = time.time()
        stop.set()
        monitor.join(timeout=2.0)
    require_clear_slot("immediately after timed ANE prediction block")
    if time.perf_counter() - measurement_started < args.minimum_duration_s:
        raise RuntimeError("maximum repetitions reached before minimum duration")
    if determinism_max_abs != 0.0:
        raise RuntimeError("ANE measured vectors were not byte-identical")

    digest = hashlib.sha256(first.astype("<f4", copy=False).tobytes()).hexdigest()
    return {
        "case_id": case.get("id"),
        "lane": "rotated-coreml-ane",
        "compute_units": "CPU_AND_NE (GPU excluded)",
        "real_tokens": real_tokens,
        "executed_shape": [1, sequence_length],
        "load_wall_s": load_wall_s,
        "first_use_wall_s": first_use_wall_s,
        "warmup_wall_s": warmup_wall_s,
        "measured_wall_s": walls,
        "measurement_window_unix_s": [measurement_start_unix_s, measurement_end_unix_s],
        "vector_sha256": digest,
        "repeated_determinism_max_abs": determinism_max_abs,
        "vector": first.tolist(),
        "unexpected_ane_worker_detected": False,
    }


def parse_metal_shape(log_path: Path) -> list[int]:
    shapes = {
        (int(match.group(1)), int(match.group(2)))
        for match in PROFILE_SHAPE.finditer(log_path.read_text(encoding="utf-8", errors="replace"))
    }
    if len(shapes) != 1:
        raise RuntimeError(f"Metal profile did not prove one executed shape: {sorted(shapes)}")
    batch, sequence = next(iter(shapes))
    return [batch, sequence]


def timing_summary(walls: list[float], real_tokens: int) -> dict[str, Any]:
    if not walls or any(not math.isfinite(value) or value <= 0.0 for value in walls):
        raise ValueError("timing samples must be finite and positive")
    median = statistics.median(walls)
    return {
        "samples": len(walls),
        "median_row_wall_ms": median * 1000.0,
        "mean_row_wall_ms": statistics.fmean(walls) * 1000.0,
        "p95_row_wall_ms": percentile(walls, 0.95) * 1000.0,
        "min_row_wall_ms": min(walls) * 1000.0,
        "max_row_wall_ms": max(walls) * 1000.0,
        "real_tokens_per_s_at_median": real_tokens / median,
    }


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must have equal nonzero dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("vectors must be nonzero")
    return dot / (left_norm * right_norm)


def host_report() -> dict[str, Any]:
    def output(*command: str) -> str:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "machine": output("sysctl", "-n", "hw.model"),
        "chip": output("sysctl", "-n", "machdep.cpu.brand_string"),
        "physical_memory_bytes": int(output("sysctl", "-n", "hw.memsize")),
        "macos": output("sw_vers", "-productVersion"),
        "macos_build": output("sw_vers", "-buildVersion"),
        "python": platform.python_version(),
    }


def capture_power_state() -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, check=True
    )
    text = completed.stdout.strip()
    source = re.search(r"Now drawing from '([^']+)'", text)
    battery = re.search(r"(?<!\d)(\d{1,3})%", text)
    if source is None or battery is None:
        raise RuntimeError(f"could not parse power state: {text}")
    return {"power_source": source.group(1), "battery_percent": int(battery.group(1))}


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_repetitions <= 0 or args.minimum_duration_s <= 0.0:
        raise ValueError("measurement repetition and duration bounds must be positive")
    if args.maximum_repetitions < args.minimum_repetitions:
        raise ValueError("maximum repetitions must cover minimum repetitions")
    model = args.model.expanduser().resolve()
    binary = args.metal_binary.expanduser().resolve()
    macmon = Path(shutil.which(str(args.macmon)) or args.macmon).expanduser().resolve()
    if not model.is_dir() or not (model / "model.safetensors").is_file():
        raise FileNotFoundError(f"model snapshot is missing: {model}")
    if sha256_file(model / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("Metal model snapshot digest differs")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Metal probe binary is missing or not executable: {binary}")
    if not macmon.is_file() or not os.access(macmon, os.X_OK):
        raise FileNotFoundError(f"macmon is missing or not executable: {macmon}")

    specs = parse_stage_specs(args.ane_stage)
    artifacts = args.artifacts.expanduser().resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    cases = build_cases(specs, artifacts / "cases")
    initial_memory = memory_snapshot()
    minimum_available = int(args.minimum_available_gib * 1024**3)
    if initial_memory["available_bytes"] < minimum_available:
        raise RuntimeError(
            f"preflight available memory {initial_memory['available_bytes']} is below {minimum_available}"
        )
    require_clear_slot("before acquiring the matched measurement slot")
    power_state = capture_power_state()
    if power_state["power_source"].casefold() == "battery power" and power_state["battery_percent"] < 20:
        raise RuntimeError("refusing measurement below 20% battery")

    guard: dict[str, float | int] = {
        "abort_available_bytes": int(args.abort_available_gib * 1024**3),
        "abort_free_bytes": int(args.abort_free_gib * 1024**3),
        "abort_wired_fraction": args.abort_wired_fraction,
        "abort_owned_rss_bytes": int(args.abort_owned_rss_gib * 1024**3),
        "timeout_s": args.child_timeout_minutes * 60.0,
    }
    order = [
        (length, round_number, lane)
        for length in CASE_LENGTHS
        for round_number in (1, 2)
        for lane in ("ane", "metal")
    ]
    raw_results: dict[str, dict[str, Any]] = {}
    condition_reports: dict[str, dict[str, Any]] = {}
    lock = acquire_bench_lock()
    try:
        for length, round_number, lane in order:
            label = f"{length}-r{round_number}-{lane}"
            require_clear_slot(f"before {label}")
            condition_reports[label] = {"before": ambient_snapshot()}
            if not args.ambient_load:
                condition_reports[label]["idle_gate"] = idle_preflight(
                    macmon, args.max_idle_cpu_percent, args.max_idle_gpu_percent
                )
            case_path = Path(cases[length]["path"])
            report_path = artifacts / f"{label}.json"
            log_path = artifacts / f"{label}.log"
            macmon_path = artifacts / f"{label}-macmon.jsonl"
            if lane == "ane":
                stage = int(cases[length]["ane_sequence_length"])
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_ane",
                    "--compiled",
                    str(specs[stage].compiled),
                    "--case",
                    str(case_path),
                    "--report",
                    str(report_path),
                    "--minimum-repetitions",
                    str(args.minimum_repetitions),
                    "--minimum-duration-s",
                    str(args.minimum_duration_s),
                    "--maximum-repetitions",
                    str(args.maximum_repetitions),
                ]
            else:
                command = [
                    "/usr/bin/env",
                    "SYNAPSE_EMBED_PROFILE=1",
                    str(binary),
                    str(model),
                    str(case_path),
                    str(report_path),
                    str(args.metal_cache.expanduser().resolve()),
                    str(args.minimum_repetitions),
                    str(args.minimum_duration_s),
                    str(args.maximum_repetitions),
                ]
            if args.ambient_load:
                process = run_guarded(
                    command,
                    log_path,
                    abort_available_bytes=int(guard["abort_available_bytes"]),
                    abort_free_bytes=int(guard["abort_free_bytes"]),
                    abort_wired_fraction=float(guard["abort_wired_fraction"]),
                    abort_owned_rss_bytes=int(guard["abort_owned_rss_bytes"]),
                    timeout_s=float(guard["timeout_s"]),
                )
                report = load_json(report_path)
            else:
                process, report = run_powered_guarded(
                    command, report_path, log_path, macmon, macmon_path, guard
                )
            condition_reports[label]["after"] = ambient_snapshot()
            require_clear_slot(f"after {label}")
            if lane == "metal":
                report["executed_shape"] = parse_metal_shape(log_path)
                if report.get("bucket_policy_version") != EXPECTED_BUCKET_POLICY:
                    raise RuntimeError("Metal probe did not use bucket policy v2")
                walls = report.pop("measured_engine_wall_s")
                report["measured_wall_s"] = walls
            window = report.get("measurement_window_unix_s")
            if not isinstance(window, list) or len(window) != 2:
                raise RuntimeError(f"{label} did not report a measurement window")
            report["power"] = (
                {
                    "status": "not_measured",
                    "reason": "Concurrent foreign CPU work made host-level power unattributable to either lane.",
                }
                if args.ambient_load
                else summarize_power(
                    macmon_path,
                    [float(window[0]), float(window[1])],
                    reject_gpu_activity=lane == "ane",
                )
            )
            report["guard"] = {
                "peak_owned_rss_bytes": process["peak_owned_rss_bytes"],
                "minimum_system_available_bytes": process["minimum_system_available_bytes"],
                "maximum_system_wired_bytes": process["maximum_system_wired_bytes"],
                "abort_reason": process["abort_reason"],
            }
            raw_results[label] = report
    finally:
        lock.rmdir()

    rows = []
    for length in CASE_LENGTHS:
        ane_arms = [raw_results[f"{length}-r{round_number}-ane"] for round_number in (1, 2)]
        metal_arms = [raw_results[f"{length}-r{round_number}-metal"] for round_number in (1, 2)]
        ane_walls = [
            float(value) for arm in ane_arms for value in arm["measured_wall_s"]
        ]
        metal_walls = [
            float(value) for arm in metal_arms for value in arm["measured_wall_s"]
        ]
        ane_summary = timing_summary(ane_walls, length)
        metal_summary = timing_summary(metal_walls, length)
        ane_shape = [int(value) for value in ane_arms[0]["executed_shape"]]
        metal_shape = [int(value) for value in metal_arms[0]["executed_shape"]]
        if any(arm["executed_shape"] != ane_shape for arm in ane_arms):
            raise RuntimeError(f"ANE shape changed between {length}-token arms")
        if any(arm["executed_shape"] != metal_shape for arm in metal_arms):
            raise RuntimeError(f"Metal shape changed between {length}-token arms")
        if len({str(arm["vector_sha256"]) for arm in ane_arms}) != 1:
            raise RuntimeError(f"ANE vectors changed between {length}-token arms")
        if len({str(arm["vector_sha256"]) for arm in metal_arms}) != 1:
            raise RuntimeError(f"Metal vectors changed between {length}-token arms")
        paired_ratios = [
            statistics.median(ane_arms[index]["measured_wall_s"])
            / statistics.median(metal_arms[index]["measured_wall_s"])
            for index in range(2)
        ]
        rows.append(
            {
                "real_tokens": length,
                "input_ids_sha256": cases[length]["input_ids_sha256"],
                "ane": {
                    "executed_batch_shape": ane_shape,
                    "padded_tokens": ane_shape[0] * ane_shape[1],
                    "timing": ane_summary,
                    "arms": [
                        {
                            "round": index + 1,
                            "timing": timing_summary(
                                [float(value) for value in arm["measured_wall_s"]], length
                            ),
                            "ambient_conditions": condition_reports[f"{length}-r{index + 1}-ane"],
                        }
                        for index, arm in enumerate(ane_arms)
                    ],
                    "power": ane_arms[0]["power"],
                    "vector_sha256": ane_arms[0]["vector_sha256"],
                    "guard": [arm["guard"] for arm in ane_arms],
                },
                "metal": {
                    "executed_batch_shape": metal_shape,
                    "padded_tokens": metal_shape[0] * metal_shape[1],
                    "timing": metal_summary,
                    "arms": [
                        {
                            "round": index + 1,
                            "timing": timing_summary(
                                [float(value) for value in arm["measured_wall_s"]], length
                            ),
                            "ambient_conditions": condition_reports[f"{length}-r{index + 1}-metal"],
                        }
                        for index, arm in enumerate(metal_arms)
                    ],
                    "power": metal_arms[0]["power"],
                    "vector_sha256": metal_arms[0]["vector_sha256"],
                    "guard": [arm["guard"] for arm in metal_arms],
                },
                "vector_agreement_diagnostic": {
                    "cosine": cosine(
                        [float(value) for value in ane_arms[0]["vector"]],
                        [float(value) for value in metal_arms[0]["vector"]],
                    ),
                    "max_abs": max(
                        abs(float(left) - float(right))
                        for left, right in zip(ane_arms[0]["vector"], metal_arms[0]["vector"])
                    ),
                    "gate": False,
                },
                "metal_over_ane_median_speed_ratio": (
                    ane_summary["median_row_wall_ms"] / metal_summary["median_row_wall_ms"]
                ),
                "paired_round_speed_ratios": paired_ratios,
                "paired_round_median_speed_ratio": statistics.median(paired_ratios),
            }
        )

    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    portable = {
        "status": "passed_matched_long_row_measurement",
        "measured_at": time.strftime("%Y-%m-%d", time.localtime()),
        "source_revision": source_revision,
        "hardware": host_report(),
        "power_state": power_state,
        "measurement_slot": args.measurement_slot_authorized,
        "model": {
            "snapshot": model.name,
            "model_safetensors_sha256": EXPECTED_MODEL_SHA256,
            "ane_rotation": {
                "kind": "hadamard",
                "seed": 0,
                "matrix_sha256": EXPECTED_ROTATION_SHA256,
            },
            "metal_dtype": "f16",
            "metal_bucket_policy_version": EXPECTED_BUCKET_POLICY,
        },
        "boundaries": {
            "shared_input": "Identical pre-tokenized active token IDs; tokenization and text transport are outside both timers.",
            "ane": "Caller wall around CompiledMLModel.predict on a prebuilt padded input dictionary through normalized embedding return. Stable-path load, first use, warmup, worker checks, and IPC are outside the timer.",
            "metal": "Caller wall around production OwnedMetalEmbedEngine.embed_batch on a prebuilt TokenBatch through bucket selection, padding/mask construction, MPSGraph execution, readback, and normalized CLS return. Load, first use, warmup, and IPC are outside the timer.",
            "unavoidable_difference": "The rotated ANE package is a spike artifact, not a production worker lane, so no honest common worker-IPC boundary exists. ANE includes Python/Core ML bridge overhead; Metal includes Rust bucket planning. Both numbers use the nearest in-process complete embedding-call wall boundary.",
        },
        "measurement_protocol": {
            "order": [
                f"{length}-r{round_number}-{lane}"
                for length, round_number, lane in order
            ],
            "minimum_repetitions": args.minimum_repetitions,
            "minimum_timed_duration_s": args.minimum_duration_s,
            "ambient_load": args.ambient_load,
            "condition_capture": "One-minute load average and CPU idle percentage were sampled for one second immediately before and after every lane arm.",
            "conditions": condition_reports,
            "power": (
                "Not measured: concurrent foreign CPU work made host-level power unattributable. ANE's efficiency case remains unmeasured."
                if args.ambient_load
                else "macmon 10 Hz host PMU samples were restricted to each timed prediction block."
            ),
            "contention": "The paired arms were interleaved ANE, Metal, ANE, Metal at every length so ambient drift landed on both lanes. A benchmark lock excluded other registered runs; Runner.Worker was absent; every ANE timed block was checked before, monitored during, and checked after for ck-synapse-worker-ane. Absolute values are ambient-load observations; the paired ratio is the comparison result.",
        },
        "ane_packages": [
            {
                "sequence_length": length,
                "package_sha256": specs[length].package_sha256,
                "input_sha256": specs[length].input_sha256,
                "compiled_stable_path_verified_against_compile_report": True,
            }
            for length in STAGE_LENGTHS
        ],
        "rows": rows,
        "safety": {
            "initial_memory": initial_memory,
            "minimum_available_preflight_bytes": minimum_available,
            **guard,
            "one_model_process_at_a_time": True,
            "unexpected_ane_worker_detected": False,
            "guard_abort_count": 0,
            "limitation": "Owned RSS and system wired/available observations cannot guarantee that ANE allocation is bounded or prevent workstation pressure.",
        },
    }
    write_json(args.portable_report, portable)
    raw = {
        "portable_report": str(args.portable_report.resolve()),
        "stage_specs": {
            length: {
                "sequence_length": spec.sequence_length,
                "package": str(spec.package),
                "compiled": str(spec.compiled),
                "input_path": str(spec.input_path),
                "export_report": str(spec.export_report),
                "compile_report": str(spec.compile_report),
                "package_sha256": spec.package_sha256,
                "input_sha256": spec.input_sha256,
            }
            for length, spec in specs.items()
        },
        "cases": {
            length: {key: str(value) if isinstance(value, Path) else value for key, value in case.items()}
            for length, case in cases.items()
        },
        "results": raw_results,
    }
    write_json(artifacts / "run.json", raw)
    return portable


def main() -> int:
    args = parse_args()
    try:
        report = command_ane(args) if args.command == "_ane" else command_run(args)
        if args.command == "_ane":
            write_json(args.report, report)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
