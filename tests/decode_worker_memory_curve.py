#!/usr/bin/env python3
"""Measure one decode worker across repeated GENERATE requests on macOS.

The curve records process RSS, task phys_footprint, and system wired bytes after
LOAD and after every request. Run one configuration per invocation; the worker
is always terminated on exit and the wired-growth guard aborts a leaking run
before it can consume the machine.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import statistics
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

GIB = 1024**3
RUSAGE_INFO_V4 = 4


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_checkpoint = os.environ.get("SYNAPSE_OWNED_DECODE_QWEN3_0_6B")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--worker",
        type=Path,
        default=root / "target/release/ck-synapse-worker-decode",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(default_checkpoint) if default_checkpoint else None,
        required=default_checkpoint is None,
    )
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1, choices=(1, 64))
    parser.add_argument("--weight-quant", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--chain-k", type=int, choices=(1, 16), default=1)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--constraint-schema")
    parser.add_argument(
        "--constraint-compiler",
        type=Path,
        default=root / "target/release/compile_constraint",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-samples", type=int, default=5)
    parser.add_argument("--max-wired-growth-gib", type=float, default=12.0)
    return parser.parse_args()


def load_machine_driver(root: Path) -> Any:
    path = root / "tests/ane-prefill-certification/machine_driver.py"
    spec = importlib.util.spec_from_file_location("ane_prefill_machine_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import machine driver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_memory(pid: int) -> tuple[int, int]:
    """Return resident_size and phys_footprint from proc_pid_rusage."""

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    proc_pid_rusage = libproc.proc_pid_rusage
    proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    proc_pid_rusage.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(512)
    if proc_pid_rusage(pid, RUSAGE_INFO_V4, buffer) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # rusage_info_v4 starts with a 16-byte UUID, then u64 counters. Resident
    # size and physical footprint are counters 7 and 8 respectively.
    resident_size = struct.unpack_from("=Q", buffer, 64)[0]
    phys_footprint = struct.unpack_from("=Q", buffer, 72)[0]
    return resident_size, phys_footprint


def system_wired_bytes() -> int:
    output = subprocess.check_output(["vm_stat"], text=True)
    lines = output.splitlines()
    if not lines:
        raise RuntimeError("vm_stat returned no output")
    page_size = int(lines[0].split("page size of ", 1)[1].split(" bytes", 1)[0])
    for line in lines[1:]:
        if line.startswith("Pages wired down:"):
            pages = int(line.split(":", 1)[1].strip().rstrip("."))
            return pages * page_size
    raise RuntimeError("vm_stat did not report wired pages")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def deterministic_prompt(length: int) -> list[int]:
    seed = hashlib.sha256(f"decode-worker-memory-w{length}".encode()).digest()
    return [
        1024 + ((seed[index % len(seed)] * 257 + index * 313) % 120_000)
        for index in range(length)
    ]


def descendant_pids(parent_pid: int) -> list[int]:
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        pid_text, parent_text = line.split()
        children.setdefault(int(parent_text), []).append(int(pid_text))
    descendants: list[int] = []
    pending = list(children.get(parent_pid, []))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children.get(pid, []))
    return descendants


def sample_memory(pid: int, request: int) -> dict[str, int]:
    resident_size, phys_footprint = task_memory(pid)
    descendant_resident = 0
    descendant_footprint = 0
    descendants = descendant_pids(pid)
    for descendant in descendants:
        try:
            child_resident, child_footprint = task_memory(descendant)
        except OSError:
            continue
        descendant_resident += child_resident
        descendant_footprint += child_footprint
    return {
        "request": request,
        "resident_bytes": resident_size,
        "phys_footprint_bytes": phys_footprint,
        "descendant_count": len(descendants),
        "descendant_resident_bytes": descendant_resident,
        "descendant_phys_footprint_bytes": descendant_footprint,
        "process_tree_resident_bytes": resident_size + descendant_resident,
        "process_tree_phys_footprint_bytes": phys_footprint + descendant_footprint,
        "system_wired_bytes": system_wired_bytes(),
    }


def slope(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = statistics.fmean(values)
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator


def summarize(
    config: dict[str, Any], samples: list[dict[str, int]], warm_samples: int
) -> dict[str, Any]:
    start = min(warm_samples, len(samples) - 1)
    steady = samples[start:]
    baseline = samples[0]
    final = samples[-1]
    metrics = (
        "resident_bytes",
        "phys_footprint_bytes",
        "descendant_resident_bytes",
        "descendant_phys_footprint_bytes",
        "process_tree_resident_bytes",
        "process_tree_phys_footprint_bytes",
        "system_wired_bytes",
    )
    request_span = steady[-1]["request"] - steady[0]["request"]
    return {
        "type": "summary",
        "config": config,
        "sample_count": len(samples),
        "steady_curve_start_request": steady[0]["request"],
        "delta_bytes": {metric: final[metric] - baseline[metric] for metric in metrics},
        "steady_linear_slope_bytes_per_request": {
            metric: slope([row[metric] for row in steady]) for metric in metrics
        },
        "steady_endpoint_bytes_per_request": {
            metric: (steady[-1][metric] - steady[0][metric]) / max(request_span, 1)
            for metric in metrics
        },
        "steady_median_step_bytes": {
            metric: statistics.median(
                later[metric] - earlier[metric]
                for earlier, later in zip(steady, steady[1:])
            )
            if len(steady) > 1
            else 0
            for metric in metrics
        },
    }


def main() -> int:
    args = parse_args()
    if sys.platform != "darwin":
        raise RuntimeError(
            "decode worker memory curves require macOS task and vm statistics"
        )
    if args.iterations < 1 or args.prompt_length < 1:
        raise ValueError("iterations and prompt length must be positive")
    if (args.compiled is None) != (args.sidecar is None):
        raise ValueError("--compiled and --sidecar must be supplied together")
    if args.constraint_schema is not None and not args.constraint_compiler.is_file():
        raise FileNotFoundError(
            f"constraint compiler is missing: {args.constraint_compiler}"
        )
    if not args.worker.is_file():
        raise FileNotFoundError(f"worker binary is missing: {args.worker}")
    checkpoint = args.checkpoint.resolve()
    compiled = args.compiled.resolve() if args.compiled else None
    sidecar = args.sidecar.resolve() if args.sidecar else args.worker.resolve()
    driver = load_machine_driver(args.root.resolve())
    config = {
        "label": args.label,
        "worker_sha256": file_sha256(args.worker),
        "iterations": args.iterations,
        "prompt_length": args.prompt_length,
        "max_tokens": args.max_tokens,
        "weight_quant": args.weight_quant,
        "chain_k": args.chain_k,
        "prefill": "ane-split" if compiled else "gpu",
        "constrained": args.constraint_schema is not None,
    }
    client = driver.WorkerClient(
        args.worker.resolve(),
        sidecar,
        args.constraint_compiler.resolve(),
        checkpoint,
        compiled,
        args.prompt_length,
        "f16-step" if args.weight_quant == "f16" else "q8-step",
        args.chain_k,
    )
    samples: list[dict[str, int]] = []
    token_digest: str | None = None
    try:
        pid = client._child.pid
        baseline = sample_memory(pid, 0)
        samples.append(baseline)
        print(json.dumps({"type": "sample", **baseline}), flush=True)
        prompt = deterministic_prompt(args.prompt_length)
        max_wired_growth = int(args.max_wired_growth_gib * GIB)
        for request in range(1, args.iterations + 1):
            tokens, _, _ = client.generate(
                prompt, args.max_tokens, grammar=args.constraint_schema
            )
            digest = hashlib.sha256(struct.pack(f"={len(tokens)}I", *tokens)).hexdigest()
            if token_digest is None:
                token_digest = digest
            elif digest != token_digest:
                raise RuntimeError(f"token output changed at request {request}")
            row = sample_memory(pid, request)
            samples.append(row)
            print(json.dumps({"type": "sample", **row}), flush=True)
            wired_growth = row["system_wired_bytes"] - baseline["system_wired_bytes"]
            if wired_growth > max_wired_growth:
                limit = args.max_wired_growth_gib
                raise RuntimeError(
                    f"wired memory grew by more than {limit:.1f} GiB; "
                    "aborting bounded run"
                )
    finally:
        client.close()

    result = summarize(config, samples, args.warm_samples)
    result["generated_token_sha256"] = token_digest
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
