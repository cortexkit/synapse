#!/usr/bin/env python3
"""Capture named macOS footprint stages for one owned decode worker.

The worker loads lazily enough that a normal LOAD separates spawn from model
residency. ANE installation is lazy too: its sidecar is created and installs the
Core ML program while processing the first split-prefill request, so that stage
is intentionally recorded as one atomic post-request observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from decode_worker_memory_curve import deterministic_prompt, descendant_pids, sample_memory

SIZE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)"
    r"\s+\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|TB)"
    r"\s+\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|TB)"
    r"\s+\d+\s+(.+?)\s*$"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--worker", type=Path, default=root / "target/release/ck-synapse-worker-decode"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--weight-quant", choices=("f16", "q8_0"), required=True)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--steady-requests", type=int, default=5)
    return parser.parse_args()


def size_bytes(value: str, unit: str) -> int:
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return round(float(value) * scale[unit])


def footprint_categories(pid: int) -> dict[str, int]:
    """Return categories printed by footprint(1), not virtual VM sizes."""

    import subprocess

    completed = subprocess.run(
        ["footprint", str(pid)], text=True, capture_output=True, check=True
    )
    categories: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        matched = SIZE_RE.match(line)
        if matched is None:
            continue
        value, unit, category = matched.groups()
        categories[category] = size_bytes(value, unit)
    return categories


def snapshot(pid: int, stage: str, request: int) -> dict[str, Any]:
    memory = sample_memory(pid, request)
    pids = [pid, *descendant_pids(pid)]
    categories: dict[str, int] = {}
    per_process: dict[str, Any] = {}
    for process_pid in pids:
        try:
            process_categories = footprint_categories(process_pid)
        except Exception as error:  # A just-exited sidecar is not measurement failure.
            per_process[str(process_pid)] = {"unavailable": str(error)}
            continue
        per_process[str(process_pid)] = process_categories
        for category, value in process_categories.items():
            categories[category] = categories.get(category, 0) + value
    return {
        "stage": stage,
        "pids": pids,
        "memory": memory,
        "footprint_categories_bytes": categories,
        "per_process_footprint_categories_bytes": per_process,
    }


def main() -> int:
    args = parse_args()
    if (args.compiled is None) != (args.sidecar is None):
        raise ValueError("--compiled and --sidecar must be supplied together")
    if args.steady_requests < 1:
        raise ValueError("--steady-requests must be positive")

    sys.path.insert(0, str(args.root / "tests/ane-prefill-certification"))
    from machine_driver import WorkerClient  # pylint: disable=import-outside-toplevel

    client = WorkerClient(
        args.worker.resolve(),
        args.sidecar.resolve() if args.sidecar else args.worker.resolve(),
        args.root / "target/release/compile_constraint",
        args.checkpoint.resolve(),
        args.compiled.resolve() if args.compiled else None,
        args.prompt_length,
        "f16-step" if args.weight_quant == "f16" else "q8-step",
        1,
        load=False,
    )
    rows: list[dict[str, Any]] = []
    try:
        pid = client._child.pid
        rows.append(snapshot(pid, "post-spawn", 0))
        client.load()
        rows.append(
            snapshot(
                pid,
                "post-model-load" if args.weight_quant == "f16" else "post-q8-ingest-and-load",
                0,
            )
        )
        prompt = deterministic_prompt(args.prompt_length)
        expected_digest: str | None = None
        for request in range(1, args.steady_requests + 1):
            tokens, _, _ = client.generate(prompt, 1)
            digest = hashlib.sha256(
                b"".join(token.to_bytes(4, "little") for token in tokens)
            ).hexdigest()
            if expected_digest is None:
                expected_digest = digest
            elif digest != expected_digest:
                raise RuntimeError(f"token output changed at request {request}")
            if request == 1:
                stage = (
                    "post-first-decode/post-ANE-install"
                    if args.compiled
                    else "post-first-decode"
                )
                rows.append(snapshot(pid, stage, request))
        rows.append(snapshot(pid, "steady", args.steady_requests))
    finally:
        client.close()

    result = {
        "label": args.label,
        "weight_quant": args.weight_quant,
        "prefill": "ane-split" if args.compiled else "gpu",
        "prompt_length": args.prompt_length,
        "steady_requests": args.steady_requests,
        "generated_token_sha256": expected_digest,
        "stages": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
