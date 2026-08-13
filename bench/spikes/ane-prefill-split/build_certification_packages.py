#!/usr/bin/env python3
"""Build W128/W256/W512 certification packages with bounded retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

BUCKETS = (128, 256, 512)
MAX_ATTEMPTS = 2


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        hasher.update(child.relative_to(path).as_posix().encode())
        hasher.update(b"\0")
        with child.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def size_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def run(command: list[str], log: Path, environment: dict[str, str]) -> tuple[int, float]:
    started = time.monotonic()
    with log.open("wb") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=environment)
    return result.returncode, time.monotonic() - started


def build_bucket(args: argparse.Namespace, bucket: int) -> dict[str, Any]:
    package = args.artifacts / f"qwen3-prefill-w{bucket}.mlpackage"
    compiled = args.artifacts / f"qwen3-prefill-w{bucket}.mlmodelc"
    conversion_report = args.artifacts / f"conversion-w{bucket}.json"
    compile_report = args.artifacts / f"compile-w{bucket}.json"
    attempts: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["DEVELOPER_DIR"] = args.developer_dir
    for attempt in range(1, MAX_ATTEMPTS + 1):
        for path in (package, compiled, conversion_report, compile_report):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        conversion_log = args.artifacts / f"conversion-w{bucket}-attempt{attempt}.log"
        code, elapsed = run(
            [
                str(args.python),
                str(args.spike / "convert_qwen3_prefill.py"),
                "--model",
                str(args.model),
                "--window",
                str(bucket),
                "--out",
                str(package),
                "--report-json",
                str(conversion_report),
            ],
            conversion_log,
            environment,
        )
        row: dict[str, Any] = {
            "attempt": attempt,
            "conversion_exit_code": code,
            "conversion_wall_s": elapsed,
            "conversion_log": str(conversion_log),
        }
        if code != 0:
            attempts.append(row)
            continue
        compile_log = args.artifacts / f"compile-w{bucket}-attempt{attempt}.log"
        code, compile_elapsed = run(
            [
                str(args.runner),
                "compile",
                "--model",
                str(package),
                "--out",
                str(compiled),
                "--stats",
                str(compile_report),
            ],
            compile_log,
            environment,
        )
        row.update(
            {
                "compile_exit_code": code,
                "compile_wall_s": compile_elapsed,
                "compile_log": str(compile_log),
            }
        )
        attempts.append(row)
        if code == 0:
            return {
                "bucket": bucket,
                "status": "ready",
                "attempts": attempts,
                "package_size_bytes": size_bytes(package),
                "package_sha256": sha256_path(package),
                "compiled_size_bytes": size_bytes(compiled),
                "compiled_sha256": sha256_path(compiled),
            }
    return {
        "bucket": bucket,
        "status": "absent",
        "absence_reason": "compile_or_load_failure",
        "detail": "conversion or compilation failed in both permitted attempts; inspect the recorded logs",
        "attempts": attempts,
    }


def parse_args() -> argparse.Namespace:
    spike = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spike", type=Path, default=spike)
    parser.add_argument("--artifacts", type=Path, default=spike / "artifacts")
    parser.add_argument("--python", type=Path, default=spike / ".venv/bin/python")
    parser.add_argument("--runner", type=Path, default=spike / ".build/ane-prefill-runner")
    parser.add_argument("--developer-dir", default="/Applications/Xcode.app/Contents/Developer")
    parser.add_argument("--output", type=Path, default=spike / "artifacts/package-build-record.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_revision": 1,
        "maximum_attempts_per_bucket": MAX_ATTEMPTS,
        "buckets": [build_bucket(args, bucket) for bucket in BUCKETS],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(row["status"] == "ready" for row in result["buckets"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
