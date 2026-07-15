#!/usr/bin/env python3
"""Attempt a fixed-window MLState LFM2 decode conversion and map its failure boundary."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import coremltools as ct  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import transformers

from lfm2_decode import LFM2StatefulDecode  # pyright: ignore[reportMissingImports]
from lfm2_model import build_prefill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def directory_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64).ravel()
    right64 = right.astype(np.float64).ravel()
    return float(
        np.dot(left64, right64)
        / max(float(np.linalg.norm(left64) * np.linalg.norm(right64)), 1e-12)
    )


def main() -> int:
    args = parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive")
    for path in (args.out, args.report_json):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
            remove_path(path)
    report: dict[str, Any] = {
        "status": "started",
        "stage": "build",
        "model": str(args.model.expanduser().resolve()),
        "window": args.window,
        "environment": environment(),
        "frontend": "torch.export",
        "compute_precision": "float16",
        "compute_units": "CPU_AND_NE",
        "minimum_deployment_target": "macOS15",
    }
    try:
        prefill, config = build_prefill(args.model.expanduser().resolve(), seq_len=1)
        decode = LFM2StatefulDecode(prefill, window=args.window).eval()
        token_ids = torch.tensor([[config.bos_token_id]], dtype=torch.int32)
        position = torch.tensor([0], dtype=torch.int32)
        valid_length = torch.tensor([1], dtype=torch.int32)
        example = (token_ids, position, valid_length)
        decode.reset_state()
        with torch.inference_mode():
            eager_first = decode(*example).detach().cpu().float().numpy()
        decode.reset_state()

        report["stage"] = "torch_export"
        started = time.monotonic()
        exported = torch.export.export(decode, example, strict=False)
        report["export_s"] = time.monotonic() - started
        signature = exported.graph_signature
        report["mutable_buffers"] = dict(signature.buffers_to_mutate)
        report["mutable_buffer_count"] = len(signature.buffers_to_mutate)
        report["export_inputs"] = [str(spec) for spec in signature.input_specs]

        report["stage"] = "coreml_convert"
        started = time.monotonic()
        mlmodel = ct.convert(
            exported,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        report["conversion_s"] = time.monotonic() - started
        outputs = list(mlmodel.output_description)
        if len(outputs) != 1:
            raise RuntimeError(f"expected one normal output, found {outputs}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(args.out)
        report["package_size_bytes"] = directory_size(args.out)

        report["stage"] = "stateful_runtime"
        state = mlmodel.make_state()
        prediction = mlmodel.predict(
            {
                "token_ids": token_ids.numpy(),
                "position": position.numpy(),
                "valid_length": valid_length.numpy(),
            },
            state=state,
        )
        coreml_first = np.asarray(next(iter(prediction.values())), dtype=np.float32)
        report["first_step_cosine"] = cosine(eager_first, coreml_first)
        report["status"] = "converted_and_ran"
        report["stage"] = "complete"
        write_report(args.report_json, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        if report["stage"] == "stateful_runtime" and args.out.exists():
            try:
                cpu_model = ct.models.MLModel(str(args.out), compute_units=ct.ComputeUnit.CPU_ONLY)
                cpu_state = cpu_model.make_state()
                started = time.monotonic()
                cpu_prediction = cpu_model.predict(
                    {
                        "token_ids": np.asarray([[1]], dtype=np.int32),
                        "position": np.asarray([0], dtype=np.int32),
                        "valid_length": np.asarray([1], dtype=np.int32),
                    },
                    state=cpu_state,
                )
                cpu_output = np.asarray(next(iter(cpu_prediction.values())))
                report["cpu_only_runtime"] = {
                    "status": "passed",
                    "wall_s": time.monotonic() - started,
                    "output_shape": list(cpu_output.shape),
                }
            except Exception as cpu_error:
                report["cpu_only_runtime"] = {
                    "status": "failed",
                    "error_type": type(cpu_error).__name__,
                    "error": str(cpu_error),
                }
        write_report(args.report_json, report)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
