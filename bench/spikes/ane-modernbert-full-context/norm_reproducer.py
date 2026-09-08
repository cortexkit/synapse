#!/usr/bin/env python3
"""Minimal actual-value embedding LayerNorm reproducer for Core ML backends."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open

EPSILON = 1e-5
OUTPUT_NAMES = (
    "native_sequence_last",
    "explicit_sequence_last",
    "native_channel_input",
    "explicit_channel_input",
    "native_after_gather",
    "explicit_after_gather",
)


class NormVariants(torch.nn.Module):
    """Evaluate equivalent norms from sequence, channel, and gathered actual values."""

    def __init__(self, weight: torch.Tensor, gather_table: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.float(), requires_grad=False)
        self.gather_table = torch.nn.Parameter(gather_table.float(), requires_grad=False)

    def _native_sequence(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            value, (self.weight.numel(),), self.weight, None, EPSILON
        )

    def _explicit_sequence(self, value: torch.Tensor) -> torch.Tensor:
        centered = value - value.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + EPSILON) * self.weight

    def forward(
        self,
        sequence_last: torch.Tensor,
        channel_input: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        native_sequence_last = self._native_sequence(sequence_last)
        explicit_sequence_last = self._explicit_sequence(sequence_last)

        channel_sequence = channel_input.squeeze(2).transpose(1, 2).contiguous()
        native_channel_input = self._native_sequence(channel_sequence)
        centered_channel = channel_input - channel_input.mean(dim=1, keepdim=True)
        variance_channel = (centered_channel * centered_channel).mean(dim=1, keepdim=True)
        explicit_channel_input = (
            centered_channel
            * torch.rsqrt(variance_channel + EPSILON)
            * self.weight.reshape(1, -1, 1, 1)
        ).squeeze(2).transpose(1, 2).contiguous()

        gathered = torch.nn.functional.embedding(position_ids, self.gather_table).contiguous()
        native_after_gather = self._native_sequence(gathered)
        explicit_after_gather = self._explicit_sequence(gathered)
        return (
            native_sequence_last,
            explicit_sequence_last,
            native_channel_input,
            explicit_channel_input,
            native_after_gather,
            explicit_after_gather,
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--model", type=Path, required=True)
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--report", type=Path, required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--capture", type=Path, required=True)
    export.add_argument("--precision", choices=("float16", "float32"), required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--report", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")

    reload_parser = subparsers.add_parser("reload")
    reload_parser.add_argument("--capture", type=Path, required=True)
    reload_parser.add_argument("--package", type=Path, required=True)
    reload_parser.add_argument("--compute-units", choices=("cpu-only", "cpu-and-ne"), required=True)
    reload_parser.add_argument("--out", type=Path, required=True)
    reload_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def environment_report() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "coremltools", "numpy", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "macos_build": subprocess.run(
            ["sw_vers", "-buildVersion"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "machine": platform.machine(),
        "packages": packages,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        line = handle.readline()
    if not line:
        raise ValueError("input JSONL is empty")
    return json.loads(line)


def tensor_summary(value: np.ndarray) -> dict[str, Any]:
    flattened = value.reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "variance": float(value.var()),
        "first_16_values": [float(item) for item in flattened[:16]],
    }


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference_rows = reference.reshape(-1, reference.shape[-1]).astype(np.float32)
    candidate_rows = candidate.reshape(-1, candidate.shape[-1]).astype(np.float32)
    numerator = np.sum(reference_rows * candidate_rows, axis=-1)
    denominator = np.linalg.norm(reference_rows, axis=-1) * np.linalg.norm(
        candidate_rows, axis=-1
    )
    cosine = numerator / np.maximum(denominator, 1e-12)
    difference = np.abs(reference_rows - candidate_rows)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "worst_token_index": int(np.argmin(cosine)),
    }


def run_variants(
    model: torch.nn.Module,
    sequence_last: torch.Tensor,
    channel_input: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[np.ndarray, ...]:
    with torch.inference_mode():
        return tuple(
            output.detach().cpu().float().numpy()
            for output in model(sequence_last, channel_input, position_ids)
        )



def command_capture(args: argparse.Namespace) -> dict[str, Any]:
    row = first_jsonl_row(args.input)
    input_ids = torch.tensor([row["input_ids"]], dtype=torch.int64)
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as reader:
        embedding_weight = reader.get_tensor("embeddings.tok_embeddings.weight").float()
        norm_weight = reader.get_tensor("embeddings.norm.weight").float()
    sequence_last = torch.nn.functional.embedding(input_ids, embedding_weight)
    channel_input = sequence_last.transpose(1, 2).unsqueeze(2).contiguous()
    position_ids = torch.arange(sequence_last.shape[1], dtype=torch.int32).unsqueeze(0)
    gather_table = sequence_last.squeeze(0).contiguous()
    model = NormVariants(norm_weight, gather_table).eval()
    eager = run_variants(model, sequence_last, channel_input, position_ids)
    half_model = NormVariants(norm_weight.half(), gather_table.half()).half().eval()
    eager_half = run_variants(
        half_model, sequence_last.half(), channel_input.half(), position_ids
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequence_last": sequence_last.cpu().numpy().astype(np.float32),
        "channel_input": channel_input.cpu().numpy().astype(np.float32),
        "position_ids": position_ids.cpu().numpy().astype(np.int32),
        "norm_weight": norm_weight.cpu().numpy().astype(np.float32),
        **{f"eager_{name}": value for name, value in zip(OUTPUT_NAMES, eager, strict=True)},
        **{f"eager_half_{name}": value for name, value in zip(OUTPUT_NAMES, eager_half, strict=True)},
    }
    np.savez_compressed(str(args.out), **payload)
    native = eager[0]
    return {
        "status": "passed",
        "source": "actual full-context row token embeddings and checkpoint embeddings.norm.weight",
        "row_id": row["id"],
        "actual_token_count": row["actual_token_count"],
        "input_ids_sha256": hashlib.sha256(
            np.asarray(row["input_ids"], dtype=np.int32).tobytes()
        ).hexdigest(),
        "sequence_last": tensor_summary(sequence_last.cpu().numpy()),
        "channel_input": tensor_summary(channel_input.cpu().numpy()),
        "position_ids": tensor_summary(position_ids.cpu().numpy()),
        "norm_weight": tensor_summary(norm_weight.cpu().numpy()),
        "eager_equivalent_formulations": {
            name: comparison(native, value)
            for name, value in zip(OUTPUT_NAMES, eager, strict=True)
        },
        "eager_float16_vs_float32": {
            name: comparison(value, half_value)
            for name, value, half_value in zip(OUTPUT_NAMES, eager, eager_half, strict=True)
        },
        "input_jsonl_sha256": sha256_file(args.input),
        "capture_npz_sha256": sha256_file(args.out),
        "model_safetensors_sha256": sha256_file(args.model / "model.safetensors"),
        "environment": environment_report(),
    }



def exported_program(capture: Any) -> tuple[torch.export.ExportedProgram, tuple[np.ndarray, ...]]:
    sequence_last = torch.from_numpy(capture["sequence_last"])
    channel_input = torch.from_numpy(capture["channel_input"])
    position_ids = torch.from_numpy(capture["position_ids"])
    gather_table = sequence_last.squeeze(0).contiguous()
    model = NormVariants(torch.from_numpy(capture["norm_weight"]), gather_table).eval()
    inputs = (sequence_last, channel_input, position_ids)
    exported = torch.export.export(model, inputs, strict=False).run_decompositions(
        {torch.ops.aten.alias.default: lambda value: value}
    )
    exported_values = run_variants(exported.module(), *inputs)
    return exported, exported_values



def variable_shape(variable: Any) -> list[int]:
    try:
        return [int(dimension) for dimension in variable.shape]
    except (TypeError, ValueError):
        return []


def mil_inspection(mlmodel: Any) -> dict[str, Any]:
    from coremltools.converters.mil.mil import types

    def dtype_label(dtype: Any) -> str:
        if dtype == types.fp16:
            return "float16"
        if dtype == types.fp32:
            return "float32"
        if dtype == types.int32:
            return "int32"
        if dtype == types.uint16:
            return "uint16"
        return str(dtype)

    program = getattr(mlmodel, "_mil_program", None)
    if program is None:
        return {"available": False, "reason": "coremltools did not retain the MIL program"}
    layer_norms = []
    casts = []
    operator_counts: dict[str, int] = {}
    for function in program.functions.values():
        for operation in function.operations:
            operator_counts[operation.op_type] = operator_counts.get(operation.op_type, 0) + 1
            if operation.op_type == "layer_norm":
                axes_value = getattr(getattr(operation, "axes", None), "val", None)
                axes = [] if axes_value is None else [int(value) for value in np.asarray(axes_value)]
                layer_norms.append(
                    {
                        "name": operation.name,
                        "input_shape": variable_shape(operation.x),
                        "input_dtype": dtype_label(operation.x.dtype),
                        "axes": axes,
                        "output_shape": variable_shape(operation.outputs[0]),
                        "output_dtype": dtype_label(operation.outputs[0].dtype),
                    }
                )
            if operation.op_type == "cast":
                casts.append(
                    {
                        "name": operation.name,
                        "input_name": operation.x.name,
                        "input_dtype": dtype_label(operation.x.dtype),
                        "output_name": operation.outputs[0].name,
                        "output_dtype": dtype_label(operation.outputs[0].dtype),
                        "shape": variable_shape(operation.outputs[0]),
                    }
                )
    return {
        "available": True,
        "operator_counts": operator_counts,
        "layer_norms": layer_norms,
        "cast_boundaries": casts,
        "mil_op_evaluation": {
            "available": False,
            "reason": "coremltools 9.0 exposes conversion IR but no public arbitrary-input MIL interpreter",
        },
    }


def command_export(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct

    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite")
        shutil.rmtree(args.out) if args.out.is_dir() else args.out.unlink()
    capture = np.load(args.capture)
    exported, exported_values = exported_program(capture)
    eager_values = tuple(capture[f"eager_{name}"] for name in OUTPUT_NAMES)
    export_parity = {
        name: comparison(eager, candidate)
        for name, eager, candidate in zip(OUTPUT_NAMES, eager_values, exported_values, strict=True)
    }
    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    started = time.perf_counter()
    mlmodel = cast(
        Any,
        ct.convert(
            exported,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=precision,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            skip_model_load=True,
        ),
    )
    conversion_s = time.perf_counter() - started
    output_names = list(mlmodel.output_description)
    if len(output_names) != len(OUTPUT_NAMES):
        raise RuntimeError(f"expected {len(OUTPUT_NAMES)} outputs, got {output_names}")
    for original, replacement in zip(output_names, OUTPUT_NAMES, strict=True):
        ct.utils.rename_feature(mlmodel._spec, original, replacement)
    inspection = mil_inspection(mlmodel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.out)
    return {
        "status": "passed",
        "precision": args.precision,
        "skip_model_load": True,
        "conversion_s": conversion_s,
        "eager_vs_exported": export_parity,
        "mil": inspection,
        "capture_npz_sha256": sha256_file(args.capture),
        "package_sha256": sha256_tree(args.out),
        "environment": environment_report(),
    }


def command_reload(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct

    capture = np.load(args.capture)
    sequence_last = np.asarray(capture["sequence_last"], dtype=np.float32)
    channel_input = np.asarray(capture["channel_input"], dtype=np.float32)
    position_ids = np.asarray(capture["position_ids"], dtype=np.int32)
    compute_units = (
        ct.ComputeUnit.CPU_ONLY if args.compute_units == "cpu-only" else ct.ComputeUnit.CPU_AND_NE
    )
    load_started = time.perf_counter()
    model = ct.models.MLModel(str(args.package), compute_units=compute_units)
    load_s = time.perf_counter() - load_started
    predict_started = time.perf_counter()
    model_inputs = {
        "sequence_last": sequence_last,
        "channel_input": channel_input,
        "position_ids": position_ids,
    }
    prediction = model.predict(model_inputs)
    first_predict_s = time.perf_counter() - predict_started
    repeated = model.predict(model_inputs)
    outputs = tuple(np.asarray(prediction[name], dtype=np.float32) for name in OUTPUT_NAMES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        name: value for name, value in zip(OUTPUT_NAMES, outputs, strict=True)
    }
    cast(Any, np.savez_compressed)(str(args.out), **output_payload)
    return {
        "status": "passed",
        "compute_units": args.compute_units,
        "timing_s": {"load": load_s, "first_predict": first_predict_s},
        "vs_eager_float32": {
            name: comparison(capture[f"eager_{name}"], value)
            for name, value in zip(OUTPUT_NAMES, outputs, strict=True)
        },
        "vs_eager_float16": {
            name: comparison(capture[f"eager_half_{name}"], value)
            for name, value in zip(OUTPUT_NAMES, outputs, strict=True)
        },
        "runtime_native_vs_explicit": {
            "sequence_last": comparison(outputs[0], outputs[1]),
            "channel_input": comparison(outputs[2], outputs[3]),
            "after_gather": comparison(outputs[4], outputs[5]),
            "native_sequence_vs_channel": comparison(outputs[0], outputs[2]),
            "native_sequence_vs_gather": comparison(outputs[0], outputs[4]),
            "explicit_sequence_vs_channel": comparison(outputs[1], outputs[3]),
            "explicit_sequence_vs_gather": comparison(outputs[1], outputs[5]),
        },
        "repeated_determinism": {
            name: comparison(value, np.asarray(repeated[name], dtype=np.float32))
            for name, value in zip(OUTPUT_NAMES, outputs, strict=True)
        },
        "raw_outputs": {name: tensor_summary(value) for name, value in zip(OUTPUT_NAMES, outputs, strict=True)},
        "capture_npz_sha256": sha256_file(args.capture),
        "package_sha256": sha256_tree(args.package),
        "output_npz_sha256": sha256_file(args.out),
        "environment": environment_report(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "capture":
        return command_capture(args)
    if args.command == "export":
        return command_export(args)
    if args.command == "reload":
        return command_reload(args)
    raise AssertionError(args.command)


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
        write_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "environment": environment_report(),
        }
        write_json(args.report, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
