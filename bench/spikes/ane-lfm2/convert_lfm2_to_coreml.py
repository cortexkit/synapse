#!/usr/bin/env python3
"""Convert LFM2 prefill to fixed-shape fp16 Core ML through torch.export only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import coremltools as ct  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import transformers

from lfm2_model import LFM2Config, build_prefill

PINNED_VERSIONS = {
    "torch": "2.5.1",
    "coremltools": "8.3.0",
    "transformers": "4.51.3",
}


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    macos: str
    machine: str
    torch: str
    coremltools: str
    transformers: str
    numpy: str


@dataclass(frozen=True)
class Comparison:
    prompt_min_cosine: float
    prompt_mean_cosine: float
    token_min_cosine: float
    token_mean_cosine: float
    max_abs: float


@dataclass(frozen=True)
class ParityReport:
    rows: int
    active_tokens: int
    reference_eager: Comparison
    eager_export: Comparison
    reference_coreml: Comparison


@dataclass(frozen=True)
class TimingReport:
    wrapper_load_s: float
    export_s: float
    conversion_s: float
    coreml_predictions_s: float


@dataclass(frozen=True)
class ConversionReport:
    parity_gate_passed: bool
    source_model: str
    source_revision: str
    source_model_sha256: str
    reference_path: str
    seq_len: int
    output_path: str
    output_name: str
    frontend: str
    compute_precision: str
    compute_units: str
    minimum_deployment_target: str
    package_size_bytes: int
    model_config: dict[str, object]
    parity: ParityReport
    timing: TimingReport
    environment: EnvironmentReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Hugging Face snapshot")
    parser.add_argument("--seq-len", type=int, required=True, choices=(128, 256))
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Destination .mlpackage")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--silu-fp32",
        action="store_true",
        help="Keep SiLU in fp32 to diagnose Core ML 8.3's inaccurate fp16 SiLU.",
    )
    parser.add_argument(
        "--allow-parity-failure",
        action="store_true",
        help="Save a diagnostic package and report even when cosine is below the viability gate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_report() -> EnvironmentReport:
    return EnvironmentReport(
        python=platform.python_version(),
        macos=platform.mac_ver()[0],
        machine=platform.machine(),
        torch=torch.__version__,
        coremltools=ct.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
    )


def require_pinned_environment(environment: EnvironmentReport) -> None:
    actual = {
        "torch": environment.torch.split("+")[0],
        "coremltools": environment.coremltools,
        "transformers": environment.transformers,
    }
    mismatches = [
        f"{name}={actual[name]} (expected {expected})"
        for name, expected in PINNED_VERSIONS.items()
        if actual[name] != expected
    ]
    if mismatches:
        raise RuntimeError("conversion environment is not pinned: " + ", ".join(mismatches))


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def directory_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def load_reference(
    path: Path,
    seq_len: int,
    checkpoint_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        input_ids = archive["input_ids"].astype(np.int32)
        attention_mask = archive["attention_mask"].astype(np.int32)
        hidden_states = archive["hidden_states"].astype(np.float32)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if input_ids.shape != attention_mask.shape or input_ids.shape[1] != seq_len:
        raise ValueError(
            f"reference input shapes {input_ids.shape}/{attention_mask.shape} do not match seq{seq_len}"
        )
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError(f"reference hidden shape {hidden_states.shape} does not match inputs")
    if int(metadata["seq_len"]) != seq_len:
        raise ValueError("reference metadata sequence length does not match requested bucket")
    if metadata["source_model_sha256"] != checkpoint_sha256:
        raise ValueError("reference checkpoint hash does not match conversion checkpoint")
    if input_ids.shape[0] != 20:
        raise ValueError(f"the parity gate requires 20 prompts, found {input_ids.shape[0]}")
    return input_ids, attention_mask, hidden_states, metadata


def tensor_rows(module: torch.nn.Module, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for row_ids, row_mask in zip(ids, mask, strict=True):
            value = module(
                torch.from_numpy(row_ids[None, :]),
                torch.from_numpy(row_mask[None, :]),
            )
            rows.append(value.detach().cpu().float().numpy())
    return np.concatenate(rows, axis=0)


def compare(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> Comparison:
    if reference.shape != candidate.shape:
        raise ValueError(f"parity shape mismatch: {reference.shape} versus {candidate.shape}")
    prompt_cosines: list[float] = []
    token_cosines: list[float] = []
    max_abs = 0.0
    for reference_row, candidate_row, row_mask in zip(reference, candidate, mask, strict=True):
        active = row_mask.astype(bool)
        expected = reference_row[active].astype(np.float64)
        actual = candidate_row[active].astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(expected - actual))))
        prompt_numerator = float(np.sum(expected * actual))
        prompt_denominator = float(np.linalg.norm(expected) * np.linalg.norm(actual))
        prompt_cosines.append(prompt_numerator / max(prompt_denominator, 1e-12))
        numerators = np.sum(expected * actual, axis=-1)
        denominators = np.linalg.norm(expected, axis=-1) * np.linalg.norm(actual, axis=-1)
        token_cosines.extend((numerators / np.maximum(denominators, 1e-12)).tolist())
    return Comparison(
        prompt_min_cosine=float(np.min(prompt_cosines)),
        prompt_mean_cosine=float(np.mean(prompt_cosines)),
        token_min_cosine=float(np.min(token_cosines)),
        token_mean_cosine=float(np.mean(token_cosines)),
        max_abs=max_abs,
    )


def coreml_rows(
    mlmodel: ct.models.MLModel,
    output_name: str,
    ids: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rows = []
    for row_ids, row_mask in zip(ids, mask, strict=True):
        prediction = mlmodel.predict(
            {
                "input_ids": row_ids[None, :].astype(np.int32),
                "attention_mask": row_mask[None, :].astype(np.int32),
            }
        )
        value = prediction.get(output_name)
        if value is None and len(prediction) == 1:
            value = next(iter(prediction.values()))
        if value is None:
            raise RuntimeError(f"Core ML prediction has no {output_name!r} output: {list(prediction)}")
        rows.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(rows, axis=0)


def ensure_metadata(mlmodel: ct.models.MLModel, report: ConversionReport) -> None:
    mlmodel.short_description = "Fixed-bucket LFM2 prefill hidden states for Apple Neural Engine."
    mlmodel.author = "Synapse bench/spikes/ane-lfm2"
    mlmodel.license = "Apache-2.0; see the referenced Liquid AI checkpoint."
    metadata = mlmodel.user_defined_metadata
    metadata["synapse.source_model"] = report.source_model
    metadata["synapse.source_revision"] = report.source_revision
    metadata["synapse.source_model_sha256"] = report.source_model_sha256
    metadata["synapse.seq_len"] = str(report.seq_len)
    metadata["synapse.frontend"] = report.frontend
    metadata["synapse.output_name"] = report.output_name
    metadata["synapse.layer_types"] = json.dumps(report.model_config["layer_types"])
    metadata["synapse.actual_intermediate_size"] = str(report.model_config["intermediate_size"])


def main() -> int:
    args = parse_args()
    environment = environment_report()
    require_pinned_environment(environment)
    model_path = args.model.expanduser().resolve()
    for path in (args.out, args.report_json):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
            remove_path(path)
    checkpoint_sha256 = sha256(model_path / "model.safetensors")
    input_ids, attention_mask, reference_hidden, reference_metadata = load_reference(
        args.reference_npz,
        args.seq_len,
        checkpoint_sha256,
    )

    started = time.monotonic()
    wrapper, config = build_prefill(model_path, args.seq_len)
    wrapper_load_s = time.monotonic() - started
    example = (
        torch.from_numpy(input_ids[:1]),
        torch.from_numpy(attention_mask[:1]),
    )
    eager_hidden = tensor_rows(wrapper, input_ids, attention_mask)
    started = time.monotonic()
    exported = torch.export.export(wrapper, example, strict=False)
    export_s = time.monotonic() - started
    exported_hidden = tensor_rows(exported.module(), input_ids, attention_mask)

    started = time.monotonic()
    compute_precision: Any
    if args.silu_fp32:
        # Core ML 8.3's fp16 SiLU approximation compounds across all 16
        # SwiGLU blocks. This diagnostic keeps only SiLU in fp32 while leaving
        # every projection and convolution fp16.
        compute_precision = ct.transform.FP16ComputePrecision(
            op_selector=lambda operation: operation.op_type != "silu"
        )
    else:
        compute_precision = ct.precision.FLOAT16
    mlmodel = ct.convert(
        exported,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    conversion_s = time.monotonic() - started
    existing_outputs = list(mlmodel.output_description)
    if len(existing_outputs) != 1:
        raise RuntimeError(f"expected one model output, found {existing_outputs!r}")
    converted_output = existing_outputs[0]
    output_name = "hidden_states"
    if converted_output != output_name:
        ct.utils.rename_feature(mlmodel._spec, converted_output, output_name)

    started = time.monotonic()
    converted_hidden = coreml_rows(mlmodel, output_name, input_ids, attention_mask)
    coreml_predictions_s = time.monotonic() - started
    parity = ParityReport(
        rows=input_ids.shape[0],
        active_tokens=int(np.sum(attention_mask)),
        reference_eager=compare(reference_hidden, eager_hidden, attention_mask),
        eager_export=compare(eager_hidden, exported_hidden, attention_mask),
        reference_coreml=compare(reference_hidden, converted_hidden, attention_mask),
    )
    if parity.reference_eager.prompt_min_cosine < 0.99999:
        raise RuntimeError(f"custom LFM2 wrapper disagrees with Transformers: {parity.reference_eager}")
    if parity.eager_export.max_abs > 1e-5:
        raise RuntimeError(f"torch.export parity failed: {parity.eager_export}")
    parity_gate_passed = parity.reference_coreml.prompt_min_cosine >= 0.999
    if not parity_gate_passed and not args.allow_parity_failure:
        raise RuntimeError(f"Core ML parity failed: {parity.reference_coreml}")

    timing = TimingReport(
        wrapper_load_s=wrapper_load_s,
        export_s=export_s,
        conversion_s=conversion_s,
        coreml_predictions_s=coreml_predictions_s,
    )
    placeholder_report = ConversionReport(
        parity_gate_passed=parity_gate_passed,
        source_model=str(model_path),
        source_revision=str(reference_metadata["source_revision"]),
        source_model_sha256=checkpoint_sha256,
        reference_path=str(args.reference_npz),
        seq_len=args.seq_len,
        output_path=str(args.out),
        output_name=output_name,
        frontend="torch.export",
        compute_precision="float16_except_silu_fp32" if args.silu_fp32 else "float16",
        compute_units="CPU_AND_NE",
        minimum_deployment_target="macOS15",
        package_size_bytes=0,
        model_config=config.to_json(),
        parity=parity,
        timing=timing,
        environment=environment,
    )
    ensure_metadata(mlmodel, placeholder_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.out)
    report = ConversionReport(
        **{
            **asdict(placeholder_report),
            "package_size_bytes": directory_size(args.out),
            "parity": parity,
            "timing": timing,
            "environment": environment,
        }
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
