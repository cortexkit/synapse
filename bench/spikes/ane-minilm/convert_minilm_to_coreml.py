#!/usr/bin/env python3
"""Convert all-MiniLM-L6-v2 into a fixed-shape Core ML package.

The conversion always uses `torch.export`; trace-built encoder packages are
forbidden because the original spike demonstrated catastrophic parity loss.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import coremltools as ct  # pyright: ignore[reportMissingImports]
import torch
import transformers
from transformers import AutoModel

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
)
DEFAULT_OUTPUT_NAME = "last_hidden_state"
SUPPORTED_FRONTENDS = ("export",)


@dataclass
class ConversionReport:
    source_model: str
    seq_len: int
    output_path: str
    output_name: str
    requested_frontend: str
    frontend_used: str
    trace_status: str
    compute_precision: str
    compute_units: str
    torch_version: str
    coremltools_version: str
    transformers_version: str
    macos_version: str


class MiniLMEncoder(torch.nn.Module):
    """Expose only the last hidden state with explicit zero token-type ids."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_type_ids = torch.zeros_like(input_ids)
        last_hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )[0]
        return last_hidden_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=(
            "HF repo id or local snapshot directory. Default: sentence-transformers/all-MiniLM-L6-v2 "
            "(prefers the local HF cache when present)."
        ),
    )
    parser.add_argument("--seq-len", type=int, required=True, choices=(256, 512))
    parser.add_argument("--out", type=Path, required=True, help="Destination .mlpackage path")
    parser.add_argument(
        "--frontend",
        choices=SUPPORTED_FRONTENDS,
        default="export",
        help="Conversion frontend. Only torch.export is supported.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Rename the single model output to this feature name. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path for a machine-readable conversion report.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads when the requested model is not already cached.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output package/report instead of failing.",
    )
    return parser.parse_args()


def resolve_model_ref(requested: str) -> tuple[str, str]:
    requested_path = Path(requested).expanduser()
    if requested_path.exists():
        resolved = str(requested_path)
        return resolved, resolved

    if requested == DEFAULT_MODEL_ID and DEFAULT_MODEL_CACHE.exists():
        snapshots = sorted(path for path in DEFAULT_MODEL_CACHE.iterdir() if path.is_dir())
        if snapshots:
            resolved = str(snapshots[-1])
            return resolved, resolved

    return requested, requested


def load_model(model_ref: str, allow_download: bool) -> MiniLMEncoder:
    model = AutoModel.from_pretrained(
        model_ref,
        local_files_only=not allow_download,
        attn_implementation="eager",
    )
    return MiniLMEncoder(model).eval()


def build_example_inputs(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    example_ids = torch.ones((1, seq_len), dtype=torch.int64)
    example_mask = torch.ones((1, seq_len), dtype=torch.int64)
    return example_ids, example_mask


def convert_with_export(wrapper: MiniLMEncoder, seq_len: int, output_name: str) -> ct.models.MLModel:
    example_ids, example_mask = build_example_inputs(seq_len)
    with torch.no_grad():
        exported = torch.export.export(wrapper, (example_ids, example_mask), strict=False)

    mlmodel = ct.convert(
        exported,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    existing_outputs = list(mlmodel.output_description)
    if len(existing_outputs) != 1:
        raise RuntimeError(f"expected one model output, found {existing_outputs!r}")
    if existing_outputs[0] != output_name:
        ct.utils.rename_feature(mlmodel._spec, existing_outputs[0], output_name)
    return mlmodel


def ensure_metadata(mlmodel: ct.models.MLModel, report: ConversionReport) -> None:
    mlmodel.short_description = f"Synapse ANE spike MiniLM encoder with fixed sequence length {report.seq_len}."
    mlmodel.author = "Synapse bench/spikes/ane-minilm"
    mlmodel.license = "Source model license follows sentence-transformers/all-MiniLM-L6-v2"
    mlmodel.user_defined_metadata["synapse.source_model"] = report.source_model
    mlmodel.user_defined_metadata["synapse.seq_len"] = str(report.seq_len)
    mlmodel.user_defined_metadata["synapse.frontend"] = report.frontend_used
    mlmodel.user_defined_metadata["synapse.output_name"] = report.output_name
    mlmodel.user_defined_metadata["synapse.torch_version"] = report.torch_version
    mlmodel.user_defined_metadata["synapse.coremltools_version"] = report.coremltools_version
    mlmodel.user_defined_metadata["synapse.transformers_version"] = report.transformers_version
    mlmodel.user_defined_metadata["synapse.macos_version"] = report.macos_version


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_report(report: ConversionReport, report_path: Path | None, overwrite: bool) -> None:
    if report_path is None:
        return
    if report_path.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite {report_path}; pass --overwrite to replace it")
        remove_path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite to replace it")
        remove_path(args.out)

    resolved_model_ref, report_model_ref = resolve_model_ref(args.model)
    wrapper = load_model(resolved_model_ref, allow_download=args.allow_download)

    requested_frontend = args.frontend
    mlmodel = convert_with_export(wrapper, args.seq_len, args.output_name)
    frontend_used = "export"
    trace_status = "disabled: trace-built packages are forbidden by the parity evidence"

    report = ConversionReport(
        source_model=report_model_ref,
        seq_len=args.seq_len,
        output_path=str(args.out),
        output_name=args.output_name,
        requested_frontend=requested_frontend,
        frontend_used=frontend_used,
        trace_status=trace_status,
        compute_precision="float16",
        compute_units="CPU_AND_NE",
        torch_version=torch.__version__,
        coremltools_version=ct.__version__,
        transformers_version=transformers.__version__,
        macos_version=platform.mac_ver()[0],
    )
    ensure_metadata(mlmodel, report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.out)
    write_report(report, args.report_json, overwrite=args.overwrite)

    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
