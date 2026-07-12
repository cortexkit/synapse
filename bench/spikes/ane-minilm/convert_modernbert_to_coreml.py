#!/usr/bin/env python3
"""Convert fixed-shape GTE ModernBERT embedding or reranking models to Core ML."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import coremltools as ct  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

EMBEDDER_MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
RERANKER_MODEL_ID = "Alibaba-NLP/gte-reranker-modernbert-base"
MODEL_IDS = {"embedder": EMBEDDER_MODEL_ID, "reranker": RERANKER_MODEL_ID}
SMOKE_QUERY = "what is rust?"
SMOKE_DOCUMENTS = (
    "Rust is a systems programming language.",
    "A banana is a yellow fruit.",
)
SMOKE_EMBED_TEXTS = (
    "Represent this sentence for searching relevant passages: what is rust?",
    "Rust is a systems programming language.",
)


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
class ParityReport:
    rows: int
    eager_export_max_abs: float
    eager_export_mean_cosine: float | None
    eager_coreml_max_abs: float
    eager_coreml_mean_cosine: float | None
    eager_coreml_pearson: float | None


@dataclass(frozen=True)
class ConversionReport:
    model_kind: str
    source_model: str
    seq_len: int
    output_path: str
    output_name: str
    frontend: str
    compute_precision: str
    compute_units: str
    parity: ParityReport
    environment: EnvironmentReport


class ModernBertEmbedder(torch.nn.Module):
    """Return the checkpoint's CLS embedding with L2 normalization in the graph."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        return torch.nn.functional.normalize(hidden[:, 0, :], p=2, dim=-1)


class ModernBertReranker(torch.nn.Module):
    """Return one raw regression logit after the checkpoint's masked-mean classification head."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        return logits[:, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument(
        "--model",
        help="HF repo id or local snapshot directory; defaults to the model selected by --kind.",
    )
    parser.add_argument("--seq-len", type=int, required=True, choices=(128, 256, 512))
    parser.add_argument("--out", type=Path, required=True, help="Destination .mlpackage path")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads when the requested model is not cached.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and report paths.",
    )
    return parser.parse_args()


def cached_model_snapshot(model_id: str) -> Path | None:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
    )
    if not cache_root.exists():
        return None
    snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def resolve_model_ref(requested: str) -> tuple[str, str]:
    requested_path = Path(requested).expanduser()
    if requested_path.exists():
        resolved = str(requested_path.resolve())
        return resolved, resolved
    cached = cached_model_snapshot(requested)
    if cached is not None:
        resolved = str(cached.resolve())
        return resolved, resolved
    return requested, requested


def load_wrapper(
    kind: Literal["embedder", "reranker"], model_ref: str, allow_download: bool
) -> torch.nn.Module:
    common: dict[str, Any] = {
        "local_files_only": not allow_download,
        "attn_implementation": "eager",
    }
    if kind == "embedder":
        model = AutoModel.from_pretrained(model_ref, **common)
        return ModernBertEmbedder(model).eval()
    model = AutoModelForSequenceClassification.from_pretrained(model_ref, **common)
    return ModernBertReranker(model).eval()


def smoke_inputs(
    kind: Literal["embedder", "reranker"], model_ref: str, seq_len: int, allow_download: bool
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=not allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 50283
    encoded_rows: list[dict[str, Any]] = []
    if kind == "embedder":
        for text in SMOKE_EMBED_TEXTS:
            encoded_rows.append(
                tokenizer(
                    text,
                    truncation=True,
                    max_length=seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
    else:
        for document in SMOKE_DOCUMENTS:
            encoded_rows.append(
                tokenizer(
                    SMOKE_QUERY,
                    document,
                    truncation=True,
                    max_length=seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
    return [(row["input_ids"], row["attention_mask"]) for row in encoded_rows]


def tensor_rows(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(array.shape[0], -1)


def mean_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = np.sum(reference * candidate, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def pearson(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    if reference_flat.size < 2 or np.std(reference_flat) == 0 or np.std(candidate_flat) == 0:
        return None
    return float(np.corrcoef(reference_flat, candidate_flat)[0, 1])


def convert_and_verify(
    wrapper: torch.nn.Module,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    output_name: str,
    kind: Literal["embedder", "reranker"],
) -> tuple[ct.models.MLModel, ParityReport]:
    example = examples[0]
    with torch.inference_mode():
        exported = torch.export.export(wrapper, example, strict=False)
        eager_rows = [tensor_rows(wrapper(*inputs)) for inputs in examples]
        exported_rows = [tensor_rows(exported.module()(*inputs)) for inputs in examples]

    mlmodel = ct.convert(
        exported,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    existing_outputs = list(mlmodel.output_description)
    if len(existing_outputs) != 1:
        raise RuntimeError(f"expected one model output, found {existing_outputs!r}")
    converted_output = existing_outputs[0]
    if converted_output != output_name:
        ct.utils.rename_feature(mlmodel._spec, converted_output, output_name)

    coreml_rows = []
    for input_ids, attention_mask in examples:
        prediction = mlmodel.predict(
            {
                "input_ids": input_ids.detach().cpu().numpy().astype(np.int32),
                "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int32),
            }
        )
        coreml_rows.append(tensor_rows(prediction[converted_output]))

    eager = np.concatenate(eager_rows, axis=0)
    exported_output = np.concatenate(exported_rows, axis=0)
    coreml = np.concatenate(coreml_rows, axis=0)
    export_max_abs = float(np.max(np.abs(eager - exported_output)))
    coreml_max_abs = float(np.max(np.abs(eager - coreml)))
    export_cosine = mean_cosine(eager, exported_output) if kind == "embedder" else None
    coreml_cosine = mean_cosine(eager, coreml) if kind == "embedder" else None
    coreml_pearson = pearson(eager, coreml) if kind == "reranker" else None

    if export_max_abs > 1e-6:
        raise RuntimeError(f"torch.export parity failed: max_abs={export_max_abs}")
    if kind == "embedder" and (coreml_cosine is None or coreml_cosine < 0.999):
        raise RuntimeError(f"Core ML conversion parity failed: mean_cosine={coreml_cosine}")
    if kind == "reranker" and (coreml_pearson is None or coreml_pearson < 0.999):
        raise RuntimeError(f"Core ML conversion parity failed: pearson={coreml_pearson}")

    return mlmodel, ParityReport(
        rows=eager.shape[0],
        eager_export_max_abs=export_max_abs,
        eager_export_mean_cosine=export_cosine,
        eager_coreml_max_abs=coreml_max_abs,
        eager_coreml_mean_cosine=coreml_cosine,
        eager_coreml_pearson=coreml_pearson,
    )


def environment_report() -> EnvironmentReport:
    macos_version = platform.mac_ver()[0]
    return EnvironmentReport(
        python=platform.python_version(),
        macos=macos_version,
        machine=platform.machine(),
        torch=torch.__version__,
        coremltools=ct.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
    )


def ensure_metadata(mlmodel: ct.models.MLModel, report: ConversionReport) -> None:
    mlmodel.short_description = (
        f"Synapse fixed-sequence GTE ModernBERT {report.model_kind} for Apple Neural Engine."
    )
    mlmodel.author = "Synapse bench/spikes/ane-minilm"
    mlmodel.license = "Source model license follows the referenced Hugging Face checkpoint."
    metadata = mlmodel.user_defined_metadata
    metadata["synapse.source_model"] = report.source_model
    metadata["synapse.model_kind"] = report.model_kind
    metadata["synapse.seq_len"] = str(report.seq_len)
    metadata["synapse.frontend"] = report.frontend
    metadata["synapse.output_name"] = report.output_name
    metadata["synapse.torch_version"] = report.environment.torch
    metadata["synapse.coremltools_version"] = report.environment.coremltools
    metadata["synapse.transformers_version"] = report.environment.transformers
    metadata["synapse.macos_version"] = report.environment.macos


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    args = parse_args()
    kind: Literal["embedder", "reranker"] = args.kind
    requested_model = args.model or MODEL_IDS[kind]
    model_ref, report_model_ref = resolve_model_ref(requested_model)
    for path in (args.out, args.report_json):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
            remove_path(path)

    wrapper = load_wrapper(kind, model_ref, args.allow_download)
    examples = smoke_inputs(kind, model_ref, args.seq_len, args.allow_download)
    output_name = "embedding" if kind == "embedder" else "score"
    mlmodel, parity = convert_and_verify(wrapper, examples, output_name, kind)
    report = ConversionReport(
        model_kind=kind,
        source_model=report_model_ref,
        seq_len=args.seq_len,
        output_path=str(args.out),
        output_name=output_name,
        frontend="torch.export",
        compute_precision="float16",
        compute_units="CPU_AND_NE",
        parity=parity,
        environment=environment_report(),
    )
    ensure_metadata(mlmodel, report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.out)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
