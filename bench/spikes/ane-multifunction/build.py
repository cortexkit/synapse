#!/usr/bin/env python3
"""Build single-bucket and multi-function Core ML packages for gte-modernbert-base.

The single-bucket packages go through the same conversion code the production
ANE embedding packages were built with (bench/spikes/ane-minilm/
convert_modernbert_to_coreml.py): torch.export, fp16 compute precision,
CPU_AND_NE, macOS 14 deployment target, CLS pooling plus L2 normalization in the
graph, output renamed to `embedding`. The multi-function packages are then
assembled from those exact single-bucket packages with
`ct.utils.MultiFunctionDescriptor` and `ct.utils.save_multifunction`, which is
the coremltools path documented as deduplicating weights shared across
functions.

Artifacts are large and are written outside the repository, under
`~/.local/share/cortexkit/synapse/ane-multifunction/` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERTER_PATH = REPO_ROOT / "bench" / "spikes" / "ane-minilm" / "convert_modernbert_to_coreml.py"
DEFAULT_ROOT = Path.home() / ".local" / "share" / "cortexkit" / "synapse" / "ane-multifunction"


def load_converter() -> Any:
    """Import the production converter module by path so its code is reused verbatim."""
    spec = importlib.util.spec_from_file_location("convert_modernbert_to_coreml", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree_bytes(path: Path) -> int:
    """Apparent byte size of every regular file under `path`."""
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files(path: Path) -> dict[str, dict[str, Any]]:
    """Size and SHA-256 of every file inside a package or compiled bundle."""
    files: dict[str, dict[str, Any]] = {}
    for entry in sorted(path.rglob("*")):
        if entry.is_file():
            files[str(entry.relative_to(path))] = {
                "bytes": entry.stat().st_size,
                "sha256": sha256_file(entry),
            }
    return files


def build_single(converter: Any, model_ref: str, seq_len: int, out: Path) -> dict[str, Any]:
    """Convert one fixed-shape bucket through the production conversion function.

    The production converter refuses to save a package whose Core ML output drifts
    below 0.999 mean cosine from eager PyTorch on its smoke rows. That gate is kept
    for every bucket; when it trips (expected risk at 1024 because fp16 activations
    can overflow without rotation conditioning) the failure is recorded and the
    package is rebuilt with the identical `ct.convert` arguments but without the
    gate, so the multi-function experiment can still report what it sees.
    """
    import coremltools as ct  # pyright: ignore[reportMissingImports]
    import torch

    wrapper = converter.load_wrapper("embedder", model_ref, False)
    examples = converter.smoke_inputs("embedder", model_ref, seq_len, False)
    started = time.perf_counter()
    gate_error: str | None = None
    parity: dict[str, Any] | None = None
    try:
        mlmodel, parity_report = converter.convert_and_verify(
            wrapper, examples, "embedding", "embedder"
        )
        parity = parity_report.__dict__
    except RuntimeError as error:
        gate_error = str(error)
        with torch.inference_mode():
            exported = torch.export.export(wrapper, examples[0], strict=False)
        mlmodel = ct.convert(
            exported,
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        outputs = list(mlmodel.output_description)
        if outputs != ["embedding"]:
            ct.utils.rename_feature(mlmodel._spec, outputs[0], "embedding")
    convert_s = time.perf_counter() - started
    report = converter.ConversionReport(
        model_kind="embedder",
        source_model=model_ref,
        seq_len=seq_len,
        output_path=str(out),
        output_name="embedding",
        frontend="torch.export",
        compute_precision="float16",
        compute_units="CPU_AND_NE",
        parity=converter.ParityReport(**parity) if parity else converter.ParityReport(0, 0, None, 0, None, None),
        environment=converter.environment_report(),
    )
    converter.ensure_metadata(mlmodel, report)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    return {
        "seq_len": seq_len,
        "path": str(out),
        "convert_s": convert_s,
        "production_parity_gate": "passed" if gate_error is None else "failed",
        "production_parity_gate_error": gate_error,
        "parity": parity,
        "package_bytes": tree_bytes(out),
        "files": package_files(out),
    }


def build_multi(sources: dict[str, Path], default_function: str, out: Path) -> dict[str, Any]:
    """Assemble one multi-function package whose functions are the given single-bucket mains."""
    import coremltools as ct  # pyright: ignore[reportMissingImports]

    names = list(sources)
    descriptor = ct.utils.MultiFunctionDescriptor()
    for name in names:
        descriptor.add_function(str(sources[name]), src_function_name="main", target_function_name=name)
    descriptor.default_function_name = default_function
    if out.exists():
        shutil.rmtree(out)
    started = time.perf_counter()
    ct.utils.save_multifunction(descriptor, str(out))
    elapsed = time.perf_counter() - started
    loaded = ct.models.MLModel(str(out), skip_model_load=True)
    spec = loaded.get_spec()
    return {
        "functions": names,
        "default_function": default_function,
        "path": str(out),
        "save_multifunction_s": elapsed,
        "specification_version": spec.specificationVersion,
        "function_descriptions": [
            {
                "name": function.name,
                "inputs": [
                    {"name": feature.name, "shape": list(feature.type.multiArrayType.shape)}
                    for feature in function.input
                ],
                "outputs": [feature.name for feature in function.output],
            }
            for function in spec.description.functions
        ],
        "package_bytes": tree_bytes(out),
        "files": package_files(out),
    }


def environment() -> dict[str, Any]:
    import coremltools as ct  # pyright: ignore[reportMissingImports]
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "coremltools": ct.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--buckets", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--skip-singles", action="store_true", help="reuse existing single packages")
    args = parser.parse_args()

    converter = load_converter()
    model_ref, _ = converter.resolve_model_ref(converter.EMBEDDER_MODEL_ID)
    single_dir = args.root / "packages" / "single"
    multi_dir = args.root / "packages" / "multi"
    report: dict[str, Any] = {
        "environment": environment(),
        "source_model": converter.EMBEDDER_MODEL_ID,
        "source_snapshot": Path(model_ref).name,
        "singles": {},
        "multis": {},
    }
    singles: dict[int, Path] = {}
    for bucket in args.buckets:
        out = single_dir / f"embedder-seq{bucket}.mlpackage"
        singles[bucket] = out
        if args.skip_singles and out.exists():
            report["singles"][str(bucket)] = {
                "seq_len": bucket,
                "path": str(out),
                "package_bytes": tree_bytes(out),
                "files": package_files(out),
            }
            continue
        print(f"converting seq{bucket}", flush=True)
        report["singles"][str(bucket)] = build_single(converter, model_ref, bucket, out)

    variants = {"mf-128-256-512": [128, 256, 512]}
    if 1024 in singles:
        variants["mf-128-256-512-1024"] = [128, 256, 512, 1024]
    for name, buckets in variants.items():
        print(f"assembling {name}", flush=True)
        sources = {f"seq{bucket}": singles[bucket] for bucket in buckets}
        report["multis"][name] = build_multi(sources, "seq128", multi_dir / f"{name}.mlpackage")

    report_path = args.root / "reports" / "build.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "singles" and k != "multis"}, indent=2))
    for name, entry in {**report["singles"], **report["multis"]}.items():
        print(name, entry["package_bytes"])
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
