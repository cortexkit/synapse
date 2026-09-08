#!/usr/bin/env python3
"""Prepare, reference, export, and reload the bounded ModernBERT Core ML spike."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from tokenizers import Tokenizer  # pyright: ignore[reportMissingImports]

from modernbert_tiled import ModernBertConfig, build_embedder

MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
STAGES = (1024, 2048, 4096, 8192)
FIXTURE_TEXTS = (
    (
        "systems",
        "Represent this sentence for searching relevant passages: how does memory safety work? "
        "Rust enforces ownership and borrowing at compile time while permitting explicit, reviewed unsafe blocks.",
    ),
    (
        "multilingual",
        "İstanbul'da yağmur başladı. 東京では新しい鉄道路線を評価しています. "
        "Les chercheurs comparent la précision, la latence et la consommation d'énergie.",
    ),
    (
        "structured",
        "Experiment 17 measured 2.71828 ms; status=ready, retries=0. "
        "The control uses JSON {\"enabled\": true}, Unicode λ, and a final partial batch.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local GTE snapshot directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cpu = subparsers.add_parser("cpu-check")
    cpu.add_argument("--seq-len", type=int, default=129)
    cpu.add_argument("--query-tile", type=int, default=32)
    cpu.add_argument("--report", type=Path, required=True)

    prepare = subparsers.add_parser("prepare-input")
    _add_stage_shape_args(prepare)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--report", type=Path, required=True)

    reference = subparsers.add_parser("reference")
    _add_stage_shape_args(reference)
    reference.add_argument("--input", type=Path, required=True)
    reference.add_argument("--out", type=Path, required=True)
    reference.add_argument("--report", type=Path, required=True)

    export = subparsers.add_parser("export")
    _add_stage_shape_args(export)
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--reference", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--report", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")

    reload_parser = subparsers.add_parser("reload")
    reload_parser.add_argument("--package", type=Path, required=True)
    reload_parser.add_argument("--input", type=Path, required=True)
    reload_parser.add_argument("--reference", type=Path, required=True)
    reload_parser.add_argument("--warm-repetitions", type=int, default=3)
    reload_parser.add_argument(
        "--compute-units", choices=("cpu-and-ne", "cpu-only"), default="cpu-and-ne"
    )
    reload_parser.add_argument("--vectors-out", type=Path, required=True)
    reload_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _add_stage_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seq-len", type=int, required=True, choices=STAGES)
    parser.add_argument("--query-tile", type=int, default=256)
    parser.add_argument("--key-tile", type=int, default=256)


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


def model_digests(snapshot: Path) -> dict[str, str]:
    files = ("config.json", "model.safetensors", "tokenizer.json")
    values = {name: sha256_file(snapshot / name) for name in files}
    combined = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return {**values, "combined_sha256": combined}


def environment_report() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "coremltools", "transformers", "tokenizers", "safetensors", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    macos_build = subprocess.run(
        ["sw_vers", "-buildVersion"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "macos_build": macos_build,
        "machine": platform.machine(),
        "packages": packages,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tokenizer_for(snapshot: Path) -> Tokenizer:
    return Tokenizer.from_file(str(snapshot / "tokenizer.json"))


def _content_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def benchmark_rows(
    snapshot: Path, config: ModernBertConfig, sequence_length: int, query_tile_size: int
) -> list[dict[str, Any]]:
    tokenizer = tokenizer_for(snapshot)
    corpus = "\n".join(text for _, text in FIXTURE_TEXTS)
    corpus_ids = _content_ids(tokenizer, corpus)
    if not corpus_ids:
        raise RuntimeError("tokenizer produced no benchmark content tokens")

    max_content = sequence_length - 2
    repeated = (corpus_ids * ((max_content + len(corpus_ids) - 1) // len(corpus_ids)))[:max_content]
    targets = (
        ("full-context", repeated),
        (FIXTURE_TEXTS[1][0], _content_ids(tokenizer, FIXTURE_TEXTS[1][1])),
        (
            "partial-tile",
            (corpus_ids * 4)[: max(1, min(max_content, query_tile_size + 15))],
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, content in targets:
        active = [config.cls_token_id, *content[:max_content], config.sep_token_id]
        mask = [1] * len(active) + [0] * (sequence_length - len(active))
        ids = active + [config.pad_token_id] * (sequence_length - len(active))
        unique_ids = len(set(active))
        if unique_ids < min(16, len(active)):
            raise RuntimeError(f"benchmark row {label} is insufficiently diverse: {unique_ids} ids")
        rows.append(
            {
                "id": label,
                "input_ids": ids,
                "attention_mask": mask,
                "actual_token_count": len(active),
                "unique_active_token_ids": unique_ids,
            }
        )
    return rows


def row_tensors(rows: list[dict[str, Any]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.tensor([row["input_ids"]], dtype=torch.int32),
            torch.tensor([row["attention_mask"]], dtype=torch.int32),
        )
        for row in rows
    ]


def tensor_rows(values: Iterable[torch.Tensor | np.ndarray]) -> np.ndarray:
    arrays = []
    for value in values:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().float().numpy()
        arrays.append(np.asarray(value, dtype=np.float32).reshape(1, -1))
    return np.concatenate(arrays, axis=0)


def cosine_rows(reference: np.ndarray, candidate: np.ndarray) -> list[float]:
    numerator = np.sum(reference * candidate, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    return [float(value) for value in numerator / np.maximum(denominator, 1e-12)]


def parity_report(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    cosine = cosine_rows(reference, candidate)
    return {
        "max_abs": float(np.max(np.abs(reference - candidate))),
        "mean_abs": float(np.mean(np.abs(reference - candidate))),
        "cosine_by_row": cosine,
        "min_cosine": min(cosine),
    }


def save_vectors(path: Path, rows: list[dict[str, Any]], vectors: np.ndarray) -> None:
    write_jsonl(
        path,
        ({"id": row["id"], "vec": vector.tolist()} for row, vector in zip(rows, vectors, strict=True)),
    )


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    rows = read_jsonl(path)
    return [str(row["id"]) for row in rows], np.asarray([row["vec"] for row in rows], dtype=np.float32)


def inspect_exported_program(
    exported: torch.export.ExportedProgram, sequence_length: int, query_tile_size: int
) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    attention_tensors: list[dict[str, Any]] = []
    for node in exported.graph.nodes:
        values = node.meta.get("val")
        values = values if isinstance(values, (tuple, list)) else (values,)
        for value in values:
            if not isinstance(value, torch.Tensor):
                continue
            shape = [int(dimension) for dimension in value.shape]
            item = {
                "node": node.name,
                "target": str(node.target),
                "shape": shape,
                "numel": int(value.numel()),
                "dtype": str(value.dtype),
            }
            tensors.append(item)
            target = str(node.target)
            if "einsum" in target or "softmax" in target or "matmul" in target:
                attention_tensors.append(item)
    forbidden = [
        item
        for item in attention_tensors
        if sequence_length > query_tile_size and item["shape"].count(sequence_length) >= 2
    ]
    if forbidden:
        raise RuntimeError(f"export contains an unbounded sequence-square attention tensor: {forbidden[:3]}")
    return {
        "graph_sha256": hashlib.sha256(str(exported.graph).encode()).hexdigest(),
        "node_count": len(tuple(exported.graph.nodes)),
        "largest_tensor": max(tensors, key=lambda item: item["numel"]),
        "largest_attention_tensor": max(attention_tensors, key=lambda item: item["numel"]),
        "attention_tensor_count": len(attention_tensors),
        "forbidden_sequence_square_attention_tensors": forbidden,
    }


def inspect_mil_program(mlmodel: Any, sequence_length: int, query_tile_size: int) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    program = getattr(mlmodel, "_mil_program", None)
    if program is None:
        return {"available": False, "reason": "coremltools did not retain a MIL program"}
    for function in program.functions.values():
        for operation in function.operations:
            for output in operation.outputs:
                shape = []
                try:
                    shape = [int(dimension) for dimension in output.shape]
                except (TypeError, ValueError):
                    pass
                operations.append(
                    {
                        "operator": operation.op_type,
                        "name": operation.name,
                        "shape": shape,
                        "numel": int(np.prod(shape)) if shape else 0,
                    }
                )
    attention_ops = [
        item
        for item in operations
        if item["operator"] in {"matmul", "einsum", "softmax"}
    ]
    forbidden = [
        item
        for item in attention_ops
        if sequence_length > query_tile_size and item["shape"].count(sequence_length) >= 2
    ]
    if forbidden:
        raise RuntimeError(f"MIL lowering materialized sequence-square attention: {forbidden[:3]}")
    counts: dict[str, int] = {}
    for item in operations:
        counts[item["operator"]] = counts.get(item["operator"], 0) + 1
    return {
        "available": True,
        "operation_count": len(operations),
        "operator_counts": counts,
        "largest_attention_output": max(attention_ops, key=lambda item: item["numel"]),
        "forbidden_sequence_square_attention_outputs": forbidden,
    }


def command_prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = ModernBertConfig.from_snapshot(args.model)
    rows = benchmark_rows(args.model, config, args.seq_len, args.query_tile)
    write_jsonl(args.out, rows)
    return {
        "status": "passed",
        "sequence_length": args.seq_len,
        "query_tile_size": args.query_tile,
        "rows": [
            {
                "id": row["id"],
                "actual_token_count": row["actual_token_count"],
                "unique_active_token_ids": row["unique_active_token_ids"],
            }
            for row in rows
        ],
        "input_sha256": sha256_file(args.out),
        "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
        "model": model_digests(args.model),
        "environment": environment_report(),
    }


def _run_model(model: torch.nn.Module, inputs: list[tuple[torch.Tensor, torch.Tensor]]) -> np.ndarray:
    with torch.inference_mode():
        return tensor_rows(model(*item) for item in inputs)


def command_cpu_check(args: argparse.Namespace) -> dict[str, Any]:
    config = ModernBertConfig.from_snapshot(args.model)
    rows = benchmark_rows(args.model, config, args.seq_len, args.query_tile)
    inputs = row_tensors(rows)

    tiled, _ = build_embedder(args.model, args.seq_len, args.query_tile, attention_kind="query_tiled")
    tiled_vectors = _run_model(tiled, inputs)
    del tiled
    gc.collect()

    streaming, _ = build_embedder(
        args.model,
        args.seq_len,
        args.query_tile,
        key_tile_size=31,
        attention_kind="streaming_reference",
    )
    streaming_vectors = _run_model(streaming, inputs)
    del streaming
    gc.collect()

    from transformers import AutoModel  # pyright: ignore[reportMissingImports]

    hf_model = AutoModel.from_pretrained(
        args.model, local_files_only=True, attn_implementation="eager", dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        hf_vectors = tensor_rows(
            functional_normalized_cls(hf_model(ids.long(), attention_mask=mask.long()).last_hidden_state)
            for ids, mask in inputs
        )
    del hf_model
    gc.collect()

    tiled_streaming = parity_report(streaming_vectors, tiled_vectors)
    tiled_hf = parity_report(hf_vectors, tiled_vectors)
    if tiled_streaming["min_cosine"] < 0.999999:
        raise RuntimeError(f"tiled/streaming CPU parity failed: {tiled_streaming}")
    if tiled_hf["min_cosine"] < 0.99999:
        raise RuntimeError(f"tiled/Hugging Face CPU parity failed: {tiled_hf}")
    return {
        "status": "passed",
        "sequence_length": args.seq_len,
        "query_tile_size": args.query_tile,
        "fixtures": [row["id"] for row in rows],
        "tiled_vs_streaming_full_context": tiled_streaming,
        "tiled_vs_huggingface_eager": tiled_hf,
        "model": model_digests(args.model),
        "environment": environment_report(),
    }


def functional_normalized_cls(hidden: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(hidden[:, 0, :].float(), p=2, dim=-1)


def command_reference(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    inputs = row_tensors(rows)
    started = time.perf_counter()
    model, _ = build_embedder(
        args.model,
        args.seq_len,
        args.query_tile,
        key_tile_size=args.key_tile,
        attention_kind="streaming_reference",
    )
    vectors = _run_model(model, inputs)
    elapsed = time.perf_counter() - started
    save_vectors(args.out, rows, vectors)
    return {
        "status": "passed",
        "algorithm": "online-softmax full-context reference",
        "dense_8192_baseline_allocated": False,
        "sequence_length": args.seq_len,
        "query_tile_size": args.query_tile,
        "key_tile_size": args.key_tile,
        "rows": len(rows),
        "actual_token_counts": [row["actual_token_count"] for row in rows],
        "elapsed_s": elapsed,
        "input_sha256": sha256_file(args.input),
        "reference_sha256": sha256_file(args.out),
        "model": model_digests(args.model),
        "environment": environment_report(),
    }


def command_export(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct  # pyright: ignore[reportMissingImports]

    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite")
        shutil.rmtree(args.out) if args.out.is_dir() else args.out.unlink()
    rows = read_jsonl(args.input)
    reference_ids, reference = load_vectors(args.reference)
    if reference_ids != [row["id"] for row in rows]:
        raise ValueError("reference row ordering differs from input")
    inputs = row_tensors(rows)
    model, _ = build_embedder(args.model, args.seq_len, args.query_tile, attention_kind="query_tiled")
    eager = _run_model(model, inputs)
    eager_reference = parity_report(reference, eager)
    if eager_reference["min_cosine"] < 0.99999:
        raise RuntimeError(f"query-tiled eager/reference parity failed: {eager_reference}")

    export_started = time.perf_counter()
    exported = torch.export.export(model, inputs[0], strict=False).run_decompositions(
        {torch.ops.aten.alias.default: lambda value: value}
    )
    export_latency = time.perf_counter() - export_started
    exported_rows = _run_model(exported.module(), inputs)
    exported_parity = parity_report(eager, exported_rows)
    if exported_parity["max_abs"] > 1e-5:
        raise RuntimeError(f"torch.export parity failed: {exported_parity}")
    export_inspection = inspect_exported_program(exported, args.seq_len, args.query_tile)
    del model
    gc.collect()

    conversion_started = time.perf_counter()
    mlmodel = ct.convert(
        exported,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    conversion_latency = time.perf_counter() - conversion_started
    outputs = list(mlmodel.output_description)
    if len(outputs) != 1:
        raise RuntimeError(f"expected one model output, got {outputs}")
    if outputs[0] != "embedding":
        ct.utils.rename_feature(mlmodel._spec, outputs[0], "embedding")
    mil_inspection = inspect_mil_program(mlmodel, args.seq_len, args.query_tile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    mlmodel.save(args.out)
    save_latency = time.perf_counter() - save_started
    return {
        "status": "passed",
        "sequence_length": args.seq_len,
        "query_tile_size": args.query_tile,
        "key_tile_size": args.key_tile,
        "frontend": "torch.export",
        "compute_precision": "float16",
        "compute_units": "CPU_AND_NE",
        "skip_model_load": True,
        "timing_s": {
            "torch_export": export_latency,
            "coreml_conversion": conversion_latency,
            "package_save": save_latency,
        },
        "parity": {
            "eager_vs_streaming_reference": eager_reference,
            "eager_vs_exported": exported_parity,
        },
        "export_inspection": export_inspection,
        "mil_inspection": mil_inspection,
        "input_sha256": sha256_file(args.input),
        "reference_sha256": sha256_file(args.reference),
        "package_sha256": sha256_tree(args.out),
        "model": model_digests(args.model),
        "environment": environment_report(),
    }


def command_reload(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct  # pyright: ignore[reportMissingImports]

    rows = read_jsonl(args.input)
    reference_ids, reference = load_vectors(args.reference)
    if reference_ids != [row["id"] for row in rows]:
        raise ValueError("reference row ordering differs from input")
    input_dicts = [
        {
            "input_ids": np.asarray([row["input_ids"]], dtype=np.int32),
            "attention_mask": np.asarray([row["attention_mask"]], dtype=np.int32),
        }
        for row in rows
    ]
    load_started = time.perf_counter()
    compute_units = (
        ct.ComputeUnit.CPU_AND_NE if args.compute_units == "cpu-and-ne" else ct.ComputeUnit.CPU_ONLY
    )
    model = ct.models.MLModel(str(args.package), compute_units=compute_units)
    load_latency = time.perf_counter() - load_started

    first_started = time.perf_counter()
    first = model.predict(input_dicts[0])["embedding"]
    first_latency = time.perf_counter() - first_started
    predictions = [first]
    row_latencies = [first_latency]
    for item in input_dicts[1:]:
        started = time.perf_counter()
        predictions.append(model.predict(item)["embedding"])
        row_latencies.append(time.perf_counter() - started)
    warm_vectors = []
    warm_latencies = []
    for _ in range(args.warm_repetitions):
        started = time.perf_counter()
        warm_vectors.append(np.asarray(model.predict(input_dicts[0])["embedding"], dtype=np.float32))
        warm_latencies.append(time.perf_counter() - started)

    candidate = tensor_rows(predictions)
    parity = parity_report(reference, candidate)
    parity_threshold = 0.999
    status = "passed" if parity["min_cosine"] >= parity_threshold else "parity_failed"
    determinism_max_abs = max(
        (float(np.max(np.abs(warm_vectors[0] - item))) for item in warm_vectors[1:]), default=0.0
    )
    save_vectors(args.vectors_out, rows, candidate)
    return {
        "status": status,
        "compute_units": (
            "CPU_AND_NE (GPU excluded)" if args.compute_units == "cpu-and-ne" else "CPU_ONLY"
        ),
        "placement_claim": "runtime success alone does not prove ANE residency; join with MLComputePlan report",
        "timing_s": {
            "separate_process_load": load_latency,
            "first_predict": first_latency,
            "remaining_fixture_predicts": row_latencies[1:],
            "warm_predicts": warm_latencies,
        },
        "actual_token_counts": [row["actual_token_count"] for row in rows],
        "parity_threshold_min_cosine": parity_threshold,
        "parity_vs_streaming_full_context_reference": parity,
        "repeated_determinism_max_abs": determinism_max_abs,
        "input_sha256": sha256_file(args.input),
        "reference_sha256": sha256_file(args.reference),
        "vectors_sha256": sha256_file(args.vectors_out),
        "package_sha256": sha256_tree(args.package),
        "environment": environment_report(),
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "cpu-check":
        return command_cpu_check(args)
    if args.command == "prepare-input":
        return command_prepare(args)
    if args.command == "reference":
        return command_reference(args)
    if args.command == "export":
        return command_export(args)
    if args.command == "reload":
        return command_reload(args)
    raise AssertionError(args.command)


def main() -> int:
    args = parse_args()
    report_path: Path = args.report
    try:
        report = run_command(args)
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "status": "failed",
            "command": args.command,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "environment": environment_report(),
        }
        write_json(report_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
