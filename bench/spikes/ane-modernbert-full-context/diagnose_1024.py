#!/usr/bin/env python3
"""Isolate 1024-token divergence at embedding, attention, MLP, norm, and pooling boundaries."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import modernbert_tiled as tiled
from spike import (
    environment_report,
    inspect_exported_program,
    inspect_mil_program,
    model_digests,
    parity_report,
    read_jsonl,
    row_tensors,
    sha256_file,
    sha256_tree,
    write_json,
)

SEQUENCE_LENGTH = 1024
QUERY_TILE = 256
KEY_TILE = 256


class DiagnosticWrapper(torch.nn.Module):
    def __init__(self, model: tiled.ModernBertEmbedder) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_diagnostics(input_ids, attention_mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--out", type=Path, required=True)
    reference.add_argument("--report", type=Path, required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--report", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")

    ranges = subparsers.add_parser("ranges")
    ranges.add_argument("--report", type=Path, required=True)
    ranges.add_argument("--rotation", choices=("none", "hadamard"), default="none")
    ranges.add_argument("--rotation-seed", type=int, default=0)

    reload_parser = subparsers.add_parser("reload")
    reload_parser.add_argument("--package", type=Path, required=True)
    reload_parser.add_argument("--reference", type=Path, required=True)
    reload_parser.add_argument("--compute-units", choices=("cpu-and-ne", "cpu-only"), required=True)
    reload_parser.add_argument("--out", type=Path, required=True)
    reload_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sampled(value: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    return value[:, positions, :]


def hf_checkpoints(
    snapshot: Path,
    inputs: list[tuple[torch.Tensor, torch.Tensor]],
    positions: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, attn_implementation="eager", dtype=torch.float32
    ).eval()
    all_checkpoints = []
    all_embeddings = []
    for input_ids, attention_mask in inputs:
        layer_inputs: dict[int, torch.Tensor] = {}
        attention_outputs: dict[int, torch.Tensor] = {}
        layer_outputs: dict[int, torch.Tensor] = {}
        token_embedding_output: list[torch.Tensor] = []
        embedding_output: list[torch.Tensor] = []
        final_output: list[torch.Tensor] = []
        handles = [
            model.embeddings.tok_embeddings.register_forward_hook(
                lambda _module, _inputs, output: token_embedding_output.append(output.detach())
            ),
            model.embeddings.register_forward_hook(
                lambda _module, _inputs, output: embedding_output.append(output.detach())
            ),
            model.final_norm.register_forward_hook(
                lambda _module, _inputs, output: final_output.append(output.detach())
            ),
        ]
        for index, layer in enumerate(model.layers):
            handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, arguments, layer_index=index: layer_inputs.__setitem__(
                        layer_index, arguments[0].detach()
                    )
                )
            )
            handles.append(
                layer.attn.register_forward_hook(
                    lambda _module, _inputs, output, layer_index=index: attention_outputs.__setitem__(
                        layer_index, (output[0] if isinstance(output, tuple) else output).detach()
                    )
                )
            )
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, layer_index=index: layer_outputs.__setitem__(
                        layer_index, (output[0] if isinstance(output, tuple) else output).detach()
                    )
                )
            )
        with torch.inference_mode():
            result = model(input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        for handle in handles:
            handle.remove()
        checkpoints = [
            sampled(token_embedding_output[0], positions),
            sampled(embedding_output[0], positions),
        ]
        for index in range(len(model.layers)):
            checkpoints.append(sampled(layer_inputs[index] + attention_outputs[index], positions))
            checkpoints.append(sampled(layer_outputs[index], positions))
        checkpoints.append(sampled(final_output[0], positions))
        all_checkpoints.append(torch.stack(checkpoints, dim=1).cpu().float().numpy())
        all_embeddings.append(
            torch.nn.functional.normalize(result[:, 0, :].float(), p=2, dim=-1).cpu().numpy()
        )
    del model
    gc.collect()
    return np.concatenate(all_checkpoints, axis=0), np.concatenate(all_embeddings, axis=0)


def checkpoint_report(
    reference: np.ndarray, candidate: np.ndarray, labels: tuple[str, ...]
) -> dict[str, Any]:
    checkpoints = []
    for index, label in enumerate(labels):
        reference_rows = reference[:, index].reshape(-1, reference.shape[-1])
        candidate_rows = candidate[:, index].reshape(-1, candidate.shape[-1])
        metrics = parity_report(reference_rows, candidate_rows)
        checkpoints.append(
            {
                "checkpoint": label,
                "reference_abs_max": float(np.max(np.abs(reference_rows))),
                "candidate_abs_max": float(np.max(np.abs(candidate_rows))),
                **metrics,
            }
        )
    first_over_1e5 = next(
        (item["checkpoint"] for item in checkpoints if item["max_abs"] > 1e-5), None
    )
    first_below_0999 = next(
        (item["checkpoint"] for item in checkpoints if item["min_cosine"] < 0.999), None
    )
    return {
        "first_checkpoint_max_abs_over_1e-5": first_over_1e5,
        "first_checkpoint_min_cosine_below_0.999": first_below_0999,
        "checkpoints": checkpoints,
    }


def active_position_focus(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: tuple[str, ...],
    rows: list[dict[str, Any]],
    positions: tuple[int, ...],
) -> dict[str, Any]:
    entries = []
    for row_index, row in enumerate(rows):
        for position_index, position in enumerate(positions):
            if position >= int(row["actual_token_count"]):
                continue
            reference_path = reference[row_index, :, position_index]
            candidate_path = candidate[row_index, :, position_index]
            numerator = np.sum(reference_path * candidate_path, axis=-1)
            denominator = np.linalg.norm(reference_path, axis=-1) * np.linalg.norm(
                candidate_path, axis=-1
            )
            cosine = numerator / np.maximum(denominator, 1e-12)
            first_bad = np.flatnonzero(cosine < 0.999)
            worst_index = int(np.argmin(cosine))
            entries.append(
                {
                    "row_id": row["id"],
                    "position": int(position),
                    "first_checkpoint_below_0.999": (
                        labels[int(first_bad[0])] if first_bad.size else None
                    ),
                    "worst_checkpoint": labels[worst_index],
                    "worst_cosine": float(cosine[worst_index]),
                }
            )
    return {"active_sampled_positions": entries}


def mask_and_rope_checks(
    model: tiled.ModernBertEmbedder,
    inputs: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    input_ids, attention_mask = inputs[0]
    sequence_length = input_ids.shape[1]
    global_tiles = []
    local_tiles = []
    for start, end in tiled.tile_ranges(sequence_length, QUERY_TILE):
        global_tiles.append(
            tiled._tile_additive_mask(
                attention_mask, start, end, 0, sequence_length, None, torch.float32
            ).permute(0, 2, 3, 1).expand(1, 1, end - start, sequence_length)
        )
        local_key_start = max(0, start - 64)
        local_key_end = min(sequence_length, end + 64)
        local_full = torch.full(
            (1, 1, end - start, sequence_length), tiled.MASK_MIN_VALUE
        )
        local_full[:, :, :, local_key_start:local_key_end] = tiled._tile_additive_mask(
            attention_mask,
            start,
            end,
            local_key_start,
            local_key_end,
            64,
            torch.float32,
        ).permute(0, 2, 3, 1)
        local_tiles.append(local_full)
    reconstructed_global = torch.cat(global_tiles, dim=2)
    reconstructed_local = torch.cat(local_tiles, dim=2)
    padding = (1.0 - attention_mask.float()).reshape(1, 1, 1, sequence_length)
    dense_global = padding * tiled.MASK_MIN_VALUE
    positions = torch.arange(sequence_length)
    permitted = (positions[:, None] - positions[None, :]).abs() <= 64
    dense_local = torch.where(
        permitted.reshape(1, 1, sequence_length, sequence_length),
        dense_global,
        torch.full((1, 1, sequence_length, sequence_length), tiled.MASK_MIN_VALUE),
    )

    with torch.inference_mode():
        hidden = model._embed(input_ids)
        layer = cast(tiled.ModernBertLayer, model.layers[0])
        qkv = layer.qkv(layer.attention_norm(hidden)).reshape(
            1, 3, layer.heads, layer.head_dim, sequence_length
        )
        query, key, _ = qkv.unbind(dim=1)
        rope_query, rope_key = tiled.apply_rope(
            query, key, model.global_cos, model.global_sin
        )
        query_conventional = query.permute(0, 1, 3, 2)
        key_conventional = key.permute(0, 1, 3, 2)
        cos = model.global_cos.permute(0, 1, 3, 2)
        sin = model.global_sin.permute(0, 1, 3, 2)

        def independent_rope(value: torch.Tensor) -> torch.Tensor:
            half = value.shape[-1] // 2
            rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
            return (value * cos + rotated * sin).permute(0, 1, 3, 2)

        expected_query = independent_rope(query_conventional)
        expected_key = independent_rope(key_conventional)
    return {
        "global_mask_max_abs": float((reconstructed_global - dense_global).abs().max()),
        "local_mask_max_abs": float((reconstructed_local - dense_local).abs().max()),
        "global_mask_shape": list(reconstructed_global.shape),
        "local_dense_allocated_only_for_1024_diagnostic": True,
        "rope_query_max_abs": float((rope_query - expected_query).abs().max()),
        "rope_key_max_abs": float((rope_key - expected_key).abs().max()),
    }


def activation_stats(value: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(value)
    absolute = value.detach().float().abs()
    absolute_max = float(absolute.max())
    bounded_for_spacing = min(absolute_max, float(torch.finfo(torch.float16).max))
    return {
        "shape": list(value.shape),
        "abs_max": absolute_max,
        "nonfinite_count": int((~finite).sum()),
        "above_float16_max_count": int((absolute > torch.finfo(torch.float16).max).sum()),
        "float16_spacing_at_abs_max": float(np.spacing(np.float16(bounded_for_spacing))),
    }


def command_ranges(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    inputs = row_tensors(rows)
    model, _ = tiled.build_embedder(
        args.model,
        SEQUENCE_LENGTH,
        QUERY_TILE,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
    )
    rotation = model.rotation_report()
    input_ids, attention_mask = inputs[0]
    layers = []
    with torch.inference_mode():
        hidden = model._embed(input_ids)
        embedding_stats = activation_stats(hidden)
        for index, layer_module in enumerate(model.layers):
            layer = cast(tiled.ModernBertLayer, layer_module)
            attention_normalized = layer.attention_norm(hidden)
            qkv = layer.qkv(attention_normalized)
            after_attention = layer._attention_block(
                hidden,
                attention_mask,
                model.global_cos,
                model.global_sin,
                model.local_cos,
                model.local_sin,
            )
            mlp_normalized = layer.mlp_norm(after_attention)
            mlp_input = layer.mlp_input(mlp_normalized)
            activation, gate = mlp_input.chunk(2, dim=1)
            gated = torch.nn.functional.gelu(activation) * gate
            output = layer._mlp_block(after_attention)
            layers.append(
                {
                    "layer": index,
                    "attention_kind": "global" if layer.local_window is None else "local",
                    "residual_input": activation_stats(hidden),
                    "attention_normalized": activation_stats(attention_normalized),
                    "qkv": activation_stats(qkv),
                    "after_attention_residual": activation_stats(after_attention),
                    "mlp_normalized": activation_stats(mlp_normalized),
                    "mlp_input": activation_stats(mlp_input),
                    "gated_mlp": activation_stats(gated),
                    "layer_output": activation_stats(output),
                }
            )
            hidden = output
        final_normalized = model.output_unrotate(model.final_norm(hidden))
    return {
        "status": "passed",
        "row_id": rows[0]["id"],
        "actual_token_count": rows[0]["actual_token_count"],
        "rotation": rotation,
        "embedding_norm": embedding_stats,
        "layers": layers,
        "final_norm": activation_stats(final_normalized),
        "input_sha256": sha256_file(args.input),
        "model": model_digests(args.model),
        "environment": environment_report(),
    }


def command_reference(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    inputs = row_tensors(rows)
    model, config = tiled.build_embedder(args.model, SEQUENCE_LENGTH, QUERY_TILE)
    positions = tiled.diagnostic_positions(SEQUENCE_LENGTH)
    labels = tiled.diagnostic_checkpoint_labels(config.num_hidden_layers)
    with torch.inference_mode():
        custom_outputs = [model.forward_diagnostics(*item) for item in inputs]
        custom_checkpoints = np.concatenate(
            [output[0].cpu().float().numpy() for output in custom_outputs], axis=0
        )
        custom_embeddings = np.concatenate(
            [output[1].cpu().float().numpy() for output in custom_outputs], axis=0
        )
        exported = torch.export.export(DiagnosticWrapper(model), inputs[0], strict=False).run_decompositions(
            {torch.ops.aten.alias.default: lambda value: value}
        )
        exported_outputs = [exported.module()(*item) for item in inputs]
        exported_checkpoints = np.concatenate(
            [output[0].cpu().float().numpy() for output in exported_outputs], axis=0
        )
        exported_embeddings = np.concatenate(
            [output[1].cpu().float().numpy() for output in exported_outputs], axis=0
        )
        structure_checks = mask_and_rope_checks(model, inputs)
    del model, exported
    gc.collect()
    hf_values, hf_embeddings = hf_checkpoints(args.model, inputs, positions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        custom_checkpoints=custom_checkpoints,
        custom_embeddings=custom_embeddings,
        exported_checkpoints=exported_checkpoints,
        exported_embeddings=exported_embeddings,
        hf_checkpoints=hf_values,
        hf_embeddings=hf_embeddings,
        positions=np.asarray(positions, dtype=np.int32),
        labels=np.asarray(labels),
    )
    return {
        "status": "passed",
        "sequence_length": SEQUENCE_LENGTH,
        "row_ids": [row["id"] for row in rows],
        "positions": positions,
        "checkpoint_labels": labels,
        "structure_checks": structure_checks,
        "custom_vs_huggingface": checkpoint_report(hf_values, custom_checkpoints, labels),
        "custom_vs_huggingface_active_focus": active_position_focus(
            hf_values, custom_checkpoints, labels, rows, positions
        ),
        "exported_vs_custom": checkpoint_report(custom_checkpoints, exported_checkpoints, labels),
        "pooling": {
            "custom_vs_huggingface": parity_report(hf_embeddings, custom_embeddings),
            "exported_vs_custom": parity_report(custom_embeddings, exported_embeddings),
        },
        "input_sha256": sha256_file(args.input),
        "diagnostic_npz_sha256": sha256_file(args.out),
        "model": model_digests(args.model),
        "environment": environment_report(),
    }


def command_export(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct

    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite")
        shutil.rmtree(args.out) if args.out.is_dir() else args.out.unlink()
    rows = read_jsonl(args.input)
    inputs = row_tensors(rows)
    model, _ = tiled.build_embedder(args.model, SEQUENCE_LENGTH, QUERY_TILE)
    started = time.perf_counter()
    exported = torch.export.export(DiagnosticWrapper(model), inputs[0], strict=False).run_decompositions(
        {torch.ops.aten.alias.default: lambda value: value}
    )
    export_s = time.perf_counter() - started
    graph = inspect_exported_program(exported, SEQUENCE_LENGTH, QUERY_TILE)
    del model
    gc.collect()
    started = time.perf_counter()
    mlmodel = cast(
        Any,
        ct.convert(
            exported,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            skip_model_load=True,
        ),
    )
    conversion_s = time.perf_counter() - started
    outputs = list(mlmodel.output_description)
    if len(outputs) != 2:
        raise RuntimeError(f"diagnostic model expected two outputs, got {outputs}")
    ct.utils.rename_feature(mlmodel._spec, outputs[0], "checkpoints")
    ct.utils.rename_feature(mlmodel._spec, outputs[1], "embedding")
    mil = inspect_mil_program(mlmodel, SEQUENCE_LENGTH, QUERY_TILE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mlmodel.save(args.out)
    save_s = time.perf_counter() - started
    return {
        "status": "passed",
        "skip_model_load": True,
        "timing_s": {"torch_export": export_s, "coreml_conversion": conversion_s, "save": save_s},
        "export_inspection": graph,
        "mil_inspection": mil,
        "package_sha256": sha256_tree(args.out),
        "input_sha256": sha256_file(args.input),
        "environment": environment_report(),
    }


def command_reload(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct

    reference = np.load(args.reference)
    rows = read_jsonl(args.input)
    compute_units = (
        ct.ComputeUnit.CPU_AND_NE if args.compute_units == "cpu-and-ne" else ct.ComputeUnit.CPU_ONLY
    )
    load_started = time.perf_counter()
    model = ct.models.MLModel(str(args.package), compute_units=compute_units)
    load_s = time.perf_counter() - load_started
    checkpoint_rows = []
    embedding_rows = []
    predict_s = []
    for row in rows:
        values = {
            "input_ids": np.asarray([row["input_ids"]], dtype=np.int32),
            "attention_mask": np.asarray([row["attention_mask"]], dtype=np.int32),
        }
        started = time.perf_counter()
        prediction = model.predict(values)
        predict_s.append(time.perf_counter() - started)
        checkpoint_rows.append(np.asarray(prediction["checkpoints"], dtype=np.float32))
        embedding_rows.append(np.asarray(prediction["embedding"], dtype=np.float32).reshape(1, -1))
    checkpoints = np.concatenate(checkpoint_rows, axis=0)
    embeddings = np.concatenate(embedding_rows, axis=0)
    labels = tuple(str(value) for value in reference["labels"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, checkpoints=checkpoints, embeddings=embeddings)
    return {
        "status": "passed",
        "compute_units": args.compute_units,
        "timing_s": {"load": load_s, "predict_by_row": predict_s},
        "vs_custom": checkpoint_report(reference["custom_checkpoints"], checkpoints, labels),
        "vs_custom_active_focus": active_position_focus(
            reference["custom_checkpoints"],
            checkpoints,
            labels,
            rows,
            tuple(int(value) for value in reference["positions"]),
        ),
        "vs_huggingface": checkpoint_report(reference["hf_checkpoints"], checkpoints, labels),
        "pooling": {
            "vs_custom": parity_report(reference["custom_embeddings"], embeddings),
            "vs_huggingface": parity_report(reference["hf_embeddings"], embeddings),
        },
        "output_npz_sha256": sha256_file(args.out),
        "reference_npz_sha256": sha256_file(args.reference),
        "package_sha256": sha256_tree(args.package),
        "environment": environment_report(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "ranges":
        return command_ranges(args)
    if args.command == "reference":
        return command_reference(args)
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
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
