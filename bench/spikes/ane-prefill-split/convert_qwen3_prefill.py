#!/usr/bin/env python3
"""Export fixed-window Qwen3 prefill with every layer's K/V as normal outputs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import coremltools as ct
import numpy as np
import torch
from torch.nn import functional
from transformers import AutoModelForCausalLM, AutoTokenizer


SPIKE_DIR = Path(__file__).resolve().parent
SHARED_CONVERTER = SPIKE_DIR.parent / "ane-spec-decode" / "convert_qwen3_to_coreml.py"
MODEL_ID = "Qwen/Qwen3-0.6B"
SMOKE_TEXT = "Explain why fixed-shape inference can reduce accelerator dispatch overhead."


def load_shared_converter() -> Any:
    spec = importlib.util.spec_from_file_location("ane_spec_decode_converter", SHARED_CONVERTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared Qwen3 converter from {SHARED_CONVERTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = load_shared_converter()


@dataclass(frozen=True)
class TensorParity:
    name: str
    shape: list[int]
    eager_hf_max_abs: float
    eager_hf_mean_cosine: float
    eager_export_max_abs: float
    eager_coreml_max_abs: float
    eager_coreml_mean_cosine: float


@dataclass(frozen=True)
class ConversionReport:
    status: str
    source_model: str
    source_model_sha256: str
    window: int
    frontend: str
    compute_precision: str
    compute_units: str
    minimum_deployment_target: str
    output_path: str
    output_count: int
    output_names: list[str]
    kv_output_elements: int
    kv_output_bytes_f16: int
    package_size_bytes: int
    conversion_s: float
    model_config: Any
    tokenizer_policy: Any
    environment: Any
    parity: list[TensorParity]


class Qwen3PrefillLayer(shared.Qwen3Layer):
    """A Qwen3 block that exposes the post-RoPE key and projected value tensors."""

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden
        normalized = self.input_norm(hidden)

        query = self.q_proj(normalized)
        key = self.k_proj(normalized)
        value = self.v_proj(normalized)
        query = query.reshape(1, self.query_heads, self.head_dim, self.window).permute(0, 1, 3, 2)
        key = key.reshape(1, self.kv_heads, self.head_dim, self.window).permute(0, 1, 3, 2)
        value = value.reshape(1, self.kv_heads, self.head_dim, self.window).permute(0, 1, 3, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query = shared.apply_rope(query, cos, sin)
        key = shared.apply_rope(key, cos, sin)
        attention_key = key.repeat_interleave(self.kv_repetition, dim=1)
        attention_value = value.repeat_interleave(self.kv_repetition, dim=1)

        padding_mask = (1.0 - attention_mask.to(dtype=torch.float32)).reshape(
            1, 1, 1, self.window
        )
        scores = torch.matmul(query.float(), attention_key.float().transpose(-1, -2)) * self.scale
        scores = scores + causal_mask + padding_mask * -10_000.0
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attended = torch.matmul(probabilities, attention_value.float())
        attended = attended.permute(0, 1, 3, 2).reshape(
            1, self.projection_size, 1, self.window
        )
        hidden = residual + self.o_proj(attended)

        residual = hidden
        normalized = self.post_attention_norm(hidden)
        gate = functional.silu(self.gate_proj(normalized))
        hidden = residual + self.down_proj(gate * self.up_proj(normalized))
        return hidden, key, value


class Qwen3Prefill(torch.nn.Module):
    """Fixed-window causal prefill returning logits followed by 28 K/V pairs."""

    def __init__(self, config: Any, tensors: dict[str, torch.Tensor], window: int) -> None:
        super().__init__()
        self.config = config
        self.window = window
        self.embedding_weight = torch.nn.Parameter(
            shared.tensor_for(tensors, "embed_tokens.weight"), requires_grad=False
        )
        layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"layers.{index}"
            layers.append(
                Qwen3PrefillLayer(
                    config=config,
                    input_norm=shared.tensor_for(tensors, f"{prefix}.input_layernorm.weight"),
                    post_attention_norm=shared.tensor_for(
                        tensors, f"{prefix}.post_attention_layernorm.weight"
                    ),
                    q_proj=shared.tensor_for(tensors, f"{prefix}.self_attn.q_proj.weight"),
                    q_norm=shared.tensor_for(tensors, f"{prefix}.self_attn.q_norm.weight"),
                    k_proj=shared.tensor_for(tensors, f"{prefix}.self_attn.k_proj.weight"),
                    k_norm=shared.tensor_for(tensors, f"{prefix}.self_attn.k_norm.weight"),
                    v_proj=shared.tensor_for(tensors, f"{prefix}.self_attn.v_proj.weight"),
                    o_proj=shared.tensor_for(tensors, f"{prefix}.self_attn.o_proj.weight"),
                    gate_proj=shared.tensor_for(tensors, f"{prefix}.mlp.gate_proj.weight"),
                    up_proj=shared.tensor_for(tensors, f"{prefix}.mlp.up_proj.weight"),
                    down_proj=shared.tensor_for(tensors, f"{prefix}.mlp.down_proj.weight"),
                    window=window,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.final_norm = shared.ChannelRmsNorm(
            shared.tensor_for(tensors, "norm.weight"), config.rms_norm_eps
        )
        lm_head_weight = tensors.get("lm_head.weight")
        if lm_head_weight is None:
            lm_head_weight = tensors.get("model.lm_head.weight")
        if lm_head_weight is None:
            if not config.tie_word_embeddings:
                raise KeyError("checkpoint has no lm_head.weight and embeddings are not tied")
            lm_head_weight = shared.tensor_for(tensors, "embed_tokens.weight")
        self.lm_head = shared.Conv1x1(lm_head_weight)
        cos, sin = shared.build_rope_tables(window, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)
        causal = torch.zeros((1, 1, window, window), dtype=torch.float32)
        causal = causal.masked_fill(
            torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1), -10_000.0
        )
        self.register_buffer("causal_mask", causal, persistent=True)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        hidden = functional.embedding(input_ids, self.embedding_weight)
        hidden = hidden.transpose(1, 2).unsqueeze(2)
        caches: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, key, value = layer(
                hidden, attention_mask, self.causal_mask, self.cos, self.sin
            )
            caches.extend((key, value))
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden[:, :, :, -1:]).squeeze(3)
        return (logits, *caches)


def output_names(layer_count: int) -> list[str]:
    names = ["logits"]
    for layer in range(layer_count):
        names.extend((f"key_{layer:02d}", f"value_{layer:02d}"))
    return names


def flatten(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def hf_outputs(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    with torch.inference_mode():
        result = model(
            input_ids=input_ids.to(dtype=torch.long),
            attention_mask=attention_mask.to(dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
    caches = result.past_key_values
    legacy = caches.to_legacy_cache() if hasattr(caches, "to_legacy_cache") else caches
    values: list[torch.Tensor] = [result.logits[:, -1, :]]
    for key, value in legacy:
        values.extend((key, value))
    return tuple(values)


def ensure_metadata(mlmodel: ct.models.MLModel, report: ConversionReport) -> None:
    mlmodel.short_description = "Stateless Qwen3-0.6B prefill with explicit per-layer K/V outputs."
    mlmodel.author = "Synapse bench/spikes/ane-prefill-split"
    mlmodel.license = "Source model license follows the referenced Hugging Face checkpoint."
    metadata = mlmodel.user_defined_metadata
    metadata["synapse.source_model"] = report.source_model
    metadata["synapse.source_model_sha256"] = report.source_model_sha256
    metadata["synapse.window"] = str(report.window)
    metadata["synapse.frontend"] = report.frontend
    metadata["synapse.output_kind"] = "logits-and-explicit-kv"
    metadata["synapse.output_count"] = str(report.output_count)
    metadata["synapse.tokenizer_policy"] = json.dumps(
        asdict(report.tokenizer_policy), sort_keys=True
    )
    metadata["synapse.torch_version"] = report.environment.torch
    metadata["synapse.coremltools_version"] = report.environment.coremltools
    metadata["synapse.transformers_version"] = report.environment.transformers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF repo id or local snapshot")
    parser.add_argument("--window", type=int, required=True, choices=(32, 128, 256, 512))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = shared.environment_report()
    shared.require_environment(environment)
    model_ref, report_model_ref = shared.resolve_model_ref(args.model)
    model_path = Path(model_ref)
    if not model_path.exists():
        raise FileNotFoundError(f"Qwen3 local snapshot is required: {model_ref}")
    for path in (args.out, args.report_json):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
            shared.remove_path(path)

    config = shared.read_config(model_ref)
    wrapper = Qwen3Prefill(config, shared.load_tensors(model_ref), args.window).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, local_files_only=not args.allow_download
    )
    input_ids, attention_mask, tokenizer_policy = shared.prepare_input(
        tokenizer, config, (SMOKE_TEXT + " ") * args.window, args.window
    )
    if not bool(torch.all(attention_mask == 1)):
        raise RuntimeError("conversion smoke input must fill the fixed window without padding")
    names = output_names(config.num_hidden_layers)

    started = time.monotonic()
    with torch.inference_mode():
        eager = wrapper(input_ids, attention_mask)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=not args.allow_download,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval()
    hf = hf_outputs(hf_model, input_ids, attention_mask)
    del hf_model
    gc.collect()
    with torch.inference_mode():
        exported = torch.export.export(
            wrapper, (input_ids, attention_mask), strict=False
        )
        exported_values = exported.module()(input_ids, attention_mask)
    if len(eager) != len(names):
        raise RuntimeError(f"wrapper returned {len(eager)} outputs, expected {len(names)}")

    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, args.window), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, args.window), dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    converted_names = list(mlmodel.output_description)
    if len(converted_names) != len(names):
        raise RuntimeError(
            f"Core ML produced {len(converted_names)} outputs, expected {len(names)}: {converted_names}"
        )
    for old, new in zip(converted_names, names, strict=True):
        if old != new:
            ct.utils.rename_feature(mlmodel._spec, old, new)

    del exported, wrapper
    gc.collect()
    prediction = mlmodel.predict(
        {
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attention_mask.numpy().astype(np.int32),
        }
    )
    if len(hf) != len(names):
        raise RuntimeError(f"HF returned {len(hf)} outputs, expected {len(names)}")

    parity: list[TensorParity] = []
    for name, prediction_name, eager_value, exported_value, hf_value in zip(
        names, converted_names, eager, exported_values, hf, strict=True
    ):
        eager_row = flatten(eager_value)
        exported_row = flatten(exported_value)
        hf_row = flatten(hf_value)
        coreml_row = flatten(prediction[prediction_name])
        if not (eager_row.shape == exported_row.shape == hf_row.shape == coreml_row.shape):
            raise RuntimeError(
                f"{name} shape mismatch: eager={eager_row.shape}, export={exported_row.shape}, "
                f"hf={hf_row.shape}, coreml={coreml_row.shape}"
            )
        parity.append(
            TensorParity(
                name=name,
                shape=list(eager_value.shape),
                eager_hf_max_abs=float(np.max(np.abs(eager_row - hf_row))),
                eager_hf_mean_cosine=cosine(eager_row, hf_row),
                eager_export_max_abs=float(np.max(np.abs(eager_row - exported_row))),
                eager_coreml_max_abs=float(np.max(np.abs(eager_row - coreml_row))),
                eager_coreml_mean_cosine=cosine(eager_row, coreml_row),
            )
        )

    worst_hf_cosine = min(row.eager_hf_mean_cosine for row in parity)
    worst_export_abs = max(row.eager_export_max_abs for row in parity)
    if worst_hf_cosine < 0.99999:
        worst_rows = sorted(parity, key=lambda row: row.eager_hf_mean_cosine)[:5]
        raise RuntimeError(
            "custom prefill disagrees with HF: "
            f"worst cosine {worst_hf_cosine}; rows={worst_rows!r}"
        )
    if worst_export_abs > 1e-5:
        raise RuntimeError(f"torch.export parity failed: worst max abs {worst_export_abs}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    provisional = ConversionReport(
        status="converted_and_ran",
        source_model=report_model_ref,
        source_model_sha256=shared.sha256(model_path / "model.safetensors"),
        window=args.window,
        frontend="torch.export",
        compute_precision="float16",
        compute_units="CPU_AND_NE",
        minimum_deployment_target="macOS14",
        output_path=str(args.out),
        output_count=len(names),
        output_names=names,
        kv_output_elements=sum(value.numel() for value in eager[1:]),
        kv_output_bytes_f16=sum(value.numel() for value in eager[1:]) * 2,
        package_size_bytes=0,
        conversion_s=time.monotonic() - started,
        model_config=config,
        tokenizer_policy=tokenizer_policy,
        environment=environment,
        parity=parity,
    )
    ensure_metadata(mlmodel, provisional)
    mlmodel.save(args.out)
    report = ConversionReport(
        **{**asdict(provisional), "package_size_bytes": shared.directory_size(args.out)}
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
