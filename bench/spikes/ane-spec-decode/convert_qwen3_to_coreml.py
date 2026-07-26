#!/usr/bin/env python3
"""Convert Qwen3-0.6B causal decoding windows to fixed-shape Core ML.

The graph re-encodes a left-padded token window and returns logits for the final
K positions.  In ``--unroll-k`` mode it explicitly repeats the greedy pass K
times inside the exported graph, returning the K generated token IDs.  It
deliberately has no mutable KV state: Phase A measures the known-good stateless
workaround for the Core ML 8.3 stateful ANE failure.
"""

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
import torch.nn.functional as functional
import transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
SMOKE_TEXTS = (
    "Explain why fixed-shape neural network graphs can be easier to deploy.",
    "The quick brown fox jumps over the lazy dog.",
)
PINNED_TORCH = "2.5.1"
PINNED_TRANSFORMERS = "4.51.3"


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
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    eos_token_id: int
    pad_token_id: int
    tie_word_embeddings: bool


@dataclass(frozen=True)
class TokenizerPolicy:
    add_special_tokens: bool
    pad_token_id: int
    padding_side: str
    position_policy: str


@dataclass(frozen=True)
class ParityReport:
    rows: int
    last_k: int
    wrapper_hf_max_abs: float
    wrapper_hf_mean_cosine: float
    eager_export_max_abs: float
    eager_export_mean_cosine: float
    eager_coreml_max_abs: float
    eager_coreml_mean_cosine: float
    mode: str = "last_k"
    token_count: int = 0
    wrapper_hf_token_agreements: int = 0
    eager_export_token_agreements: int = 0
    eager_coreml_token_agreements: int = 0


@dataclass(frozen=True)
class ConversionReport:
    source_model: str
    source_model_sha256: str
    window: int
    last_k: int
    unroll_k: int | None
    mode: str
    output_path: str
    output_name: str
    output_kind: str
    frontend: str
    compute_precision: str
    compute_units: str
    minimum_deployment_target: str
    package_size_bytes: int
    model_config: ModelConfig
    tokenizer_policy: TokenizerPolicy
    parity: ParityReport
    environment: EnvironmentReport
    conversion_s: float


class Conv1x1(torch.nn.Module):
    """Represent a frozen linear projection as an ANE-friendly 1x1 convolution."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"projection weight must be rank 2, got {tuple(weight.shape)}")
        self.weight = torch.nn.Parameter(
            weight.to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1), requires_grad=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(hidden, self.weight)


class ChannelRmsNorm(torch.nn.Module):
    """Qwen RMSNorm over channels of a [batch, hidden, 1, window] tensor."""

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            weight.to(dtype=torch.float32).reshape(1, -1, 1, 1), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=1, keepdim=True)
        return hidden * torch.rsqrt(variance + self.eps) * self.weight


class HeadRmsNorm(torch.nn.Module):
    """Qwen Q/K RMSNorm over each attention head's feature dimension."""

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            weight.to(dtype=torch.float32).reshape(1, 1, 1, -1), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + self.eps) * self.weight


class Qwen3Layer(torch.nn.Module):
    """One fixed-window Qwen3 decoder block with causal grouped-query attention."""

    def __init__(
        self,
        *,
        config: ModelConfig,
        input_norm: torch.Tensor,
        post_attention_norm: torch.Tensor,
        q_proj: torch.Tensor,
        q_norm: torch.Tensor,
        k_proj: torch.Tensor,
        k_norm: torch.Tensor,
        v_proj: torch.Tensor,
        o_proj: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        window: int,
    ) -> None:
        super().__init__()
        self.input_norm = ChannelRmsNorm(input_norm, config.rms_norm_eps)
        self.post_attention_norm = ChannelRmsNorm(post_attention_norm, config.rms_norm_eps)
        self.q_proj = Conv1x1(q_proj)
        self.q_norm = HeadRmsNorm(q_norm, config.rms_norm_eps)
        self.k_proj = Conv1x1(k_proj)
        self.k_norm = HeadRmsNorm(k_norm, config.rms_norm_eps)
        self.v_proj = Conv1x1(v_proj)
        self.o_proj = Conv1x1(o_proj)
        self.gate_proj = Conv1x1(gate_proj)
        self.up_proj = Conv1x1(up_proj)
        self.down_proj = Conv1x1(down_proj)
        self.projection_size = config.num_attention_heads * config.head_dim
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = window
        self.kv_repetition = self.query_heads // self.kv_heads
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
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
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        key = key.repeat_interleave(self.kv_repetition, dim=1)
        value = value.repeat_interleave(self.kv_repetition, dim=1)

        padding_mask = (1.0 - attention_mask.to(dtype=torch.float32)).reshape(1, 1, 1, self.window)
        padding_mask = padding_mask * -10_000.0
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * self.scale
        scores = scores + causal_mask + padding_mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attended = torch.matmul(probabilities, value.float())
        attended = attended.permute(0, 1, 3, 2).reshape(
            1, self.projection_size, 1, self.window
        )
        hidden = residual + self.o_proj(attended)

        residual = hidden
        normalized = self.post_attention_norm(hidden)
        gate = functional.silu(self.gate_proj(normalized))
        hidden = residual + self.down_proj(gate * self.up_proj(normalized))
        return hidden


class Qwen3SpeculativeDrafter(torch.nn.Module):
    """The causal LM body and lm_head for a stateless fixed-window draft call."""

    def __init__(
        self, config: ModelConfig, tensors: dict[str, torch.Tensor], window: int, last_k: int
    ) -> None:
        super().__init__()
        if last_k <= 0 or last_k > window:
            raise ValueError(f"last_k must be in [1, {window}], got {last_k}")
        self.config = config
        self.window = window
        self.last_k = last_k
        self.embedding_weight = torch.nn.Parameter(
            tensor_for(tensors, "embed_tokens.weight"), requires_grad=False
        )
        layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"layers.{index}"
            layers.append(
                Qwen3Layer(
                    config=config,
                    input_norm=tensor_for(tensors, f"{prefix}.input_layernorm.weight"),
                    post_attention_norm=tensor_for(tensors, f"{prefix}.post_attention_layernorm.weight"),
                    q_proj=tensor_for(tensors, f"{prefix}.self_attn.q_proj.weight"),
                    q_norm=tensor_for(tensors, f"{prefix}.self_attn.q_norm.weight"),
                    k_proj=tensor_for(tensors, f"{prefix}.self_attn.k_proj.weight"),
                    k_norm=tensor_for(tensors, f"{prefix}.self_attn.k_norm.weight"),
                    v_proj=tensor_for(tensors, f"{prefix}.self_attn.v_proj.weight"),
                    o_proj=tensor_for(tensors, f"{prefix}.self_attn.o_proj.weight"),
                    gate_proj=tensor_for(tensors, f"{prefix}.mlp.gate_proj.weight"),
                    up_proj=tensor_for(tensors, f"{prefix}.mlp.up_proj.weight"),
                    down_proj=tensor_for(tensors, f"{prefix}.mlp.down_proj.weight"),
                    window=window,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.final_norm = ChannelRmsNorm(tensor_for(tensors, "norm.weight"), config.rms_norm_eps)
        lm_head_weight = tensors.get("lm_head.weight")
        if lm_head_weight is None:
            lm_head_weight = tensors.get("model.lm_head.weight")
        if lm_head_weight is None:
            if not config.tie_word_embeddings:
                raise KeyError("checkpoint has no lm_head.weight and does not tie word embeddings")
            lm_head_weight = tensor_for(tensors, "embed_tokens.weight")
        self.lm_head = Conv1x1(lm_head_weight)
        cos, sin = build_rope_tables(window, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)
        causal = torch.zeros((1, 1, window, window), dtype=torch.float32)
        causal = causal.masked_fill(
            torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1), -10_000.0
        )
        self.register_buffer("causal_mask", causal, persistent=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = functional.embedding(input_ids, self.embedding_weight)
        hidden = hidden.transpose(1, 2).unsqueeze(2)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, self.causal_mask, self.cos, self.sin)
        hidden = self.final_norm(hidden)
        last_hidden = hidden[:, :, :, self.window - self.last_k :]
        logits = self.lm_head(last_hidden)
        return logits.squeeze(2).transpose(1, 2)


class Qwen3AutoregressiveUnrolledDrafter(torch.nn.Module):
    """Run a fixed number of greedy draft steps inside one exported graph."""

    def __init__(self, drafter: Qwen3SpeculativeDrafter, unroll_k: int) -> None:
        super().__init__()
        if unroll_k <= 0:
            raise ValueError(f"unroll_k must be positive, got {unroll_k}")
        self.drafter = drafter
        self.unroll_k = unroll_k
        self.config = drafter.config
        self.window = drafter.window

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        current_ids = input_ids
        current_mask = attention_mask
        logits = self.drafter(current_ids, current_mask)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        next_token_ids = next_token.to(dtype=current_ids.dtype)
        token_rows = next_token_ids.unsqueeze(1)
        current_ids = torch.cat((current_ids[:, 1:], next_token_ids.unsqueeze(1)), dim=1)
        current_mask = torch.cat(
            (
                current_mask[:, 1:],
                torch.ones_like(next_token_ids, dtype=current_mask.dtype).unsqueeze(1),
            ),
            dim=1,
        )
        for _ in range(1, self.unroll_k):
            logits = self.drafter(current_ids, current_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            next_token_ids = next_token.to(dtype=current_ids.dtype)
            token_rows = torch.cat((token_rows, next_token_ids.unsqueeze(1)), dim=1)
            current_ids = torch.cat((current_ids[:, 1:], next_token_ids.unsqueeze(1)), dim=1)
            current_mask = torch.cat(
                (
                    current_mask[:, 1:],
                    torch.ones_like(next_token_ids, dtype=current_mask.dtype).unsqueeze(1),
                ),
                dim=1,
            )
        return token_rows


def apply_rope(hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = hidden.shape[-1] // 2
    rotated = torch.cat((-hidden[..., half:], hidden[..., :half]), dim=-1)
    return hidden * cos + rotated * sin


def build_rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.float32)
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    return (
        angles.cos().reshape(1, 1, seq_len, head_dim),
        angles.sin().reshape(1, 1, seq_len, head_dim),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF repo id or local snapshot directory")
    parser.add_argument("--window", type=int, required=True, choices=(32, 64, 128))
    output_mode = parser.add_mutually_exclusive_group(required=True)
    output_mode.add_argument("--last-k", type=int, choices=(1, 4, 8), help="Return logits for the final K positions")
    output_mode.add_argument("--unroll-k", type=int, help="Greedily unroll K fixed-window passes inside one graph")
    parser.add_argument("--out", type=Path, required=True, help="Destination .mlpackage path")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def read_config(model_ref: str) -> ModelConfig:
    raw = json.loads((Path(model_ref) / "config.json").read_text(encoding="utf-8"))
    required = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "eos_token_id",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Qwen3 config is missing {missing}")
    pad_token_id = raw.get("pad_token_id", raw.get("eos_token_id"))
    config = ModelConfig(
        **{key: raw[key] for key in required},
        pad_token_id=int(pad_token_id),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
    )
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("Qwen3 query heads must divide evenly across KV heads")
    return config


def load_tensors(model_ref: str) -> dict[str, torch.Tensor]:
    path = Path(model_ref) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Qwen3 safetensors checkpoint not found: {path}")
    with safe_open(path, framework="pt", device="cpu") as reader:
        return {name: reader.get_tensor(name).to(dtype=torch.float32) for name in reader.keys()}


def tensor_for(tensors: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    for key in (suffix, f"model.{suffix}"):
        if key in tensors:
            return tensors[key]
    raise KeyError(f"checkpoint has no tensor named {suffix!r} or {('model.' + suffix)!r}")


def build_drafter(
    model_ref: str, window: int, last_k: int, unroll_k: int | None = None
) -> tuple[torch.nn.Module, ModelConfig]:
    config = read_config(model_ref)
    tensors = load_tensors(model_ref)
    if unroll_k is not None:
        model: torch.nn.Module = Qwen3AutoregressiveUnrolledDrafter(
            Qwen3SpeculativeDrafter(config, tensors, window, 1), unroll_k
        )
    else:
        model = Qwen3SpeculativeDrafter(config, tensors, window, last_k)
    return model.eval(), config


def prepare_input(
    tokenizer: Any, config: ModelConfig, text: str, window: int
) -> tuple[torch.Tensor, torch.Tensor, TokenizerPolicy]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=window,
        padding=False,
        return_attention_mask=False,
    )
    ids = [int(value) for value in encoded["input_ids"]][-window:]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = config.pad_token_id
    pad_count = window - len(ids)
    return (
        torch.tensor([[int(pad_token_id)] * pad_count + ids], dtype=torch.int32),
        torch.tensor([[0] * pad_count + [1] * len(ids)], dtype=torch.int32),
        TokenizerPolicy(
            add_special_tokens=True,
            pad_token_id=int(pad_token_id),
            padding_side="left",
            position_policy="fixed positions 0..window-1; left padding preserves active relative positions",
        ),
    )


def tensor_rows(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(array.shape[0], -1)


def mean_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = np.sum(reference * candidate, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def hf_logits(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, last_k: int
) -> torch.Tensor:
    with torch.inference_mode():
        output = model(
            input_ids=input_ids.to(dtype=torch.long),
            attention_mask=attention_mask.to(dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).logits
    return output[:, -last_k:, :].float()


def hf_unrolled_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    unroll_k: int,
) -> torch.Tensor:
    current_ids = input_ids.to(dtype=torch.long)
    current_mask = attention_mask.to(dtype=torch.long)
    tokens: list[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(unroll_k):
            logits = hf_logits(model, current_ids, current_mask, 1)
            token = torch.argmax(logits[:, -1, :], dim=-1)
            tokens.append(token)
            current_ids = torch.cat((current_ids[:, 1:], token.unsqueeze(1)), dim=1)
            current_mask = torch.cat(
                (current_mask[:, 1:], torch.ones_like(token).unsqueeze(1)), dim=1
            )
    return torch.stack(tokens, dim=1)


def convert_and_verify(
    wrapper: torch.nn.Module,
    model_ref: str,
    window: int,
    last_k: int,
    unroll_k: int | None,
    allow_download: bool,
) -> tuple[ct.models.MLModel, ParityReport, TokenizerPolicy, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=not allow_download)
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    tokenizer_policy: TokenizerPolicy | None = None
    for text in SMOKE_TEXTS:
        input_ids, attention_mask, tokenizer_policy = prepare_input(tokenizer, wrapper.config, text, window)
        examples.append((input_ids, attention_mask))
    assert tokenizer_policy is not None

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=not allow_download,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval()
    with torch.inference_mode():
        eager_rows = [tensor_rows(wrapper(*inputs)) for inputs in examples]
        if unroll_k is None:
            hf_rows = [tensor_rows(hf_logits(hf_model, *inputs, last_k)) for inputs in examples]
        else:
            hf_rows = [tensor_rows(hf_unrolled_tokens(hf_model, *inputs, unroll_k)) for inputs in examples]
        exported = torch.export.export(wrapper, examples[0], strict=False)
        exported_rows = [tensor_rows(exported.module()(*inputs)) for inputs in examples]
    del hf_model

    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, window), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, window), dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    existing_outputs = list(mlmodel.output_description)
    if len(existing_outputs) != 1:
        raise RuntimeError(f"expected one model output, found {existing_outputs!r}")
    converted_output = existing_outputs[0]
    output_name = "token_ids" if unroll_k is not None else "logits"
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
        value = prediction.get(output_name)
        if value is None and len(prediction) == 1:
            value = next(iter(prediction.values()))
        if value is None:
            raise RuntimeError(f"Core ML prediction has no {output_name} output: {list(prediction)}")
        coreml_rows.append(tensor_rows(value))

    eager = np.concatenate(eager_rows, axis=0)
    hf = np.concatenate(hf_rows, axis=0)
    exported_output = np.concatenate(exported_rows, axis=0)
    coreml = np.concatenate(coreml_rows, axis=0)
    mode = "autoregressive_unroll" if unroll_k is not None else "last_k"
    token_count = int(eager.size) if unroll_k is not None else 0
    token_agreements = (
        {
            "wrapper_hf": int(np.sum(eager == hf)),
            "eager_export": int(np.sum(eager == exported_output)),
            "eager_coreml": int(np.sum(eager == coreml)),
        }
        if unroll_k is not None
        else {"wrapper_hf": 0, "eager_export": 0, "eager_coreml": 0}
    )
    report = ParityReport(
        rows=eager.shape[0],
        last_k=unroll_k if unroll_k is not None else last_k,
        wrapper_hf_max_abs=float(np.max(np.abs(eager - hf))),
        wrapper_hf_mean_cosine=mean_cosine(hf, eager),
        eager_export_max_abs=float(np.max(np.abs(eager - exported_output))),
        eager_export_mean_cosine=mean_cosine(eager, exported_output),
        eager_coreml_max_abs=float(np.max(np.abs(eager - coreml))),
        eager_coreml_mean_cosine=mean_cosine(eager, coreml),
        mode=mode,
        token_count=token_count,
        wrapper_hf_token_agreements=token_agreements["wrapper_hf"],
        eager_export_token_agreements=token_agreements["eager_export"],
        eager_coreml_token_agreements=token_agreements["eager_coreml"],
    )
    if unroll_k is None and report.wrapper_hf_mean_cosine < 0.99999:
        raise RuntimeError(f"custom Qwen3 wrapper disagrees with HF: {report.wrapper_hf_mean_cosine}")
    if unroll_k is not None and report.wrapper_hf_token_agreements != token_count:
        raise RuntimeError(
            "autoregressive wrapper disagrees with HF: "
            f"{report.wrapper_hf_token_agreements}/{token_count} token ids"
        )
    if report.eager_export_max_abs > 1e-5:
        raise RuntimeError(f"torch.export parity failed: max_abs={report.eager_export_max_abs}")
    return mlmodel, report, tokenizer_policy, output_name


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


def require_environment(environment: EnvironmentReport) -> None:
    torch_version = environment.torch.split("+")[0]
    if torch_version != PINNED_TORCH:
        raise RuntimeError(f"torch={torch_version}; expected pinned {PINNED_TORCH}")
    if environment.transformers != PINNED_TRANSFORMERS:
        raise RuntimeError(
            f"transformers={environment.transformers}; expected pinned {PINNED_TRANSFORMERS}"
        )
    pieces = tuple(int(part) for part in environment.coremltools.split(".")[:2])
    if pieces < (8, 3):
        raise RuntimeError(f"coremltools={environment.coremltools}; expected 8.3 or newer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def ensure_metadata(mlmodel: ct.models.MLModel, report: ConversionReport) -> None:
    mlmodel.short_description = "Stateless fixed-window Qwen3-0.6B greedy draft unroll for Apple Neural Engine."
    mlmodel.author = "Synapse bench/spikes/ane-spec-decode"
    mlmodel.license = "Source model license follows the referenced Hugging Face checkpoint."
    metadata = mlmodel.user_defined_metadata
    metadata["synapse.source_model"] = report.source_model
    metadata["synapse.source_model_sha256"] = report.source_model_sha256
    metadata["synapse.window"] = str(report.window)
    metadata["synapse.last_k"] = str(report.last_k)
    metadata["synapse.unroll_k"] = str(report.unroll_k or 0)
    metadata["synapse.mode"] = report.mode
    metadata["synapse.frontend"] = report.frontend
    metadata["synapse.output_name"] = report.output_name
    metadata["synapse.output_kind"] = report.output_kind
    metadata["synapse.tokenizer_policy"] = json.dumps(asdict(report.tokenizer_policy), sort_keys=True)
    metadata["synapse.torch_version"] = report.environment.torch
    metadata["synapse.coremltools_version"] = report.environment.coremltools
    metadata["synapse.transformers_version"] = report.environment.transformers


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    args = parse_args()
    if args.last_k is not None and args.last_k > args.window:
        raise ValueError("--last-k cannot exceed --window")
    if args.unroll_k is not None and args.unroll_k <= 0:
        raise ValueError("--unroll-k must be positive")
    effective_last_k = args.last_k if args.last_k is not None else 1
    mode = "autoregressive_unroll" if args.unroll_k is not None else "last_k"
    output_kind = "token_ids" if args.unroll_k is not None else "logits"
    environment = environment_report()
    require_environment(environment)
    model_ref, report_model_ref = resolve_model_ref(args.model)
    model_path = Path(model_ref)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Qwen3 must be available as a local snapshot for safetensors conversion: {model_ref}"
        )
    for path in (args.out, args.report_json):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
            remove_path(path)

    started = time.monotonic()
    wrapper, config = build_drafter(model_ref, args.window, effective_last_k, args.unroll_k)
    mlmodel, parity, tokenizer_policy, output_name = convert_and_verify(
        wrapper, model_ref, args.window, effective_last_k, args.unroll_k, args.allow_download
    )
    conversion_s = time.monotonic() - started
    report = ConversionReport(
        source_model=report_model_ref,
        source_model_sha256=sha256(model_path / "model.safetensors"),
        window=args.window,
        last_k=int(args.unroll_k) if args.unroll_k is not None else effective_last_k,
        unroll_k=args.unroll_k,
        mode=mode,
        output_path=str(args.out),
        output_name=output_name,
        frontend="torch.export",
        compute_precision="float16",
        compute_units="CPU_AND_NE",
        minimum_deployment_target="macOS14",
        package_size_bytes=0,
        output_kind=output_kind,
        model_config=config,
        tokenizer_policy=tokenizer_policy,
        parity=parity,
        environment=environment,
        conversion_s=conversion_s,
    )
    ensure_metadata(mlmodel, report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.out)
    report = ConversionReport(**{**asdict(report), "package_size_bytes": directory_size(args.out)})
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
