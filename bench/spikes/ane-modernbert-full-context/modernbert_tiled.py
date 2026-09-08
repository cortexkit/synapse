#!/usr/bin/env python3
"""Exact, bounded-attention GTE ModernBERT modules for the Core ML spike."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn.functional as functional
from safetensors import safe_open

MASK_MIN_VALUE = -10_000.0
AttentionKind = Literal["query_tiled", "streaming_reference"]
RotationKind = Literal["none", "hadamard"]


@dataclass(frozen=True)
class ModernBertConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    global_attn_every_n_layers: int
    global_rope_theta: float
    local_attention: int
    local_rope_theta: float
    max_position_embeddings: int
    norm_eps: float
    pad_token_id: int
    cls_token_id: int
    sep_token_id: int
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "ModernBertConfig":
        raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        fields = tuple(cls.__dataclass_fields__)
        missing = [field for field in fields if field not in raw]
        if missing:
            raise ValueError(f"ModernBERT config is missing {missing}")
        config = cls(**{field: raw[field] for field in fields})
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must divide evenly across attention heads")
        if config.local_attention <= 0 or config.local_attention % 2:
            raise ValueError("local_attention must be a positive even window size")
        return config


@dataclass(frozen=True)
class RotationWeights:
    layer0_residual: torch.Tensor
    output_unrotate: torch.Tensor
    seed: int
    matrix_sha256: str


def _hadamard12() -> torch.Tensor:
    """Return the order-12 matrix used by the reference ModernBERT exporter."""
    return torch.tensor(
        [
            [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
            [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
            [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
            [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
            [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
            [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
            [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
            [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
            [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
            [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
            [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
        ],
        dtype=torch.float64,
    )


def _hadamard_base(size: int, transpose: bool = False) -> tuple[torch.Tensor | None, int]:
    if size % 12 == 0 and (size // 12) & (size // 12 - 1) == 0:
        matrix = _hadamard12()
        return (matrix.T if transpose else matrix), 12
    if size > 0 and size & (size - 1) == 0:
        return None, 1
    raise ValueError("Hadamard rotation requires hidden size 12*2^n or 2^n")


def _matmul_hadamard(value: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    size = value.shape[-1]
    base, block_size = _hadamard_base(size, transpose)
    source = value.clone().reshape(-1, size, 1)
    target = source.clone()
    while source.shape[1] > block_size:
        source = source.reshape(source.shape[0], source.shape[1] // 2, 2, source.shape[2])
        target = target.reshape(source.shape)
        target[:, :, 0, :] = source[:, :, 0, :] + source[:, :, 1, :]
        target[:, :, 1, :] = source[:, :, 0, :] - source[:, :, 1, :]
        target = target.reshape(source.shape[0], source.shape[1], -1)
        source, target = target, source
    if base is not None:
        source = base.reshape(1, block_size, block_size).to(source) @ source
    return source.reshape(value.shape) / math.sqrt(size)


def randomized_hadamard_matrix(size: int, seed: int) -> torch.Tensor:
    """Build the reference randomized Hadamard rotation with a reproducible sign draw."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator).to(torch.float64) * 2 - 1
    return _matmul_hadamard(torch.diag(signs))


def _replace_weight(
    tensors: dict[str, torch.Tensor], name: str, transformed: torch.Tensor
) -> None:
    tensors[name] = transformed.to(dtype=tensors[name].dtype, device="cpu")


def apply_hadamard_rotation(
    config: ModernBertConfig, tensors: dict[str, torch.Tensor], seed: int
) -> RotationWeights:
    """Fold the reference residual-stream rotation into raw checkpoint weights."""
    hidden_size = config.hidden_size
    rotation = randomized_hadamard_matrix(hidden_size, seed)
    centering = torch.eye(hidden_size, dtype=torch.float64) - (1.0 / hidden_size)

    embedding_name = "embeddings.tok_embeddings.weight"
    embedding = tensors[embedding_name].double()
    _replace_weight(tensors, embedding_name, embedding - embedding.mean(dim=-1, keepdim=True))

    embedding_scale = torch.diag(tensors["embeddings.norm.weight"].double())
    residual = (centering @ embedding_scale).to(torch.float32)
    first_qkv = "layers.0.attn.Wqkv.weight"
    _replace_weight(
        tensors,
        first_qkv,
        tensors[first_qkv].double() * tensors["embeddings.norm.weight"].double(),
    )

    for layer_index in range(config.num_hidden_layers):
        prefix = f"layers.{layer_index}"
        qkv_name = f"{prefix}.attn.Wqkv.weight"
        attention_output_name = f"{prefix}.attn.Wo.weight"
        mlp_input_name = f"{prefix}.mlp.Wi.weight"
        mlp_output_name = f"{prefix}.mlp.Wo.weight"
        if layer_index > 0:
            _replace_weight(
                tensors,
                qkv_name,
                tensors[qkv_name].double() * tensors[f"{prefix}.attn_norm.weight"].double(),
            )
        _replace_weight(
            tensors,
            mlp_input_name,
            tensors[mlp_input_name].double() * tensors[f"{prefix}.mlp_norm.weight"].double(),
        )
        attention_output = tensors[attention_output_name].double()
        _replace_weight(
            tensors,
            attention_output_name,
            attention_output - attention_output.mean(dim=-2, keepdim=True),
        )
        mlp_output = tensors[mlp_output_name].double()
        _replace_weight(
            tensors,
            mlp_output_name,
            mlp_output - mlp_output.mean(dim=-2, keepdim=True),
        )

    output_unrotate = torch.diag(tensors["final_norm.weight"].double()).to(torch.float32)
    _replace_weight(tensors, embedding_name, tensors[embedding_name].double() @ rotation)
    residual = rotation.T @ residual.double() @ rotation
    for layer_index in range(config.num_hidden_layers):
        prefix = f"layers.{layer_index}"
        for name in (f"{prefix}.attn.Wqkv.weight", f"{prefix}.mlp.Wi.weight"):
            _replace_weight(tensors, name, tensors[name].double() @ rotation)
        for name in (f"{prefix}.attn.Wo.weight", f"{prefix}.mlp.Wo.weight"):
            _replace_weight(tensors, name, rotation.T @ tensors[name].double())
    output_unrotate = output_unrotate.double() @ rotation
    matrix_sha256 = hashlib.sha256(rotation.numpy().tobytes()).hexdigest()
    return RotationWeights(
        layer0_residual=residual.to(torch.float32),
        output_unrotate=output_unrotate.to(torch.float32),
        seed=seed,
        matrix_sha256=matrix_sha256,
    )


def tile_ranges(length: int, tile_size: int) -> tuple[tuple[int, int], ...]:
    """Return complete, ordered half-open tiles, including a final partial tile."""
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive")
    return tuple((start, min(start + tile_size, length)) for start in range(0, length, tile_size))


def diagnostic_positions(sequence_length: int) -> tuple[int, ...]:
    candidates = (0, 63, 64, 255, 256, sequence_length - 1)
    return tuple(sorted({position for position in candidates if 0 <= position < sequence_length}))


def diagnostic_checkpoint_labels(num_layers: int) -> tuple[str, ...]:
    labels = ["token_embeddings", "embeddings_norm"]
    for layer_index in range(num_layers):
        labels.extend((f"layer_{layer_index}.attention_residual", f"layer_{layer_index}.output"))
    labels.append("final_norm")
    return tuple(labels)


def build_rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.float32)
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    frequencies = torch.outer(positions, inverse_frequency)
    frequencies = torch.cat((frequencies, frequencies), dim=-1).transpose(0, 1)
    shape = (1, 1, head_dim, seq_len)
    return frequencies.cos().reshape(shape), frequencies.sin().reshape(shape)


def apply_rope(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ModernBERT's split-half RoPE to tensors shaped [B, H, D, S]."""
    half = query.shape[-2] // 2
    query_rotated = torch.cat((-query[:, :, half:, :], query[:, :, :half, :]), dim=2)
    key_rotated = torch.cat((-key[:, :, half:, :], key[:, :, :half, :]), dim=2)
    return query * cos + query_rotated * sin, key * cos + key_rotated * sin


def _tile_additive_mask(
    attention_mask: torch.Tensor,
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    local_radius: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    key_padding = (1.0 - attention_mask[:, key_start:key_end].to(torch.float32)).reshape(
        attention_mask.shape[0], key_end - key_start, 1, 1
    )
    key_padding = key_padding * MASK_MIN_VALUE
    if local_radius is None:
        return key_padding.to(dtype)

    query_positions = torch.arange(query_start, query_end, device=attention_mask.device)
    key_positions = torch.arange(key_start, key_end, device=attention_mask.device)
    permitted = (key_positions[:, None] - query_positions[None, :]).abs() <= local_radius
    permitted = permitted.reshape(1, key_end - key_start, 1, query_end - query_start)
    outside = torch.full_like(key_padding, MASK_MIN_VALUE).expand(
        attention_mask.shape[0], key_end - key_start, 1, query_end - query_start
    )
    return torch.where(permitted, key_padding, outside).to(dtype)


def query_tiled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    query_tile_size: int,
    local_window: int | None,
) -> torch.Tensor:
    """Compute exact attention while bounding every score tensor to one head/query tile.

    Query, key, and value use [B, H, D, S]. Global query tiles retain the full
    key/value sequence. Local tiles retain the complete radius-sized halo, so a
    tile boundary never changes which keys are visible.
    """
    sequence_length = query.shape[-1]
    local_radius = None if local_window is None else local_window // 2
    scale = query.shape[-2] ** -0.5
    attended_heads: list[torch.Tensor] = []

    for head_index in range(query.shape[1]):
        query_head = query[:, head_index, :, :].unsqueeze(2)
        key_head = key[:, head_index, :, :]
        value_head = value[:, head_index, :, :].unsqueeze(2)
        attended_tiles: list[torch.Tensor] = []
        for query_start, query_end in tile_ranges(sequence_length, query_tile_size):
            if local_radius is None:
                key_start, key_end = 0, sequence_length
            else:
                key_start = max(0, query_start - local_radius)
                key_end = min(sequence_length, query_end + local_radius)

            query_tile = query_head[:, :, :, query_start:query_end]
            key_tile = key_head[:, :, key_start:key_end].transpose(1, 2).unsqueeze(2)
            value_tile = value_head[:, :, :, key_start:key_end]
            scores = torch.einsum("bchq,bkhc->bkhq", query_tile, key_tile) * scale
            scores = scores + _tile_additive_mask(
                attention_mask,
                query_start,
                query_end,
                key_start,
                key_end,
                local_radius,
                scores.dtype,
            )
            probabilities = functional.softmax(scores, dim=1, dtype=torch.float32).to(query.dtype)
            attended_tiles.append(torch.einsum("bkhq,bchk->bchq", probabilities, value_tile))
        attended_heads.append(torch.cat(attended_tiles, dim=3))
    return torch.cat(attended_heads, dim=1)


def streaming_reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    query_tile_size: int,
    key_tile_size: int,
    local_window: int | None,
) -> torch.Tensor:
    """Independent online-softmax reference with bounded query and key blocks."""
    sequence_length = query.shape[-1]
    local_radius = None if local_window is None else local_window // 2
    scale = query.shape[-2] ** -0.5
    query_bhsd = query.permute(0, 1, 3, 2).float()
    key_bhsd = key.permute(0, 1, 3, 2).float()
    value_bhsd = value.permute(0, 1, 3, 2).float()
    output_tiles: list[torch.Tensor] = []

    for query_start, query_end in tile_ranges(sequence_length, query_tile_size):
        query_tile = query_bhsd[:, :, query_start:query_end, :]
        running_max = torch.full(
            (*query_tile.shape[:-1], 1), -torch.inf, dtype=torch.float32, device=query.device
        )
        denominator = torch.zeros_like(running_max)
        numerator = torch.zeros_like(query_tile)
        for key_start, key_end in tile_ranges(sequence_length, key_tile_size):
            key_tile = key_bhsd[:, :, key_start:key_end, :]
            value_tile = value_bhsd[:, :, key_start:key_end, :]
            scores = torch.matmul(query_tile, key_tile.transpose(-1, -2)) * scale
            tile_mask = _tile_additive_mask(
                attention_mask,
                query_start,
                query_end,
                key_start,
                key_end,
                local_radius,
                scores.dtype,
            ).permute(0, 2, 3, 1)
            scores = scores + tile_mask
            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            block_weights = torch.exp(scores - new_max)
            numerator = numerator * old_scale + torch.matmul(block_weights, value_tile)
            denominator = denominator * old_scale + block_weights.sum(dim=-1, keepdim=True)
            running_max = new_max
        output_tiles.append(numerator / denominator.clamp_min(1e-30))
    output = torch.cat(output_tiles, dim=2).to(query.dtype)
    return output.permute(0, 1, 3, 2).reshape(
        query.shape[0], query.shape[1] * query.shape[2], 1, sequence_length
    )


class Conv1x1(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            weight.float().unsqueeze(-1).unsqueeze(-1), requires_grad=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(hidden, self.weight)


class ChannelLayerNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.float(), requires_grad=False)
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        sequence_last = hidden.squeeze(2).transpose(1, 2)
        normalized = functional.layer_norm(
            sequence_last, (self.weight.numel(),), self.weight, None, self.eps
        )
        return normalized.transpose(1, 2).unsqueeze(2)


class ChannelRMSNorm(torch.nn.Module):
    """Parameter-free RMSNorm formulated to avoid squaring large residual values."""

    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones((1, size, 1, 1), dtype=torch.float32), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden.dtype
        value = hidden.float()
        eps_channel = torch.ones(
            (value.shape[0], 1, value.shape[2], value.shape[3]),
            dtype=value.dtype,
            device=value.device,
        ) * math.sqrt(self.eps * value.shape[1])
        denominator = torch.linalg.vector_norm(torch.cat((value, eps_channel), dim=1), dim=1, keepdim=True)
        normalized = value / denominator * math.sqrt(value.shape[1])
        return normalized.to(original_dtype) * self.weight


class ModernBertLayer(torch.nn.Module):
    attention_norm: torch.nn.Module
    qkv: Conv1x1
    attention_output: Conv1x1
    mlp_input: Conv1x1
    mlp_output: Conv1x1
    residual_transform: torch.nn.Module

    def __init__(
        self,
        *,
        config: ModernBertConfig,
        tensors: dict[str, torch.Tensor],
        layer_index: int,
        sequence_length: int,
        query_tile_size: int,
        key_tile_size: int,
        attention_kind: AttentionKind,
        rotation_weights: RotationWeights | None,
    ) -> None:
        super().__init__()
        prefix = f"layers.{layer_index}"
        if layer_index == 0:
            self.attention_norm = torch.nn.Identity()
        elif rotation_weights is None:
            self.attention_norm = ChannelLayerNorm(
                tensors[f"{prefix}.attn_norm.weight"], config.norm_eps
            )
        else:
            self.attention_norm = ChannelRMSNorm(config.hidden_size, config.norm_eps)
        self.qkv = Conv1x1(tensors[f"{prefix}.attn.Wqkv.weight"])
        self.attention_output = Conv1x1(tensors[f"{prefix}.attn.Wo.weight"])
        self.mlp_norm = (
            ChannelLayerNorm(tensors[f"{prefix}.mlp_norm.weight"], config.norm_eps)
            if rotation_weights is None
            else ChannelRMSNorm(config.hidden_size, config.norm_eps)
        )
        self.mlp_input = Conv1x1(tensors[f"{prefix}.mlp.Wi.weight"])
        self.mlp_output = Conv1x1(tensors[f"{prefix}.mlp.Wo.weight"])
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sequence_length = sequence_length
        self.query_tile_size = query_tile_size
        self.key_tile_size = key_tile_size
        self.local_window = (
            None if layer_index % config.global_attn_every_n_layers == 0 else config.local_attention
        )
        self.attention_kind = attention_kind
        self.residual_transform = (
            Conv1x1(rotation_weights.layer0_residual)
            if layer_index == 0 and rotation_weights is not None
            else torch.nn.Identity()
        )

    def _attention_block(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        global_cos: torch.Tensor,
        global_sin: torch.Tensor,
        local_cos: torch.Tensor,
        local_sin: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        qkv = self.qkv(normalized).reshape(
            hidden.shape[0], 3, self.heads, self.head_dim, self.sequence_length
        )
        query, key, value = qkv.unbind(dim=1)
        if self.local_window is None:
            query, key = apply_rope(query, key, global_cos, global_sin)
        else:
            query, key = apply_rope(query, key, local_cos, local_sin)

        if self.attention_kind == "query_tiled":
            attended = query_tiled_attention(
                query,
                key,
                value,
                attention_mask,
                query_tile_size=self.query_tile_size,
                local_window=self.local_window,
            )
        else:
            attended = streaming_reference_attention(
                query,
                key,
                value,
                attention_mask,
                query_tile_size=self.query_tile_size,
                key_tile_size=self.key_tile_size,
                local_window=self.local_window,
            )
        return self.residual_transform(hidden) + self.attention_output(attended)

    def _mlp_block(self, hidden: torch.Tensor) -> torch.Tensor:
        mlp_input = self.mlp_input(self.mlp_norm(hidden))
        activation, gate = mlp_input.chunk(2, dim=1)
        return hidden + self.mlp_output(functional.gelu(activation) * gate)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        global_cos: torch.Tensor,
        global_sin: torch.Tensor,
        local_cos: torch.Tensor,
        local_sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._attention_block(
            hidden, attention_mask, global_cos, global_sin, local_cos, local_sin
        )
        return self._mlp_block(hidden)

    def forward_checkpoints(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        global_cos: torch.Tensor,
        global_sin: torch.Tensor,
        local_cos: torch.Tensor,
        local_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        after_attention = self._attention_block(
            hidden, attention_mask, global_cos, global_sin, local_cos, local_sin
        )
        return after_attention, self._mlp_block(after_attention)



class ModernBertEmbedder(torch.nn.Module):
    """GTE ModernBERT with raw checkpoint weights and normalized CLS pooling."""

    global_cos: torch.Tensor
    global_sin: torch.Tensor
    local_cos: torch.Tensor
    local_sin: torch.Tensor

    def __init__(
        self,
        config: ModernBertConfig,
        tensors: dict[str, torch.Tensor],
        sequence_length: int,
        query_tile_size: int,
        key_tile_size: int = 256,
        attention_kind: AttentionKind = "query_tiled",
        rotation: RotationKind = "none",
        rotation_seed: int = 0,
    ) -> None:
        super().__init__()
        if sequence_length > config.max_position_embeddings:
            raise ValueError("sequence length exceeds the checkpoint's context limit")
        if rotation not in ("none", "hadamard"):
            raise ValueError(f"unsupported rotation: {rotation}")
        rotation_weights = (
            apply_hadamard_rotation(config, tensors, rotation_seed)
            if rotation == "hadamard"
            else None
        )
        self.rotation_kind = rotation
        self.rotation_seed = rotation_seed
        self.rotation_matrix_sha256 = (
            rotation_weights.matrix_sha256 if rotation_weights is not None else None
        )
        self.embedding_weight = torch.nn.Parameter(
            tensors["embeddings.tok_embeddings.weight"].float(), requires_grad=False
        )
        self.embedding_norm = (
            ChannelLayerNorm(tensors["embeddings.norm.weight"], config.norm_eps)
            if rotation_weights is None
            else ChannelRMSNorm(config.hidden_size, config.norm_eps)
        )
        self.layers = torch.nn.ModuleList(
            ModernBertLayer(
                config=config,
                tensors=tensors,
                layer_index=index,
                sequence_length=sequence_length,
                query_tile_size=query_tile_size,
                key_tile_size=key_tile_size,
                attention_kind=attention_kind,
                rotation_weights=rotation_weights,
            )
            for index in range(config.num_hidden_layers)
        )
        self.final_norm = (
            ChannelLayerNorm(tensors["final_norm.weight"], config.norm_eps)
            if rotation_weights is None
            else ChannelRMSNorm(config.hidden_size, config.norm_eps)
        )
        self.output_unrotate = (
            torch.nn.Identity()
            if rotation_weights is None
            else Conv1x1(rotation_weights.output_unrotate)
        )
        self.sequence_length = sequence_length
        global_cos, global_sin = build_rope_tables(
            sequence_length, config.head_dim, config.global_rope_theta
        )
        local_cos, local_sin = build_rope_tables(
            sequence_length, config.head_dim, config.local_rope_theta
        )
        self.register_buffer("global_cos", global_cos, persistent=True)
        self.register_buffer("global_sin", global_sin, persistent=True)
        self.register_buffer("local_cos", local_cos, persistent=True)
        self.register_buffer("local_sin", local_sin, persistent=True)

    def rotation_report(self) -> dict[str, str | int]:
        report: dict[str, str | int] = {"kind": self.rotation_kind}
        if self.rotation_matrix_sha256 is not None:
            report["seed"] = self.rotation_seed
            report["matrix_sha256"] = self.rotation_matrix_sha256
        return report

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = functional.embedding(input_ids, self.embedding_weight)
        return self.embedding_norm(hidden.transpose(1, 2).unsqueeze(2))

    def _normalize_cls(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden[:, :, 0, 0]
        norm = torch.sqrt(torch.sum(pooled.float() * pooled.float(), dim=1, keepdim=True))
        return pooled / norm.clamp_min(1e-12)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self._embed(input_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask,
                self.global_cos,
                self.global_sin,
                self.local_cos,
                self.local_sin,
            )
        return self._normalize_cls(self.output_unrotate(self.final_norm(hidden)))

    def forward_diagnostics(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sampled hidden checkpoints plus the normal production-spike embedding."""
        positions = diagnostic_positions(self.sequence_length)

        def sample(value: torch.Tensor) -> torch.Tensor:
            return torch.stack([value[:, :, 0, position] for position in positions], dim=1)

        token_embeddings = functional.embedding(input_ids, self.embedding_weight)
        token_embeddings = token_embeddings.transpose(1, 2).unsqueeze(2)
        hidden = self.embedding_norm(token_embeddings)
        checkpoints = [sample(token_embeddings), sample(hidden)]
        for layer_module in self.layers:
            layer = cast(ModernBertLayer, layer_module)
            after_attention, hidden = layer.forward_checkpoints(
                hidden,
                attention_mask,
                self.global_cos,
                self.global_sin,
                self.local_cos,
                self.local_sin,
            )
            checkpoints.extend((sample(after_attention), sample(hidden)))
        final_hidden = self.output_unrotate(self.final_norm(hidden))
        checkpoints.append(sample(final_hidden))
        return torch.stack(checkpoints, dim=1), self._normalize_cls(final_hidden)



def load_tensors(snapshot: Path) -> dict[str, torch.Tensor]:
    checkpoint = snapshot / "model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        return {name: reader.get_tensor(name).float() for name in reader.keys()}


def build_embedder(
    snapshot: Path,
    sequence_length: int,
    query_tile_size: int,
    *,
    key_tile_size: int = 256,
    attention_kind: AttentionKind = "query_tiled",
    rotation: RotationKind = "none",
    rotation_seed: int = 0,
) -> tuple[ModernBertEmbedder, ModernBertConfig]:
    config = ModernBertConfig.from_snapshot(snapshot)
    model = ModernBertEmbedder(
        config,
        load_tensors(snapshot),
        sequence_length,
        query_tile_size,
        key_tile_size,
        attention_kind,
        rotation,
        rotation_seed,
    ).eval()
    return model, config
