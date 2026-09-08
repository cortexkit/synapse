#!/usr/bin/env python3
"""Exact, bounded-attention GTE ModernBERT modules for the Core ML spike."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as functional  # pyright: ignore[reportMissingImports]
from safetensors import safe_open  # pyright: ignore[reportMissingImports]

MASK_MIN_VALUE = -10_000.0
AttentionKind = Literal["query_tiled", "streaming_reference"]


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


class ModernBertLayer(torch.nn.Module):
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
    ) -> None:
        super().__init__()
        prefix = f"layers.{layer_index}"
        self.attention_norm = (
            torch.nn.Identity()
            if layer_index == 0
            else ChannelLayerNorm(tensors[f"{prefix}.attn_norm.weight"], config.norm_eps)
        )
        self.qkv = Conv1x1(tensors[f"{prefix}.attn.Wqkv.weight"])
        self.attention_output = Conv1x1(tensors[f"{prefix}.attn.Wo.weight"])
        self.mlp_norm = ChannelLayerNorm(tensors[f"{prefix}.mlp_norm.weight"], config.norm_eps)
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
        return hidden + self.attention_output(attended)

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

    def __init__(
        self,
        config: ModernBertConfig,
        tensors: dict[str, torch.Tensor],
        sequence_length: int,
        query_tile_size: int,
        key_tile_size: int = 256,
        attention_kind: AttentionKind = "query_tiled",
    ) -> None:
        super().__init__()
        if sequence_length > config.max_position_embeddings:
            raise ValueError("sequence length exceeds the checkpoint's context limit")
        self.embedding_weight = torch.nn.Parameter(
            tensors["embeddings.tok_embeddings.weight"].float(), requires_grad=False
        )
        self.embedding_norm = ChannelLayerNorm(tensors["embeddings.norm.weight"], config.norm_eps)
        self.layers = torch.nn.ModuleList(
            ModernBertLayer(
                config=config,
                tensors=tensors,
                layer_index=index,
                sequence_length=sequence_length,
                query_tile_size=query_tile_size,
                key_tile_size=key_tile_size,
                attention_kind=attention_kind,
            )
            for index in range(config.num_hidden_layers)
        )
        self.final_norm = ChannelLayerNorm(tensors["final_norm.weight"], config.norm_eps)
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
        return self._normalize_cls(self.final_norm(hidden))

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
        for layer in self.layers:
            after_attention, hidden = layer.forward_checkpoints(
                hidden,
                attention_mask,
                self.global_cos,
                self.global_sin,
                self.local_cos,
                self.local_sin,
            )
            checkpoints.extend((sample(after_attention), sample(hidden)))
        final_hidden = self.final_norm(hidden)
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
) -> tuple[ModernBertEmbedder, ModernBertConfig]:
    config = ModernBertConfig.from_snapshot(snapshot)
    model = ModernBertEmbedder(
        config,
        load_tensors(snapshot),
        sequence_length,
        query_tile_size,
        key_tile_size,
        attention_kind,
    ).eval()
    return model, config
