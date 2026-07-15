"""Fixed-window stateful LFM2 decode graph used by the Phase B probe."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional

from lfm2_model import (
    ChannelRmsNorm,
    Conv1x1,
    HeadRmsNorm,
    LFM2AttentionMixer,
    LFM2Config,
    LFM2ConvMixer,
    LFM2Layer,
    LFM2Prefill,
    apply_rope,
)


class StatefulConvMixer(torch.nn.Module):
    """One-token short convolution with its three-sample history as model state."""

    in_proj: Conv1x1
    conv_weight: torch.Tensor
    out_proj: Conv1x1
    conv_state: torch.Tensor

    def __init__(self, source: LFM2ConvMixer, hidden_size: int) -> None:
        super().__init__()
        self.in_proj = source.in_proj
        self.conv_weight = source.conv_weight
        self.out_proj = source.out_proj
        self.hidden_size = hidden_size
        self.register_buffer(
            "conv_state",
            torch.zeros((1, hidden_size, source.kernel_size, 1), dtype=torch.float32),
            persistent=True,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.in_proj(hidden)
        b_gate, c_gate, values = projected.chunk(3, dim=1)
        product = b_gate * values
        updated = torch.cat((self.conv_state[:, :, 1:, :], product), dim=2)
        self.conv_state.copy_(updated)
        convolved = functional.conv2d(updated, self.conv_weight, groups=self.hidden_size)
        return self.out_proj(c_gate * convolved)


class StatefulAttentionMixer(torch.nn.Module):
    """One-token GQA with fixed rolling K/V tensors exposed as Core ML state."""

    q_proj: Conv1x1
    q_norm: HeadRmsNorm
    k_proj: Conv1x1
    k_norm: HeadRmsNorm
    v_proj: Conv1x1
    out_proj: Conv1x1
    key_state: torch.Tensor
    value_state: torch.Tensor
    rope_frequencies: torch.Tensor
    cache_indices: torch.Tensor

    def __init__(self, source: LFM2AttentionMixer, config: LFM2Config, window: int) -> None:
        super().__init__()
        self.q_proj = source.q_proj
        self.q_norm = source.q_norm
        self.k_proj = source.k_proj
        self.k_norm = source.k_norm
        self.v_proj = source.v_proj
        self.out_proj = source.out_proj
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.projection_size = self.query_heads * self.head_dim
        self.kv_repetition = self.query_heads // self.kv_heads
        self.scale = self.head_dim**-0.5
        self.window = window
        state_shape = (1, self.kv_heads, window, self.head_dim)
        self.register_buffer("key_state", torch.zeros(state_shape), persistent=True)
        self.register_buffer("value_state", torch.zeros(state_shape), persistent=True)
        frequencies = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("rope_frequencies", frequencies, persistent=False)
        self.register_buffer("cache_indices", torch.arange(window, dtype=torch.float32), persistent=False)

    def forward(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> torch.Tensor:
        query = self.q_proj(hidden).reshape(
            1, self.query_heads, self.head_dim, 1
        ).permute(0, 1, 3, 2)
        key = self.k_proj(hidden).reshape(
            1, self.kv_heads, self.head_dim, 1
        ).permute(0, 1, 3, 2)
        value = self.v_proj(hidden).reshape(
            1, self.kv_heads, self.head_dim, 1
        ).permute(0, 1, 3, 2)
        angles = position.to(dtype=torch.float32).reshape(1, 1, 1, 1) * self.rope_frequencies.reshape(
            1, 1, 1, -1
        )
        cos = torch.cat((angles, angles), dim=-1).cos()
        sin = torch.cat((angles, angles), dim=-1).sin()
        query = apply_rope(self.q_norm(query), cos, sin)
        key = apply_rope(self.k_norm(key), cos, sin)
        updated_keys = torch.cat((self.key_state[:, :, 1:, :], key), dim=2)
        updated_values = torch.cat((self.value_state[:, :, 1:, :], value), dim=2)
        self.key_state.copy_(updated_keys)
        self.value_state.copy_(updated_values)

        repeated_keys = updated_keys.repeat_interleave(self.kv_repetition, dim=1)
        repeated_values = updated_values.repeat_interleave(self.kv_repetition, dim=1)
        scores = torch.matmul(query.float(), repeated_keys.float().transpose(-1, -2)) * self.scale
        first_valid = self.window - valid_length.to(dtype=torch.float32)
        invalid = (self.cache_indices < first_valid).to(dtype=torch.float32).reshape(1, 1, 1, -1)
        probabilities = torch.softmax(scores + invalid * -10_000.0, dim=-1, dtype=torch.float32)
        attended = torch.matmul(probabilities.to(dtype=query.dtype), repeated_values)
        attended = attended.permute(0, 1, 3, 2).reshape(1, self.projection_size, 1, 1)
        return self.out_proj(attended)


class StatefulDecodeLayer(torch.nn.Module):
    operator_norm: ChannelRmsNorm
    ffn_norm: ChannelRmsNorm
    w1: Conv1x1
    w2: Conv1x1
    w3: Conv1x1
    mixer: StatefulConvMixer | StatefulAttentionMixer

    def __init__(self, source: LFM2Layer, config: LFM2Config, window: int) -> None:
        super().__init__()
        self.operator_norm = source.operator_norm
        self.ffn_norm = source.ffn_norm
        self.w1 = source.w1
        self.w2 = source.w2
        self.w3 = source.w3
        if isinstance(source.mixer, LFM2ConvMixer):
            self.mixer = StatefulConvMixer(source.mixer, config.hidden_size)
        elif isinstance(source.mixer, LFM2AttentionMixer):
            self.mixer = StatefulAttentionMixer(source.mixer, config, window)
        else:
            raise TypeError(f"unsupported LFM2 mixer {type(source.mixer).__name__}")

    def forward(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.operator_norm(hidden)
        if isinstance(self.mixer, StatefulAttentionMixer):
            mixed = self.mixer(normalized, position, valid_length)
        else:
            mixed = self.mixer(normalized)
        hidden = hidden + mixed
        normalized = self.ffn_norm(hidden)
        return hidden + self.w2(functional.silu(self.w1(normalized)) * self.w3(normalized))


class LFM2StatefulDecode(torch.nn.Module):
    """Token/position/valid-length to logits with rolling conv and KV state."""

    embedding_weight: torch.Tensor
    layers: torch.nn.ModuleList
    final_norm: ChannelRmsNorm

    def __init__(self, prefill: LFM2Prefill, window: int = 512) -> None:
        super().__init__()
        self.config = prefill.config
        self.window = window
        self.embedding_weight = prefill.embedding_weight
        self.layers = torch.nn.ModuleList(
            StatefulDecodeLayer(cast(LFM2Layer, layer), self.config, window)
            for layer in prefill.layers
        )
        self.final_norm = prefill.final_norm

    def forward(
        self,
        token_ids: torch.Tensor,
        position: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> torch.Tensor:
        hidden = functional.embedding(token_ids, self.embedding_weight)
        hidden = hidden.transpose(1, 2).unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, position, valid_length)
        hidden = self.final_norm(hidden)
        logits = functional.conv2d(
            hidden,
            self.embedding_weight.unsqueeze(-1).unsqueeze(-1),
        )
        return logits.squeeze(-1).squeeze(-1)

    def reset_state(self) -> None:
        for module in self.modules():
            if isinstance(module, StatefulConvMixer):
                module.conv_state.zero_()
            elif isinstance(module, StatefulAttentionMixer):
                module.key_state.zero_()
                module.value_state.zero_()
