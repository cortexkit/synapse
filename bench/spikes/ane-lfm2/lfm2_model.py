"""Static-shape LFM2 modules shared by the Core ML conversion and tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open


@dataclass(frozen=True)
class LFM2Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    conv_kernel_size: int
    layer_types: tuple[str, ...]
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["layer_types"] = list(self.layer_types)
        return result


class Conv1x1(torch.nn.Module):
    """A bias-free linear projection represented as an ANE-native convolution."""

    weight: torch.Tensor

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
    """LFM2 RMSNorm over channels in the ANE [batch, channel, sequence, 1] layout."""

    weight: torch.Tensor

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            weight.to(dtype=torch.float32).reshape(1, -1, 1, 1), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        normalized = hidden.float()
        variance = normalized.square().mean(dim=1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=input_dtype) * self.weight


class HeadRmsNorm(torch.nn.Module):
    """LFM2 Q/K RMSNorm over each attention head's feature dimension."""

    weight: torch.Tensor

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            weight.to(dtype=torch.float32).reshape(1, 1, 1, -1), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        normalized = hidden.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=input_dtype) * self.weight


class LFM2ConvMixer(torch.nn.Module):
    """The gated depthwise causal short-convolution used by LFM2."""

    in_proj: Conv1x1
    conv_weight: torch.Tensor
    out_proj: Conv1x1

    def __init__(
        self,
        *,
        hidden_size: int,
        in_proj: torch.Tensor,
        conv_weight: torch.Tensor,
        out_proj: torch.Tensor,
    ) -> None:
        super().__init__()
        if tuple(conv_weight.shape[:2]) != (hidden_size, 1):
            raise ValueError(f"convolution is not depthwise: {tuple(conv_weight.shape)}")
        self.in_proj = Conv1x1(in_proj)
        self.conv_weight = torch.nn.Parameter(
            conv_weight.to(dtype=torch.float32).unsqueeze(-1), requires_grad=False
        )
        self.out_proj = Conv1x1(out_proj)
        self.hidden_size = hidden_size
        self.kernel_size = int(conv_weight.shape[-1])

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=hidden.dtype).reshape(1, 1, -1, 1)
        projected = self.in_proj(hidden * mask)
        b_gate, c_gate, values = projected.chunk(3, dim=1)
        product = b_gate * values
        padded = functional.pad(product, (0, 0, self.kernel_size - 1, 0))
        convolved = functional.conv2d(
            padded,
            self.conv_weight,
            groups=self.hidden_size,
        )
        return self.out_proj(c_gate * convolved)


class LFM2AttentionMixer(torch.nn.Module):
    """Static causal grouped-query attention in the ANE channel-first layout."""

    q_proj: Conv1x1
    q_norm: HeadRmsNorm
    k_proj: Conv1x1
    k_norm: HeadRmsNorm
    v_proj: Conv1x1
    out_proj: Conv1x1

    def __init__(
        self,
        *,
        config: LFM2Config,
        q_proj: torch.Tensor,
        q_norm: torch.Tensor,
        k_proj: torch.Tensor,
        k_norm: torch.Tensor,
        v_proj: torch.Tensor,
        out_proj: torch.Tensor,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.q_proj = Conv1x1(q_proj)
        self.q_norm = HeadRmsNorm(q_norm, config.rms_norm_eps)
        self.k_proj = Conv1x1(k_proj)
        self.k_norm = HeadRmsNorm(k_norm, config.rms_norm_eps)
        self.v_proj = Conv1x1(v_proj)
        self.out_proj = Conv1x1(out_proj)
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.seq_len = seq_len
        self.projection_size = self.query_heads * self.head_dim
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
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        query = query.reshape(1, self.query_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        key = key.reshape(1, self.kv_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        value = value.reshape(1, self.kv_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        query = apply_rope(self.q_norm(query), cos, sin)
        key = apply_rope(self.k_norm(key), cos, sin)
        key = key.repeat_interleave(self.kv_repetition, dim=1)
        value = value.repeat_interleave(self.kv_repetition, dim=1)
        padding_mask = (1.0 - attention_mask.to(dtype=torch.float32)).reshape(
            1, 1, 1, self.seq_len
        ) * -10_000.0
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * self.scale
        probabilities = torch.softmax(
            scores + causal_mask + padding_mask,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=query.dtype)
        attended = torch.matmul(probabilities, value)
        attended = attended.permute(0, 1, 3, 2).reshape(
            1, self.projection_size, self.seq_len, 1
        )
        return self.out_proj(attended)


class LFM2Layer(torch.nn.Module):
    """One LFM2 residual block with either short convolution or GQA attention."""

    operator_norm: ChannelRmsNorm
    ffn_norm: ChannelRmsNorm
    mixer: LFM2ConvMixer | LFM2AttentionMixer
    w1: Conv1x1
    w2: Conv1x1
    w3: Conv1x1

    def __init__(
        self,
        *,
        config: LFM2Config,
        operator_norm: torch.Tensor,
        ffn_norm: torch.Tensor,
        mixer: LFM2ConvMixer | LFM2AttentionMixer,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> None:
        super().__init__()
        self.operator_norm = ChannelRmsNorm(operator_norm, config.rms_norm_eps)
        self.ffn_norm = ChannelRmsNorm(ffn_norm, config.rms_norm_eps)
        self.mixer = mixer
        self.w1 = Conv1x1(w1)
        self.w2 = Conv1x1(w2)
        self.w3 = Conv1x1(w3)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.operator_norm(hidden)
        if isinstance(self.mixer, LFM2AttentionMixer):
            mixed = self.mixer(normalized, attention_mask, causal_mask, cos, sin)
        else:
            mixed = self.mixer(normalized, attention_mask)
        hidden = hidden + mixed
        normalized = self.ffn_norm(hidden)
        hidden = hidden + self.w2(functional.silu(self.w1(normalized)) * self.w3(normalized))
        return hidden


class LFM2Prefill(torch.nn.Module):
    """Fixed-bucket LFM2 prefill graph returning every final hidden state."""

    embedding_weight: torch.Tensor
    layers: torch.nn.ModuleList
    final_norm: ChannelRmsNorm
    causal_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        config: LFM2Config,
        tensors: dict[str, torch.Tensor],
        seq_len: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.embedding_weight = torch.nn.Parameter(
            tensors["embed_tokens.weight"].to(dtype=torch.float32), requires_grad=False
        )
        layers: list[LFM2Layer] = []
        for index, layer_type in enumerate(config.layer_types):
            prefix = f"layers.{index}"
            if layer_type == "conv":
                mixer: torch.nn.Module = LFM2ConvMixer(
                    hidden_size=config.hidden_size,
                    in_proj=tensors[f"{prefix}.conv.in_proj.weight"],
                    conv_weight=tensors[f"{prefix}.conv.conv.weight"],
                    out_proj=tensors[f"{prefix}.conv.out_proj.weight"],
                )
            elif layer_type == "full_attention":
                mixer = LFM2AttentionMixer(
                    config=config,
                    q_proj=tensors[f"{prefix}.self_attn.q_proj.weight"],
                    q_norm=tensors[f"{prefix}.self_attn.q_layernorm.weight"],
                    k_proj=tensors[f"{prefix}.self_attn.k_proj.weight"],
                    k_norm=tensors[f"{prefix}.self_attn.k_layernorm.weight"],
                    v_proj=tensors[f"{prefix}.self_attn.v_proj.weight"],
                    out_proj=tensors[f"{prefix}.self_attn.out_proj.weight"],
                    seq_len=seq_len,
                )
            else:
                raise ValueError(f"unsupported LFM2 layer type {layer_type!r}")
            layers.append(
                LFM2Layer(
                    config=config,
                    operator_norm=tensors[f"{prefix}.operator_norm.weight"],
                    ffn_norm=tensors[f"{prefix}.ffn_norm.weight"],
                    mixer=mixer,
                    w1=tensors[f"{prefix}.feed_forward.w1.weight"],
                    w2=tensors[f"{prefix}.feed_forward.w2.weight"],
                    w3=tensors[f"{prefix}.feed_forward.w3.weight"],
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.final_norm = ChannelRmsNorm(tensors["embedding_norm.weight"], config.rms_norm_eps)
        cos, sin = build_rope_tables(seq_len, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)
        causal = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
        future = torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal.masked_fill(future, -10_000.0), persistent=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = functional.embedding(input_ids, self.embedding_weight)
        hidden = hidden.transpose(1, 2).unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, self.causal_mask, self.cos, self.sin)
        return self.final_norm(hidden).squeeze(-1).transpose(1, 2)


def apply_rope(hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply LFM2's half-split, non-interleaved rotary embedding."""
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


def _canonical_name(name: str) -> str:
    for prefix in ("model.lfm.", "model.", "lfm."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def load_tensors(model_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = model_path / "model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"LFM2 safetensors checkpoint not found: {checkpoint}")
    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        return {
            _canonical_name(name): reader.get_tensor(name).to(dtype=torch.float32)
            for name in reader.keys()
        }


def read_config(model_path: Path, tensors: dict[str, torch.Tensor]) -> LFM2Config:
    raw_outer = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    raw = raw_outer.get("lfm", raw_outer)
    hidden_size = int(raw.get("hidden_size", raw.get("block_dim")))
    num_heads = int(raw.get("num_attention_heads", raw.get("num_heads")))
    num_layers = int(raw["num_hidden_layers"])
    if "layer_types" in raw:
        layer_types = tuple(
            "full_attention" if value in ("full_attention", "attention") else "conv"
            for value in raw["layer_types"]
        )
    else:
        attention_indices = {int(value) for value in raw["full_attn_idxs"]}
        layer_types = tuple(
            "full_attention" if index in attention_indices else "conv"
            for index in range(num_layers)
        )
    actual_widths = {
        int(tensors[f"layers.{index}.feed_forward.w1.weight"].shape[0])
        for index in range(num_layers)
    }
    if len(actual_widths) != 1:
        raise ValueError(f"LFM2 checkpoint has inconsistent MLP widths: {sorted(actual_widths)}")
    intermediate_size = actual_widths.pop()
    config = LFM2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        num_key_value_heads=int(raw["num_key_value_heads"]),
        head_dim=hidden_size // num_heads,
        rms_norm_eps=float(raw.get("norm_eps", raw.get("block_norm_eps", 1e-5))),
        rope_theta=float(raw["rope_theta"]),
        vocab_size=int(raw["vocab_size"]),
        conv_kernel_size=int(raw.get("conv_L_cache", 3)),
        layer_types=layer_types,
        bos_token_id=int(raw.get("bos_token_id", 1)),
        eos_token_id=int(raw.get("eos_token_id", 7)),
        pad_token_id=int(raw.get("pad_token_id", 0)),
    )
    validate_config(config, tensors)
    return config


def validate_config(config: LFM2Config, tensors: dict[str, torch.Tensor]) -> None:
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError("hidden size must divide evenly across query heads")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("query heads must divide evenly across KV heads")
    if len(config.layer_types) != config.num_hidden_layers:
        raise ValueError("layer type count does not match num_hidden_layers")
    expected_embedding = (config.vocab_size, config.hidden_size)
    if tuple(tensors["embed_tokens.weight"].shape) != expected_embedding:
        raise ValueError("embedding tensor shape does not match config")
    for index, layer_type in enumerate(config.layer_types):
        prefix = f"layers.{index}"
        for suffix, expected in (
            ("feed_forward.w1.weight", (config.intermediate_size, config.hidden_size)),
            ("feed_forward.w2.weight", (config.hidden_size, config.intermediate_size)),
            ("feed_forward.w3.weight", (config.intermediate_size, config.hidden_size)),
        ):
            actual = tuple(tensors[f"{prefix}.{suffix}"].shape)
            if actual != expected:
                raise ValueError(f"{prefix}.{suffix} has shape {actual}, expected {expected}")
        if layer_type == "conv":
            actual_kernel = tuple(tensors[f"{prefix}.conv.conv.weight"].shape)
            expected_kernel = (config.hidden_size, 1, config.conv_kernel_size)
            if actual_kernel != expected_kernel:
                raise ValueError(f"{prefix} convolution has shape {actual_kernel}, expected {expected_kernel}")


def build_prefill(model_path: Path, seq_len: int) -> tuple[LFM2Prefill, LFM2Config]:
    tensors = load_tensors(model_path)
    config = read_config(model_path, tensors)
    model = LFM2Prefill(config, tensors, seq_len).eval()
    return model, config
