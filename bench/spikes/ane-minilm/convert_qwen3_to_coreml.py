#!/usr/bin/env python3
"""Convert official Qwen3-Embedding-0.6B to a fixed-shape ANE Core ML package.

The exported graph preserves Qwen3's causal decoder semantics: Q/K RMSNorm,
RoPE, grouped-query attention, causal-plus-padding masking, terminal-EOS
inputs, last-token pooling, and L2 normalization.  It intentionally uses
``torch.export`` only; trace-built encoder packages can pass conversion while
producing an unusable embedding fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import coremltools as ct  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import torch.nn.functional as functional
import transformers
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
SMOKE_TEXTS = (
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


@dataclass(frozen=True)
class TokenizerPolicy:
    add_special_tokens: bool
    terminal_eos_token_id: int
    pad_token_id: int
    padding_side: str


@dataclass(frozen=True)
class ParityReport:
    rows: int
    wrapper_hf_max_abs: float
    wrapper_hf_mean_cosine: float
    eager_export_max_abs: float
    eager_export_mean_cosine: float
    eager_coreml_max_abs: float
    eager_coreml_mean_cosine: float


@dataclass(frozen=True)
class ConversionReport:
    source_model: str
    source_model_sha256: str
    seq_len: int
    output_path: str
    output_name: str
    frontend: str
    compute_precision: str
    compute_units: str
    model_config: ModelConfig
    tokenizer_policy: TokenizerPolicy
    parity: ParityReport
    environment: EnvironmentReport


class Conv1x1(torch.nn.Module):
    """A frozen linear projection represented as ANE-friendly 1x1 convolution."""

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
    """Qwen RMSNorm over the channel dimension of [B, hidden, 1, seq]."""

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
    """One static-shape Qwen3 decoder block with causal grouped-query attention."""

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
        seq_len: int,
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
        self.hidden_size = config.hidden_size
        self.projection_size = config.num_attention_heads * config.head_dim
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.seq_len = seq_len
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
        query = query.reshape(1, self.query_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        key = key.reshape(1, self.kv_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)
        value = value.reshape(1, self.kv_heads, self.head_dim, self.seq_len).permute(0, 1, 3, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        key = key.repeat_interleave(self.kv_repetition, dim=1)
        value = value.repeat_interleave(self.kv_repetition, dim=1)

        padding_mask = (1.0 - attention_mask.to(dtype=torch.float32)).reshape(
            1, 1, 1, self.seq_len
        ) * -10_000.0
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * self.scale
        scores = scores + causal_mask + padding_mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attended = torch.matmul(probabilities, value.float())
        attended = attended.permute(0, 1, 3, 2).reshape(
            1, self.projection_size, 1, self.seq_len
        )
        hidden = residual + self.o_proj(attended)

        residual = hidden
        normalized = self.post_attention_norm(hidden)
        gate = functional.silu(self.gate_proj(normalized))
        hidden = residual + self.down_proj(gate * self.up_proj(normalized))
        return hidden


class Qwen3Embedder(torch.nn.Module):
    """The official Qwen3 embedding encoder as a fixed-bucket Core ML graph."""

    def __init__(self, config: ModelConfig, tensors: dict[str, torch.Tensor], seq_len: int) -> None:
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.embedding_weight = torch.nn.Parameter(
            tensors["embed_tokens.weight"].to(dtype=torch.float32), requires_grad=False
        )
        layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"layers.{index}"
            layers.append(
                Qwen3Layer(
                    config=config,
                    input_norm=tensors[f"{prefix}.input_layernorm.weight"],
                    post_attention_norm=tensors[f"{prefix}.post_attention_layernorm.weight"],
                    q_proj=tensors[f"{prefix}.self_attn.q_proj.weight"],
                    q_norm=tensors[f"{prefix}.self_attn.q_norm.weight"],
                    k_proj=tensors[f"{prefix}.self_attn.k_proj.weight"],
                    k_norm=tensors[f"{prefix}.self_attn.k_norm.weight"],
                    v_proj=tensors[f"{prefix}.self_attn.v_proj.weight"],
                    o_proj=tensors[f"{prefix}.self_attn.o_proj.weight"],
                    gate_proj=tensors[f"{prefix}.mlp.gate_proj.weight"],
                    up_proj=tensors[f"{prefix}.mlp.up_proj.weight"],
                    down_proj=tensors[f"{prefix}.mlp.down_proj.weight"],
                    seq_len=seq_len,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.final_norm = ChannelRmsNorm(tensors["norm.weight"], config.rms_norm_eps)
        cos, sin = build_rope_tables(seq_len, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)
        causal = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
        causal = causal.masked_fill(torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1), -10_000.0)
        self.register_buffer("causal_mask", causal, persistent=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = functional.embedding(input_ids, self.embedding_weight)
        hidden = hidden.transpose(1, 2).unsqueeze(2)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, self.causal_mask, self.cos, self.sin)
        hidden = self.final_norm(hidden).squeeze(2).transpose(1, 2)
        # Inputs are left-padded, so the terminal EOS token is always at -1.
        pooled = hidden[:, -1, :]
        norm = torch.sqrt(torch.sum(pooled.float() * pooled.float(), dim=-1, keepdim=True))
        return pooled / torch.clamp(norm, min=1e-12)


def apply_rope(hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = hidden.shape[-1] // 2
    rotated = torch.cat((-hidden[..., half:], hidden[..., :half]), dim=-1)
    return hidden * cos + rotated * sin


def build_rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.float32)
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().reshape(1, 1, seq_len, head_dim), angles.sin().reshape(1, 1, seq_len, head_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF repo id or local snapshot directory")
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


def read_config(model_ref: str) -> ModelConfig:
    config_path = Path(model_ref) / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
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
    config = ModelConfig(**{key: raw[key] for key in required})
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("Qwen3 query heads must divide evenly across KV heads")
    return config


def load_tensors(model_ref: str) -> dict[str, torch.Tensor]:
    path = Path(model_ref) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Qwen3 safetensors checkpoint not found: {path}")
    with safe_open(path, framework="pt", device="cpu") as reader:
        return {name: reader.get_tensor(name).to(dtype=torch.float32) for name in reader.keys()}


def build_embedder(model_ref: str, seq_len: int) -> tuple[Qwen3Embedder, ModelConfig]:
    config = read_config(model_ref)
    model = Qwen3Embedder(config, load_tensors(model_ref), seq_len).eval()
    return model, config


def prepare_input(
    tokenizer: Any,
    config: ModelConfig,
    text: str,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, TokenizerPolicy]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=seq_len - 1,
        padding=False,
        return_attention_mask=False,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    if ids and ids[-1] == config.eos_token_id:
        ids.pop()
    ids = ids[: seq_len - 1]
    ids.append(config.eos_token_id)
    if len(ids) > seq_len:
        raise AssertionError("terminal-EOS policy exceeded fixed bucket")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = config.eos_token_id
    pad_count = seq_len - len(ids)
    input_ids = [int(pad_token_id)] * pad_count + ids
    attention_mask = [0] * pad_count + [1] * len(ids)
    policy = TokenizerPolicy(
        add_special_tokens=True,
        terminal_eos_token_id=config.eos_token_id,
        pad_token_id=int(pad_token_id),
        padding_side="left",
    )
    return (
        torch.tensor([input_ids], dtype=torch.int32),
        torch.tensor([attention_mask], dtype=torch.int32),
        policy,
    )


def tensor_rows(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32).reshape(value.shape[0], -1)


def mean_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = np.sum(reference * candidate, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def hf_embedding(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    with torch.inference_mode():
        hidden = model(
            input_ids=input_ids.to(dtype=torch.long),
            attention_mask=attention_mask.to(dtype=torch.long),
            return_dict=False,
        )[0]
        pooled = hidden[:, -1, :]
        return functional.normalize(pooled.float(), p=2, dim=-1)


def convert_and_verify(
    wrapper: Qwen3Embedder,
    model_ref: str,
    seq_len: int,
    allow_download: bool,
) -> tuple[ct.models.MLModel, ParityReport, TokenizerPolicy]:
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=not allow_download)
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    tokenizer_policy: TokenizerPolicy | None = None
    for text in SMOKE_TEXTS:
        input_ids, attention_mask, tokenizer_policy = prepare_input(
            tokenizer, wrapper.config, text, seq_len
        )
        examples.append((input_ids, attention_mask))
    assert tokenizer_policy is not None

    hf_model = AutoModel.from_pretrained(
        model_ref,
        local_files_only=not allow_download,
        attn_implementation="eager",
    ).eval()
    with torch.inference_mode():
        eager_rows = [tensor_rows(wrapper(*inputs)) for inputs in examples]
        hf_rows = [tensor_rows(hf_embedding(hf_model, *inputs)) for inputs in examples]
        exported = torch.export.export(wrapper, examples[0], strict=False)
        exported_rows = [tensor_rows(exported.module()(*inputs)) for inputs in examples]
    del hf_model

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
    if converted_output != "embedding":
        ct.utils.rename_feature(mlmodel._spec, converted_output, "embedding")

    coreml_rows = []
    for input_ids, attention_mask in examples:
        prediction = mlmodel.predict(
            {
                "input_ids": input_ids.detach().cpu().numpy().astype(np.int32),
                "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int32),
            }
        )
        value = prediction.get("embedding", prediction.get(converted_output))
        if value is None:
            raise RuntimeError(f"Core ML prediction has no embedding output: {list(prediction)}")
        coreml_rows.append(tensor_rows(value))

    eager = np.concatenate(eager_rows, axis=0)
    hf = np.concatenate(hf_rows, axis=0)
    exported_output = np.concatenate(exported_rows, axis=0)
    coreml = np.concatenate(coreml_rows, axis=0)
    report = ParityReport(
        rows=eager.shape[0],
        wrapper_hf_max_abs=float(np.max(np.abs(eager - hf))),
        wrapper_hf_mean_cosine=mean_cosine(hf, eager),
        eager_export_max_abs=float(np.max(np.abs(eager - exported_output))),
        eager_export_mean_cosine=mean_cosine(eager, exported_output),
        eager_coreml_max_abs=float(np.max(np.abs(eager - coreml))),
        eager_coreml_mean_cosine=mean_cosine(eager, coreml),
    )
    if report.wrapper_hf_mean_cosine < 0.99999:
        raise RuntimeError(f"custom Qwen3 wrapper disagrees with HF: {report.wrapper_hf_mean_cosine}")
    if report.eager_export_max_abs > 1e-6:
        raise RuntimeError(f"torch.export parity failed: max_abs={report.eager_export_max_abs}")
    if report.eager_coreml_mean_cosine < 0.999:
        raise RuntimeError(f"Core ML conversion parity failed: mean_cosine={report.eager_coreml_mean_cosine}")
    return mlmodel, report, tokenizer_policy


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_metadata(mlmodel: ct.models.MLModel, report: ConversionReport) -> None:
    mlmodel.short_description = "Synapse fixed-sequence Qwen3-Embedding-0.6B for Apple Neural Engine."
    mlmodel.author = "Synapse bench/spikes/ane-minilm"
    mlmodel.license = "Source model license follows the referenced Hugging Face checkpoint."
    metadata = mlmodel.user_defined_metadata
    metadata["synapse.source_model"] = report.source_model
    metadata["synapse.source_model_sha256"] = report.source_model_sha256
    metadata["synapse.seq_len"] = str(report.seq_len)
    metadata["synapse.frontend"] = report.frontend
    metadata["synapse.output_name"] = report.output_name
    metadata["synapse.tokenizer_policy"] = json.dumps(asdict(report.tokenizer_policy), sort_keys=True)
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

    wrapper, config = build_embedder(model_ref, args.seq_len)
    mlmodel, parity, tokenizer_policy = convert_and_verify(
        wrapper, model_ref, args.seq_len, args.allow_download
    )
    report = ConversionReport(
        source_model=report_model_ref,
        source_model_sha256=sha256(model_path / "model.safetensors"),
        seq_len=args.seq_len,
        output_path=str(args.out),
        output_name="embedding",
        frontend="torch.export",
        compute_precision="float16",
        compute_units="CPU_AND_NE",
        model_config=config,
        tokenizer_policy=tokenizer_policy,
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
