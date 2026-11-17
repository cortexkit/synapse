"""CPU-only Q8 reference and explicit half-boundary calibration, not an engine."""

import contextlib
import json
import math
from pathlib import Path
import types

from qwen3_asr_reference import Reference, model_pin, sha256

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "qwen3-asr-quant-profile.json"
SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def q8_names(layers):
    return [f"thinker.model.layers.{i}.{suffix}" for i in range(layers)
            for suffix in SUFFIXES] + ["thinker.lm_head.weight"]


def verified_artifacts(directory):
    """Never trust a previously verified cache, or a mutable lineage tensor index."""
    directory = Path(directory)
    pin = model_pin()
    pins = pin["derived_artifacts"]
    for name, key in (("weights.q8_0", "q8_0_digest"),
                      ("remaining.f16.safetensors", "f16_digest")):
        if sha256(directory / name) != pins[key].removeprefix("sha256:"):
            raise ValueError("artifact_poisoned: " + name)
    lineage = json.loads((directory / "lineage.json").read_text())
    if (lineage["format"] != "synapse-owned-decode-q8_0-v1"
            or lineage["family"] != "qwen3-asr-1.7b"
            or lineage["source_manifest_digest"].removeprefix("sha256:") != pin["checkpoint_digest"].removeprefix("sha256:")
            or lineage["quantizer_revision"] != pins["quantizer_revision"]
            or lineage["derived_digest"] != pins["q8_0_digest"].removeprefix("sha256:")
            or lineage["object_file"] != "weights.q8_0"):
        raise ValueError("Q8 lineage identity mismatch")
    return lineage


def dequantize(raw, shape):
    """Decode Rust's 34-byte blocks; Python never quantizes the source weights."""
    import numpy as np
    if len(shape) != 2 or shape[1] % 32 or len(raw) != math.prod(shape) // 32 * 34:
        raise ValueError("invalid Q8 matrix layout")
    blocks = np.frombuffer(raw, dtype=np.dtype([("scale", "<f2"), ("q", "i1", 32)]))
    return (blocks["scale"].astype("<f4")[:, None]
            * blocks["q"].astype("<f4")).reshape(shape)


def install_weights(reference, directory, lineage):
    import torch
    from safetensors import safe_open
    model = reference.model
    layers = model.config.thinker_config.text_config.num_hidden_layers
    names = q8_names(layers)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    offset, expected = 0, []
    for name in names:
        shape = list(parameters[name].shape)
        length = math.prod(shape) // 32 * 34
        expected.append({"name": name, "shape": shape,
                         "byte_offset": offset, "byte_length": length})
        offset += length
    if lineage["tensors"] != expected or (directory / "weights.q8_0").stat().st_size != offset:
        raise ValueError("Q8 lineage tensor inventory mismatch")

    def replace(name, value):
        # Rebind rather than copy: the official loader ties head and embedding,
        # but the reference needs a Q8 head and an independent f16 embedding.
        parent, attr = name.rsplit(".", 1)
        model.get_submodule(parent)._parameters[attr] = torch.nn.Parameter(value, requires_grad=False)

    with safe_open(directory / "remaining.f16.safetensors", framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != set(parameters) - set(names):
            raise ValueError("f16 remainder tensor inventory mismatch")
        for name in tensors.keys():
            value = tensors.get_tensor(name)
            if value.dtype != torch.float16 or value.shape != parameters[name].shape:
                raise ValueError("invalid f16 remainder tensor")
            replace(name, value.float())
    # Do not retain the original fp32 parameter table while replacing Q8 matrices.
    del parameters
    with (directory / "weights.q8_0").open("rb") as stream:
        for record in expected:
            replace(record["name"], torch.from_numpy(dequantize(stream.read(record["byte_length"]), record["shape"])))


def half(value):
    import torch
    return value.to(torch.float16).to(torch.float32)


@contextlib.contextmanager
def numerical_profile(model):
    """Install only the boundaries enumerated in qwen3-asr-quant-profile.json."""
    import torch
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as upstream
    originals, handles = [], []

    def patch(obj, name, value):
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def rounded(_module, _args, output):
        return half(output)

    def decoder(layer, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(hidden_states),
            position_embeddings=position_embeddings, attention_mask=attention_mask, **kwargs)
        hidden_states = half(residual + hidden_states)
        return half(hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states)))

    def encoder(layer, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = layer.self_attn(hidden_states=layer.self_attn_layer_norm(hidden_states),
                                       cu_seqlens=cu_seqlens, attention_mask=attention_mask, **kwargs)
        hidden_states = half(residual + hidden_states)
        residual = hidden_states
        hidden_states = layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden_states))))
        return (half(residual + hidden_states),)

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **_kwargs):
        if dropout or module.training:
            raise ValueError("calibration requires eval mode without dropout")
        key = upstream.repeat_kv(key, module.num_key_value_groups)
        value = upstream.repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :key.shape[-2]]
        probabilities = half(torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32))
        output = torch.matmul(probabilities, value.float()).transpose(1, 2).contiguous()
        return output, probabilities

    try:
        for name, module in model.named_modules():
            if ((isinstance(module, torch.nn.Linear) and name != "thinker.lm_head")
                    or isinstance(module, (torch.nn.LayerNorm, upstream.Qwen3ASRThinkerTextRMSNorm,
                                           upstream.Qwen3ASRTextRMSNorm))):
                handles.append(module.register_forward_hook(rounded))
        for layer in model.thinker.model.layers:
            patch(layer, "forward", types.MethodType(decoder, layer))
        for layer in model.thinker.audio_tower.layers:
            patch(layer, "forward", types.MethodType(encoder, layer))
        # Indexing the positional sum is exact; this pre-hook rounds that sum
        # before the first encoder layer consumes it.
        handles.append(model.thinker.audio_tower.layers[0].register_forward_pre_hook(
            lambda _m, args: (half(args[0]), *args[1:])))
        patch(upstream, "eager_attention_forward", attention)
        yield
    finally:
        for handle in handles:
            handle.remove()
        for obj, name, value in reversed(originals):
            setattr(obj, name, value)


class QuantReference:
    def __init__(self, model_directory, artifacts):
        artifacts = Path(artifacts)
        lineage = verified_artifacts(artifacts)
        self.reference = Reference(Path(model_directory))
        install_weights(self.reference, artifacts, lineage)
        self.torch = self.reference.torch
        self.model = self.reference.model
        self.processor = self.reference.processor
        self.stop_ids = self.reference.stop_ids

    def inputs(self, case, audio):
        torch = self.torch
        context = ", ".join(case.get("terms", []))
        if case.get("prompt"):
            context += ("\n" if context else "") + case["prompt"]
        prompt = self.reference.wrapper._build_text_prompt("", case.get("language_hint"))
        inputs = self.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
        prefix = self.processor.tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
        ids = inputs["input_ids"][0].tolist()
        if ids[:len(prefix)] != prefix:
            raise ValueError("pinned template prefix changed")
        context_ids = self.processor.tokenizer.encode(context, add_special_tokens=False, split_special_tokens=True)
        inputs["input_ids"] = torch.tensor([prefix + context_ids + ids[len(prefix):]])
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs, len(context_ids)

    def run(self, case, audio, *, forced_ids=None, profile=False):
        """One prefill position plus one per forced id; never force free-decode ids."""
        from transformers import DynamicCache
        from qwen_asr.inference.utils import parse_asr_output
        from qwen3_asr_corpus import audio_tokens, max_new_tokens
        torch = self.torch
        inputs, context_tokens = self.inputs(case, audio)
        ids = inputs["input_ids"][0].tolist()
        positions = [i for i, token in enumerate(ids) if token == 151676]
        count = audio_tokens(int(inputs["feature_attention_mask"].sum()))
        if not positions or positions != list(range(positions[0], positions[0] + count)):
            raise ValueError("invalid audio span")
        cap = max_new_tokens(len(audio), 25)
        limit = len(forced_ids) + 1 if forced_ids is not None else cap
        if limit > cap:
            raise ValueError("forced trace exceeds generation cap")

        class ProfileCache(DynamicCache):
            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                return super().update(half(key_states), half(value_states), layer_idx, cache_kwargs)

        cache = (ProfileCache if profile else DynamicCache)(config=self.model.thinker.model.config)
        top, margins = [], []
        context = numerical_profile(self.model) if profile else contextlib.nullcontext()
        with context, torch.inference_mode():
            encoder = self.model.thinker.get_audio_features(
                inputs["input_features"], feature_attention_mask=inputs["feature_attention_mask"])
            if profile:
                encoder = half(encoder)
            embeddings = self.model.thinker.get_input_embeddings()(inputs["input_ids"])
            embeddings[:, positions[0]:positions[0] + count] = encoder
            for step in range(limit):
                output = self.model.thinker.model(inputs_embeds=embeddings, past_key_values=cache, use_cache=True)
                logits = self.model.thinker.lm_head(output.last_hidden_state[:, -1]).float()[0]
                values, _indices = torch.topk(logits, 2)
                token = int(logits.argmax())
                margin = float(values[0] - values[1])
                if not math.isfinite(margin) or margin < 0:
                    raise ValueError("invalid top-two margin")
                top.append(token)
                margins.append(margin)
                if forced_ids is None and token in self.stop_ids:
                    break
                if step + 1 < limit:
                    next_id = forced_ids[step] if forced_ids is not None else token
                    embeddings = self.model.thinker.get_input_embeddings()(torch.tensor([[next_id]]))
        raw = self.processor.tokenizer.decode(top, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        language, text = (None, None) if forced_ids is not None else parse_asr_output(raw, user_language=case.get("language_hint"))
        return {"id": case["id"], "split": case["split"], "generated_ids": top,
                "top_two_margins": margins, "text": text, "language": language or None,
                "prompt_ids": ids, "span": {"start": positions[0], "len": count},
                "context_tokens": context_tokens, "cap": cap,
                "stop_reason": "forced" if forced_ids is not None else ("stop" if top[-1] in self.stop_ids else "cap"),
                "mode": "f16-boundaries" if profile else "q8-cpu-teacher",
                "forced_ids": forced_ids}
