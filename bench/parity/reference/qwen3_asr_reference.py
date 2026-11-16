"""Pinned CPU reference and content-addressed tensor/trace serialization."""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import urllib.request

from qwen3_asr_corpus import SLUG, audio_tokens, max_new_tokens

HERE = Path(__file__).resolve().parent
PARITY = HERE.parent
VERSIONS = {"qwen-asr": "0.0.6", "transformers": "4.57.6", "torch": "2.9.1",
            "numpy": "2.2.6", "soundfile": "0.13.1", "librosa": "0.11.0"}
_TORCH = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_pin():
    return json.loads((PARITY / "models.json").read_text())["models"][SLUG]


def fetch_model(cache):
    from prepare_parakeet import fetch
    pin = model_pin()
    directory = cache / "model" / SLUG
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in pin["files"].items():
        source = fetch(f"https://huggingface.co/{pin['hf_repo']}/resolve/{pin['hf_revision']}/{name}",
                       digest, cache / "downloads")
        target = directory / name
        if not target.exists():
            os.link(source, target)
        if sha256(target) != digest:
            raise ValueError("model digest mismatch: " + name)
    return directory


def merged_tensor_index(headers):
    merged = {}
    for header in headers:
        for name, tensor in header.items():
            if name in merged and (name != "__metadata__" or merged[name] != tensor):
                raise ValueError("duplicate tensor in checkpoint shards: " + name)
            merged[name] = tensor
    return merged


def generation_config_record(config):
    """Keep effective policy and runtime token IDs, omitting only length budgets."""
    def json_value(value):
        if hasattr(value, "detach") and hasattr(value, "tolist"):
            # Transformers 4.57.6 adds token tensors before constructing its
            # logits processors; GenerationConfig.to_dict leaves those intact.
            return json_value(value.detach().cpu().tolist())
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        return value
    return {key: json_value(value) for key, value in config.to_dict().items()
            if key not in ("max_length", "max_new_tokens")}


def generation_stop_reason(ids, cap, stop_ids):
    """EOS wins at the final allowed position; non-EOS prefixes must fill the cap."""
    if not ids or len(ids) > cap or any(token in stop_ids for token in ids[:-1]):
        raise ValueError("invalid reference continuation")
    if ids[-1] in stop_ids:
        return "stop"
    if len(ids) == cap:
        return "cap"
    raise ValueError("reference ended early without a stop token")


def fetch_tensor_index():
    """Fetch just each immutable safetensors header, never the full shard."""
    pin = model_pin()
    headers = []
    for name in sorted(pin["files"]):
        if not name.endswith(".safetensors"):
            continue
        url = f"https://huggingface.co/{pin['hf_repo']}/resolve/{pin['hf_revision']}/{name}"
        def read_range(start, end):
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise ValueError("server did not honor checkpoint header range")
                raw = response.read(end - start + 2)
            if len(raw) != end - start + 1:
                raise ValueError("checkpoint header range size mismatch")
            return raw
        length = struct.unpack("<Q", read_range(0, 7))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("unreasonable checkpoint header length")
        headers.append(json.loads(read_range(8, length + 7)))
    return merged_tensor_index(headers)


def deterministic_runtime():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for package, expected in VERSIONS.items():
        actual = importlib.metadata.version(package).split("+")[0]
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, got {actual}")
    import numpy as np
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError("reference requires the pinned CPU Torch wheel, not a CUDA build")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.set_float32_matmul_precision("highest")
    _TORCH = torch
    return torch


def artifact(output, raw, relative, external=False):
    digest = hashlib.sha256(raw).hexdigest()
    relative = str(Path(SLUG) / (digest + ".f32le")) if external else relative
    destination = output / ("external" if external else "") / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent shards can produce the same audio tensor. Publish complete
    # immutable bytes atomically, never an in-progress content-addressed file.
    temporary = destination.with_name(destination.name + f".partial-{os.getpid()}")
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    return {"cache_path" if external else "path": relative, "sha256": digest, "bytes": len(raw)}


def pcm(case, cache):
    import numpy as np
    import generate_parakeet
    if case.get("recipe") in ("near-silence", "noise"):
        scale = 1e-6 if case["recipe"] == "near-silence" else 0.05
        rng = np.random.Generator(np.random.PCG64(case["seed"]))
        return (rng.standard_normal(case["sample_count"]) * scale).astype(np.float32)
    if case.get("recipe") == "concatenate":
        audio = np.concatenate([pcm(part, cache) for part in case["parts"]])
        return np.pad(audio, (0, case["sample_count"] - len(audio)))
    generate_parakeet.CACHE = cache
    return generate_parakeet.pcm(case)


def checkpoint_weight_facts(directory, model):
    """Bind on-disk head/embedding ranges to the weights the official loader uses."""
    import numpy as np
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    names = ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight")
    sources = {}
    for name in names:
        shard = directory / weight_map[name]
        with shard.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
            info = header[name]
            if info["dtype"] != "BF16":
                raise ValueError("unexpected source embedding/head dtype")
            start, end = info["data_offsets"]
            stream.seek(8 + length + start)
            raw_hash, fp32_hash = hashlib.sha256(), hashlib.sha256()
            remaining = end - start
            while remaining:
                raw = stream.read(min(remaining, 8 * 1024 * 1024))
                if not raw:
                    raise ValueError("truncated source head/embedding")
                remaining -= len(raw)
                raw_hash.update(raw)
                # Exact BF16 expansion: put its 16 bits in the high half of f32.
                bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
                fp32_hash.update(bits.astype("<u4", copy=False).tobytes())
        sources[name] = {"shard": weight_map[name], "data_offsets": [start, end],
                         "sha256": raw_hash.hexdigest(), "fp32_sha256": fp32_hash.hexdigest()}
    embedding = model.thinker.model.embed_tokens.weight
    head = model.thinker.lm_head.weight
    loaded = {}
    for name, tensor in zip(names, (embedding, head)):
        digest = hashlib.sha256()
        for chunk in tensor.detach().split(1024):
            digest.update(chunk.contiguous().numpy().astype("<f4", copy=False).tobytes())
        loaded[name] = digest.hexdigest()
    matches = [name for name in names if sources[name]["fp32_sha256"] == loaded[names[1]]]
    if not matches:
        raise ValueError("official loaded logits weight matches neither pinned source tensor")
    return {"source_tensors": sources,
            "source_tensors_byte_identical": sources[names[0]]["sha256"] == sources[names[1]]["sha256"],
            "loaded_fp32_sha256": loaded,
            "loaded_head_embedding_share_storage": head.untyped_storage().data_ptr() == embedding.untyped_storage().data_ptr(),
            "logits_weight_matches_sources": matches,
            "engine_rule": "Use the matrix the official loader exposes as thinker.lm_head.weight, even if config tying changes which source tensor supplies it."}


class Reference:
    def __init__(self, model_directory):
        self.torch = deterministic_runtime()
        from qwen_asr import Qwen3ASRModel
        self.wrapper = Qwen3ASRModel.from_pretrained(
            str(model_directory), dtype=self.torch.float32, device_map="cpu",
            attn_implementation="eager", max_inference_batch_size=1)
        self.model = self.wrapper.model.eval()
        self.processor = self.wrapper.processor
        if any(p.dtype != self.torch.float32 for p in self.model.parameters()):
            raise ValueError("reference contains non-fp32 parameters")
        self.stop_ids = json.loads((model_directory / "generation_config.json").read_text())["eos_token_id"]
        self.pinned_generation_config = json.loads((model_directory / "generation_config.json").read_text())
        self.generation_policy = None
        self.generation_budget = None
        original = self.model.thinker._get_logits_processor
        def capture_processors(*args, **kwargs):
            processors = original(*args, **kwargs)
            # Empty actual processor lists prove that the larger reference cap
            # cannot alter logits (for example by forcing EOS near the limit).
            if processors:
                raise ValueError("unexpected generation logits processors: " +
                                 ", ".join(type(p).__name__ for p in processors))
            config = kwargs.get("generation_config", args[0] if args else None)
            if config is None:
                raise ValueError("missing effective generation config")
            if config.get_generation_mode().value != "greedy_search":
                raise ValueError("reference generation mode is not greedy search")
            self.generation_budget = {"max_new_tokens": config.max_new_tokens, "max_length": config.max_length}
            effective = generation_config_record(config)
            policy = {"pinned_generation_config": self.pinned_generation_config,
                      "pinned_generation_config_sha256": sha256(model_directory / "generation_config.json"),
                      "effective_generation_config_without_length_limits": effective,
                      "actual_logits_processors": [], "do_sample": False,
                      "generation_mode": "greedy_search",
                      "cap_independence": "Empty logits processors and greedy search: an unused larger cap cannot change stopped continuations. Capped prefixes retain their exact reference cap as identity."}
            if self.generation_policy is not None and self.generation_policy != policy:
                raise ValueError("generation policy changed across fixtures")
            self.generation_policy = policy
            return processors
        self.model.thinker._get_logits_processor = capture_processors

    def run(self, case, audio, output, tokens_per_second=6):
        from qwen_asr.inference.utils import parse_asr_output
        torch = self.torch
        context = ", ".join(case.get("terms", []))
        if case.get("prompt"):
            context += ("\n" if context else "") + case["prompt"]
        prompt = self.wrapper._build_text_prompt("", case.get("language_hint"))
        inputs = self.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
        prefix = self.processor.tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
        original_ids = inputs["input_ids"][0].tolist()
        if original_ids[:len(prefix)] != prefix:
            raise ValueError("pinned template system prefix changed")
        # Parse framing tokens only in the template, never in caller-provided text.
        context_ids = self.processor.tokenizer.encode(context, add_special_tokens=False,
                                                      split_special_tokens=True)
        inputs["input_ids"] = torch.tensor([prefix + context_ids + original_ids[len(prefix):]])
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        ids = inputs["input_ids"][0].tolist()
        positions = [i for i, token in enumerate(ids) if token == 151676]
        valid_frames = int(inputs["feature_attention_mask"].sum())
        count = audio_tokens(valid_frames)
        if not positions or positions != list(range(positions[0], positions[0] + count)):
            raise ValueError("non-contiguous or incorrectly sized audio placeholder")
        projected = []
        handle = self.model.thinker.audio_tower.proj2.register_forward_hook(
            lambda _module, _args, value: projected.append(value.detach().clone()))
        cap = max_new_tokens(len(audio), tokens_per_second)
        self.generation_budget = None
        try:
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                                                eos_token_id=self.stop_ids, output_scores=True)
        finally:
            handle.remove()
        continuation = generated.sequences[0, len(ids):].tolist()
        stop_reason = generation_stop_reason(continuation, cap, self.stop_ids)
        if self.generation_policy is None:
            raise ValueError("generation processor evidence was not captured")
        if self.generation_budget != {"max_new_tokens": cap, "max_length": len(ids) + cap}:
            raise ValueError("official generation budget differs from the requested cap")
        if case.get("language_hint"):
            language_prefix = 0
        else:
            if continuation.count(151704) != 1:
                raise ValueError("missing or repeated <asr_text> language marker: " + case["id"])
            language_prefix = continuation.index(151704) + 1
        if len(projected) != 1 or tuple(projected[0].shape) != (count, 2048):
            raise ValueError("unexpected proj2 output")
        margins, top_ids = [], []
        for logits in generated.scores:
            values, indices = torch.topk(logits[0].float(), 2)
            margins.append(float(values[0] - values[1]))
            top_ids.append(int(logits[0].argmax()))
        if not all(math.isfinite(margin) and margin >= 0 for margin in margins):
            raise ValueError("non-finite reference margin")
        if top_ids != continuation:
            raise ValueError("greedy ids differ from recorded top-1 logits")
        raw_text = self.processor.tokenizer.decode(continuation, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
        language, text = parse_asr_output(raw_text, user_language=case.get("language_hint"))
        mel = inputs["input_features"][0].T.contiguous().numpy().astype("<f4").tobytes()
        encoder = projected[0].contiguous().numpy().astype("<f4").tobytes()
        return {"id": case["id"], "split": case["split"], "tier": case.get("tier"), "sample_count": len(audio),
                "ground_truth": case["ground_truth"], "prompt_ids": ids,
                "span": {"start": positions[0], "len": count}, "valid_mel_frames": valid_frames,
                "mel_frames": inputs["input_features"].shape[-1], "encoder_width": 2048,
                "generated_ids": continuation, "top_two_margins": margins,
                "language_prefix_tokens": language_prefix, "context_tokens": len(context_ids),
                "text": text, "language": language or None, "max_new_tokens": cap,
                "stop_reason": stop_reason, "cap": cap,
                "generation_max_length": len(ids) + cap,
                "tokens_per_second": tokens_per_second,
                "generation_policy_sha256": hashlib.sha256(canonical(self.generation_policy)).hexdigest(),
                "mel": artifact(output, mel, "", True),
                "encoder": artifact(output, encoder, "", True)}
