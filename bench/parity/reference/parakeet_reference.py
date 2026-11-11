#!/usr/bin/env python3
"""Independent, trace-producing fp32 Parakeet TDT reference.

Numerical libraries are imported only after the single-thread environment is set.
The bounded loop also accepts ordinary Python stub joints, so its termination
rule is tested without downloading a checkpoint.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

MODEL_REPO = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
MODEL_SHA256 = "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"
TRANSFORMERS_VERSION = "5.16.1"
TORCH_VERSION = "2.14.0"
TRANSFORMERS_WHEEL_SHA256 = "2f2d5b98a5ad3718713653734298fa620754ed683702a635ebb587df3ed29c7e"
BLANK = 8192
# Transformers 5.16.1 generation_parakeet.py:280-299 deliberately bypasses
# RNNT's per-frame count. Its K=10 only sizes a global output buffer. Our
# bounded adaptation forces progress after K zero-advance evaluations. Real
# fixtures must match HF generate and must never reach K emissions at a frame.
MAX_SYMBOLS_PER_FRAME = 10
DURATIONS = (0, 1, 2, 3, 4)
RUNGS = (128, 256, 512, 1024, 2048, 4096, 8192)


def frame_counts(samples):
    """Actual mel, valid mel, valid encoder counts for unpadded 16-kHz PCM."""
    valid_mel = samples // 160
    return valid_mel + 1, valid_mel, (valid_mel + 7) // 8


def ladder():
    """Smallest millisecond-aligned sample count for each boundary frame."""
    rows = []
    for rung in RUNGS:
        for frames in (rung - 1, rung, rung + 1):
            samples = ((frames - 1) * 8 + 1) * 160
            assert samples % 16 == 0 and frame_counts(samples)[2] == frames
            rows.append({"id": f"ladder-{frames}", "sample_count": samples,
                         "duration_ms": samples // 16, "encoder_frames": frames})
    rows.append({"id": "ladder-600000ms", "sample_count": 9_600_000,
                 "duration_ms": 600_000, "encoder_frames": 7500})
    return rows


def top_two(logits):
    """Highest logit first, lowest id wins ties, including for the runner-up."""
    if len(logits) < 2 or not all(math.isfinite(float(x)) for x in logits):
        raise ValueError("joint head must contain at least two finite logits")
    first = max(range(len(logits)), key=lambda i: (float(logits[i]), -i))
    second = max((i for i in range(len(logits)) if i != first),
                 key=lambda i: (float(logits[i]), -i))
    return first, second, float(logits[first]), float(logits[second])


def greedy(joint, valid_encoder_frames, blank=BLANK, cap=MAX_SYMBOLS_PER_FRAME,
           durations=DURATIONS):
    """Drive a stateful joint(frame, previous_token), recording every decision."""
    if valid_encoder_frames < 1 or cap < 1:
        raise ValueError("positive encoder length and per-frame cap required")
    trace, triples = [], []
    frame, previous, evaluations = 0, blank, 0
    while frame < valid_encoder_frames:
        token_logits, duration_logits = joint(frame, previous)
        tok1, tok2, tl1, tl2 = top_two(token_logits)
        dur1, dur2, dl1, dl2 = top_two(duration_logits)
        duration = durations[dur1]
        emitted = tok1 != blank
        evaluations += 1
        advance = duration if emitted else max(duration, 1)
        cap_hit = evaluations == cap and advance == 0
        trace.append({"frame_index": frame, "token_top1": tok1, "token_top2": tok2,
                      "duration_top1": dur1, "duration_top2": dur2,
                      "token_logit1": tl1, "token_logit2": tl2,
                      "duration_logit1": dl1, "duration_logit2": dl2,
                      "emitted": emitted, "cap_hit": cap_hit})
        if emitted:
            triples.append([tok1, frame, duration])
        previous = tok1
        if advance or cap_hit:
            frame += advance or 1
            evaluations = 0
    return trace, triples


def words_from_triples(tokenizer, triples, sample_count):
    """SentencePiece word boundaries and 80-ms timestamps, clamped to PCM."""
    groups = []
    for triple in triples:
        piece = tokenizer.convert_ids_to_tokens(triple[0])
        if not groups or piece.startswith("▁"):
            groups.append([])
        groups[-1].append(triple)
    duration_ms = sample_count / 16
    words = []
    for group in groups:
        first, last = group[0], group[-1]
        words.append({"word": tokenizer.decode([x[0] for x in group]).strip(),
                      "start_ms": min(80 * first[1], duration_ms),
                      "end_ms": min(80 * (last[1] + max(last[2], 1)), duration_ms)})
    return {"text": tokenizer.decode([x[0] for x in triples]) if triples else "",
            "words": words, "no_speech": not triples}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_runtime():
    import torch
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError("reference requires transformers==" + TRANSFORMERS_VERSION)
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise RuntimeError("reference requires torch==" + TORCH_VERSION)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    return torch


def load_reference(cache):
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, ParakeetForTDT
    from transformers.models.parakeet.generation_parakeet import ParakeetRNNTDecoderCache
    torch = configure_runtime()
    files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
             "processor_config.json", "generation_config.json"]
    path = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
                                  cache_dir=cache, allow_patterns=files))
    if sha256_file(path / "model.safetensors") != MODEL_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    model = ParakeetForTDT.from_pretrained(path, dtype=torch.float32,
                                         attn_implementation="eager").cpu().eval()
    processor = AutoProcessor.from_pretrained(path)
    return torch, model, processor, ParakeetRNNTDecoderCache, path


def run_fixture(runtime, audio, case_id):
    torch, model, processor, cache_class, _ = runtime
    from transformers.models.parakeet import modeling_parakeet
    import parakeet_mask_fix
    features = processor(audio, sampling_rate=16000, return_tensors="pt")
    actual_mel, valid_mel, valid_encoder = frame_counts(len(audio))
    actual_encoder = (actual_mel + 7) // 8
    upstream_function = modeling_parakeet.eager_attention_forward
    upstream_nan = False
    if actual_encoder > valid_encoder:
        with torch.inference_mode():
            unpatched = model.get_audio_features(**features, output_attention_mask=True)
        upstream_nan = not bool(torch.isfinite(unpatched.last_hidden_state[:, :valid_encoder]).all())
        if not upstream_nan:
            raise RuntimeError(f"{case_id}: padded-input exception was not reproduced")
        modeling_parakeet.eager_attention_forward = parakeet_mask_fix.eager_attention_forward
    captured = []
    handle = model.encoder.register_forward_hook(lambda module, args, out: captured.append(out))
    try:
        with torch.inference_mode():
            upstream = model.generate(**features, do_sample=False, num_beams=1)
    finally:
        handle.remove()
        modeling_parakeet.eager_attention_forward = upstream_function
    if len(captured) != 1:
        raise RuntimeError("expected exactly one upstream encoder evaluation")
    encoded = captured[0]
    if not upstream_nan:
        modeling_parakeet.eager_attention_forward = parakeet_mask_fix.eager_attention_forward
        try:
            with torch.inference_mode():
                patched = model.get_audio_features(**features, output_attention_mask=True)
            if not torch.equal(patched.last_hidden_state, encoded.last_hidden_state):
                raise RuntimeError(f"{case_id}: mask fix changed an unpadded encoder output")
        finally:
            modeling_parakeet.eager_attention_forward = upstream_function
    if not bool(torch.isfinite(encoded.last_hidden_state[:, :valid_encoder]).all()):
        raise RuntimeError(f"{case_id}: nonfinite valid encoder output")
    if features.input_features.shape[1] != actual_mel:
        raise RuntimeError("actual mel length disagrees with confirmed frame function")
    if features.attention_mask.sum().item() != valid_mel:
        raise RuntimeError("valid mel length disagrees with confirmed frame function")
    if encoded.attention_mask.sum().item() != valid_encoder:
        raise RuntimeError("valid encoder length disagrees with confirmed frame function")
    with torch.inference_mode():
        projected = model.encoder_projector(encoded.last_hidden_state)
        cache = cache_class(model.config)

        def joint(frame, previous):
            prediction = model.decoder(torch.tensor([[previous]]), cache=cache)
            logits = model.joint(prediction, projected[:, frame:frame + 1, :]).flatten()
            return logits[:model.config.vocab_size].tolist(), logits[model.config.vocab_size:].tolist()

        trace, triples = greedy(joint, valid_encoder)
    expected_tokens = upstream.sequences[0].tolist()[1:]
    expected_advances = upstream.durations[0].tolist()[1:]
    if [step["token_top1"] for step in trace] != expected_tokens:
        raise RuntimeError(f"{case_id}: independent loop differs from HF greedy tokens")
    advances = [DURATIONS[s["duration_top1"]] if s["emitted"] else
                max(DURATIONS[s["duration_top1"]], 1) for s in trace]
    if advances != expected_advances:
        raise RuntimeError(f"{case_id}: independent loop differs from HF greedy durations")
    maximum = max(Counter(x[1] for x in triples).values(), default=0)
    if maximum >= MAX_SYMBOLS_PER_FRAME:
        raise RuntimeError(f"{case_id}: natural fixture reaches per-frame cap")
    result = {"id": case_id, "sample_count": len(audio), "mel_frames": actual_mel,
              "valid_mel_frames": valid_mel, "valid_encoder_frames": valid_encoder,
              "trace": trace, "triples": triples, "hf_greedy_equal": True,
              "upstream_exception": "upstream NaN on padded input" if upstream_nan else None,
              "mask_fix": parakeet_mask_fix.REVISION,
              "max_emissions_per_frame": maximum,
              "minimum_token_margin": min(s["token_logit1"] - s["token_logit2"] for s in trace),
              "minimum_duration_margin": min(s["duration_logit1"] - s["duration_logit2"] for s in trace)}
    result.update(words_from_triples(processor.tokenizer, triples, len(audio)))
    mel = features.input_features[0].numpy().astype("<f4").tobytes()
    encoder = encoded.last_hidden_state[0, :valid_encoder].numpy().astype("<f4").tobytes()
    return result, mel, encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    runtime = load_reference(args.cache)
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=True)
    result, mel, encoder = run_fixture(runtime, np.zeros(16000, dtype=np.float32), "silence-1s")
    (args.output / "silence-1s.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "silence-1s.mel.f32le").write_bytes(mel)
    (args.output / "silence-1s.encoder.f32le").write_bytes(encoder)
    print(json.dumps({"seconds": time.monotonic() - start, "steps": len(result["trace"]),
                      "text": result["text"], "token_margin": result["minimum_token_margin"],
                      "duration_margin": result["minimum_duration_margin"]}), flush=True)


if __name__ == "__main__":
    main()
