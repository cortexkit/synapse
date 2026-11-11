#!/usr/bin/env python3
"""Assemble measured reference outputs without changing legacy profile entries.

Run after the corpus and all four ladder partitions have been materialized.
Generated cache artifacts stay outside git; the index records their local mirror
paths and cryptographic seals. --prepare omits package pins for the converter's
first pass; only the default, fully sealed output is suitable for committing.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct

import parakeet_reference as ref

SLUG = "parakeet-tdt-0.6b-v3"


def digest(path):
    return ref.sha256_file(path)


def write(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=None if compact else 2,
                               allow_nan=False) + "\n")


def artifact(path, root):
    return {"path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size}


def confirmations(reference_dir):
    return {
        "transformers_version": ref.TRANSFORMERS_VERSION,
        "transformers_wheel_sha256": ref.TRANSFORMERS_WHEEL_SHA256,
        "torch_version": ref.TORCH_VERSION,
        "environment": {"device": "cpu", "dtype": "fp32", "intraop_threads": 1,
                        "interop_threads": 1, "deterministic_algorithms": True,
                        "attention_implementation": "eager", "seed": 0},
        "frontend": {
            "source": "transformers 5.16.1 feature_extraction_parakeet.py:30-31,96-125,214-245",
            "sample_rate": 16000, "preemphasis": 0.97, "preemphasis_first_sample": "unchanged",
            "window": "Hann, periodic=False", "window_samples": 400, "hop_samples": 160,
            "n_fft": 512, "stft_center": True, "stft_pad_mode": "constant (zero)",
            "spectrum": "sqrt(re^2+im^2), then square, fp32",
            "mel_bins": 128, "mel_scale": "Slaney (librosa htk=False)", "mel_norm": "slaney",
            "mel_filter_dtype": "fp32", "frequency_range_hz": [0, 8000],
            "log": "natural", "log_guard": "2^-24 (5.9604644775390625e-8)",
            "normalization": "per feature, mean and sample standard deviation over valid frames only",
            "mean_divisor": "valid_mel_frames", "std_divisor": "valid_mel_frames - 1",
            "normalization_denominator": "std + 1e-5", "dither": 0,
            "padding_after_normalization": "invalid mel frames set to 0.0",
        },
        "frame_counts": {
            "samples": "mono 16 kHz sample frames after reference resampling",
            "actual_mel_frames": "floor(samples / 160) + 1",
            "valid_mel_frames": "floor(samples / 160)",
            "stride2": "floor((L - 1) / 2) + 1 = ceil(L / 2)",
            "valid_encoder_frames": "ceil(ceil(ceil(valid_mel_frames / 2) / 2) / 2)",
            "source": "feature_extraction_parakeet.py:191-245; modeling_parakeet.py:365-431,515-541",
        },
        "masking": {
            "before_first_subsampling_convolution": "no extra mask beyond frontend zero padding",
            "between_subsampling_stages": "zero invalid time positions after every Conv2d, including pointwise stages",
            "conformer_convolution": "zero invalid positions after GLU, immediately before depthwise convolution",
            "attention": "pairwise valid query AND valid key; finite -10000 mask before softmax, then zero masked probabilities",
            "patch_revision": "synapse-parakeet-mask-fix-v1",
            "patch_sha256": digest(reference_dir / "parakeet_mask_fix.py"),
            "replaced_function": "transformers.models.parakeet.modeling_parakeet.eager_attention_forward:241-263",
            "reason": "upstream all-masked padded query rows produce NaN under eager softmax; padded convolution neighbours contaminate valid frames",
            "nemo_revision": "2381f42f6979449b5b99538f8f80135831009b51",
            "nemo_attention_source": "nemo/collections/asr/parts/submodules/multi_head_attention.py:125-146",
            "nemo_convolution_source": "nemo/collections/asr/parts/submodules/conformer_modules.py:320-345",
            "nemo_behavior": "finite -10000 sentinel, zero masked probabilities after softmax, zero padded frames after GLU before depthwise convolution",
            "convolution_patch_needed": False,
        },
        "decode_rule": {
            "revision": "synapse-parakeet-tdt-greedy-v1", "blank_token_id": 8192,
            "durations": [0, 1, 2, 3, 4], "argmax_ties": "lowest index",
            "a": "zero LSTM state, initial predictor input blank; blank preserves predictor state and output",
            "b": "count every joint evaluation at a frame; a blank always advances by max(duration, 1)",
            "c": "force +1 only when the K-th evaluation otherwise advances by zero; no extra +1 after positive duration",
            "d": "K=10 is a bounded adaptation, not the upstream TDT per-frame rule",
            "K": 10, "maximum_evaluations": "valid_encoder_frames * K",
            "transformers_deviation": "5.16.1 generation_parakeet.py:280-299 bypasses the RNNT frame counter; max_symbols_per_step=10 sizes only the global output buffer",
            "nemo_deviation": "v2.4.0 tdt greedy counts per-frame symbols but adds +1 even after a positive duration at the cap; this bounded rule matches neither upstream exactly",
            "natural_fixture_guard": "fail if any fixture reaches K emissions at a single frame",
            "hf_equality": "tokens and durations must equal HF generate on every fixture with finite unpatched encoder output",
            "hf_exception_class": "upstream NaN on padded input; compare independent loop against patched HF generate instead",
        },
        "word_rule": {
            "boundary": "first emitted token, then each token piece beginning with U+2581",
            "word": "tokenizer decode of word token ids, leading/trailing whitespace stripped",
            "text": "tokenizer decode of all emitted token ids",
            "start_ms": "80 * first token frame_index, clamped to audio duration",
            "end_ms": "80 * (last token frame_index + max(last token duration, 1)), clamped to audio duration",
            "no_speech": "true exactly when no token was emitted; text empty and words empty",
        },
        "wer_normalizer": "synapse-wer-norm-v1",
        "synthetic_wav_metadata": "libsndfile FLOAT WAV PEAK timestamp is zeroed; encoded samples and peak values are unchanged",
        "duration_ladder": ref.ladder(),
        "integer_tensors": "retain BatchNorm num_batches_tracked scalar I64 counters losslessly in v1 packages",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--parity-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--package-seals", type=Path)
    args = parser.parse_args()
    root, generated = args.parity_dir, args.generated
    summary_paths = [generated / f"summary-{group}-{part}.json"
                     for group, part in [("corpus", 0)] + [("ladder", i) for i in range(4)]]
    if args.prepare:
        summary_paths = [path for path in summary_paths if path.exists()]
    summaries = [json.loads(path.read_text()) for path in summary_paths]
    cases = sorted([case for summary in summaries for case in summary["cases"]], key=lambda c: c["id"])
    if len({c["id"] for c in cases}) != len(cases):
        raise RuntimeError("duplicate case ids across partitions")
    corpus = json.loads((generated / "reference/parakeet-corpus.json").read_text())
    if len(corpus["cases"]) != 1200 or {c["id"] for c in corpus["cases"]} - {c["id"] for c in cases}:
        raise RuntimeError("the full 1,200-case corpus is required")
    if not args.prepare and {c["id"] for c in cases if c["id"].startswith("ladder-")} != {c["id"] for c in ref.ladder()}:
        raise RuntimeError("duration ladder is incomplete")
    for case in cases:
        if case["max_emissions_per_frame"] >= ref.MAX_SYMBOLS_PER_FRAME:
            raise RuntimeError("a natural fixture reached the bounded cap")
        for key in ("trace", "log_mel", "encoder"):
            seal = case.get(key)
            if not seal:
                continue
            source = generated / ("external" if "cache_path" in seal else "") / seal.get("path", seal.get("cache_path"))
            if source.stat().st_size != seal["bytes"] or digest(source) != seal["sha256"]:
                raise RuntimeError(f"invalid generated seal: {source}")
            destination = root / (".cache-out" if "cache_path" in seal else "") / seal.get("path", seal.get("cache_path"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    shutil.copyfile(generated / "reference/parakeet-corpus.json", root / "reference/parakeet-corpus.json")
    subset_path = root / "fixtures" / SLUG / "self-check.json"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated / "fixtures" / SLUG / "self-check.json", subset_path)
    subset = json.loads(subset_path.read_text())
    for case in subset["cases"]:
        if case["minimum_margin"] < 2 * subset["tau_margin"]:
            raise RuntimeError("self-check subset margin below 2*tau_margin")
        if "wav" in case:
            path = case["wav"]["path"]
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generated / path, root / path)
        trace_path = case["trace"]["path"]
        shutil.copyfile(generated / trace_path, root / trace_path)
    payload_bytes = sum(c.get("wav", {}).get("bytes", 0) + c["trace"]["bytes"] for c in subset["cases"])
    while True:
        raw = json.dumps(subset, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        subset_bytes = len(raw.encode()) + payload_bytes
        if subset.get("sealed_bytes") == subset_bytes:
            break
        subset["sealed_bytes"] = subset_bytes
    subset_path.write_text(raw)
    if subset_bytes > 256 * 1024:
        raise RuntimeError("self-check subset exceeds 256 KiB")
    by_id = {c["id"]: c for c in cases}
    max_resample = 0.0
    for case in cases:
        if case["input"].get("recipe") != "resampled":
            continue
        base = by_id[case["input"]["base"]["id"]]
        actual = (root / case["log_mel"]["path"]).read_bytes()
        original = (root / base["log_mel"]["path"]).read_bytes()
        frames = min(case["valid_mel_frames"], base["valid_mel_frames"])
        values = frames * 128 * 4
        max_resample = max(max_resample, max(abs(a[0] - b[0]) for a, b in zip(
            struct.iter_unpack("<f", actual[:values]), struct.iter_unpack("<f", original[:values]))))
    contract = confirmations(root / "reference")
    # Keep exact measured values as decimal strings in descriptive metadata.
    # The manifest's legacy JSON parser does not promise long-float round trips.
    contract["measured_resample_max_abs_decimal"] = format(max_resample, ".17g")
    tau_resample = math.ceil((max_resample + 0.001) * 1_000_000) / 1_000_000
    contract["self_check_sealed_bytes"] = subset_bytes
    contract["maximum_emissions_per_frame"] = max(c["max_emissions_per_frame"] for c in cases)
    fixture_id = f"{SLUG}.ref-v1.transformers-5.16.1.seed-0"
    fixture_path = root / "fixtures" / SLUG / "reference.json"
    write(fixture_path, {"fixture_set_id": fixture_id, "model": SLUG, "cases": cases,
                         "hf_revision": ref.MODEL_REVISION, "checkpoint_digest": "sha256:" + ref.MODEL_SHA256,
                         "tokenizer_digest": "sha256:" + digest(root / "checkpoints" / SLUG / "tokenizer.json"),
                         "upstream_nan_padded_input_cases": [c["id"] for c in cases if c.get("upstream_exception")],
                         "maximum_emissions_per_frame": contract["maximum_emissions_per_frame"]}, compact=True)
    index_path = root / "fixtures/index.json"
    index = json.loads(index_path.read_text())
    index[fixture_id] = {**artifact(fixture_path, root), "model": SLUG, "transformers_version": "5.16.1", "seed": 0,
                         "artifacts": {c["id"]: {k: c[k] for k in ("log_mel", "encoder", "trace") if k in c} for c in cases}}
    write(index_path, index)
    manifest = json.loads((root / "models.json").read_text())
    checkpoint = root / "checkpoints" / SLUG
    config = json.loads((checkpoint / "config.json").read_text())
    encoder_keys = ("num_hidden_layers hidden_size num_attention_heads intermediate_size num_mel_bins "
                    "subsampling_conv_channels subsampling_conv_kernel_size subsampling_conv_stride "
                    "subsampling_factor conv_kernel_size scale_input attention_bias convolution_bias").split()
    top_keys = "vocab_size blank_token_id decoder_hidden_size num_decoder_layers max_symbols_per_step".split()
    files = {p.name: digest(p) for p in checkpoint.iterdir() if p.name != "tensor-index.json"}
    files["model.safetensors"] = ref.MODEL_SHA256
    manifest["models"][SLUG] = {
        "architecture": {"class": "ParakeetForTDT", "family": "parakeet_tdt",
                         "params": {**{k: config["encoder_config"][k] for k in encoder_keys}, **{k: config[k] for k in top_keys}}},
        "checkpoint_digest": "sha256:" + ref.MODEL_SHA256, "files": files,
        "hf_repo": ref.MODEL_REPO, "hf_revision": ref.MODEL_REVISION, "operation": "transcribe",
        "tensor_index_sha256": digest(checkpoint / "tensor-index.json"), "tensor_prefix": "",
        "tokenizer_digest": "sha256:" + files["tokenizer.json"],
    }
    tensor_names = json.loads((checkpoint / "tensor-index.json").read_text())
    package_seals = json.loads(args.package_seals.read_text()) if args.package_seals else None
    if package_seals is not None and (package_seals.get("checkpoint_sha256") != ref.MODEL_SHA256
                                     or package_seals.get("conversion_rule") != "v1"):
        raise RuntimeError("converted-package receipt has the wrong checkpoint or converter rule")
    for lane in ("owned-metal", "owned-cuda", "owned-vulkan", "ane-direct-worker"):
        profile_id = SLUG + "." + lane
        profile = {"model": SLUG, "lane": lane, "storage_dtype": "f16", "compute_dtype": "f16",
                   "conversion_rule": "none" if lane == "owned-metal" else "v1",
                   "rerank_tolerance_class": "fp16", "fp32_tensors": [],
                   "transcribe": {"max_duration_ms": 600000, "encoder_rungs": list(ref.RUNGS),
                                  "tau_feat": 0.001, "tau_resample": tau_resample,
                                  "tau_margin": subset["tau_margin"], "confirmations": contract,
                                  "self_check": artifact(subset_path, root)}}
        if lane == "owned-cuda":
            profile.update(cuda_min_compute_major=7, cuda_min_compute_minor=5, cuda_min_driver_api=13020)
        if lane == "ane-direct-worker":
            profile.update(cpu_stages=["tdt_predictor", "tdt_joint"], rotation="none", gelu_lowering="none")
            profile["fp32_tensors"] = sorted(k for k in tensor_names if k.startswith(("decoder.", "joint.")))
        if lane != "owned-metal":
            package = generated / (profile_id + ".safetensors")
            if package.exists():
                profile["converted_package_digest"] = "sha256:" + digest(package)
            elif package_seals is not None:
                receipt = package_seals["profiles"][profile_id]
                if not re.fullmatch("[0-9a-f]{64}", receipt["sha256"]) or receipt["bytes"] < 1_000_000_000:
                    raise RuntimeError(f"invalid converted-package receipt for {profile_id}")
                profile["converted_package_digest"] = "sha256:" + receipt["sha256"]
            elif not args.prepare:
                raise RuntimeError(f"missing real converted package {package}")
        manifest["profiles"][profile_id] = profile
    if package_seals is not None:
        write(root / "reference/parakeet-packages.json", package_seals)
    write(root / "models.json", manifest)
    fixture_bytes = sum(p.stat().st_size for p in (root / "fixtures").rglob("*") if p.is_file())
    if fixture_bytes > 64 * 1024 * 1024:
        raise RuntimeError(f"committed fixture bytes exceed 64 MiB: {fixture_bytes}")
    print(json.dumps({"fixtures": len(cases), "committed_fixture_bytes": fixture_bytes,
                      "self_check_bytes": subset_bytes, "max_resample": max_resample,
                      "maximum_emissions_per_frame": contract["maximum_emissions_per_frame"]}))


if __name__ == "__main__":
    main()
