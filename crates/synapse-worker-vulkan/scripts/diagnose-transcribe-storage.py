#!/usr/bin/env python3
"""Compare fp32 and fp16-storage/fp32-arithmetic encoders without changing seals.

Run in the pinned reference environment. Only the checkpoint's persistent
floating tensors are rounded: computed positional frequencies stay fp32, just
as in the Vulkan engine. Outputs are diagnostic scratch artifacts, not seals.
"""

import argparse
import copy
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

TRACE = struct.Struct("<iiiiiffffI")
FIELDS = ("frame_index", "token_top1", "token_top2", "duration_top1", "duration_top2",
          "token_logit1", "token_logit2", "duration_logit1", "duration_logit2")


def read_artifact(artifact, parity, mirror):
    if bool(artifact.get("path")) == bool(artifact.get("cache_path")):
        raise ValueError("artifact must name exactly one path")
    root, name = ((parity, artifact["path"]) if artifact.get("path")
                  else (mirror, artifact["cache_path"]))
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path escapes its root")
    raw = (root / relative).read_bytes()
    if len(raw) != artifact["bytes"] or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
        raise ValueError("artifact seal mismatch: " + name)
    return raw


def first_finding(reference, actual, tau):
    for index in range(max(len(reference), len(actual))):
        if index >= len(reference) or index >= len(actual):
            return {"step": index, "error": "trace length differs", "margin_exempt": False}
        r, a = reference[index], actual[index]
        if r["frame_index"] != a["frame_index"]:
            return {"step": index, "reference": r, "engine": a,
                    "error": "frame index differs", "margin_exempt": False}
        heads = []
        for head in ("token", "duration"):
            if r[head + "_top1"] != a[head + "_top1"]:
                heads.append({"head": head,
                              "reference_margin": r[head + "_logit1"] - r[head + "_logit2"],
                              "reference_pair": [r[head + "_top1"], r[head + "_top2"]],
                              "engine_pair": [a[head + "_top1"], a[head + "_top2"]]})
        if heads:
            return {"step": index, "reference": r, "engine": a, "heads": heads,
                    "margin_exempt": all(h["reference_margin"] < tau for h in heads)}
    return None


def save_tensor(array, directory):
    raw = array.astype("<f4").tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    path = directory / (digest + ".f32le")
    path.write_bytes(raw)
    return {"path": path.name, "sha256": digest, "bytes": len(raw), "frames": len(array)}


def encoder_metrics(reference, actual, per_frame=False):
    import numpy as np
    r, a = reference.astype("float64"), actual.astype("float64")
    rn, an = np.linalg.norm(r, axis=1), np.linalg.norm(a, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = (r * a).sum(axis=1) / (rn * an)
    worst = int(np.nanargmin(cosines)) if np.isfinite(cosines).any() else 0
    result = {"max_abs": float(np.max(np.abs(r - a))),
            "minimum_cosine": float(cosines[worst]) if np.isfinite(cosines[worst]) else None,
            "worst_frame": worst, "reference_norm_at_worst": float(rn[worst]),
            "actual_norm_at_worst": float(an[worst]),
            "reference_norm_min": float(rn.min()), "reference_norm_max": float(rn.max()),
            "actual_norm_min": float(an.min()), "actual_norm_max": float(an.max()),
            "bad_frames": [{"frame": int(i), "frames_from_end": len(r) - 1 - int(i),
                            "cosine": float(cosines[i]) if np.isfinite(cosines[i]) else None,
                            "reference_norm": float(rn[i]), "actual_norm": float(an[i])}
                           for i in np.flatnonzero(~np.isfinite(cosines) | (cosines < 0.999))]}
    if per_frame:
        result["reference_frame_norms"] = rn.tolist()
        result["actual_frame_norms"] = an.tolist()
    return result


def keep_fp32(name, mode):
    if mode == "f32-last-three":
        return any(name.startswith(f"encoder.layers.{i}.") for i in (21, 22, 23))
    if mode == "f32-attention":
        return name.startswith("encoder.layers.") and ".self_attn." in name and name.endswith(".weight")
    if mode == "f32-feed-forward":
        return name.startswith("encoder.layers.") and ".feed_forward" in name and name.endswith(".weight")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--vulkan-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", action="append", help="restrict diagnostics to these fixture ids")
    parser.add_argument("--reuse-captures", action="store_true", help="reuse SHA-checked encoder captures in the output directory")
    parser.add_argument("--mixed", action="store_true", help="also measure three selective-fp32 storage modes on native failures")
    args = parser.parse_args()
    parity = args.repo / "bench/parity"
    sys.path.insert(0, str(parity / "reference"))
    import numpy as np
    import parakeet_mask_fix
    import parakeet_reference as ref
    from transformers.models.parakeet import modeling_parakeet

    torch, model, processor, cache_class, checkpoint = ref.load_reference(str(args.hf_cache))
    modeling_parakeet.eager_attention_forward = parakeet_mask_fix.eager_attention_forward
    rows = json.loads((parity / "fixtures/parakeet-tdt-0.6b-v3/reference.json").read_text())["cases"]
    manifest = json.loads((parity / "models.json").read_text())
    profile = manifest["profiles"]["parakeet-tdt-0.6b-v3.owned-vulkan"]["transcribe"]
    maximum = profile["max_duration_ms"]
    tau = profile["tau_margin"]
    native = json.loads(args.vulkan_summary.read_text())
    failures = {r["fixture"] for r in native["trace_failures"]}
    tensor_rows = [r for r in rows if r.get("encoder") and r["duration_ms"] <= maximum]
    short = min((r for r in tensor_rows if r["id"].startswith("librispeech-")),
                key=lambda r: (r["sample_count"], r["id"]))["id"]
    layer_ids = {"ladder-1025", "ladder-4096", "silence-30s", short}
    selected = [r for r in rows if r["id"] in failures | layer_ids
                or (r.get("encoder") and r["duration_ms"] <= maximum)]
    if args.fixture:
        selected = [r for r in selected if r["id"] in args.fixture]
        if {r["id"] for r in selected} != set(args.fixture):
            raise ValueError("unknown or out-of-scope fixture")
    args.out.mkdir(parents=True, exist_ok=True)
    audio_index = json.loads((args.audio / "inputs.json").read_text())
    reference_heads = copy.deepcopy((model.encoder_projector, model.decoder, model.joint))
    report = {"diagnostic_only": True, "torch": torch.__version__,
              "checkpoint": str(checkpoint), "checkpoint_sha256": ref.MODEL_SHA256,
              "mask_fix": parakeet_mask_fix.REVISION, "tau_margin": tau,
              "layer_fixtures": sorted(layer_ids), "tensor_count": len(tensor_rows),
              "native_trace_failure_count": len(failures), "selected": len(selected),
              "rounding": "persistent floating checkpoint tensors only; computed buffers unchanged",
              "cases": {}}
    report_path = args.out / "storage-summary.json"

    def write_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def features(row):
        valid = row["valid_mel_frames"]
        if row.get("log_mel"):
            raw = read_artifact(row["log_mel"], parity, args.mirror)
            values = np.frombuffer(raw, dtype="<f4").copy().reshape(row["mel_frames"], 128)
        else:
            import soundfile as sf
            entry = audio_index[row["id"]]
            path = args.audio / entry["path"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("prepared WAV hash mismatch: " + row["id"])
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16000 or audio.ndim != 1 or len(audio) != row["sample_count"]:
                raise ValueError("malformed prepared WAV")
            return processor(audio, sampling_rate=16000, return_tensors="pt")
        mask = torch.arange(row["mel_frames"]).unsqueeze(0) < valid
        return {"input_features": torch.from_numpy(values).unsqueeze(0), "attention_mask": mask.long()}

    def encode(row, mode):
        directory = args.out / mode / row["id"]
        directory.mkdir(parents=True, exist_ok=True)
        if args.reuse_captures and (directory / "layers.json").exists():
            index = json.loads((directory / "layers.json").read_text())
            raw = read_artifact(index["layer-23"], directory, directory)
            values = np.frombuffer(raw, dtype="<f4").copy().reshape(1, row["valid_encoder_frames"], 1024)
            print(mode, row["id"], "reused SHA-checked encoder", flush=True)
            return torch.from_numpy(values), directory
        taps = {}
        handles = []
        valid = row["valid_encoder_frames"]

        def capture(name, tensor):
            array = tensor[0, :valid].detach().cpu().numpy()
            taps[name] = save_tensor(array, directory)
            print(mode, row["id"], name, "captured", flush=True)

        if row["id"] in layer_ids:
            handles.append(model.encoder.layers[0].register_forward_pre_hook(
                lambda module, inputs: capture("subsampling", inputs[0])))
            for index, layer in enumerate(model.encoder.layers):
                handles.append(layer.register_forward_hook(
                    lambda module, inputs, output, name=f"layer-{index:02}": capture(name, output)))
        try:
            with torch.inference_mode():
                encoded = model.get_audio_features(**features(row), output_attention_mask=True)
        finally:
            for handle in handles:
                handle.remove()
        encoder = encoded.last_hidden_state[:, :valid].contiguous()
        if int(encoded.attention_mask.sum()) != valid or not bool(torch.isfinite(encoder).all()):
            raise ValueError("nonfinite encoder or invalid frame count")
        if not taps:
            taps["layer-23"] = save_tensor(encoder[0].numpy(), directory)
        (directory / "layers.json").write_text(json.dumps(taps, indent=2) + "\n")
        return encoder, directory

    def decode(encoder, heads, row, directory, label):
        projector, decoder, joint_head = heads
        cache = cache_class(model.config)
        with torch.inference_mode():
            projected = projector(encoder)

            def joint(frame, previous):
                prediction = decoder(torch.tensor([[previous]]), cache=cache)
                logits = joint_head(prediction, projected[:, frame:frame + 1]).flatten()
                return logits[:model.config.vocab_size].tolist(), logits[model.config.vocab_size:].tolist()

            trace, triples = ref.greedy(joint, row["valid_encoder_frames"])
        raw = b"".join(TRACE.pack(*(s[k] for k in FIELDS), int(s["emitted"]) | (int(s["cap_hit"]) << 1)) for s in trace)
        name = "trace." + label + ".bin"
        (directory / name).write_bytes(raw)
        index_path = directory / "traces.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        index[label] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "triples": triples}
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        raw = read_artifact(row["trace"], parity, args.mirror)
        reference = [dict(zip(FIELDS, TRACE.unpack_from(raw, offset)[:9]))
                     for offset in range(0, len(raw), TRACE.size)]
        return first_finding(reference, trace, tau)

    started = time.monotonic()
    for row in selected:
        if row["id"] in layer_ids:
            encoder, directory = encode(row, "fp32")
            report["cases"][row["id"]] = {"fp32_layers": str(directory / "layers.json")}
            if row.get("encoder"):
                raw = read_artifact(row["encoder"], parity, args.mirror)
                reference = np.frombuffer(raw, dtype="<f4").reshape(-1, 1024)
                report["cases"][row["id"]]["fp32_vs_sealed_encoder"] = encoder_metrics(reference, encoder[0].numpy(), row["id"] == "silence-30s")
            write_report()
    # state_dict excludes nonpersistent, computed position-frequency buffers.
    # Round the exact tensors present in the checkpoint, not every model buffer.
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()} if args.mixed else None
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                rounded = tensor.to(torch.float16).to(torch.float32)
                if not bool(torch.isfinite(rounded).all()):
                    raise ValueError("fp16 storage overflow")
                tensor.copy_(rounded)
    for row in selected:
        encoder, directory = encode(row, "f16-storage")
        entry = report["cases"].setdefault(row["id"], {})
        entry["shadow_layers"] = str(directory / "layers.json")
        if row["id"] in layer_ids:
            fp32_directory = args.out / "fp32" / row["id"]
            fp32_layers = json.loads((fp32_directory / "layers.json").read_text())
            shadow_layers = json.loads((directory / "layers.json").read_text())
            entry["per_layer_storage_error"] = {
                name: encoder_metrics(
                    np.frombuffer(read_artifact(fp32_layers[name], fp32_directory, fp32_directory), dtype="<f4").reshape(-1, 1024),
                    np.frombuffer(read_artifact(artifact, directory, directory), dtype="<f4").reshape(-1, 1024))
                for name, artifact in shadow_layers.items()}
        if row.get("encoder"):
            raw = read_artifact(row["encoder"], parity, args.mirror)
            reference = np.frombuffer(raw, dtype="<f4").reshape(-1, 1024)
            entry["shadow_vs_sealed_encoder"] = encoder_metrics(reference, encoder[0].numpy(), row["id"] == "silence-30s")
        if row["id"] in failures:
            entry["shadow_encoder_fp32_joint_trace"] = decode(encoder, reference_heads, row, directory, "fp32-joint")
        entry["shadow_encoder_f16_storage_joint_trace"] = decode(
            encoder, (model.encoder_projector, model.decoder, model.joint), row, directory, "f16-storage-joint")
        report["elapsed_s"] = time.monotonic() - started
        write_report()
        print(json.dumps({"fixture": row["id"], "elapsed_s": report["elapsed_s"],
                          "shadow_min_cosine": entry.get("shadow_vs_sealed_encoder", {}).get("minimum_cosine"),
                          "shadow_bad_frames": len(entry.get("shadow_vs_sealed_encoder", {}).get("bad_frames", [])),
                          "fp32_joint_finding": entry.get("shadow_encoder_fp32_joint_trace"),
                          "f16_storage_joint_finding": entry.get("shadow_encoder_f16_storage_joint_trace")}), flush=True)
    if args.mixed:
        sensitivity_ids = failures | {r["fixture"] for r in native["encoder_failures"]}
        sensitivity_rows = [r for r in selected if r["id"] in sensitivity_ids]
        report["sensitivity_fixtures"] = [r["id"] for r in sensitivity_rows]
        report["sensitivity"] = {}
        for mode in ("f32-last-three", "f32-attention", "f32-feed-forward"):
            kept = []
            with torch.no_grad():
                for name, tensor in model.state_dict().items():
                    source = original[name]
                    if source.is_floating_point() and not keep_fp32(name, mode):
                        source = source.to(torch.float16).to(torch.float32)
                    elif source.is_floating_point():
                        kept.append(name)
                    tensor.copy_(source)
            mode_report = report["sensitivity"][mode] = {"fp32_tensors": kept, "cases": {}}
            for row in sensitivity_rows:
                encoder, directory = encode(row, mode)
                entry = mode_report["cases"][row["id"]] = {}
                if row.get("encoder"):
                    raw = read_artifact(row["encoder"], parity, args.mirror)
                    reference = np.frombuffer(raw, dtype="<f4").reshape(-1, 1024)
                    entry["encoder"] = encoder_metrics(reference, encoder[0].numpy(), row["id"] == "silence-30s")
                entry["trace"] = decode(encoder, (model.encoder_projector, model.decoder, model.joint), row, directory, mode)
                report["elapsed_s"] = time.monotonic() - started
                write_report()
                print(mode, row["id"], "minimum_cosine", entry.get("encoder", {}).get("minimum_cosine"),
                      "trace_finding", entry["trace"], flush=True)
    print("diagnostic summary:", report_path, flush=True)


if __name__ == "__main__":
    main()
