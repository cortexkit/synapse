#!/usr/bin/env python3
"""Generate sealed speech fixtures, with disjoint deterministic work partitions.

Install requirements-parakeet.txt (use the PyTorch CPU index for torch), then run
from the repository root with --cache, --output and --group corpus or ladder.
Each worker is single-threaded; scheduling never enters a fixture's identity.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import struct
import subprocess
import time

import parakeet_reference as ref
from prepare_parakeet import prepare

SLUG = "parakeet-tdt-0.6b-v3"
TAU_MARGIN = 0.001
TRACE_FIELDS = ("frame_index", "token_top1", "token_top2", "duration_top1", "duration_top2",
                "token_logit1", "token_logit2", "duration_logit1", "duration_logit2")
RUNTIME = None
CACHE = None
OUTPUT = None


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n")


def seal(raw, relative, external=False):
    digest = hashlib.sha256(raw).hexdigest()
    if external:
        relative = Path(SLUG) / (digest + ".f32le")
        destination = OUTPUT / "external" / relative
    else:
        destination = OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return {"cache_path" if external else "path": str(relative), "sha256": digest, "bytes": len(raw)}


def resample(audio, source_rate, target_rate):
    """fp64 Hann-windowed sinc, six crossings, 0.99 lower-Nyquist cutoff."""
    import torch
    if source_rate == target_rate:
        return audio.copy()
    signal = torch.as_tensor(audio, dtype=torch.float64)
    count = (len(audio) * target_rate + source_rate - 1) // source_rate
    cutoff = 0.99 * min(source_rate, target_rate) / source_rate
    radius = int(6 / cutoff) + 1
    offsets = torch.arange(-radius, radius + 1)
    chunks = []
    for start in range(0, count, 2048):
        positions = torch.arange(start, min(start + 2048, count), dtype=torch.float64) * source_rate / target_rate
        indices = positions.floor().long()[:, None] + offsets
        distance = (indices - positions[:, None]) * cutoff
        inside = distance.abs() < 6
        window = torch.cos(distance.clamp(-6, 6) * (torch.pi / 12)).square()
        kernel = cutoff * torch.sinc(distance) * window * inside
        valid = (indices >= 0) & (indices < len(signal))
        chunks.append((signal[indices.clamp(0, len(signal) - 1)] * kernel * valid).sum(dim=1))
    return torch.cat(chunks).float().numpy()


def wav_bytes(audio, sample_rate, subtype):
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype=subtype)
    raw = bytearray(buffer.getvalue())
    position = 12
    while position + 8 <= len(raw):
        chunk, size = struct.unpack_from("<4sI", raw, position)
        if position + 8 + size > len(raw):
            raise RuntimeError("audio writer returned a truncated RIFF chunk")
        if chunk == b"PEAK" and size >= 8:
            # libsndfile writes wall time in FLOAT WAV's PEAK metadata. Zero
            # only that timestamp; the encoded samples and peak values stay put.
            struct.pack_into("<I", raw, position + 12, 0)
        position += 8 + size + (size & 1)
    return bytes(raw)


def pcm(case):
    import numpy as np
    import soundfile as sf
    if case.get("recipe") == "silence":
        return np.zeros(case["sample_count"], dtype=np.float32)
    if "audio_sha256" in case:
        source = CACHE / "audio" / case["audio_sha256"]
        if ref.sha256_file(source) != case["audio_sha256"]:
            raise RuntimeError("audio member digest mismatch")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        return resample(audio, rate, 16000)
    if case["recipe"] == "ladder":
        parts = [pcm(row) for row in case["parts"]]
        return np.resize(np.concatenate(parts), case["sample_count"])
    base = pcm(case["base"])
    if case["recipe"] == "leading-silence":
        return np.concatenate([np.zeros(16000, dtype=np.float32), base])
    if case["recipe"] == "hop-boundary":
        return np.resize(base, case["sample_count"])
    if case["recipe"] == "excerpt":
        return base[:case["sample_count"]]
    if case["recipe"] == "resampled":
        high = resample(base, 16000, case["sample_rate"])
        stereo = np.stack([high, high], axis=1)
        raw = wav_bytes(stereo, case["sample_rate"], case["subtype"])
        digest = hashlib.sha256(raw).hexdigest()
        target = CACHE / "audio" / digest
        target.write_bytes(raw)
        audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        case["generated_audio_sha256"] = digest
        case["input_sample_count"] = len(audio)
        return resample(audio.mean(axis=1), rate, 16000)
    raise RuntimeError("unknown fixture recipe")


def initialize(cache, output):
    global RUNTIME, CACHE, OUTPUT
    CACHE, OUTPUT = Path(cache), Path(output)
    RUNTIME = ref.load_reference(CACHE / "hf")


def generate_case(case):
    audio = pcm(case)
    result, mel, encoder = ref.run_fixture(RUNTIME, audio, case["id"])
    result["input"] = case
    result["duration_ms"] = (len(audio) + 15) // 16
    if "split" in case:
        result["split"] = case["split"]
        result["ground_truth"] = case["ground_truth"]
    trace = b"".join(struct.pack("<iiiiiffffI", *(step[key] for key in TRACE_FIELDS),
                                int(step["emitted"]) | (int(step["cap_hit"]) << 1))
                     for step in result.pop("trace"))
    prefix = Path("fixtures") / SLUG / case["id"]
    result["trace"] = seal(trace, prefix.with_suffix(".trace"))
    if case.get("tensors") or (case.get("recipe") == "ladder"):
        external = case.get("recipe") == "ladder" and result["valid_encoder_frames"] > 513
        result["log_mel"] = seal(mel, prefix.with_suffix(".mel.f32le"), external)
        result["encoder"] = seal(encoder, prefix.with_suffix(".encoder.f32le"), external)
    dump(OUTPUT / prefix.with_suffix(".json"), result)
    print(case["id"], result["valid_encoder_frames"], result["max_emissions_per_frame"], flush=True)
    return result


def plans(corpus):
    corpus_cases = corpus["cases"]
    # The tensor budget is deterministic and independent of model predictions.
    shortest = sorted(corpus_cases, key=lambda c: (c["sample_count"] / c["sample_rate"], c["id"]))
    tensor_ids = {case["id"] for case in shortest[:32]}
    cases = [dict(c, tensors=c["id"] in tensor_ids) for c in corpus_cases]
    synthetic = [{"id": f"silence-{seconds}s", "recipe": "silence", "sample_count": seconds * 16000,
                  "tensors": True} for seconds in (1, 30)]
    base = shortest[0]
    synthetic.append({"id": "leading-silence", "recipe": "leading-silence", "base": base, "tensors": True})
    for hops in (8, 100):
        for delta in (-1, 0, 1):
            synthetic.append({"id": f"hop-{hops}-{delta + 1}", "recipe": "hop-boundary", "base": base,
                              "sample_count": hops * 160 + delta, "tensors": True})
    for index, base in enumerate(shortest[:4]):
        for rate in (44100, 48000):
            for subtype in ("PCM_16", "FLOAT"):
                synthetic.append({"id": f"resample-{index}-{rate}-{subtype}", "recipe": "resampled",
                                  "sample_rate": rate, "subtype": subtype, "channels": 2,
                                  "base": base, "tensors": True})
    parts = [case for case in corpus_cases if case["split"] == "librispeech-test-clean"][:4]
    ladders = [dict(row, recipe="ladder", parts=parts) for row in ref.ladder()]
    return cases + synthetic, ladders


def self_check(corpus, results):
    selected = []
    by_id = {result["id"]: result for result in results}
    for split, limit, check_id in (("librispeech-test-clean", 3000, "speech-clean"),
                                   ("fleurs-de-test", 2000, "speech-de")):
        candidates = sorted((c for c in corpus["cases"] if c["split"] == split),
                            key=lambda c: (c["sample_count"] / c["sample_rate"], c["id"]))
        found = None
        for source in candidates:
            original = by_id[source["id"]]
            samples = len(pcm(source))
            if samples > limit * 16:
                if split != "fleurs-de-test":
                    continue
                boundaries = [w["end_ms"] for w in original["words"] if 0 < w["end_ms"] <= limit]
                if not boundaries:
                    continue
                samples = int(max(boundaries) * 16)
            case = {"id": "self-check-" + check_id, "recipe": "excerpt", "base": source,
                    "sample_count": samples}
            candidate = generate_case(case)
            if min(candidate["minimum_token_margin"], candidate["minimum_duration_margin"]) < 2 * TAU_MARGIN:
                continue
            # The compiled WAV, not a pre-quantized array, is the self-check input.
            raw = wav_bytes(pcm(case), 16000, "PCM_16")
            digest = hashlib.sha256(raw).hexdigest()
            (CACHE / "audio" / digest).write_bytes(raw)
            candidate = generate_case({"id": case["id"], "audio_sha256": digest})
            if min(candidate["minimum_token_margin"], candidate["minimum_duration_margin"]) < 2 * TAU_MARGIN:
                continue
            candidate["wav"] = seal(raw, Path("fixtures") / SLUG / "self-check" / (check_id + ".wav"))
            candidate["source"] = case
            candidate["minimum_margin"] = min(candidate["minimum_token_margin"], candidate["minimum_duration_margin"])
            found = candidate
            break
        if found is None:
            raise RuntimeError("STOP: no self-check clip qualifies for " + split)
        selected.append(found)
    silence = by_id["silence-1s"]
    silence["minimum_margin"] = min(silence["minimum_token_margin"], silence["minimum_duration_margin"])
    if silence["minimum_margin"] < 2 * TAU_MARGIN:
        raise RuntimeError("STOP: silence self-check margin is below 2*tau_margin")
    subset = {"schema": "synapse-parakeet-self-check-v1", "tau_margin": TAU_MARGIN,
              "cases": [silence] + selected}
    raw = json.dumps(subset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    total = len(raw) + sum(case["wav"]["bytes"] for case in selected)
    if total > 256 * 1024:
        raise RuntimeError(f"STOP: self-check subset exceeds 256 KiB: {total}")
    subset["sealed_bytes"] = total
    dump(OUTPUT / "fixtures" / SLUG / "self-check.json", subset)
    return subset


def pack_output(output, bundle):
    """Seal files and normalize transport metadata, independently of wall time."""
    contents = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            entry = {"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                     "sha256": ref.sha256_file(path)}
            contents.append(entry)
            print("ARTIFACT " + json.dumps(entry, sort_keys=True), flush=True)
    dump(bundle.with_suffix(".seals.json"), contents)
    subprocess.run(["tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner",
                    "-cf", str(bundle), "-C", str(output), "."], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("corpus", "ladder"), required=True)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    start = time.monotonic()
    if not 0 <= args.partition < args.partitions or not 1 <= args.workers <= 4:
        parser.error("invalid partition or worker count")
    corpus_path = args.output / "reference" / "parakeet-corpus.json"
    corpus = prepare(args.cache, corpus_path)
    corpus_plan, ladder_plan = plans(corpus)
    selected = (corpus_plan if args.group == "corpus" else ladder_plan)[args.partition::args.partitions]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(args.cache), str(args.output))) as pool:
        results = list(pool.map(generate_case, selected))
    if args.group == "corpus":
        initialize(str(args.cache), str(args.output))
        subset = self_check(corpus, results)
    else:
        subset = None
    seconds = time.monotonic() - start
    summary = {"group": args.group, "partition": args.partition, "partitions": args.partitions,
               "fixtures": len(results),
               "maximum_emissions_per_frame": max(r["max_emissions_per_frame"] for r in results),
               "cases": results, "self_check": subset}
    dump(args.output / f"summary-{args.group}-{args.partition}.json", summary)
    bundle = args.output.parent / f"parakeet-{args.group}-{args.partition}.tar"
    timing = {"group": args.group, "partition": args.partition, "seconds": seconds}
    dump(bundle.with_suffix(".timing.json"), timing)
    pack_output(args.output, bundle)
    print(json.dumps({"bundle": str(bundle), "bytes": bundle.stat().st_size,
                       "fixtures": len(results), "seconds": seconds}), flush=True)


if __name__ == "__main__":
    main()
