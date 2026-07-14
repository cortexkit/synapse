#!/usr/bin/env python3
"""Benchmark HTTP embedding servers with canonical token counts and GPU telemetry."""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer


def rows(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def token_count(tok, text, family, eos):
    ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=512)
    if family == "qwen3":
        if ids and ids[-1] == eos:
            ids.pop()
        ids = ids[:511] + [eos]
    return len(ids)


def post(session, flavor, port, texts):
    if flavor == "tei":
        r = session.post(
            f"http://127.0.0.1:{port}/embed", json={"inputs": texts}, timeout=600
        )
        r.raise_for_status()
        return r.json()
    r = session.post(
        f"http://127.0.0.1:{port}/v1/embeddings",
        json={"input": texts, "model": "benchmark"},
        timeout=600,
    )
    r.raise_for_status()
    return [x["embedding"] for x in r.json()["data"]]


def sample_gpu(stop, samples):
    while not stop.is_set():
        try:
            values = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                .strip()
                .split(",")
            )
            samples.append(tuple(float(x) for x in values))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(0.5)


def compute_pids():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        return sorted(
            {int(x.strip()) for x in output.splitlines() if x.strip().isdigit()}
        )
    except (OSError, subprocess.SubprocessError):
        return []


def parity(vectors, data, ref_path):
    refs = {r["id"]: np.asarray(r["vec"], dtype=np.float64) for r in rows(ref_path)}
    pairs = [
        (np.asarray(v, dtype=np.float64), refs[r["id"]])
        for r, v in zip(data, vectors)
        if r["id"] in refs
    ]
    a = np.stack([x for x, _ in pairs])
    b = np.stack([x for _, x in pairs])
    cosine = np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    sa = a @ a.T
    sb = b @ b.T
    np.fill_diagonal(sa, -np.inf)
    np.fill_diagonal(sb, -np.inf)
    overlap = []
    for i in range(len(pairs)):
        ta = set(np.argpartition(sa[i], -10)[-10:])
        tb = set(np.argpartition(sb[i], -10)[-10:])
        overlap.append(len(ta & tb) / 10)
    return {
        "matched": len(pairs),
        "mean_cosine": float(cosine.mean()),
        "top10_rank_overlap": float(np.mean(overlap)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flavor", choices=["tei", "llama", "vllm"], required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--reference")
    p.add_argument("--vectors-out")
    p.add_argument("--server-pid", type=int, required=True)
    p.add_argument("--cold-load-s", type=float, required=True)
    p.add_argument("--eos-token-id", type=int, default=151643)
    p.add_argument("--latency-iters", type=int, default=30)
    a = p.parse_args()
    data = rows(a.corpus, a.limit)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    real_tokens = sum(
        token_count(tok, r["text"], a.family, a.eos_token_id) for r in data
    )
    session = requests.Session()
    post(session, a.flavor, a.port, ["warmup varying query"])
    compute_pids_before = compute_pids()
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    sampler.start()
    load_start = os.getloadavg()
    vectors = []
    started = time.monotonic()
    for i in range(0, len(data), a.batch_size):
        vectors.extend(
            post(
                session,
                a.flavor,
                a.port,
                [r["text"] for r in data[i : i + a.batch_size]],
            )
        )
    wall = time.monotonic() - started
    load_end = os.getloadavg()
    stop.set()
    sampler.join()
    compute_pids_after = compute_pids()
    active_compute_pids = sorted(set(compute_pids_before + compute_pids_after))
    latencies = []
    for i in range(a.latency_iters):
        text = data[(i * 37) % len(data)]["text"] + f"\nquery nonce {i}"
        started = time.monotonic()
        post(session, a.flavor, a.port, [text])
        latencies.append((time.monotonic() - started) * 1000)
    result = {
        "flavor": a.flavor,
        "family": a.family,
        "items": len(data),
        "canonical_real_tokens": real_tokens,
        "batch_size": a.batch_size,
        "cold_load_s": a.cold_load_s,
        "wall_s": wall,
        "tok_per_s": real_tokens / wall,
        "single_query_p50_ms": statistics.median(latencies),
        "single_query_samples_ms": latencies,
        "host_load_start": load_start,
        "host_load_end": load_end,
        "gpu_samples": len(samples),
        "avg_gpu_watts": statistics.mean(x[0] for x in samples) if samples else None,
        "peak_gpu_watts": max((x[0] for x in samples), default=None),
        "avg_gpu_util_pct": statistics.mean(x[1] for x in samples) if samples else None,
        "peak_gpu_util_pct": max((x[1] for x in samples), default=None),
        "peak_vram_mib": max((x[2] for x in samples), default=None),
        "server_pid_namespace": a.server_pid,
        "active_compute_pids": active_compute_pids,
        "foreign_gpu_pids": active_compute_pids if len(active_compute_pids) > 1 else [],
        "contaminated": len(active_compute_pids) > 1,
    }
    if a.reference:
        result["parity"] = parity(vectors, data, a.reference)
    if a.vectors_out:
        with open(a.vectors_out, "w", encoding="utf-8") as f:
            for r, v in zip(data, vectors):
                f.write(
                    json.dumps({"id": r["id"], "vec": v}, separators=(",", ":")) + "\n"
                )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
