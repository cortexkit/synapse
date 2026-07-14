#!/usr/bin/env python3
"""Measure warm ORT-CUDA single-query latency with the lane's preprocessing policy."""

import argparse, json, statistics, time
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--pooling", choices=["mean", "cls"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4
    started = time.monotonic()
    session = ort.InferenceSession(
        a.model, sess_options=options, providers=["CUDAExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(a.tokenizer)
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    load_seconds = time.monotonic() - started
    names = {value.name for value in session.get_inputs()}
    rows = [
        json.loads(line) for line in open(a.corpus, encoding="utf-8") if line.strip()
    ]

    def run(text):
        encoding = tokenizer.encode(text, add_special_tokens=True)
        ids = np.asarray([encoding.ids], dtype=np.int64)
        mask = np.asarray([encoding.attention_mask], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(None, feeds)[0]
        vector = (
            hidden[0, 0]
            if a.pooling == "cls"
            else (hidden[0] * mask[0, :, None]).sum(axis=0) / max(mask.sum(), 1)
        )
        vector /= np.linalg.norm(vector) + 1e-12

    run("warmup varying query")
    latencies = []
    for iteration in range(30):
        text = rows[(iteration * 37) % len(rows)]["text"] + f"\nquery nonce {iteration}"
        started = time.monotonic()
        run(text)
        latencies.append((time.monotonic() - started) * 1000)
    result = {
        "load_s": load_seconds,
        "single_query_p50_ms": statistics.median(latencies),
        "single_query_samples_ms": latencies,
        "providers": session.get_providers(),
    }
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
