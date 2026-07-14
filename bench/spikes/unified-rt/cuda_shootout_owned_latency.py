#!/usr/bin/env python3
"""Measure warmed single-query latency through the owned stdio serving protocol."""

import argparse, json, statistics, struct, subprocess, time
from tokenizers import Tokenizer


def read_frame(stream):
    size = struct.unpack("<I", stream.read(4))[0]
    return json.loads(stream.read(size))


def write_frame(stream, value):
    payload = json.dumps(value, separators=(",", ":")).encode()
    stream.write(struct.pack("<I", len(payload)) + payload)
    stream.flush()


def token_count(tokenizer, text, family, eos):
    encoding = tokenizer.encode(text, add_special_tokens=True)
    ids = [token for token, mask in zip(encoding.ids, encoding.attention_mask) if mask][
        :512
    ]
    if family == "qwen3":
        if ids and ids[-1] == eos:
            ids.pop()
        ids = ids[:511] + [eos]
    return len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    shapes = [{"batch": 8, "seq": s} for s in [64, 96, 128, 160, 192, 256, 320]] + [
        {"batch": 6, "seq": 384},
        {"batch": 4, "seq": 448},
        {"batch": 3, "seq": 512},
    ]
    command = [
        args.binary,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--device",
        "cuda",
        "--dtype",
        "f16",
        "--cuda-graphs",
        "true",
        "--shapes",
        "bucketed",
        "--attention-units",
        "1000000",
        "--serve-stdio",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    ready = read_frame(process.stdout)
    ready_seconds = time.monotonic() - started
    write_frame(
        process.stdin,
        {
            "kind": "prepare_shapes",
            "workload": "embedding",
            "shapes": shapes,
            "max_length": 512,
            "force_shapes": True,
        },
    )
    prepared = read_frame(process.stdout)
    prepared_seconds = time.monotonic() - started
    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    rows = [
        json.loads(line) for line in open(args.corpus, encoding="utf-8") if line.strip()
    ]
    latencies = []
    for iteration in range(30):
        text = rows[(iteration * 37) % len(rows)]["text"] + f"\nquery nonce {iteration}"
        length = token_count(tokenizer, text, args.family, 151643)
        shape = next(shape for shape in shapes if shape["seq"] >= length)
        request = {
            "kind": "embed",
            "texts": [text],
            "max_length": 512,
            "shape_policy": "bucketed",
            "shape": shape,
        }
        request_started = time.monotonic()
        write_frame(process.stdin, request)
        response = read_frame(process.stdout)
        assert response["kind"] == "embedding", response
        latencies.append((time.monotonic() - request_started) * 1000)
    write_frame(process.stdin, {"kind": "shutdown"})
    read_frame(process.stdout)
    process.wait()
    result = {
        "family": args.family,
        "ready_s": ready_seconds,
        "prepared_s": prepared_seconds,
        "single_query_p50_ms": statistics.median(latencies),
        "single_query_samples_ms": latencies,
        "ready": ready,
        "prepared": prepared,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
