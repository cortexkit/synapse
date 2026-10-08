#!/usr/bin/env python3
"""Quiet-window, interleaved Metal serving measurements on synthetic code.

A fresh scratch Synapse module is registered under a unique module ID; the live
module is never called. Every arm owns and terminates only its own child process.
The public output contains timings, identities and high-water memory, not vectors.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

LLAMA_COMMIT = "680a036285273a3ff56032ec5d7f3352609eba4f"
GGUF_REVISION = "370f27d7550e0def9b39c1f16d3fbaa13aa67728"
HF_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


class NotQuiet(RuntimeError):
    pass


def quiet_load():
    value = os.getloadavg()[0]
    if value >= 16:
        raise NotQuiet(f"1-minute load {value:.4f} is not below 16")
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def http(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def synthetic(tokenizer, tokens, index):
    # Numeric punctuation gives code-like token density, not repeated prose.
    header = f"src/cache.rs:10-24 fn update_{index}(key: u32)\n"
    body = "let old = cache.get(&key); if old.is_none() { cache.insert(key, value); }\n"
    raw = tokenizer.encode(header + body * 100, add_special_tokens=False)[:tokens - 1]
    text = tokenizer.decode(raw)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids != raw or len(ids) + 1 != tokens:
        raise ValueError("Synthetic fixture did not round-trip at the requested length")
    return text, ids + [tokenizer.eos_token_id]


def percentile(values, p):
    values = sorted(values)
    rank = (len(values) - 1) * p
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (rank - lo) * (values[hi] - values[lo])


def memory_from_time(text):
    # Darwin time -l reports bytes for these lifetime process high-water marks.
    result = {}
    for key, label in (("max_rss_bytes", "maximum resident set size"),
                       ("peak_footprint_bytes", "peak memory footprint")):
        match = re.search(r"(\d+)\s+" + label, text)
        if match:
            result[key] = int(match.group(1))
    if "max_rss_bytes" not in result:
        raise ValueError("Darwin time -l omitted the process high-water RSS")
    return result


class Process:
    """time owns a shell that execs the model; only that known child is signaled."""
    def __init__(self, directory, command, env=None):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.pidfile = directory / "pid"
        self.log = directory / "process.log"
        self.handle = self.log.open("w")
        self.child = subprocess.Popen(
            ["/usr/bin/time", "-l", "/bin/sh", "-c", 'echo $$ > "$1"; shift; exec "$@"',
             "q8-speed", str(self.pidfile), *map(str, command)],
            env=env, stdout=self.handle, stderr=subprocess.STDOUT)
        self.pid = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.pidfile.exists() and self.pidfile.read_text().strip():
                self.pid = int(self.pidfile.read_text())
                break
            if self.child.poll() is not None:
                break
            threading.Event().wait(0.05)
        if self.pid is None:
            self.close()
            raise RuntimeError("Model process did not start; inspect the private log")

    def close(self):
        if self.child.poll() is None:
            if self.pid is not None:
                try:
                    os.kill(self.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                self.child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if self.pid is not None:
                    try:
                        os.kill(self.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.child.wait(timeout=10)
        self.handle.close()


def valid_vectors(vectors, expected):
    if len(vectors) != expected:
        raise ValueError("Serving response did not complete every row")
    for vector in vectors:
        if len(vector) != 1024 or not all(math.isfinite(v) for v in vector):
            raise ValueError("Invalid Qwen embedding vector")
        if abs(sum(v * v for v in vector) - 1) > 0.01:
            raise ValueError("Embedding output was not L2-normalized")


class Llama:
    def __init__(self, args, arm, directory, slots=64):
        self.url = f"http://127.0.0.1:{args.port}"
        path = args.f16_gguf if arm == "llama-f16" else args.q8_gguf
        self.process = Process(directory, [args.llama_server, "-m", path,
            "--host", "127.0.0.1", "--port", args.port, "--embedding", "--pooling", "last",
            "--embd-normalize", "2", "-ngl", "99", "-t", "4", "-tb", "4",
            "-c", str(slots * 512), "-b", "32768", "-ub", "32768", "-np", str(slots), "--flash-attn", "on"])
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    http(self.url + "/health")
                    text = self.process.log.read_text()
                    if not re.search(r"offloaded \d+/\d+ layers to GPU", text):
                        raise RuntimeError("llama-server did not confirm GPU layer offload")
                    return
                except urllib.error.URLError:
                    if self.process.child.poll() is not None:
                        raise RuntimeError("llama-server exited; inspect the private log")
                    threading.Event().wait(0.1)
            raise RuntimeError("llama-server startup timed out")
        except BaseException:
            self.process.close()
            raise

    def validate_fixture(self, fixtures):
        for text, ids in fixtures:
            observed = http(self.url + "/tokenize", {"content": text, "add_special": True})
            if observed["tokens"] != ids:
                raise ValueError("llama.cpp tokenizer/EOS policy differs from the common fixture")

    def request(self, texts, ids, key):
        result = http(self.url + "/v1/embeddings", {"input": texts, "model": "qwen", "encoding_format": "float"})
        data = sorted(result["data"], key=lambda x: x["index"])
        if [x["index"] for x in data] != list(range(len(texts))):
            raise ValueError("llama-server returned incomplete row indices")
        valid_vectors([x["embedding"] for x in data], len(texts))
        if result["usage"]["prompt_tokens"] != sum(map(len, ids)):
            raise ValueError("llama-server token count differs from the common fixture")


class Synapse:
    def __init__(self, args, arm, directory, slots=64):
        self.args = args
        self.module = "q8-speed-" + uuid.uuid4().hex
        self.fingerprint = None
        directory.mkdir(parents=True, exist_ok=False)
        config = {
            "preload_models": [{"model_id": "q8-speed-f16", "engine": "owned-metal",
                "profile": "qwen3-embedding-0.6b.owned-metal", "task": "embed",
                "model_path": str(args.snapshot / "model.safetensors"),
                "tokenizer_path": str(args.snapshot / "tokenizer.json"),
                "pooling": "last", "normalize": True, "dtype": "f16", "quant": "f16",
                "execution": "explicit", "attention_units": 8192 * 8192}],
            # Match production admission and bulk quanta, including job responses
            # for a 64x512-token call rather than enlarging inline limits to hide them.
            "inline": {"max_items": 64, "max_tokens": 8192, "max_concurrent_workers": 2,
                       "deadline_ms": 30000, "max_queue_ms": 5000},
            "jobs": {"bulk_quantum_tokens": 3072}}
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(config))
        env = os.environ.copy()
        env.update({"SUBC_MODULE_ID": self.module, "SYNAPSE_CONFIG_PATH": str(config_path),
                    "XDG_DATA_HOME": str(directory / "data"),
                    "CORTEXKIT_LEASE_ROOT": str(directory / "leases"),
                    "CORTEXKIT_STORE_ROOT": str(directory / "store")})
        # A supervisor's nonce descriptor is not inherited by this direct scratch launch.
        for key in list(env):
            if "NONCE" in key and key.startswith("SUBC_"):
                del env[key]
        self.process = Process(directory / "child", [args.synapse_binary, "--subc", args.subc], env)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    self.call("models.list", {})
                    return
                except (subprocess.CalledProcessError, RuntimeError):
                    if self.process.child.poll() is not None:
                        raise RuntimeError("Scratch module exited; inspect the private log")
                    threading.Event().wait(0.2)
            raise RuntimeError("Scratch module registration timed out")
        except BaseException:
            self.process.close()
            raise

    def call(self, method, params):
        # subc_call is the repository's existing management-surface client, not a
        # reimplementation of its authenticated transport. CLI startup is timed.
        reply = subprocess.run([str(self.args.subc_call), "--subc", str(self.args.subc),
            "--module", self.module, "--method", method, "--params", json.dumps(params)],
            check=True, capture_output=True, text=True, timeout=130)
        value = json.loads(reply.stdout)
        result = value.get("result", value)
        if "error" in result or "error" in value:
            raise RuntimeError("Scratch module refused the request; no latency sample accepted")
        return result

    def request(self, texts, ids, key):
        result = self.call("embed.batch", {"model": "q8-speed-f16", "input_type": "document",
            "items": [{"id": f"item-{i}", "text": text} for i, text in enumerate(texts)],
            "request_key": key, "accept_declared": True})
        if "job_id" in result:
            job = result["job_id"]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                result = self.call("embed.result", {"job_id": job})
                if result["state"] == "done":
                    break
                if result["state"] not in ("queued", "running"):
                    raise RuntimeError("Embedding job failed")
                threading.Event().wait(0.025)
            else:
                raise RuntimeError("Embedding job did not finish")
            for page in range(1, result["page_count"]):
                extra = self.call("embed.result", {"job_id": job, "page": page})
                for field in ("vectors", "real_token_counts"):
                    result[field].extend(extra[field])
        vectors = sorted(result["vectors"], key=lambda v: int(v["id"].split("-")[1]))
        if [v["id"] for v in vectors] != [f"item-{i}" for i in range(len(texts))]:
            raise ValueError("Scratch module did not complete every item identity")
        valid_vectors([v["vector"] for v in vectors], len(texts))
        if sorted(result["real_token_counts"]) != sorted(map(len, ids)):
            raise ValueError("Synapse token count differs from the common fixture")
        if self.fingerprint is not None and result["fingerprint"] != self.fingerprint:
            raise ValueError("Serving identity changed during the arm")
        self.fingerprint = result["fingerprint"]


def measure(client, fixtures, key):
    load = quiet_load()
    started = time.monotonic()
    texts, ids = map(list, zip(*fixtures))
    client.request(texts, ids, key)
    elapsed = time.monotonic() - started
    return {"elapsed_s": elapsed, "rows": len(texts), "tokens": sum(map(len, ids)),
            "load_1m": load}


def sweep(client, tokenizer, length, batch, repeats, nonce):
    fixture = [synthetic(tokenizer, length, i) for i in range(batch)]
    if isinstance(client, Llama):
        client.validate_fixture(fixture)
    measure(client, fixture, nonce + "-warmup")
    samples = [measure(client, fixture, nonce + f"-{i}") for i in range(repeats)]
    return {"samples": samples, "median_rows_per_min": statistics.median(60 * x["rows"] / x["elapsed_s"] for x in samples),
            "median_tokens_per_s": statistics.median(x["tokens"] / x["elapsed_s"] for x in samples),
            "latency_p50_s": percentile([x["elapsed_s"] for x in samples], .5),
            "latency_p90_s": percentile([x["elapsed_s"] for x in samples], .9)}


def fill(client, tokenizer, seconds, nonce):
    # Two sustained callers, 64 rows/call. Build fixtures before starting timers.
    fixture = [synthetic(tokenizer, 100 + i % 51, i) for i in range(64)]
    if isinstance(client, Llama):
        client.validate_fixture(fixture)
    measure(client, fixture, nonce + "-warmup")
    stop = threading.Event()
    started = time.monotonic()
    deadline = started + seconds

    def worker(number):
        results, seq = [], 0
        while time.monotonic() < deadline and not stop.is_set():
            try:
                results.append(measure(client, fixture, f"{nonce}-{number}-{seq}"))
                seq += 1
            except BaseException:
                stop.set()
                raise
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, i) for i in range(2)]
        samples = [sample for f in futures for sample in f.result()]
    elapsed = time.monotonic() - started
    if not samples:
        raise ValueError("No completed fill calls")
    latencies = [x["elapsed_s"] for x in samples]
    return {"seconds": elapsed, "calls": len(samples), "rows": sum(x["rows"] for x in samples),
            "rows_per_min": 60 * sum(x["rows"] for x in samples) / elapsed,
            "latency_p50_s": percentile(latencies, .5), "latency_p90_s": percentile(latencies, .9),
            "in_flight_limit": 2, "rows_per_call": 64, "samples": samples,
            "fixture_tokens_min_max": [min(len(x[1]) for x in fixture), max(len(x[1]) for x in fixture)],
            "fixture_chars_mean": statistics.mean(len(x[0]) for x in fixture)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--private-dir", type=Path, required=True)
    p.add_argument("--probe", action="store_true", help="record load without launching anything")
    p.add_argument("--llama-source", type=Path)
    p.add_argument("--llama-server", type=Path)
    p.add_argument("--f16-gguf", type=Path)
    p.add_argument("--q8-gguf", type=Path)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--synapse-binary", type=Path)
    p.add_argument("--subc-call", type=Path)
    p.add_argument("--subc", type=Path, help="explicit daemon connection; only a unique scratch module is called")
    p.add_argument("--port", type=int, default=18943)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--fill-seconds", type=int, default=180)
    args = p.parse_args()
    report = {"python": platform.python_version(), "os": platform.system(), "machine": platform.machine(),
              "load_gate": "1-minute load < 16 before every arm and request", "observed_load_1m": os.getloadavg()[0],
              "llama_commit": LLAMA_COMMIT, "arms": []}
    try:
        quiet_load()
        if args.probe:
            report["status"] = "quiet-window-available"
            return
        required = ("llama_source", "llama_server", "f16_gguf", "q8_gguf", "snapshot", "synapse_binary", "subc_call", "subc")
        if any(getattr(args, x) is None for x in required):
            p.error("all explicit artifact, binary, and connection paths are required")
        if args.repeats < 5 or args.fill_seconds < 180:
            p.error("at least five sweep rounds and three minutes per fill arm are required")
        for x in required:
            setattr(args, x, getattr(args, x).resolve())
        sha = subprocess.check_output(["git", "-C", str(args.llama_source), "rev-parse", "HEAD"], text=True).strip()
        if sha != LLAMA_COMMIT or subprocess.check_output(["git", "-C", str(args.llama_source), "status", "--porcelain"]):
            raise ValueError("llama.cpp must be clean at the declared pinned commit")
        if args.llama_server != args.llama_source / "build/bin/llama-server":
            raise ValueError("Use the binary built inside the pinned llama.cpp source")
        cache = (args.llama_source / "build/CMakeCache.txt").read_text()
        if "GGML_METAL:BOOL=ON" not in cache:
            raise ValueError("llama.cpp was not configured with Metal")
        report["llama_version"] = subprocess.check_output([str(args.llama_server), "--version"], stderr=subprocess.STDOUT, text=True).strip()
        report["artifact_sha256"] = {name: digest(getattr(args, name)) for name in
                                     ("llama_server", "synapse_binary", "subc_call", "f16_gguf", "q8_gguf")}
        if report["artifact_sha256"]["q8_gguf"] != "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439":
            raise ValueError("Not the consumer's exact Qwen Q8_0 GGUF")
        if report["artifact_sha256"]["f16_gguf"] != "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340":
            raise ValueError("Not the pinned consumer-repository f16 GGUF")
        root = Path(__file__).resolve().parents[3]
        entry = json.loads((root / "bench/parity/models.json").read_text())["models"]["qwen3-embedding-0.6b"]
        for filename, expected in entry["files"].items():
            if digest(args.snapshot / filename) != expected:
                raise ValueError("Synapse snapshot does not match the pinned model files")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
        report["hf_revision"] = HF_REVISION
        report["gguf_revision"] = GGUF_REVISION
        args.private_dir.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        # Round-robin reversal counters time drift; each sample gets an independent
        # process and warmup, so memory is attributable to one arm, not two models.
        for round_no in range(args.repeats):
            arms = ["llama-f16", "llama-q8", "synapse-f16"]
            if round_no % 2:
                arms.reverse()
            for length in (128, 512):
                for batch in (1, 16, 64):
                    for arm in arms:
                        load = quiet_load()
                        directory = args.private_dir / f"{nonce}-{round_no}-{length}-{batch}-{arm}"
                        cls = Synapse if arm == "synapse-f16" else Llama
                        client = cls(args, arm, directory)
                        try:
                            stats = sweep(client, tokenizer, length, batch, 1, f"{nonce}-{round_no}-{length}-{batch}-{arm}")
                        finally:
                            client.process.close()
                        report["arms"].append({"kind": "sweep", "arm": arm, "round": round_no,
                            "length": length, "batch": batch, "load_1m_before_arm": load, **stats,
                            "memory": memory_from_time(client.process.log.read_text())})
        for round_no in range(3):
            arms = ["synapse-f16", "llama-q8"] if round_no % 2 == 0 else ["llama-q8", "synapse-f16"]
            for arm in arms:
                load = quiet_load()
                directory = args.private_dir / f"{nonce}-fill-{round_no}-{arm}"
                client = (Synapse if arm == "synapse-f16" else Llama)(args, arm, directory, slots=128)
                try:
                    stats = fill(client, tokenizer, args.fill_seconds, f"{nonce}-fill-{round_no}-{arm}")
                finally:
                    client.process.close()
                report["arms"].append({"kind": "aft-fill", "arm": arm, "round": round_no,
                    "load_1m_before_arm": load, **stats, "memory": memory_from_time(client.process.log.read_text())})
        groups = {}
        for row in report["arms"]:
            key = f"{row['kind']}/{row['arm']}/{row.get('length', 'mixed')}/{row.get('batch', 64)}"
            groups.setdefault(key, []).append(row)
        report["medians"] = {k: {"rounds": len(rows), "rows_per_min": statistics.median(
            row.get("rows_per_min", row.get("median_rows_per_min")) for row in rows),
            "latency_p50_s": statistics.median(row["latency_p50_s"] for row in rows),
            "latency_p90_s": statistics.median(row["latency_p90_s"] for row in rows),
            "max_rss_bytes": max(row["memory"]["max_rss_bytes"] for row in rows),
            "peak_footprint_bytes": max(row["memory"].get("peak_footprint_bytes", 0) for row in rows) or None} for k, rows in groups.items()}
        report["status"] = "completed"
    except NotQuiet as e:
        report["status"] = "deferred-high-load"
        report["reason"] = str(e)
        report["observed_load_1m"] = os.getloadavg()[0]
    except Exception:
        report["status"] = "failed"
        report["reason"] = "Arm failed validation; inspect the private process logs and exception"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report.get("status", "failed"), "load:", report["observed_load_1m"])


if __name__ == "__main__":
    main()
