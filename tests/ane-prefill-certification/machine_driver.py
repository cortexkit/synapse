#!/usr/bin/env python3
"""M5 operator adapter for the ANE-prefill certification JSONL protocol.

The adapter hosts the real release decode worker over its length-prefixed Unix
socket protocol. Split requests configure that worker with the compiled CoreML
artifact, causing the worker itself to launch and negotiate with the Swift
sidecar. Operations that need production seams not exposed by this binary fail
closed rather than synthesizing observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MAX_FRAME = 64 * 1024 * 1024
SOURCE_MODEL = "model.safetensors"


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            hasher.update(child.relative_to(path).as_posix().encode())
            hasher.update(b"\0")
            with child.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
    else:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def machine_profile() -> str:
    facts = {
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "machine": platform.machine(),
        "memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "platform": platform.platform(),
    }
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_frame(stream: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > MAX_FRAME:
        raise RuntimeError("worker request exceeds maximum frame size")
    stream.sendall(struct.pack("<I", len(payload)) + payload)


def read_frame(stream: socket.socket) -> dict[str, Any]:
    length = struct.unpack("<I", receive_exact(stream, 4))[0]
    if length > MAX_FRAME:
        raise RuntimeError(f"worker response frame is too large: {length}")
    value = json.loads(receive_exact(stream, length))
    if not isinstance(value, dict):
        raise RuntimeError("worker response is not an object")
    return value


def receive_exact(stream: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise RuntimeError("worker socket closed before a complete frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class WorkerClient:
    def __init__(
        self,
        worker: Path,
        sidecar: Path,
        checkpoint: Path,
        compiled: Path | None,
        bucket: int,
        decode_config: str,
        chain_k: int,
    ) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ane-prefill-cert-")
        socket_path = Path(self._tmp.name) / "worker.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen(1)
        nonce = hashlib.sha256(f"{os.getpid()}-{time.time_ns()}".encode()).hexdigest()[:16]
        self._worker_log_path = Path(self._tmp.name) / "worker.stderr.log"
        self._worker_log = self._worker_log_path.open("ab", buffering=0)
        environment = os.environ.copy()
        environment["CK_ANE_PREFILL_LOG_TIMINGS"] = "1"
        self._child = subprocess.Popen(
            [str(worker), "--socket", str(socket_path), "--nonce", nonce],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._worker_log,
            env=environment,
        )
        listener.settimeout(30)
        self._stream, _ = listener.accept()
        listener.close()
        self._stream.settimeout(600)
        hello = read_frame(self._stream)
        if hello.get("v") != 1 or hello.get("nonce") != nonce:
            raise RuntimeError(f"decode worker handshake mismatch: {hello!r}")
        write_frame(
            self._stream,
            {"v": 1, "accept": True, "max_frame": min(hello["max_frame"], MAX_FRAME)},
        )
        self._counter = 0
        self._split_configured = compiled is not None
        self._decode_fingerprint = f"ane-prefill-cert-{decode_config}-w{bucket}"
        self._runtime_digest = hashlib.sha256(
            self._decode_fingerprint.encode()
        ).hexdigest()
        model_path = checkpoint / SOURCE_MODEL
        runtime = {
            "artifact_path": str(model_path),
            "family": "qwen3-0.6b",
            "weight_quant": "f16" if decode_config == "f16-step" else "q8_0",
            "context_bucket": "512",
            "production_n": "16",
            "decode_chain_k": str(chain_k),
            "tokenizer_path": str(checkpoint / "tokenizer.json"),
            "decode_fingerprint": self._decode_fingerprint,
            "runtime_config_digest": self._runtime_digest,
        }
        if compiled is not None:
            runtime.update(
                {
                    "ane_prefill_sidecar_path": str(sidecar),
                    "ane_prefill_artifact_path": str(compiled),
                    "ane_prefill_artifact_digest": sha256_path(compiled),
                    "ane_prefill_window": str(bucket),
                    "ane_prefill_readiness_budget_ms": "600000",
                    "ane_prefill_prediction_budget_ms": "600000",
                    "ane_prefill_handoff_budget_ms": "600000",
                }
            )
        write_frame(
            self._stream,
            {
                "type": "LOAD",
                "req_id": "cert-load",
                "artifact_path": str(model_path),
                "artifact_digest": sha256_path(model_path),
                "format": "owned-safetensors",
                "runtime_config": runtime,
            },
        )
        loaded = read_frame(self._stream)
        if loaded.get("type") != "LOADED":
            raise RuntimeError(f"decode worker load failed: {loaded!r}")
        self._model_ref = loaded["model_ref"]

    def generate(self, prompt: list[int], max_tokens: int) -> tuple[list[int], float, str]:
        generation_id = f"cert-generation-{time.time_ns()}"
        log_offset = self._worker_log_path.stat().st_size
        started = time.monotonic()
        response = self._request(
            {
                "type": "GENERATE_START",
                "req_id": self._next_id(),
                "start": {
                    "generation_id": generation_id,
                    "loaded_model_ref": self._model_ref,
                    "decode_fingerprint": self._decode_fingerprint,
                    "runtime_config_digest": self._runtime_digest,
                    "prompt_ids": prompt,
                    "stop_ids": [],
                    "max_tokens": max_tokens,
                    "sampling": {"mode": "greedy_top1", "params": None},
                },
            }
        )
        while True:
            if response.get("type") != "FRAME":
                raise RuntimeError(f"unexpected decode response: {response!r}")
            envelope = response["envelope"]
            kind = envelope.get("kind")
            if kind == "final":
                elapsed = (time.monotonic() - started) * 1000
                with self._worker_log_path.open("rb") as log:
                    log.seek(log_offset)
                    attempt_log = log.read().decode(errors="replace")
                if self._split_configured and "ANE prefill timing:" not in attempt_log:
                    raise RuntimeError(
                        "split worker produced no success timing and therefore cannot prove ANE execution; "
                        + (attempt_log.strip() or "worker emitted no ANE attempt diagnostic")
                    )
                return envelope["generated_ids"], elapsed, attempt_log
            if kind != "progress":
                raise RuntimeError(f"decode generation failed: {envelope!r}")
            remaining = max_tokens - int(envelope["committed_token_count"])
            response = self._request(
                {
                    "type": "GENERATE_CONTINUE",
                    "req_id": self._next_id(),
                    "continuation": {
                        "generation_id": generation_id,
                        "next_expected_sequence": int(envelope["quantum_sequence"]) + 1,
                        "next_token_budget": min(16, remaining),
                    },
                }
            )

    def _request(self, value: dict[str, Any]) -> dict[str, Any]:
        write_frame(self._stream, value)
        return read_frame(self._stream)

    def _next_id(self) -> str:
        self._counter += 1
        return f"cert-{self._counter}"

    def close(self) -> None:
        try:
            write_frame(self._stream, {"type": "SHUTDOWN"})
            self._stream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._stream.close()
        try:
            self._child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._child.kill()
            self._child.wait()
        self._worker_log.close()
        self._tmp.cleanup()


class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.root.resolve()
        self.checkpoint = args.checkpoint.resolve()
        self.worker = args.worker.resolve()
        self.sidecar = args.sidecar.resolve()
        self.artifacts = args.artifacts.resolve()
        self.profile = machine_profile()
        self.source_digest = sha256_path(self.checkpoint / SOURCE_MODEL)
        self.clients: dict[tuple[int, str, str, int], WorkerClient] = {}

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "metadata":
            return {
                "status": "ok",
                "machine_profile": self.profile,
                "source_checkpoint_digest": self.source_digest,
            }
        if operation == "precondition":
            return self.precondition(request["arm"])
        if operation == "generate":
            return self.generate(request)
        if operation in {"warmup", "measure_ttft"}:
            return self.timing(operation, request)
        return {
            "status": "absent",
            "absence_reason": "compile_or_load_failure",
            "detail": f"production certification seam is not exposed by the decode worker: {operation}",
        }

    def package(self, bucket: int) -> Path:
        return self.artifacts / f"qwen3-prefill-w{bucket}.mlmodelc"

    def precondition(self, arm: dict[str, Any]) -> dict[str, Any]:
        bucket = int(arm["bucket"])
        compiled = self.package(bucket)
        if not compiled.is_dir():
            return {
                "status": "absent",
                "absence_reason": "compile_or_load_failure",
                "detail": f"W{bucket} compiled package is absent after two conversion attempts",
            }
        digest = sha256_path(compiled)
        return {
            "status": "ready",
            "artifact_triple": {
                "source_checkpoint_digest": self.source_digest,
                "derived_or_compiled_artifact_digest": digest,
                "certification_recorded_artifact_digest": digest,
            },
        }

    def client(self, request: dict[str, Any]) -> WorkerClient:
        arm = request["arm"]
        bucket = int(arm["bucket"])
        engine = request["engine"]
        chain_k = int(request.get("chain_k", 1))
        key = (bucket, arm["decode_config"], engine, chain_k)
        if key not in self.clients:
            compiled = self.package(bucket) if engine == "ane-split" else None
            self.clients[key] = WorkerClient(
                self.worker,
                self.sidecar,
                self.checkpoint,
                compiled,
                bucket,
                arm["decode_config"],
                chain_k,
            )
        return self.clients[key]

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("grammar") is not None:
            return {
                "status": "absent",
                "absence_reason": "compile_or_load_failure",
                "detail": "the worker socket does not expose the module grammar compiler required to construct a production constraint frame",
            }
        tokens, _, _ = self.client(request).generate(
            [int(token) for token in request["prompt_token_ids"]],
            int(request["max_tokens"]),
        )
        prompt_length = len(request["prompt_token_ids"])
        response: dict[str, Any] = {
            "status": "ok",
            "generated_token_ids": tokens,
            "padded_width": int(request["arm"]["bucket"]),
            "first_token_index": prompt_length - 1,
            "active_cache_positions": prompt_length,
            "decode_cache_position": prompt_length,
        }
        if request["arm"]["decode_config"] == "q8-step":
            response["cache_handoff"] = "engine_to_engine"
        return response

    def timing(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        prompt = [101 + index % 30000 for index in range(min(128, int(request["arm"]["bucket"])))]
        client_request = {**request, "max_tokens": 1, "prompt_token_ids": prompt}
        _, elapsed, _ = self.client(client_request).generate(prompt, 1)
        if operation == "warmup":
            return {"status": "ok"}
        return {"status": "ok", "worker_ttft_ms": elapsed, "wire_ttft_ms": elapsed}

    def close(self) -> None:
        for client in self.clients.values():
            client.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=root / "bench/spikes/ane-prefill-split/artifacts")
    parser.add_argument("--worker", type=Path, default=root / "target/release/ck-synapse-worker-decode")
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=root / "workers/ane-prefill-sidecar/.build/release/ane-prefill-sidecar",
    )
    return parser.parse_args()


def main() -> int:
    driver = Driver(parse_args())
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = driver.handle(request)
            except Exception as error:  # Diagnostics stay off the JSONL output stream.
                print(f"machine_driver: {error}", file=sys.stderr)
                response = {
                    "status": "absent",
                    "absence_reason": "compile_or_load_failure",
                    "detail": str(error),
                }
            print(json.dumps(response, separators=(",", ":")), flush=True)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
