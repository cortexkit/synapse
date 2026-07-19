#!/usr/bin/env python3
"""Localhost voice test bench for the STT context-bias spike.

Serves index.html, accepts 16 kHz mono s16 WAV uploads, runs the owned
spike-unified-rt ASR binary twice (baseline + trie delta 6), returns JSON.
No third-party Python packages — stdlib only. Binds 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SPIKE_PACKAGE = "spike-unified-rt"
SPIKE_BIN_NAME = "spike-unified-rt"
DEFAULT_PORT = 4799
DEFAULT_HOST = "127.0.0.1"
HF_MODEL_ID = "LiquidAI/LFM2-Audio-1.5B"
TRIE_DELTA = 6.0
TRIE_WINDOW = 16
MAX_NEW_TOKENS = 64
DECODE_CACHE_BUCKET = 1024
# Single lock: the spike is not safe for concurrent invocations against one package cache.
_transcribe_lock = threading.Lock()
_preferred_device: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "metal"),
        default="auto",
        help="ASR device: auto tries Metal then falls back to CPU",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="LFM2-Audio snapshot path (default: resolve from Hugging Face hub cache)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="Path to spike-unified-rt release binary (default: target/release/...)",
    )
    return parser.parse_args()


def cached_model_snapshot(model_id: str) -> Path | None:
    """Return the newest local HF hub snapshot for model_id, or None."""
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
    )
    if not cache_root.is_dir():
        return None
    snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def resolve_model(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"--model is not a directory: {path}")
        return path
    cached = cached_model_snapshot(HF_MODEL_ID)
    if cached is None:
        raise FileNotFoundError(
            f"No local snapshot for {HF_MODEL_ID} under ~/.cache/huggingface/hub. "
            "Download the model once, or pass --model PATH."
        )
    tokenizer = cached / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer.json missing in snapshot {cached}")
    return cached


def spike_binary_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return REPO_ROOT / "target" / "release" / SPIKE_BIN_NAME


def ensure_spike_binary(binary: Path) -> Path:
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    print(f"building {SPIKE_PACKAGE} release binary at {binary} ...", flush=True)
    subprocess.run(
        ["cargo", "build", "--release", "-p", SPIKE_PACKAGE],
        check=True,
        cwd=REPO_ROOT,
    )
    if not binary.is_file():
        raise FileNotFoundError(f"cargo build finished but binary missing: {binary}")
    return binary


def build_spike_command(
    *,
    binary: Path,
    model: Path,
    manifest: Path,
    out_json: Path,
    device: str,
    trie: bool,
) -> list[str]:
    """Match bench/spikes/stt-bias/evalkit/run_eval.py flag construction."""
    command = [
        str(binary),
        "--model",
        str(model),
        "--tokenizer",
        str(model / "tokenizer.json"),
        "--asr-audio",
        str(manifest),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--decode-cache-bucket",
        str(DECODE_CACHE_BUCKET),
        "--device",
        device,
        "--dtype",
        "f32",
        "--out",
        str(out_json),
    ]
    if trie:
        command.extend(
            [
                "--asr-trie-delta",
                f"{TRIE_DELTA:g}",
                "--asr-trie-window",
                str(TRIE_WINDOW),
            ]
        )
    return command


def parse_arm_result(out_json: Path, wall_ms: float) -> dict:
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"spike produced no results in {out_json}")
    row = results[0]
    text = row.get("text") or ""
    # Prefer per-run decode wall from the spike; fall back to process wall.
    decode_s = payload.get("decode_wall_s")
    if isinstance(decode_s, (int, float)):
        ms = float(decode_s) * 1000.0
    else:
        ms = wall_ms
    return {
        "text": text,
        "ms": round(ms, 1),
        "prefill_ms": round(float(payload.get("prefill_wall_s") or 0.0) * 1000.0, 1),
        "cold_load_ms": round(float(payload.get("cold_load_s") or 0.0) * 1000.0, 1),
        "wall_ms": round(wall_ms, 1),
    }


def run_arm(
    *,
    binary: Path,
    model: Path,
    wav_path: Path,
    work_dir: Path,
    device: str,
    bias_terms: list[str],
    bias_prompt: str | None,
    trie: bool,
    arm_name: str,
) -> dict:
    row: dict = {
        "id": "voice-clip",
        "path": str(wav_path.resolve()),
    }
    if trie:
        if not bias_terms:
            raise ValueError("trie arm requires at least one bias term")
        row["bias_terms"] = bias_terms
        if bias_prompt:
            row["bias_prompt"] = bias_prompt
    manifest = work_dir / f"{arm_name}-inputs.jsonl"
    out_json = work_dir / f"{arm_name}-owned.json"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    command = build_spike_command(
        binary=binary,
        model=model,
        manifest=manifest,
        out_json=out_json,
        device=device,
        trie=trie,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit {completed.returncode}"
        raise RuntimeError(f"{arm_name} spike failed on {device}: {detail[-4000:]}")
    if not out_json.is_file():
        raise RuntimeError(f"{arm_name} spike exited 0 but missing {out_json}")
    return parse_arm_result(out_json, wall_ms)


def choose_device(requested: str) -> str:
    """Prefer Metal on macOS when auto; cache the device that actually succeeds."""
    global _preferred_device
    if requested in {"cpu", "metal"}:
        return requested
    if _preferred_device is not None:
        return _preferred_device
    # Prefer Metal on Apple Silicon; first failed metal run falls back to CPU.
    _preferred_device = "metal" if sys.platform == "darwin" else "cpu"
    return _preferred_device


def transcribe(
    *,
    wav_bytes: bytes,
    bias_terms: list[str],
    bias_prompt: str | None,
    binary: Path,
    model: Path,
    device_pref: str,
) -> dict:
    if len(wav_bytes) < 44:
        raise ValueError("WAV body too short to be valid")
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("upload must be a RIFF/WAVE file")

    with tempfile.TemporaryDirectory(prefix="stt-voice-test-") as tmp:
        work = Path(tmp)
        wav_path = work / "clip.wav"
        wav_path.write_bytes(wav_bytes)

        device = choose_device(device_pref)
        tried: list[str] = []
        last_error: Exception | None = None
        devices_to_try = [device]
        if device == "metal" and device_pref == "auto":
            devices_to_try.append("cpu")

        for candidate in devices_to_try:
            tried.append(candidate)
            try:
                baseline = run_arm(
                    binary=binary,
                    model=model,
                    wav_path=wav_path,
                    work_dir=work,
                    device=candidate,
                    bias_terms=bias_terms,
                    bias_prompt=bias_prompt,
                    trie=False,
                    arm_name="baseline",
                )
                biased = run_arm(
                    binary=binary,
                    model=model,
                    wav_path=wav_path,
                    work_dir=work,
                    device=candidate,
                    bias_terms=bias_terms,
                    bias_prompt=bias_prompt,
                    trie=True,
                    arm_name="trie",
                )
                global _preferred_device
                _preferred_device = candidate
                return {
                    "baseline": baseline,
                    "biased": biased,
                    "device": candidate,
                    "tried_devices": tried,
                }
            except Exception as error:  # noqa: BLE001 — report and try fallback
                last_error = error
                print(f"device {candidate} failed: {error}", flush=True)
                continue
        assert last_error is not None
        raise last_error


class AppState:
    def __init__(self, binary: Path, model: Path, device: str) -> None:
        self.binary = binary
        self.model = model
        self.device = device
        self.static_dir = Path(__file__).resolve().parent


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                html = (state.static_dir / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
                return
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "model": str(state.model),
                        "binary": str(state.binary),
                        "device": state.device,
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/transcribe":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                self._send_json(400, {"error": "bad Content-Length"})
                return
            if length <= 0 or length > 50 * 1024 * 1024:
                self._send_json(400, {"error": "WAV size must be between 1 byte and 50 MiB"})
                return
            raw = self.rfile.read(length)
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()

            bias_terms: list[str] = []
            bias_prompt: str | None = None
            wav_bytes: bytes

            if content_type.startswith("multipart/form-data"):
                wav_bytes, bias_terms, bias_prompt = parse_multipart(
                    raw, self.headers.get("Content-Type") or ""
                )
            elif content_type in {"audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"}:
                wav_bytes = raw
                # Optional JSON bias headers for simple clients.
                terms_header = self.headers.get("X-Bias-Terms") or ""
                bias_terms = [t.strip() for t in terms_header.split("\n") if t.strip()]
                prompt_header = self.headers.get("X-Bias-Prompt")
                bias_prompt = prompt_header.strip() if prompt_header else None
            elif content_type == "application/json":
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    self._send_json(400, {"error": f"invalid JSON body: {error}"})
                    return
                b64 = payload.get("wav_base64") or payload.get("wav")
                if not b64:
                    self._send_json(400, {"error": "JSON body needs wav_base64"})
                    return
                try:
                    wav_bytes = base64.b64decode(b64)
                except Exception as error:  # noqa: BLE001
                    self._send_json(400, {"error": f"bad wav_base64: {error}"})
                    return
                terms = payload.get("bias_terms") or []
                if isinstance(terms, str):
                    bias_terms = [t.strip() for t in terms.splitlines() if t.strip()]
                elif isinstance(terms, list):
                    bias_terms = [str(t).strip() for t in terms if str(t).strip()]
                else:
                    self._send_json(400, {"error": "bias_terms must be a list or string"})
                    return
                bp = payload.get("bias_prompt")
                bias_prompt = str(bp).strip() if bp else None
            else:
                self._send_json(
                    415,
                    {
                        "error": "Content-Type must be multipart/form-data, audio/wav, or application/json",
                    },
                )
                return

            if not bias_terms:
                self._send_json(400, {"error": "at least one bias term is required for the trie arm"})
                return

            acquired = _transcribe_lock.acquire(blocking=False)
            if not acquired:
                self._send_json(
                    503,
                    {
                        "error": "another transcription is in progress; retry shortly",
                        "busy": True,
                    },
                )
                return
            try:
                result = transcribe(
                    wav_bytes=wav_bytes,
                    bias_terms=bias_terms,
                    bias_prompt=bias_prompt,
                    binary=state.binary,
                    model=state.model,
                    device_pref=state.device,
                )
                self._send_json(200, result)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001
                traceback.print_exc()
                self._send_json(500, {"error": str(error)})
            finally:
                _transcribe_lock.release()

    return Handler


def parse_multipart(
    body: bytes, content_type: str
) -> tuple[bytes, list[str], str | None]:
    """Minimal multipart parser for browser FormData uploads."""
    # boundary=...
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        raise ValueError("multipart missing boundary")
    delim = b"--" + boundary.encode("ascii", errors="strict")
    chunks = body.split(delim)
    wav_bytes: bytes | None = None
    bias_terms: list[str] = []
    bias_prompt: str | None = None

    for chunk in chunks:
        if not chunk or chunk in (b"--\r\n", b"--", b"\r\n", b"--\r\n\r\n"):
            continue
        if chunk.startswith(b"--"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        header_blob, _, data = chunk.partition(b"\r\n\r\n")
        if not _:
            continue
        headers = header_blob.decode("utf-8", errors="replace")
        name = None
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                for token in line.split(";"):
                    token = token.strip()
                    if token.startswith("name="):
                        name = token.split("=", 1)[1].strip().strip('"')
        if data.endswith(b"\r\n"):
            data = data[:-2]
        if name == "wav" or name == "file" or name == "audio":
            wav_bytes = data
        elif name == "bias_terms":
            text = data.decode("utf-8", errors="replace")
            bias_terms = [line.strip() for line in text.splitlines() if line.strip()]
        elif name == "bias_prompt":
            text = data.decode("utf-8", errors="replace").strip()
            bias_prompt = text or None

    if wav_bytes is None:
        raise ValueError("multipart body missing wav/file/audio field")
    return wav_bytes, bias_terms, bias_prompt


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "warning: binding outside loopback exposes the local ASR endpoint; "
            "prefer 127.0.0.1",
            file=sys.stderr,
        )
    model = resolve_model(args.model)
    binary = ensure_spike_binary(spike_binary_path(args.binary))
    state = AppState(binary=binary, model=model, device=args.device)
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"stt-voice-test listening on http://{args.host}:{args.port}/\n"
        f"  model:  {model}\n"
        f"  binary: {binary}\n"
        f"  device: {args.device}\n"
        "  audio never leaves this machine",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
