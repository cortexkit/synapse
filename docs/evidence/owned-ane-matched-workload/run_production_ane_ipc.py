#!/usr/bin/env python3
"""Measure exact pretokenized GTE fixtures through the installed ANE worker IPC."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import secrets
import socket
import statistics
import struct
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024
FIXTURE_COMMIT = "0de0d95a41c1606853509f87e4f04b9f86f2d411"
FIXTURE_SOURCE_PATH = (
    "docs/evidence/owned-metal-bucket-policy-v2/representative-text-fixture.json"
)
FIXTURE_SHA256 = "1297d1cfd09464e4b71917d317490733bf66a5e9f946652f2cff0d55e5f8b62a"
METAL_COMPARISON_SHA256 = (
    "c4b1d284fc78384010ee51c8e7e39edafc98bf5ff6ef0a6c0c61385ebf2d2c69"
)
METAL_V2_RAW_SHA256 = "b1ea366e408937861c3328616afc037b2543b238f9ad6ada6865236de52f1f58"
EXPECTED_FIXTURE_DIGESTS = {
    "tokenizer_sha256": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "model_config_sha256": "8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed",
    "model_weights_sha256": "3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba",
}
ARTIFACTS = (
    (
        128,
        "8c3bf4b2a50634ec4a3eb54e986b769c84d0d11946c281ecd88d24827afd7d80",
        "gte-modernbert-base-seq128.mlmodelc",
    ),
    (
        256,
        "a8626c487794f879b88c73bf9c8fe6f7864e3cc252a1db0d79bb5e096210b9fd",
        "gte-modernbert-base-seq256.mlmodelc",
    ),
    (
        512,
        "9df6b44617b49e08a068602efc3fddcd7dadc0e46791929698b524c96d544f72",
        "gte-modernbert-base-seq512.mlmodelc",
    ),
)
EXPECTED_CASE_ORDER = (
    "text-singleton-511",
    "text-singleton-513",
    "text-singleton-603",
    "text-singleton-1203",
    "text-mixed-budget-3072",
)
EXPECTED_ROWS = {
    "text-singleton-511": (
        (
            "climate",
            511,
            "0d3367c5fc4689301998eb69f15b9f8ad4a7a120c35435885b6c9c3d7934b8d8",
        ),
    ),
    "text-singleton-513": (
        (
            "database",
            513,
            "89e903dc0ff8586bc390acfd6b682accef4899af5b380babadbcc9331acfc4a8",
        ),
    ),
    "text-singleton-603": (
        (
            "biology",
            603,
            "5a125658bba9c1a987ff9decbb2c36001220c37458f4155863ebdae77d14490e",
        ),
    ),
    "text-singleton-1203": (
        (
            "history",
            1203,
            "a6088c8bf7e86f96374232c1c29d274fea6a6335cf0849fb1cc0ef7fa9832a8e",
        ),
    ),
    "text-mixed-budget-3072": (
        (
            "music",
            127,
            "0aa0378f8f6b28bcb83317694baad81e5d17c8217ba7922c2e6acd82d8f12a56",
        ),
        (
            "astronomy",
            255,
            "44d575756ab35edb26cab6b079a24d1f837dce160c601c8dc4ff023128d2c053",
        ),
        (
            "cooking",
            383,
            "39ea347470b8ebd789f7d1b276021a052e501249a110137c76181660e6d4b3e9",
        ),
        (
            "software",
            511,
            "9767afd28e4bfbffe635f8ee1eb2fa395079ec6ed6b1a8cc8d87b0cbb7638268",
        ),
        (
            "education",
            603,
            "c6fefe7b16745f0bdcc624f1dcc5f1e25bb3a906d79eca520d6c0c2ff380d965",
        ),
        (
            "transport",
            1193,
            "4bee0b89f6cfbc852abf60d6e13e79d428de28f2db7c5eae3d311ce4a989ce83",
        ),
    ),
}


class EvidenceError(RuntimeError):
    """Raised when inputs or worker responses violate the measurement contract."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = sha256_path(path)
    if actual != expected:
        raise EvidenceError(f"{path} SHA-256 {actual} != expected {expected}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} must contain a JSON object")
    return value


def ids_digest(ids: list[int]) -> str:
    payload = b"".join(struct.pack("<I", token_id) for token_id in ids)
    return hashlib.sha256(payload).hexdigest()


def fixture_cases(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in fixture["cases"]}


def validate_fixture(path: Path) -> dict[str, Any]:
    require_sha256(path, FIXTURE_SHA256)
    fixture = load_json(path)
    if fixture.get("fixture_version") != "owned-metal-representative-text-v1":
        raise EvidenceError("unexpected fixture_version")
    for key, expected in EXPECTED_FIXTURE_DIGESTS.items():
        if fixture.get(key) != expected:
            raise EvidenceError(f"fixture {key} does not match the pinned digest")
    if tuple(fixture.get("case_order", ())) != EXPECTED_CASE_ORDER:
        raise EvidenceError("fixture case_order does not match the pinned order")

    cases = fixture_cases(fixture)
    if tuple(case["name"] for case in fixture.get("cases", ())) != EXPECTED_CASE_ORDER:
        raise EvidenceError("fixture cases are not stored in the pinned order")
    for case_name, expected_rows in EXPECTED_ROWS.items():
        case = cases[case_name]
        rows = case.get("rows", [])
        observed_rows = tuple(
            (row.get("id"), len(row.get("input_ids", [])), row.get("input_ids_u32le_sha256"))
            for row in rows
        )
        if observed_rows != expected_rows:
            raise EvidenceError(f"fixture rows changed for {case_name}")
        if case.get("real_tokens") != sum(len(row["input_ids"]) for row in rows):
            raise EvidenceError(f"fixture real_tokens is inconsistent for {case_name}")
        for row in rows:
            ids = row["input_ids"]
            if row.get("target_tokens") != len(ids):
                raise EvidenceError(f"target_tokens changed for row {row.get('id')}")
            if any(not isinstance(token_id, int) or not 0 <= token_id < 2**31 for token_id in ids):
                raise EvidenceError(f"row {row.get('id')} contains a non-i32 token id")
            if ids_digest(ids) != row.get("input_ids_u32le_sha256"):
                raise EvidenceError(f"input id digest mismatch for row {row.get('id')}")
    return fixture


def validate_metal_references(
    fixture: dict[str, Any], comparison_path: Path, raw_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_sha256(comparison_path, METAL_COMPARISON_SHA256)
    require_sha256(raw_path, METAL_V2_RAW_SHA256)
    comparison = load_json(comparison_path)
    raw = load_json(raw_path)

    for metadata_name in ("baseline_metadata", "candidate_metadata"):
        metadata = comparison.get(metadata_name, {})
        for key, expected in EXPECTED_FIXTURE_DIGESTS.items():
            if metadata.get(key) != expected:
                raise EvidenceError(f"Metal {metadata_name}.{key} changed")
        if tuple(metadata.get("case_order", ())) != EXPECTED_CASE_ORDER:
            raise EvidenceError(f"Metal {metadata_name}.case_order changed")
    if comparison.get("baseline_metadata", {}).get("bucket_policy_version") != 1:
        raise EvidenceError("Metal baseline is not bucket policy v1")
    if comparison.get("candidate_metadata", {}).get("bucket_policy_version") != 2:
        raise EvidenceError("Metal candidate is not bucket policy v2")

    raw_metadata = raw.get("metadata", {})
    if raw_metadata != comparison.get("candidate_metadata"):
        raise EvidenceError("Metal v2 raw metadata differs from comparison candidate metadata")
    fixture_by_name = fixture_cases(fixture)
    raw_cases = {case["name"]: case for case in raw.get("cases", [])}
    if tuple(raw_cases) != EXPECTED_CASE_ORDER:
        raise EvidenceError("Metal v2 raw cases changed")
    for case_name in EXPECTED_CASE_ORDER:
        raw_case = raw_cases[case_name]
        fixture_case = fixture_by_name[case_name]
        fixture_rows_without_digests = [
            {key: value for key, value in row.items() if key != "input_ids_u32le_sha256"}
            for row in fixture_case["rows"]
        ]
        if raw_case.get("rows") != fixture_rows_without_digests:
            raise EvidenceError(f"Metal v2 rows differ from fixture for {case_name}")
        vectors = raw_case.get("vectors", [])
        if len(vectors) != len(fixture_case["rows"]):
            raise EvidenceError(f"Metal v2 vector row count changed for {case_name}")
        if any(len(vector) != 768 for vector in vectors):
            raise EvidenceError(f"Metal v2 vector dimensions changed for {case_name}")
        repeats = raw_case.get("repeat_vector_sha256", [])
        if len(repeats) != 3 or any(len(repeat) != len(vectors) for repeat in repeats):
            raise EvidenceError(f"Metal v2 repeat digests changed for {case_name}")
    return comparison, raw


def validate_artifacts(blob_dir: Path) -> list[dict[str, Any]]:
    validated = []
    for bucket, digest, expected_root in ARTIFACTS:
        path = blob_dir / digest
        if not path.is_file():
            raise EvidenceError(f"missing production artifact {path}")
        require_sha256(path, digest)
        with zipfile.ZipFile(path) as archive:
            roots = sorted(
                {
                    name.split("/", 1)[0]
                    for name in archive.namelist()
                    if name and not name.startswith("__MACOSX/")
                }
            )
        if roots != [expected_root]:
            raise EvidenceError(f"artifact {digest} roots {roots} != {[expected_root]}")
        validated.append(
            {
                "bucket": bucket,
                "digest": digest,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "archive_root": expected_root,
            }
        )
    return validated


def command_text(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_worker(worker: Path, source: Path, repo_root: Path) -> dict[str, Any]:
    if not worker.is_file() or not os.access(worker, os.X_OK):
        raise EvidenceError(f"worker is not executable: {worker}")
    swift_worker = worker.with_name("ck-synapse-worker-ane-swift")
    if not swift_worker.is_file() or not os.access(swift_worker, os.X_OK):
        raise EvidenceError(f"installed Swift worker is not executable: {swift_worker}")
    if not source.is_file():
        raise EvidenceError(f"worker source is missing: {source}")
    return {
        "launcher_path": str(worker.resolve()),
        "launcher_sha256": sha256_path(worker),
        "launcher_version": command_text([str(worker), "--version"], repo_root),
        "swift_worker_path": str(swift_worker.resolve()),
        "swift_worker_sha256": sha256_path(swift_worker),
        "source_path": str(source.relative_to(repo_root)),
        "source_sha256": sha256_path(source),
    }


def read_exact(stream: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.recv(count - len(chunks))
        if not chunk:
            raise EvidenceError(f"worker socket closed after {len(chunks)} of {count} bytes")
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(stream: socket.socket, max_frame: int = MAX_FRAME_BYTES) -> bytes:
    (length,) = struct.unpack("<I", read_exact(stream, 4))
    if length > max_frame:
        raise EvidenceError(f"worker frame length {length} exceeds {max_frame}")
    return read_exact(stream, length)


def write_frame(stream: socket.socket, payload: bytes, max_frame: int = MAX_FRAME_BYTES) -> None:
    if len(payload) > max_frame:
        raise EvidenceError(f"host frame length {len(payload)} exceeds {max_frame}")
    stream.sendall(struct.pack("<I", len(payload)) + payload)


def read_json_frame(stream: socket.socket, max_frame: int = MAX_FRAME_BYTES) -> dict[str, Any]:
    value = json.loads(read_frame(stream, max_frame))
    if not isinstance(value, dict):
        raise EvidenceError("worker JSON frame was not an object")
    return value


def write_json_frame(
    stream: socket.socket, value: dict[str, Any], max_frame: int = MAX_FRAME_BYTES
) -> None:
    write_frame(stream, json.dumps(value, separators=(",", ":")).encode(), max_frame)


def require_response(response: dict[str, Any], response_type: str, req_id: str) -> None:
    if response.get("type") != response_type or response.get("req_id") != req_id:
        rendered = json.dumps(response, sort_keys=True)
        raise EvidenceError(f"expected {response_type} for {req_id}, received {rendered}")


def flatten_ids(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        struct.pack("<i", token_id) for row in rows for token_id in row["input_ids"]
    )


def parse_vectors(
    raw: bytes, n: int, dims: int
) -> tuple[list[dict[str, Any]], list[tuple[float, ...]]]:
    expected = n * dims * 4
    if len(raw) != expected:
        raise EvidenceError(f"vector payload has {len(raw)} bytes, expected {expected}")
    records = []
    values_by_row = []
    row_bytes = dims * 4
    for index in range(n):
        payload = raw[index * row_bytes : (index + 1) * row_bytes]
        values = struct.unpack(f"<{dims}f", payload)
        if any(not math.isfinite(value) for value in values):
            raise EvidenceError(f"vector row {index} contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in values))
        records.append(
            {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "l2_norm": norm,
                "f32le_base64": base64.b64encode(payload).decode("ascii"),
            }
        )
        values_by_row.append(values)
    return records, values_by_row


def request_embed(
    stream: socket.socket,
    model_ref: str,
    req_id: str,
    rows: list[dict[str, Any]],
    max_frame: int,
) -> tuple[dict[str, Any], bytes, float]:
    request = {
        "type": "EMBED_BATCH",
        "req_id": req_id,
        "model_ref": model_ref,
        "pooling": "mean",
        "normalize": True,
        "items": [{"n_tokens": len(row["input_ids"])} for row in rows],
    }
    started = time.perf_counter_ns()
    write_json_frame(stream, request, max_frame)
    write_frame(stream, flatten_ids(rows), max_frame)
    response = read_json_frame(stream, max_frame)
    raw = b""
    if response.get("type") != "ERR":
        raw = read_frame(stream, max_frame)
        if response.get("type") == "VECTORS":
            # Include conversion to host vectors in the measured IPC return boundary.
            struct.unpack(f"<{len(raw) // 4}f", raw)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return response, raw, elapsed_ms


def vector_similarity(
    left: tuple[float, ...], right: list[float] | tuple[float, ...]
) -> dict[str, float]:
    if len(left) != len(right):
        raise EvidenceError(
            f"cannot compare vectors with dimensions {len(left)} and {len(right)}"
        )
    max_abs = max(abs(a - b) for a, b in zip(left, right))
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return {"max_abs_diff": max_abs, "cosine": dot / (left_norm * right_norm)}


def summarize_determinism(invocations: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(invocations[0]["vectors"])
    exact_rows = []
    max_abs = 0.0
    min_cosine = 1.0
    for row_index in range(row_count):
        reference_record = invocations[0]["vectors"][row_index]
        reference = struct.unpack(
            f"<{len(base64.b64decode(reference_record['f32le_base64'])) // 4}f",
            base64.b64decode(reference_record["f32le_base64"]),
        )
        digests = [invocation["vectors"][row_index]["sha256"] for invocation in invocations]
        exact_rows.append(len(set(digests)) == 1)
        for invocation in invocations[1:]:
            encoded = invocation["vectors"][row_index]["f32le_base64"]
            candidate = struct.unpack(
                f"<{len(base64.b64decode(encoded)) // 4}f", base64.b64decode(encoded)
            )
            metrics = vector_similarity(reference, candidate)
            max_abs = max(max_abs, metrics["max_abs_diff"])
            min_cosine = min(min_cosine, metrics["cosine"])
    return {
        "scope": "warmup_first_observation_and_three_warm_repeats",
        "all_invocations_byte_identical": all(exact_rows),
        "rows_byte_identical": exact_rows,
        "max_abs_diff_from_first_observation": max_abs,
        "min_cosine_to_first_observation": min_cosine,
    }


def metal_vector_map(raw: dict[str, Any], case_name: str) -> dict[str, list[float]]:
    case = next(case for case in raw["cases"] if case["name"] == case_name)
    return {row["id"]: vector for row, vector in zip(case["rows"], case["vectors"])}


def measure_workload(
    stream: socket.socket,
    model_ref: str,
    max_frame: int,
    name: str,
    fixture_case: str,
    scope: str,
    rows: list[dict[str, Any]],
    metal_raw: dict[str, Any],
    req_counter: list[int],
) -> dict[str, Any]:
    invocations = []
    decoded_invocations: list[list[tuple[float, ...]]] = []
    phases = ("warmup_first_observation", "warm_repeat_1", "warm_repeat_2", "warm_repeat_3")
    for phase in phases:
        req_counter[0] += 1
        req_id = f"embed-{req_counter[0]}"
        response, raw, elapsed_ms = request_embed(
            stream, model_ref, req_id, rows, max_frame
        )
        require_response(response, "VECTORS", req_id)
        dims = response.get("dims")
        n = response.get("n")
        if dims != 768 or n != len(rows):
            raise EvidenceError(f"unexpected vector shape n={n} dims={dims} for {name}")
        vector_records, vectors = parse_vectors(raw, n, dims)
        invocations.append(
            {
                "phase": phase,
                "worker_ipc_roundtrip_ms": elapsed_ms,
                "response": response,
                "vectors": vector_records,
            }
        )
        decoded_invocations.append(vectors)

    warm_times = [item["worker_ipc_roundtrip_ms"] for item in invocations[1:]]
    real_tokens = sum(len(row["input_ids"]) for row in rows)
    metal_vectors = metal_vector_map(metal_raw, fixture_case)
    vector_comparison = []
    for index, row in enumerate(rows):
        vector_comparison.append(
            {
                "row_id": row["id"],
                **vector_similarity(decoded_invocations[1][index], metal_vectors[row["id"]]),
            }
        )

    return {
        "name": name,
        "fixture_case": fixture_case,
        "scope": scope,
        "same_as_full_fixture_case": scope == "exact_fixture_case",
        "selected_bucket": 512,
        "rows": rows,
        "row_token_counts": [len(row["input_ids"]) for row in rows],
        "real_tokens_per_call": real_tokens,
        "padded_tokens_per_call": len(rows) * 512,
        "invocations": invocations,
        "determinism": summarize_determinism(invocations),
        "warm_aggregate": {
            "repeats": 3,
            "mean_worker_ipc_roundtrip_ms": statistics.fmean(warm_times),
            "total_worker_ipc_roundtrip_ms": sum(warm_times),
            "aggregate_real_tokens_per_ipc_second": real_tokens * 3 * 1000 / sum(warm_times),
        },
        "metal_v2_vector_comparison": {
            "metal_case": fixture_case,
            "ane_phase": "warm_repeat_1",
            "interpretation": "diagnostic_only_distinct_engine_fingerprint",
            "rows": vector_comparison,
            "max_abs_diff": max(row["max_abs_diff"] for row in vector_comparison),
            "min_cosine": min(row["cosine"] for row in vector_comparison),
        },
    }


def fixture_coverage(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = []
    for case in fixture["cases"]:
        lengths = [len(row["input_ids"]) for row in case["rows"]]
        if case["name"] == "text-singleton-511":
            status = "measured_unchanged_exact_case"
            reason = None
        elif max(lengths) > 512:
            status = "unsupported_unchanged"
            reason = f"row length {max(lengths)} exceeds the production ANE maximum of 512"
        else:
            status = "not_selected"
            reason = None
        item = {
            "name": case["name"],
            "row_ids": [row["id"] for row in case["rows"]],
            "row_token_counts": lengths,
            "real_tokens": case["real_tokens"],
            "ane_status": status,
        }
        if reason:
            item["reason"] = reason
        coverage.append(item)
    return coverage


def run_metadata_command(command: list[str], repo_root: Path) -> str | None:
    try:
        return command_text(command, repo_root)
    except (OSError, subprocess.CalledProcessError):
        return None


def host_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel_release": platform.release(),
        "macos_version": run_metadata_command(["sw_vers", "-productVersion"], repo_root),
        "macos_build": run_metadata_command(["sw_vers", "-buildVersion"], repo_root),
        "chip": run_metadata_command(["sysctl", "-n", "machdep.cpu.brand_string"], repo_root),
        "hardware_model": run_metadata_command(["sysctl", "-n", "hw.model"], repo_root),
        "memory_bytes": run_metadata_command(["sysctl", "-n", "hw.memsize"], repo_root),
    }


def run_measurement(
    fixture: dict[str, Any],
    comparison: dict[str, Any],
    metal_raw: dict[str, Any],
    artifacts: list[dict[str, Any]],
    worker_metadata: dict[str, Any],
    worker: Path,
    repo_root: Path,
) -> dict[str, Any]:
    nonce = secrets.token_hex(16)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(30)
    process: subprocess.Popen[bytes] | None = None
    stream: socket.socket | None = None
    model_ref: str | None = None
    worker_stderr = ""
    req_counter = [0]

    with tempfile.TemporaryDirectory(prefix="ck-ane-ipc-") as runtime_dir:
        socket_path = str(Path(runtime_dir) / "worker.sock")
        if len(socket_path.encode()) >= 104:
            raise EvidenceError(f"temporary Unix socket path is too long: {socket_path}")
        listener.bind(socket_path)
        listener.listen(1)
        spawned = time.perf_counter_ns()
        process = subprocess.Popen(
            [str(worker), "--socket", socket_path, "--nonce", nonce],
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            stream, _ = listener.accept()
            stream.settimeout(300)
            hello = read_json_frame(stream)
            hello_elapsed_ms = (time.perf_counter_ns() - spawned) / 1_000_000
            if hello.get("v") != PROTOCOL_VERSION or hello.get("nonce") != nonce:
                raise EvidenceError(f"invalid worker HELLO: {hello}")
            engine = hello.get("engine", {})
            if engine.get("engine") != "ane-coreml-worker":
                raise EvidenceError(f"unexpected worker engine identity: {engine}")
            max_frame = min(MAX_FRAME_BYTES, int(hello.get("max_frame", MAX_FRAME_BYTES)))
            write_json_frame(
                stream,
                {"v": PROTOCOL_VERSION, "accept": True, "max_frame": max_frame},
                max_frame,
            )

            paths = [artifact["path"] for artifact in artifacts]
            digests = [artifact["digest"] for artifact in artifacts]
            load_req_id = "load-1"
            load_request = {
                "type": "LOAD",
                "req_id": load_req_id,
                "artifact_path": paths[0],
                "artifact_digest": digests[0],
                "format": "mlmodelc",
                "runtime_config": {
                    "artifact_paths": json.dumps(paths, separators=(",", ":")),
                    "artifact_digests": json.dumps(digests, separators=(",", ":")),
                },
            }
            load_started = time.perf_counter_ns()
            write_json_frame(stream, load_request, max_frame)
            load_response = read_json_frame(stream, max_frame)
            load_roundtrip_ms = (time.perf_counter_ns() - load_started) / 1_000_000
            require_response(load_response, "LOADED", load_req_id)
            if load_response.get("dims") != 768:
                raise EvidenceError(f"unexpected loaded dimensions: {load_response}")
            model_ref = load_response.get("model_ref")
            if not isinstance(model_ref, str):
                raise EvidenceError("LOAD response omitted model_ref")

            ping_req_id = "ping-1"
            write_json_frame(
                stream, {"type": "PING", "req_id": ping_req_id}, max_frame
            )
            ping_response = read_json_frame(stream, max_frame)
            require_response(ping_response, "PONG", ping_req_id)
            placement_share = ping_response.get("placement_share")
            if not isinstance(placement_share, (int, float)) or placement_share < 0.9:
                raise EvidenceError(
                    f"worker did not report production ANE placement >= 0.9: {ping_response}"
                )

            cases = fixture_cases(fixture)
            unsupported_row = cases["text-singleton-513"]["rows"]
            unsupported_response, unsupported_raw, _ = request_embed(
                stream,
                model_ref,
                "unsupported-513",
                unsupported_row,
                max_frame,
            )
            if unsupported_raw or unsupported_response.get("type") != "ERR":
                raise EvidenceError(
                    f"513-token runtime probe was not rejected: {unsupported_response}"
                )
            if "[128,256,512]" not in unsupported_response.get("msg", "").replace(" ", ""):
                raise EvidenceError(
                    f"513-token rejection did not report loaded buckets: {unsupported_response}"
                )

            singleton = measure_workload(
                stream,
                model_ref,
                max_frame,
                "text-singleton-511",
                "text-singleton-511",
                "exact_fixture_case",
                cases["text-singleton-511"]["rows"],
                metal_raw,
                req_counter,
            )
            mixed_rows = [
                row
                for row in cases["text-mixed-budget-3072"]["rows"]
                if len(row["input_ids"]) <= 512
            ]
            mixed_subset = measure_workload(
                stream,
                model_ref,
                max_frame,
                "text-mixed-budget-3072-supported-row-subset",
                "text-mixed-budget-3072",
                "supported_row_subset_not_full_fixture_case",
                mixed_rows,
                metal_raw,
                req_counter,
            )

            result = {
                "evidence_version": "owned-ane-matched-workload-v1",
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "repository_head_at_measurement": command_text(
                    ["git", "rev-parse", "HEAD"], repo_root
                ),
                "method": {
                    "timing_scope": "production_worker_ipc_roundtrip",
                    "timing_boundary": (
                        "before JSON request serialization/write through raw vector frame read "
                        "and f32 host conversion"
                    ),
                    "engine_only_timing": False,
                    "serial_inference_worker_trees": 1,
                    "warmup_runs_per_workload": 1,
                    "measured_warm_repeats_per_workload": 3,
                    "warmup_is_first_observation": True,
                    "tokenization_performed": False,
                    "truncation_or_chunking_performed": False,
                    "pooling": "mean",
                    "normalize": True,
                },
                "source": {
                    "fixture_git_commit": FIXTURE_COMMIT,
                    "fixture_git_path": FIXTURE_SOURCE_PATH,
                    "fixture_path": "representative-text-fixture.json",
                    "fixture_sha256": FIXTURE_SHA256,
                    "metal_comparison_path": "metal-text-comparison.json",
                    "metal_comparison_sha256": METAL_COMPARISON_SHA256,
                    "metal_v2_raw_path": "metal-v2-reference.raw.json",
                    "metal_v2_raw_sha256": METAL_V2_RAW_SHA256,
                },
                "host": host_metadata(repo_root),
                "worker": {**worker_metadata, "hello": hello},
                "model": {
                    "model_id": "gte-modernbert-base-ane-fp16",
                    "maximum_loaded_bucket": 512,
                    "artifacts": artifacts,
                },
                "load": {
                    "spawn_to_hello_ms": hello_elapsed_ms,
                    "worker_ipc_load_roundtrip_ms": load_roundtrip_ms,
                    "worker_reported_cold_load_ms": load_response["cold_load_ms"],
                    "response": load_response,
                    "placement_ping": ping_response,
                    "note": (
                        "LOAD includes digest verification, archive materialization, "
                        "model loading, dimension probes, and placement-plan inspection "
                        "for all three buckets."
                    ),
                },
                "runtime_maximum_probe": {
                    "fixture_case": "text-singleton-513",
                    "tokens": 513,
                    "inference_executed": False,
                    "response": unsupported_response,
                },
                "fixture_coverage": fixture_coverage(fixture),
                "workloads": [singleton, mixed_subset],
                "metal_metric_labels": {
                    "cold_load": "cold_load_s",
                    "first_use": "first_use_engine_wall_s",
                    "warm_repeats": "warm_engine_wall_s",
                    "warm_mean": "mean_warm_engine_wall_s",
                    "throughput": "aggregate_real_tokens_per_s",
                    "scope": "engine embed_batch wall time, not ANE worker IPC roundtrip",
                },
                "metal_baseline_metadata": comparison["baseline_metadata"],
                "metal_candidate_metadata": comparison["candidate_metadata"],
            }
        finally:
            if stream is not None:
                try:
                    if model_ref is not None:
                        unload_request = {
                            "type": "UNLOAD",
                            "req_id": "unload-1",
                            "model_ref": model_ref,
                        }
                        write_json_frame(stream, unload_request)
                        read_json_frame(stream)
                    write_json_frame(stream, {"type": "SHUTDOWN"})
                except (EvidenceError, OSError, socket.timeout):
                    pass
                stream.close()
            listener.close()
            if process is not None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                if process.stderr is not None:
                    stderr_bytes = process.stderr.read()
                    worker_stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
                if process.returncode not in (0, None):
                    raise EvidenceError(
                        f"worker exited {process.returncode}; stderr: {worker_stderr[-2000:]}"
                    )
    if worker_stderr:
        result["worker"]["stderr"] = worker_stderr
    return result


def parse_args() -> argparse.Namespace:
    evidence_dir = Path(__file__).resolve().parent
    repo_root = evidence_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=evidence_dir / "representative-text-fixture.json"
    )
    parser.add_argument(
        "--metal-comparison", type=Path, default=evidence_dir / "metal-text-comparison.json"
    )
    parser.add_argument(
        "--metal-v2-raw", type=Path, default=evidence_dir / "metal-v2-reference.raw.json"
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path("~/.local/share/cortexkit/bin/ck-synapse-worker-ane").expanduser(),
    )
    parser.add_argument(
        "--blob-dir",
        type=Path,
        default=Path("~/.local/share/cortexkit/models/blobs").expanduser(),
    )
    parser.add_argument(
        "--worker-source",
        type=Path,
        default=repo_root / "crates/synapse-worker-ane/swift/ane_worker.swift",
    )
    parser.add_argument(
        "--output", type=Path, default=evidence_dir / "ane-worker-ipc.raw.json"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate pinned inputs without loading Core ML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_dir = Path(__file__).resolve().parent
    repo_root = evidence_dir.parents[2]
    fixture = validate_fixture(args.fixture.resolve())
    comparison, metal_raw = validate_metal_references(
        fixture, args.metal_comparison.resolve(), args.metal_v2_raw.resolve()
    )
    artifacts = validate_artifacts(args.blob_dir.expanduser().resolve())
    worker_metadata = validate_worker(
        args.worker.expanduser().resolve(), args.worker_source.resolve(), repo_root
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "fixture_sha256": FIXTURE_SHA256,
                    "metal_comparison_sha256": METAL_COMPARISON_SHA256,
                    "metal_v2_raw_sha256": METAL_V2_RAW_SHA256,
                    "artifact_digests": [artifact["digest"] for artifact in artifacts],
                    "worker": worker_metadata,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = run_measurement(
        fixture,
        comparison,
        metal_raw,
        artifacts,
        worker_metadata,
        args.worker.expanduser().resolve(),
        repo_root,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
