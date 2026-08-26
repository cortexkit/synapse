#!/usr/bin/env bash
set -euo pipefail

# Trusted Mac controller for the gte-modernbert f16 Vulkan embedding campaign.
# The timed cells execute on the Ally over SSH; all policy and validation below
# runs in the controller's Python standard library.
exec /usr/bin/python3 - "$@" <<'PY'
from __future__ import annotations

import ctypes
import gzip
import hashlib
import heapq
import json
import math
import os
import platform
import re
import shutil
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Historical wave-3 Ally steady pass. Registration may keep this pending until
# the idle-gated baseline command is run on the box.
BASELINE_TOK_S = 4_078.676036755986
CORPUS_SHA256 = "25d1d54427030d94c882dd96a5f5d26bfda426d902028e75aa8c3d527e34a7a7"
REFERENCE_VECTORS_SHA256 = "d55221d41098aa293507c734ebedbf2df7f095c5e7c767943167403bbb520afd"
MODEL_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
MODEL_COMPONENT_SHA256 = {
    "config.json": "8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed",
    "model.safetensors": "3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba",
    "tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
}
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots"
    / MODEL_REVISION
)
def configured_remote_target() -> str:
    target = os.environ.get("SYNAPSE_CAMPAIGN_REMOTE_TARGET")
    if not target:
        raise HarnessError("set SYNAPSE_CAMPAIGN_REMOTE_TARGET to run a remote benchmark")
    return target
REMOTE_MODEL = r"C:\bench\model-modernbert"
REMOTE_SOURCE_PARENT = r"C:\bench\campaign"
REMOTE_CARGO_TARGET = r"%USERPROFILE%\cargo-target-decode"
REMOTE_CARGO = "cargo"
REMOTE_CARGO_HOME = r"%USERPROFILE%\.cargo"
REMOTE_RUSTUP_HOME = r"%USERPROFILE%\.rustup"
REMOTE_SESSION_PREFIX = "gte-modernbert-vulkan"
DEVELOPER_DIR = "/Applications/Xcode.app/Contents/Developer"
CORPUS_NAME = "embedding-corpus.jsonl"
REFERENCE_NAME = "master-reference-vectors.bin.gz"
REFERENCE_METADATA_NAME = "REFERENCE-METADATA.json"
FIXTURE_MANIFEST_NAME = "SHA256SUMS"
FIXTURE_MANIFEST = {
    CORPUS_NAME: CORPUS_SHA256,
    REFERENCE_NAME: REFERENCE_VECTORS_SHA256,
    REFERENCE_METADATA_NAME: "0e39dd057c0f8239c6386d50f15ac46516a0e8c4932079136f3c640ebd80b718",
}
EXPECTED_ROWS = 2000
EXPECTED_DIMENSIONS = 768
EXPECTED_REAL_TOKENS = 240778
MAX_LENGTH = 512
ATTENTION_UNITS = 4_000_000
BUCKET_POLICY = 1
PROCESS_RUNS = 3
FRESH_PROCESSES_PER_RUN = 2
PASSES_PER_PROCESS = 3
WARMUP_PASSES = 1
MIN_MEAN_COSINE = 0.9999
MIN_WORST_DECILE_OVERLAP = 0.974
RANK_K = 10
RANK_QUERY_SAMPLE = 200
POWER_STATE_COMMAND = ("/usr/bin/pmset", "-g", "batt")
LOW_BATTERY_THRESHOLD_PERCENT = 20
CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112


class HarnessError(RuntimeError):
    pass


class CandidateRejected(HarnessError):
    pass


class ResultWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            existing = os.lstat(str(path))
            if not stat.S_ISREG(existing.st_mode):
                raise HarnessError(f"result path exists and is not a regular file: {path}")
            os.unlink(str(path))
        except FileNotFoundError:
            pass
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self.fd = os.open(str(path), flags, 0o600)
        os.fchmod(self.fd, 0o600)
        descriptor = os.fstat(self.fd)
        self.identity = (descriptor.st_dev, descriptor.st_ino)

    def write(self, payload: Mapping[str, Any]) -> None:
        current = os.lstat(str(self.path))
        if not stat.S_ISREG(current.st_mode):
            raise HarnessError("result path stopped being a regular file")
        if (current.st_dev, current.st_ino) != self.identity:
            raise HarnessError("result file was replaced during the harness run")
        if stat.S_IMODE(current.st_mode) != 0o600:
            raise HarnessError("result file permissions changed during the harness run")
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, encoded)
        os.fsync(self.fd)

    def close(self) -> None:
        os.close(self.fd)


def initial_payload(note: str) -> Dict[str, Any]:
    return {
        "gate_passed": False,
        "parity_passed": False,
        "determinism_passed": False,
        "samples": [],
        "paired_runs": [],
        "median_tok_s": None,
        "baseline_note": note,
        "workspace_commit": "",
    }


def load_jsonl(path: Path, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise HarnessError(f"cannot read {label}: {error}") from error
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HarnessError(f"{label}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise HarnessError(f"{label}:{line_number}: expected an object")
        rows.append(value)
    if not rows:
        raise HarnessError(f"{label}: fixture is empty")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise HarnessError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def parse_power_state(output: str) -> Dict[str, Any]:
    normalized = output.strip()
    if not normalized:
        raise HarnessError("POWER_STATE_PARSE_ERROR: pmset returned no output")
    lines = normalized.splitlines()
    source_match = re.fullmatch(r"Now drawing from '([^']+)'", lines[0].strip())
    if source_match is None:
        raise HarnessError("POWER_STATE_PARSE_ERROR: pmset did not report its power source")
    percentage_match = re.search(r"(?<!\d)(\d{1,3})%", normalized)
    if percentage_match is None:
        raise HarnessError("POWER_STATE_PARSE_ERROR: pmset did not report a battery percentage")
    battery_percent = int(percentage_match.group(1))
    if battery_percent > 100:
        raise HarnessError("POWER_STATE_PARSE_ERROR: battery percentage is out of range")
    return {
        "command": "pmset -g batt",
        "output": normalized,
        "power_source": source_match.group(1),
        "battery_percent": battery_percent,
    }


def capture_power_state() -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            POWER_STATE_COMMAND,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise HarnessError(f"POWER_STATE_PREFLIGHT_FAILED: could not run pmset: {error}") from error
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise HarnessError(
            f"POWER_STATE_PREFLIGHT_FAILED: pmset exited {completed.returncode}: "
            f"{output.strip() or '<empty>'}"
        )
    return parse_power_state(output)


def power_state_note(power: Mapping[str, Any]) -> str:
    return (
        "Power state (`pmset -g batt`): "
        f"{power.get('power_source')}, {power.get('battery_percent')}% battery."
    )


def enforce_power_preflight(power: Mapping[str, Any]) -> None:
    if (
        str(power.get("power_source", "")).casefold() == "battery power"
        and int(power.get("battery_percent", 100)) < LOW_BATTERY_THRESHOLD_PERCENT
    ):
        raise HarnessError(
            "LOW_BATTERY_POWER_PREFLIGHT: refusing measurement with "
            f"{power.get('battery_percent')}% remaining on battery power"
        )


def capture_metal_toolchain() -> Dict[str, str]:
    environment = os.environ.copy()
    environment["DEVELOPER_DIR"] = DEVELOPER_DIR
    try:
        completed = subprocess.run(
            ["/usr/bin/xcrun", "-sdk", "macosx", "--find", "metal"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
    except OSError as error:
        raise HarnessError(f"METAL_TOOLCHAIN_PREFLIGHT_FAILED: {error}") from error
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0 or not output:
        raise HarnessError(
            "METAL_TOOLCHAIN_PREFLIGHT_FAILED: xcrun --find metal failed with "
            f"status {completed.returncode}; DEVELOPER_DIR={DEVELOPER_DIR}; output={output or '<empty>'}"
        )
    compiler = output.splitlines()[-1].strip()
    if not Path(compiler).is_file() or not os.access(compiler, os.X_OK):
        raise HarnessError(f"METAL_TOOLCHAIN_PREFLIGHT_FAILED: non-executable compiler {compiler}")
    return {"developer_dir": DEVELOPER_DIR, "metal_compiler": compiler}


def configured_constants() -> Optional[float]:
    baseline_text = os.environ.get("SYNAPSE_CAMPAIGN_BASELINE_TOK_S", "pending").strip()
    if baseline_text.casefold() == "pending":
        baseline = None
    else:
        try:
            baseline = float(baseline_text)
        except ValueError as error:
            raise HarnessError("configured baseline must be numeric or pending") from error
        if baseline <= 0.0 or not math.isfinite(baseline):
            raise HarnessError("configured baseline must be a finite positive tok/s value")
    if os.environ.get("SYNAPSE_CAMPAIGN_MODEL_SHA256", MODEL_COMPONENT_SHA256["model.safetensors"]) != MODEL_COMPONENT_SHA256["model.safetensors"]:
        raise HarnessError("campaign registration model digest disagrees with the pinned harness")
    if os.environ.get("SYNAPSE_CAMPAIGN_CORPUS_SHA256", CORPUS_SHA256) != CORPUS_SHA256:
        raise HarnessError("campaign registration corpus digest disagrees with the pinned harness")
    if os.environ.get("SYNAPSE_CAMPAIGN_REFERENCE_VECTORS_SHA256", REFERENCE_VECTORS_SHA256) != REFERENCE_VECTORS_SHA256:
        raise HarnessError("campaign registration reference digest disagrees with the pinned harness")
    return baseline


def verify_model_snapshot(model: Path) -> Dict[str, str]:
    if not model.is_dir():
        raise HarnessError(f"pinned model snapshot is missing: {model}")
    if model.name != MODEL_REVISION:
        raise HarnessError(f"model snapshot revision must be {MODEL_REVISION}, got {model.name}")
    observed: Dict[str, str] = {}
    for name, expected in MODEL_COMPONENT_SHA256.items():
        path = model / name
        if not path.is_file():
            raise HarnessError(f"pinned model component is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise HarnessError(f"model component digest mismatch for {name}: {actual}")
        observed[name] = actual
    return observed


def verify_regular_file(path: Path, label: str) -> None:
    try:
        mode = os.lstat(str(path)).st_mode
    except OSError as error:
        raise HarnessError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise HarnessError(f"{label} is not a regular file: {path}")


def load_reference_binary(path: Path) -> Tuple[List[str], List[List[float]]]:
    verify_regular_file(path, "reference vectors")
    if sha256_file(path) != REFERENCE_VECTORS_SHA256:
        raise HarnessError("reference vector fixture SHA-256 does not match the pinned master output")
    try:
        data = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        raise HarnessError(f"reference vector fixture is not valid gzip: {error}") from error
    if len(data) < 12 or data[:4] != b"MEV1":
        raise HarnessError("reference vector fixture has an invalid MEV1 header")
    rows, dimensions = struct.unpack_from("<II", data, 4)
    if rows != EXPECTED_ROWS or dimensions != EXPECTED_DIMENSIONS:
        raise HarnessError("reference vector fixture has the wrong row or dimension count")
    offset = 12
    ids: List[str] = []
    vectors: List[List[float]] = []
    vector_bytes = dimensions * 4
    for _ in range(rows):
        if offset + 4 > len(data):
            raise HarnessError("reference vector fixture ends inside an ID length")
        (id_length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if id_length == 0 or offset + id_length + vector_bytes > len(data):
            raise HarnessError("reference vector fixture contains a truncated row")
        try:
            identifier = data[offset : offset + id_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise HarnessError("reference vector fixture contains a non-UTF-8 ID") from error
        offset += id_length
        vector = list(struct.unpack_from(f"<{dimensions}f", data, offset))
        offset += vector_bytes
        if any(not math.isfinite(value) for value in vector):
            raise HarnessError("reference vector fixture contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise HarnessError("reference vector fixture contains a zero vector")
        ids.append(identifier)
        vectors.append(vector)
    if offset != len(data) or len(set(ids)) != rows:
        raise HarnessError("reference vector fixture has trailing bytes or duplicate IDs")
    return ids, vectors


def extract_fixtures(workspace: Path, destination: Path) -> Tuple[Path, Path, List[Dict[str, Any]], List[List[float]]]:
    configured = os.environ.get("SYNAPSE_CAMPAIGN_FIXTURES")
    if not configured:
        raise HarnessError("set SYNAPSE_CAMPAIGN_FIXTURES to a licensed fixture directory")
    source = Path(configured).expanduser().resolve()
    if not source.is_dir():
        raise HarnessError(f"embedding fixture directory is missing: {source}")
    corpus_source = source / CORPUS_NAME
    reference_source = source / REFERENCE_NAME
    metadata_source = source / REFERENCE_METADATA_NAME
    manifest_source = source / FIXTURE_MANIFEST_NAME
    verify_regular_file(corpus_source, "embedding corpus fixture")
    verify_regular_file(reference_source, "reference vector fixture")
    verify_regular_file(metadata_source, "reference metadata fixture")
    verify_regular_file(manifest_source, "fixture SHA-256 manifest")
    manifest: Dict[str, str] = {}
    for line_number, line in enumerate(manifest_source.read_text().splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None:
            raise HarnessError(f"fixture SHA-256 manifest line {line_number} is malformed")
        digest, name = match.groups()
        if name in manifest:
            raise HarnessError(f"fixture SHA-256 manifest repeats {name}")
        manifest[name] = digest
    if manifest != FIXTURE_MANIFEST:
        raise HarnessError("fixture SHA-256 manifest does not match the pinned fixtures")
    if sha256_file(corpus_source) != CORPUS_SHA256:
        raise HarnessError("embedding corpus fixture SHA-256 does not match the pinned selection")
    destination.mkdir(parents=True, exist_ok=True, mode=0o755)
    corpus = destination / CORPUS_NAME
    reference = destination / REFERENCE_NAME
    shutil.copyfile(corpus_source, corpus)
    shutil.copyfile(reference_source, reference)
    corpus.chmod(0o444)
    reference.chmod(0o444)
    try:
        metadata = json.loads(metadata_source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"reference metadata is not valid JSON: {error}") from error
    if metadata.get("corpus_sha256") != CORPUS_SHA256 or metadata.get("vectors_sha256") != REFERENCE_VECTORS_SHA256:
        raise HarnessError("reference metadata does not match the embedded fixture pins")
    rows = load_jsonl(corpus, CORPUS_NAME)
    if len(rows) != EXPECTED_ROWS:
        raise HarnessError(f"embedding corpus must contain exactly {EXPECTED_ROWS} rows")
    ids: List[str] = []
    for index, row in enumerate(rows, start=1):
        identifier = row.get("id")
        text = row.get("text")
        if not isinstance(identifier, str) or not identifier or not isinstance(text, str) or not text:
            raise HarnessError(f"embedding corpus row {index} must contain nonempty id and text strings")
        if set(row) != {"id", "text"}:
            raise HarnessError(f"embedding corpus row {index} must contain only id and text")
        ids.append(identifier)
    if len(set(ids)) != EXPECTED_ROWS:
        raise HarnessError("embedding corpus IDs must be unique")
    reference_ids, reference_vectors = load_reference_binary(reference)
    if reference_ids != ids:
        raise HarnessError("reference vectors and embedding corpus are not ordered identically")
    return corpus, reference, rows, reference_vectors


def configured_sibling_sources() -> List[Path]:
    raw = os.environ.get("SYNAPSE_CAMPAIGN_SIBLINGS")
    if raw is None or not raw.strip():
        raise HarnessError("SYNAPSE_CAMPAIGN_SIBLINGS is unset or empty")
    sources: List[Path] = []
    names = set()
    for entry in raw.split(":"):
        if not entry.strip():
            raise HarnessError("SYNAPSE_CAMPAIGN_SIBLINGS contains an empty path")
        source = Path(entry).expanduser().resolve()
        if not source.is_dir():
            raise HarnessError(f"campaign sibling source is missing: {source}")
        if source.name in names or source.name == "workspace":
            raise HarnessError(f"campaign sibling source names are ambiguous: {source}")
        names.add(source.name)
        sources.append(source)
    return sources


def run_through_runner(runner: Path, argv: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = log_path.with_name(log_path.name + ".stderr")
    with log_path.open("ab") as output, stderr_path.open("ab") as errors:
        completed = subprocess.run(
            [str(runner), *argv],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=errors,
            check=False,
        )
    return completed.returncode


def runner_output(log_path: Path) -> str:
    pieces = []
    for path in (log_path, log_path.with_name(log_path.name + ".stderr")):
        try:
            text = path.read_text(errors="replace").strip()
        except OSError:
            continue
        if text:
            pieces.append(text)
    return "\n".join(pieces)


def ssh_command() -> List[str]:
    ssh = os.environ.get("SYNAPSE_CAMPAIGN_SSH") or shutil.which("ssh")
    if not ssh:
        raise HarnessError("ssh is not available on the campaign controller")
    command = [ssh, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    identity = os.environ.get("SYNAPSE_CAMPAIGN_SSH_IDENTITY")
    if identity:
        command.extend(["-i", identity])
    known_hosts = os.environ.get("SYNAPSE_CAMPAIGN_SSH_KNOWN_HOSTS")
    if known_hosts:
        command.extend(["-o", f"UserKnownHostsFile={known_hosts}", "-o", "StrictHostKeyChecking=yes"])
    command.append(configured_remote_target())
    return command


def scp_command() -> List[str]:
    scp = os.environ.get("SYNAPSE_CAMPAIGN_SCP") or shutil.which("scp")
    if not scp:
        raise HarnessError("scp is not available on the campaign controller")
    command = [scp, "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    identity = os.environ.get("SYNAPSE_CAMPAIGN_SSH_IDENTITY")
    if identity:
        command.extend(["-i", identity])
    known_hosts = os.environ.get("SYNAPSE_CAMPAIGN_SSH_KNOWN_HOSTS")
    if known_hosts:
        command.extend(["-o", f"UserKnownHostsFile={known_hosts}", "-o", "StrictHostKeyChecking=yes"])
    command.append(configured_remote_target())
    return command


def append_completed_streams(completed: subprocess.CompletedProcess[bytes], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        output.write(completed.stdout or b"")
    with log_path.with_name(log_path.name + ".stderr").open("ab") as errors:
        errors.write(completed.stderr or b"")


def run_remote(command: str, log_path: Path) -> int:
    completed = subprocess.run(
        [*ssh_command(), "cmd", "/c", command],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    append_completed_streams(completed, log_path)
    return completed.returncode


def remote_stdout(command: str, log_path: Path, limit: int = 16_384) -> str:
    completed = subprocess.run(
        [*ssh_command(), "cmd", "/c", command],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    append_completed_streams(completed, log_path)
    if completed.returncode != 0:
        raise HarnessError(f"Ally command failed with status {completed.returncode}: {runner_output(log_path)[-4096:]}")
    output = completed.stdout.decode("utf-8", errors="strict").strip()
    if len(output.encode()) > limit:
        raise HarnessError("Ally probe returned more data than its protocol limit")
    return output


def remote_path_for_scp(path: str) -> str:
    return path.replace("\\", "/")


def windows_quote(path: str) -> str:
    return '"' + path.replace('"', '\\"') + '"'


def scp_to(local: Path, remote: str, log_path: Path) -> None:
    target = configured_remote_target()
    completed = subprocess.run(
        [*scp_command()[:-1], str(local), f"{target}:{remote_path_for_scp(remote)}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    append_completed_streams(completed, log_path)
    if completed.returncode != 0:
        raise HarnessError(f"scp upload failed with status {completed.returncode}: {runner_output(log_path)[-4096:]}")


def scp_from(remote: str, local: Path, log_path: Path) -> None:
    target = configured_remote_target()
    local.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [*scp_command()[:-1], f"{target}:{remote_path_for_scp(remote)}", str(local)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    append_completed_streams(completed, log_path)
    if completed.returncode != 0:
        raise HarnessError(f"scp download failed with status {completed.returncode}: {runner_output(log_path)[-4096:]}")


def remote_env_prefix() -> str:
    values = {
        "CARGO_TARGET_DIR": os.environ.get("SYNAPSE_CAMPAIGN_REMOTE_CARGO_TARGET", REMOTE_CARGO_TARGET),
        "CARGO_HOME": os.environ.get("SYNAPSE_CAMPAIGN_REMOTE_CARGO_HOME", REMOTE_CARGO_HOME),
        "RUSTUP_HOME": os.environ.get("SYNAPSE_CAMPAIGN_REMOTE_RUSTUP_HOME", REMOTE_RUSTUP_HOME),
        "CARGO_NET_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return "".join(f'set "{name}={value}"&&' for name, value in values.items())


def remote_idle_probe(log_root: Path) -> str:
    user_profile = remote_stdout("echo %USERPROFILE%", log_root / "ally-userprofile.log").rstrip('"')
    if not re.fullmatch(r"[A-Za-z]:\\[^\r\n]+", user_profile):
        raise HarnessError(f"Ally returned an invalid USERPROFILE: {user_profile!r}")
    process_text = remote_stdout(
        'tasklist /FI "IMAGENAME eq cargo.exe" /FO CSV /NH && '
        'tasklist /FI "IMAGENAME eq unified-rt.exe" /FO CSV /NH && '
        'tasklist /FI "IMAGENAME eq spike-unified-rt.exe" /FO CSV /NH',
        log_root / "ally-tenant-probe.log",
    )
    active = [
        line.strip()
        for line in process_text.splitlines()
        if line.strip() and "INFO:" not in line.upper() and "NO TASKS" not in line.upper()
    ]
    if active:
        raise HarnessError(
            "Ally is busy with a cargo/unified-rt tenant; refusing execution: " + " | ".join(active)
        )
    return user_profile


def remote_model_hashes(model: str, log_root: Path) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for name, expected in MODEL_COMPONENT_SHA256.items():
        path = model + "\\" + name
        output = remote_stdout(
            f"certutil -hashfile {windows_quote(path)} SHA256",
            log_root / f"model-{name}.hash.log",
        )
        hashes = re.findall(r"(?i)\b[0-9a-f]{64}\b", output)
        if expected.lower() not in {value.lower() for value in hashes}:
            raise HarnessError(f"Ally model digest mismatch for {name}: {output!r}")
        observed[name] = expected
    return observed


def verify_utf8_no_bom(path: Path, label: str) -> None:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise HarnessError(f"{label} contains a UTF-8 BOM")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HarnessError(f"{label} is not UTF-8: {error}") from error


def make_candidate_bundle(workspace: Path, temp_root: Path) -> Tuple[Path, Path, str]:
    head = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", head.stdout.strip()) is None:
        raise CandidateRejected("candidate workspace did not report one full Git commit SHA")
    untracked = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise CandidateRejected("candidate workspace contains untracked files that cannot be synced safely")
    bundle = temp_root / "candidate.bundle"
    bundled = subprocess.run(
        ["git", "-C", str(workspace), "bundle", "create", str(bundle), "HEAD"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if bundled.returncode != 0:
        raise HarnessError(f"could not create candidate Git bundle: {bundled.stderr.decode(errors='replace')[-4096:]}")
    patch = temp_root / "candidate.patch"
    diff = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--binary", "HEAD", "--"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0:
        raise HarnessError(f"could not capture candidate worktree diff: {diff.stderr.decode(errors='replace')[-4096:]}")
    patch.write_bytes(diff.stdout)
    return bundle, patch, head.stdout.strip()


def sync_candidate_to_ally(
    workspace: Path,
    temp_root: Path,
    user_profile: str,
    log_root: Path,
) -> Dict[str, str]:
    bundle, patch, expected_head = make_candidate_bundle(workspace, temp_root)
    siblings = configured_sibling_sources()
    sibling_bundles = []
    for sibling in siblings:
        sibling_temp = temp_root / "sibling-bundles" / sibling.name
        sibling_temp.mkdir(parents=True, exist_ok=True)
        sibling_bundle, sibling_patch, sibling_head = make_candidate_bundle(sibling, sibling_temp)
        sibling_bundles.append((sibling.name, sibling_bundle, sibling_patch, sibling_head))
    root = user_profile + "\\" + REMOTE_SESSION_PREFIX + "-" + str(os.getpid())
    remote_workspace = root + "\\workspace"
    remote_fixture = root + "\\fixtures"
    remote_bundle = root + "\\candidate.bundle"
    remote_patch = root + "\\candidate.patch"
    if run_remote(
        f"if exist {windows_quote(root)} rmdir /s /q {windows_quote(root)}&&mkdir {windows_quote(root)}&&mkdir {windows_quote(remote_fixture)}",
        log_root / "ally-stage-root.log",
    ) != 0:
        raise HarnessError("Ally candidate-owned staging root could not be created")
    scp_to(bundle, remote_bundle, log_root / "bundle-upload.log")
    if patch.stat().st_size:
        scp_to(patch, remote_patch, log_root / "patch-upload.log")
    for sibling_name, sibling_bundle, sibling_patch, _sibling_head in sibling_bundles:
        scp_to(
            sibling_bundle,
            root + "\\" + sibling_name + ".bundle",
            log_root / f"{sibling_name}-bundle-upload.log",
        )
        if sibling_patch.stat().st_size:
            scp_to(
                sibling_patch,
                root + "\\" + sibling_name + ".patch",
                log_root / f"{sibling_name}-patch-upload.log",
            )
    init = (
        f"mkdir {windows_quote(remote_workspace)}&&"
        f"git -C {windows_quote(remote_workspace)} init&&"
        f"git -C {windows_quote(remote_workspace)} remote add campaign {windows_quote(remote_bundle)}&&"
        f"git -C {windows_quote(remote_workspace)} fetch --no-tags campaign HEAD&&"
        f"git -C {windows_quote(remote_workspace)} checkout --detach -f FETCH_HEAD"
    )
    if run_remote(init, log_root / "ally-git-sync.log") != 0:
        raise HarnessError("Ally checkout could not be initialized from the candidate bundle")
    if patch.stat().st_size:
        if run_remote(
            f"git -C {windows_quote(remote_workspace)} apply --whitespace=nowarn {windows_quote(remote_patch)}",
            log_root / "ally-git-patch.log",
        ) != 0:
            raise CandidateRejected("candidate worktree patch could not be applied on the Ally")
    for sibling_name, _sibling_bundle, sibling_patch, _sibling_head in sibling_bundles:
        remote_sibling = root + "\\" + sibling_name
        remote_sibling_bundle = root + "\\" + sibling_name + ".bundle"
        sibling_init = (
            f"mkdir {windows_quote(remote_sibling)}&&"
            f"git -C {windows_quote(remote_sibling)} init&&"
            f"git -C {windows_quote(remote_sibling)} remote add campaign {windows_quote(remote_sibling_bundle)}&&"
            f"git -C {windows_quote(remote_sibling)} fetch --no-tags campaign HEAD&&"
            f"git -C {windows_quote(remote_sibling)} checkout --detach -f FETCH_HEAD"
        )
        if run_remote(sibling_init, log_root / f"ally-{sibling_name}-git-sync.log") != 0:
            raise HarnessError(f"Ally sibling checkout could not be initialized: {sibling_name}")
        if sibling_patch.stat().st_size:
            remote_sibling_patch = root + "\\" + sibling_name + ".patch"
            if run_remote(
                f"git -C {windows_quote(remote_sibling)} apply --whitespace=nowarn {windows_quote(remote_sibling_patch)}",
                log_root / f"ally-{sibling_name}-git-patch.log",
            ) != 0:
                raise CandidateRejected(f"Ally sibling patch could not be applied: {sibling_name}")
    observed_head = remote_stdout(
        f"git -C {windows_quote(remote_workspace)} rev-parse HEAD",
        log_root / "ally-worktree-head.log",
    )
    if observed_head != expected_head:
        raise HarnessError(f"Ally checkout HEAD mismatch: expected {expected_head}, got {observed_head}")
    status = remote_stdout(
        f"git -C {windows_quote(remote_workspace)} status --porcelain",
        log_root / "ally-worktree-status.log",
    )
    return {
        "root": root,
        "workspace": remote_workspace,
        "fixtures": remote_fixture,
        "commit": observed_head,
        "worktree_status": status,
    }


def preserve_failure_scene(temp_root: Path, result_path: Path, remote_root: str = "") -> None:
    try:
        scene_dir = result_path.parent / "failure-scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(temp_root.rglob("*")):
            if not source.is_file():
                continue
            destination = scene_dir / source.relative_to(temp_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        scene = {
            "controller_cwd": os.getcwd(),
            "remote_target": configured_remote_target(),
            "remote_root": remote_root,
            "result": str(result_path),
        }
        (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1) + "\\n")
    except OSError:
        pass


def verify_copy(runner: Path, destination: Path, log_path: Path) -> None:
    status = run_through_runner(
        runner,
        [
            "/bin/sh",
            "-c",
            "test -d \"$1\" && for entry in \"$1\"/* \"$1\"/.[!.]* \"$1\"/..?*; do if test -e \"$entry\"; then printf '%s\\n' \"$entry\"; exit 0; fi; done; exit 1",
            "verify-copy",
            str(destination),
        ],
        log_path,
    )
    if status != 0 or not log_path.read_text(errors="replace").strip():
        raise HarnessError(f"runner copy produced an empty destination: {destination}")


def copy_tree(runner: Path, source: Path, destination: Path, log_path: Path) -> None:
    # APFS clonefile first (instant, fits inside the runner's action deadline
    # for a full checkout); byte-copy fallback for non-clonable filesystems.
    # The fixtures directory must not be driver-protected in the registration:
    # a permission-blocked subtree fails both copies, and fixture integrity is
    # already enforced by the harness's pinned SHA-256 verification.
    status = run_through_runner(runner, ["/bin/cp", "-cR", str(source), str(destination)], log_path)
    if status != 0:
        if destination.exists() or destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        status = run_through_runner(runner, ["/bin/cp", "-R", str(source), str(destination)], log_path)
    if status != 0:
        raise HarnessError(f"could not stage {source}: {runner_output(log_path)[-4096:]}")
    verify_copy(runner, destination, log_path.with_name(log_path.name + ".verify"))


def stage_sources(workspace: Path, temp_root: Path, runner: Path) -> Tuple[Path, List[Tuple[str, Path]]]:
    probe_log = temp_root / "runner-probe.log"
    status = run_through_runner(runner, ["/bin/sh", "-c", "echo runner-ok"], probe_log)
    if status != 0 or probe_log.read_text(errors="replace").strip() != "runner-ok":
        raise HarnessError(f"candidate runner probe failed: {runner_output(probe_log) or '<empty>'}")
    sources = configured_sibling_sources()
    build_root = temp_root / "build"
    # The RUNNER (candidate identity) must create the build root so it owns it;
    # a harness-side mkdir leaves a controller-owned directory the candidate
    # cannot write into, and the pre-created path turns the runner's mkdir -p
    # into a silent no-op that defers the failure to the copy step.
    if run_through_runner(runner, ["/bin/mkdir", "-p", str(build_root)], temp_root / "build-mkdir.log") != 0:
        raise HarnessError("candidate runner could not create its build staging directory")
    staged_workspace = build_root / "workspace"
    copy_tree(runner, workspace, staged_workspace, temp_root / "workspace-copy.log")
    staged_siblings: List[Tuple[str, Path]] = []
    for source in sources:
        destination = build_root / source.name
        copy_tree(runner, source, destination, temp_root / f"{source.name}-copy.log")
        staged_siblings.append((source.name, destination))
    return staged_workspace, staged_siblings


def create_candidate_output_dirs(temp_root: Path, runner: Path) -> Tuple[Path, Path, Path]:
    output_root = temp_root / "candidate-output"
    target = output_root / "target"
    packages = output_root / "packages"
    status = run_through_runner(
        runner,
        ["/bin/mkdir", "-p", str(output_root), str(target), str(packages)],
        temp_root / "candidate-output-mkdir.log",
    )
    if status != 0:
        raise HarnessError("candidate runner could not create candidate output directories")
    status = run_through_runner(
        runner,
        ["/bin/chmod", "777", str(output_root), str(target), str(packages)],
        temp_root / "candidate-output-chmod.log",
    )
    if status != 0:
        raise HarnessError("candidate output directories are not writable")
    return output_root, target, packages


def candidate_environment(target_dir: Path) -> List[str]:
    values = [
        "/usr/bin/env",
        f"DEVELOPER_DIR={DEVELOPER_DIR}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "CARGO_NET_OFFLINE=true",
        "CARGO_TERM_COLOR=never",
        "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
        f"CARGO_TARGET_DIR={target_dir}",
    ]
    for name in ("RUSTUP_HOME", "CARGO_HOME"):
        if os.environ.get(name):
            values.append(f"{name}={os.environ[name]}")
    return values


def parse_commit(text: str) -> str:
    value = text.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise CandidateRejected("candidate workspace did not report one full Git commit SHA")
    return value


def workspace_commit(runner: Path, workspace: Path, log_path: Path) -> str:
    status = run_through_runner(
        runner,
        [
            "/usr/bin/git",
            "-c",
            f"safe.directory={workspace}",
            "-C",
            str(workspace),
            "rev-parse",
            "HEAD",
        ],
        log_path,
    )
    if status != 0:
        raise CandidateRejected("candidate workspace is not a Git checkout")
    return parse_commit(log_path.read_text(errors="replace"))


def acquire_bench_lock() -> Path:
    lock = Path(
        os.environ.get(
            "SYNAPSE_CAMPAIGN_BENCH_LOCK",
            str(Path(tempfile.gettempdir()) / "synapse-benchmark.lock"),
        )
    )
    measure_lock = Path(os.environ.get("SYNAPSE_CAMPAIGN_MEASURE_LOCK", "/tmp/aft-measure.lock"))
    if measure_lock.exists():
        raise HarnessError(f"measurement lock is already present: {measure_lock}")
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise HarnessError(f"benchmark lock is already present: {lock}") from error
    except OSError as error:
        raise HarnessError(f"could not acquire benchmark lock {lock}: {error}") from error
    try:
        worker = subprocess.run(
            ["/usr/bin/pgrep", "-f", "Runner.Worker"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as error:
        lock.rmdir()
        raise HarnessError(f"could not inspect Runner.Worker state: {error}") from error
    if worker.returncode == 0:
        lock.rmdir()
        raise HarnessError("Runner.Worker is active; benchmark lock released without measuring")
    return lock


def release_bench_lock(lock: Optional[Path]) -> None:
    if lock is None:
        return
    try:
        lock.rmdir()
    except OSError as error:
        print(f"warning: could not release benchmark lock {lock}: {error}", file=sys.stderr)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def load_candidate_vectors(path: Path, expected_ids: Sequence[str]) -> List[List[float]]:
    rows = load_jsonl(path, "candidate vectors")
    if len(rows) != len(expected_ids):
        raise CandidateRejected("candidate emitted the wrong number of vectors")
    by_id: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier in by_id or set(row) != {"id", "vec"}:
            raise CandidateRejected(f"candidate vector row {index} has a duplicate identity or wrong fields")
        by_id[identifier] = row
    if set(by_id) != set(expected_ids):
        raise CandidateRejected("candidate vector IDs do not match the pinned corpus IDs")
    vectors: List[List[float]] = []
    for index, expected_id in enumerate(expected_ids, start=1):
        vector = by_id[expected_id].get("vec")
        if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSIONS:
            raise CandidateRejected(f"candidate vector row {index} is not {EXPECTED_DIMENSIONS}-dimensional")
        if any(not finite_number(value) for value in vector):
            raise CandidateRejected(f"candidate vector row {index} contains NaN or infinity")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise CandidateRejected(f"candidate vector row {index} is zero or non-finite")
        vectors.append([float(value) for value in vector])
    return vectors


def load_cblas() -> Optional[Any]:
    if platform.system() != "Darwin":
        return None
    try:
        library = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        function = library.cblas_sgemm
        float_pointer = ctypes.POINTER(ctypes.c_float)
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            float_pointer,
            ctypes.c_int,
            float_pointer,
            ctypes.c_int,
            ctypes.c_float,
            float_pointer,
            ctypes.c_int,
        ]
        function.restype = None
        return function
    except (OSError, AttributeError):
        return None


class RankComputer:
    def __init__(self, reference: Sequence[Sequence[float]]) -> None:
        self.reference = [self._normalize(row) for row in reference]
        self.cblas = load_cblas()
        if len(self.reference) >= EXPECTED_ROWS and self.cblas is None:
            raise HarnessError("full 2,000-row rank gate requires Apple's Accelerate cblas_sgemm")
        self.reference_neighbors = self._neighbors(self.reference)

    @staticmethod
    def _normalize(vector: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def _neighbors_with_cblas(self, rows: Sequence[Sequence[float]]) -> List[List[int]]:
        count = len(rows)
        dimensions = len(rows[0])
        packed = (ctypes.c_float * (count * dimensions))(
            *[value for row in rows for value in row]
        )
        products = (ctypes.c_float * (count * count))()
        self.cblas(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            count,
            count,
            dimensions,
            1.0,
            packed,
            dimensions,
            packed,
            dimensions,
            0.0,
            products,
            count,
        )
        neighbors: List[List[int]] = []
        for query in range(count):
            scores = products[query * count : (query + 1) * count]
            neighbors.append([
                index
                for _, index in heapq.nlargest(
                    RANK_K + 1,
                    ((float(score), index) for index, score in enumerate(scores) if index != query),
                )[:RANK_K]
            ])
        return neighbors

    def _neighbors_fallback(self, rows: Sequence[Sequence[float]]) -> List[List[int]]:
        count = len(rows)
        query_indices = [(index * 7919) % count for index in range(min(RANK_QUERY_SAMPLE, count))]
        selected = set(query_indices)
        result: List[List[int]] = [[] for _ in range(count)]
        for query in query_indices:
            scores = []
            for index, candidate in enumerate(rows):
                if index == query:
                    continue
                score = sum(left * right for left, right in zip(rows[query], candidate))
                scores.append((score, index))
            result[query] = [index for _, index in heapq.nlargest(RANK_K, scores)]
        return result

    def _neighbors(self, rows: Sequence[Sequence[float]]) -> List[List[int]]:
        if self.cblas is not None:
            return self._neighbors_with_cblas(rows)
        return self._neighbors_fallback(rows)

    def evaluate(self, candidate: Sequence[Sequence[float]]) -> Dict[str, float]:
        candidate_normalized = [self._normalize(row) for row in candidate]
        cosines = [cosine(left, right) for left, right in zip(candidate, self.reference)]
        candidate_neighbors = self._neighbors(candidate_normalized)
        overlaps = []
        for index, reference_row in enumerate(self.reference_neighbors):
            if not reference_row:
                continue
            candidate_row = candidate_neighbors[index]
            overlaps.append(len(set(reference_row).intersection(candidate_row)) / float(RANK_K))
        if not overlaps:
            raise CandidateRejected("rank-overlap probe did not produce any query rows")
        worst_count = max(1, math.ceil(len(overlaps) * 0.10))
        worst_decile = statistics.fmean(sorted(overlaps)[:worst_count])
        mean_cosine = statistics.fmean(cosines)
        return {
            "mean_cosine": mean_cosine,
            "worst_decile_top10_rank_overlap": worst_decile,
            "rank_queries": float(len(overlaps)),
            "parity_passed": mean_cosine >= MIN_MEAN_COSINE and worst_decile >= MIN_WORST_DECILE_OVERLAP,
        }


def validate_process_result(
    payload: Mapping[str, Any],
    vector_path: Path,
    expected_ids: Sequence[str],
    ranker: RankComputer,
) -> Dict[str, Any]:
    if payload.get("items") != EXPECTED_ROWS or payload.get("real_tokens") != EXPECTED_REAL_TOKENS:
        raise CandidateRejected("candidate changed the pinned corpus size or real-token count")
    if payload.get("shape_policy") != "bucketed" or payload.get("bucket_policy_version") != BUCKET_POLICY:
        raise CandidateRejected("candidate changed the pinned bucket policy")
    passes = payload.get("passes")
    if not isinstance(passes, list) or len(passes) != PASSES_PER_PROCESS:
        raise CandidateRejected("candidate did not emit exactly seven process passes")
    timed: List[float] = []
    for index, item in enumerate(passes):
        if not isinstance(item, dict):
            raise CandidateRejected(f"candidate pass {index + 1} is not an object")
        expected_label = "first" if index == 0 else "steady" if index == PASSES_PER_PROCESS - 1 else "warm"
        if item.get("label") != expected_label:
            raise CandidateRejected(f"candidate pass {index + 1} has the wrong warmup label")
        if item.get("items") != EXPECTED_ROWS or item.get("input_tokens") != EXPECTED_REAL_TOKENS:
            raise CandidateRejected(f"candidate pass {index + 1} changed the pinned input accounting")
        wall = item.get("infer_wall_s")
        reported = item.get("tok_per_s")
        if not finite_number(wall) or float(wall) <= 0.0 or not finite_number(reported):
            raise CandidateRejected(f"candidate pass {index + 1} has invalid timing")
        computed = EXPECTED_REAL_TOKENS / float(wall)
        if not math.isclose(float(reported), computed, rel_tol=1e-9, abs_tol=1e-9):
            raise CandidateRejected(f"candidate pass {index + 1} misreported throughput")
        if index > WARMUP_PASSES - 1:
            timed.append(computed)
    final = passes[-1]
    if payload.get("infer_wall_s") != final.get("infer_wall_s") or payload.get("tok_per_s") != final.get("tok_per_s"):
        raise CandidateRejected("candidate top-level timing does not match its steady pass")
    vectors = load_candidate_vectors(vector_path, expected_ids)
    metrics = ranker.evaluate(vectors)
    if metrics["mean_cosine"] < MIN_MEAN_COSINE or metrics["worst_decile_top10_rank_overlap"] < MIN_WORST_DECILE_OVERLAP:
        raise CandidateRejected(
            "embedding parity gate failed: "
            f"mean cosine {metrics['mean_cosine']:.9f}, "
            f"worst-decile overlap {metrics['worst_decile_top10_rank_overlap']:.5f}"
        )
    return {"samples": timed, "metrics": metrics}


def validate_vulkan_process_result(
    payload: Mapping[str, Any],
    vector_path: Path,
    expected_ids: Sequence[str],
    ranker: RankComputer,
) -> Dict[str, Any]:
    if payload.get("real_tokens") != EXPECTED_REAL_TOKENS:
        raise CandidateRejected("candidate changed the pinned real-token count")
    if payload.get("shape_policy") != "bucketed" or payload.get("bucket_policy_version") != BUCKET_POLICY:
        raise CandidateRejected("candidate changed the pinned bucket policy")
    observed_items = payload.get("items")
    if observed_items != EXPECTED_ROWS:
        raise CandidateRejected(
            f"candidate changed the pinned corpus size: expected {EXPECTED_ROWS}, got {observed_items}"
        )
    passes = payload.get("passes")
    if not isinstance(passes, list) or len(passes) != PASSES_PER_PROCESS:
        raise CandidateRejected(f"candidate did not emit exactly {PASSES_PER_PROCESS} process passes")
    timed: List[float] = []
    for index, item in enumerate(passes):
        if not isinstance(item, dict):
            raise CandidateRejected(f"candidate pass {index + 1} is not an object")
        expected_label = "first" if index == 0 else "steady" if index == PASSES_PER_PROCESS - 1 else "warm"
        if item.get("label") != expected_label:
            raise CandidateRejected(f"candidate pass {index + 1} has the wrong warmup label")
        if item.get("items") != EXPECTED_ROWS or item.get("input_tokens") != EXPECTED_REAL_TOKENS:
            raise CandidateRejected(f"candidate pass {index + 1} changed pinned input accounting")
        wall = item.get("infer_wall_s")
        reported = item.get("tok_per_s")
        if not finite_number(wall) or float(wall) <= 0.0 or not finite_number(reported):
            raise CandidateRejected(f"candidate pass {index + 1} has invalid timing")
        computed = EXPECTED_REAL_TOKENS / float(wall)
        if not math.isclose(float(reported), computed, rel_tol=1e-9, abs_tol=1e-9):
            raise CandidateRejected(f"candidate pass {index + 1} misreported throughput")
        if index >= WARMUP_PASSES:
            timed.append(computed)
    vectors = load_candidate_vectors(vector_path, expected_ids)
    metrics = ranker.evaluate(vectors)
    if metrics["mean_cosine"] < MIN_MEAN_COSINE or metrics["worst_decile_top10_rank_overlap"] < MIN_WORST_DECILE_OVERLAP:
        raise CandidateRejected(
            "embedding parity gate failed: "
            f"mean cosine {metrics['mean_cosine']:.9f}, "
            f"worst-decile overlap {metrics['worst_decile_top10_rank_overlap']:.5f}"
        )
    return {"samples": timed, "metrics": metrics}


def upload_remote_fixture(
    source: Path,
    remote_path: str,
    expected_sha256: str,
    log_root: Path,
    label: str,
) -> None:
    verify_utf8_no_bom(source, label)
    scp_to(source, remote_path, log_root / f"{label}-upload.log")
    output = remote_stdout(
        f"certutil -hashfile {windows_quote(remote_path)} SHA256",
        log_root / f"{label}-remote-hash.log",
    )
    hashes = re.findall(r"(?i)\b[0-9a-f]{64}\b", output)
    if expected_sha256.lower() not in {value.lower() for value in hashes}:
        raise HarnessError(f"remote {label} digest mismatch: {output!r}")
    if run_remote(f"attrib +R {windows_quote(remote_path)}", log_root / f"{label}-readonly.log") != 0:
        raise HarnessError(f"remote {label} could not be made read-only")


def remote_candidate_command(
    binary: str,
    model: str,
    corpus: str,
    output: str,
    vectors: str,
    workspace: str,
    passes: int,
    label: str,
    extra: Sequence[str] = (),
) -> str:
    args = [
        remote_env_prefix() + windows_quote(binary),
        "--model", windows_quote(model),
        "--tokenizer", windows_quote(model + "\\tokenizer.json"),
        "--corpus", windows_quote(corpus),
        "--out", windows_quote(output),
        "--vectors-out", windows_quote(vectors),
        "--device", "vulkan",
        "--dtype", "f16",
        "--vulkan-gemm", "cooperative",
        "--execution", "explicit",
        "--shapes", "bucketed",
        "--bucket-policy", str(BUCKET_POLICY),
        "--passes", str(passes),
        "--max-length", str(MAX_LENGTH),
        "--attention-units", str(ATTENTION_UNITS),
        "--model-label", windows_quote(label),
    ]
    del workspace  # The source checkout is already synchronized before this cell.
    args.extend(extra)
    return " ".join(args)


def run_remote_cell(
    command: str,
    log_path: Path,
    remote_output: str,
    remote_vectors: str,
    local_output: Path,
    local_vectors: Path,
) -> None:
    status = run_remote(command, log_path)
    if status != 0:
        raise CandidateRejected(f"Ally cell failed with status {status}: {runner_output(log_path)[-4096:]}")
    scp_from(remote_output, local_output, log_path.with_name(log_path.name + ".output-download"))
    scp_from(remote_vectors, local_vectors, log_path.with_name(log_path.name + ".vectors-download"))
    try:
        payload = json.loads(local_output.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateRejected(f"Ally cell did not produce valid result JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CandidateRejected("Ally cell result was not a JSON object")


def frame_bytes(value: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    return struct.pack("<I", len(encoded)) + encoded


def read_exact(stream: Any, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            # Include what actually arrived: a text preamble here means some
            # layer wrote non-frame bytes onto the protocol channel, while an
            # empty capture means the process died before writing at all.
            raise CandidateRejected(
                "candidate stdio process closed before a complete protocol frame; "
                f"partial bytes ({len(chunks)} of {size}): {bytes(chunks[:200])!r}"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(stream: Any) -> Tuple[bytes, Dict[str, Any]]:
    length = struct.unpack("<I", read_exact(stream, 4))[0]
    if length > 256 * 1024 * 1024:
        raise CandidateRejected("candidate stdio frame exceeded the protocol limit")
    raw = read_exact(stream, length)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateRejected(f"candidate stdio response was not JSON: {error}") from error
    if not isinstance(value, dict):
        raise CandidateRejected("candidate stdio response was not an object")
    return raw, value


def run_determinism_probe(
    runner: Path,
    binary: Path,
    model: Path,
    package_cache: Path,
    corpus_rows: Sequence[Mapping[str, Any]],
    environment: Sequence[str],
    log_path: Path,
) -> None:
    texts = [str(row["text"]) for row in corpus_rows[:8]]
    command = [
        *environment,
        str(binary),
        "--model",
        str(model),
        "--tokenizer",
        str(model / "tokenizer.json"),
        "--device",
        "metal",
        "--dtype",
        "f16",
        "--execution",
        "explicit",
        "--shapes",
        "bucketed",
        "--bucket-policy",
        str(BUCKET_POLICY),
        "--max-length",
        str(MAX_LENGTH),
        "--attention-units",
        str(ATTENTION_UNITS),
        "--package-cache",
        str(package_cache),
        "--serve-stdio",
        "--model-label",
        "gte-modernbert-f16-determinism-probe",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as errors:
        process = subprocess.Popen(
            [str(runner), *command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            if process.stdin is None or process.stdout is None:
                raise CandidateRejected("candidate stdio process did not expose pipes")
            _, ready = read_frame(process.stdout)
            if ready.get("kind") != "ready" or ready.get("protocol_version") != 1:
                raise CandidateRejected("candidate stdio process did not send protocol v1 Ready")
            shape = {"batch": 8, "seq": 512}
            request = {
                "kind": "embed",
                "texts": texts,
                "max_length": MAX_LENGTH,
                "shape_policy": "bucketed",
                "shape": shape,
            }
            process.stdin.write(frame_bytes(request))
            process.stdin.flush()
            first_raw, first = read_frame(process.stdout)
            process.stdin.write(frame_bytes(request))
            process.stdin.flush()
            second_raw, second = read_frame(process.stdout)
            if first.get("kind") != "embedding" or second.get("kind") != "embedding":
                raise CandidateRejected("candidate determinism probe did not return embedding frames")
            first_vectors = json.dumps(first.get("vectors"), separators=(",", ":"), allow_nan=False).encode()
            second_vectors = json.dumps(second.get("vectors"), separators=(",", ":"), allow_nan=False).encode()
            if first_vectors != second_vectors:
                raise CandidateRejected("same-process embedding vectors were not byte-identical")
            process.stdin.write(frame_bytes({"kind": "shutdown"}))
            process.stdin.flush()
            read_frame(process.stdout)
            process.stdin.close()
            process.wait(timeout=60)
            if process.returncode != 0:
                raise CandidateRejected(f"candidate determinism process exited {process.returncode}")
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as error:
            process.kill()
            process.wait()
            raise CandidateRejected(f"candidate determinism probe failed: {error}") from error
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def run_optional_reference_cells(
    runner: Path,
    staged_workspace: Path,
    model: Path,
    corpus: Path,
    output_root: Path,
    environment: Sequence[str],
    log_root: Path,
) -> Dict[str, Any]:
    references: Dict[str, Any] = {}
    mlx_python = Path(
        os.environ.get("MLX_PYTHON", "/tmp/synapse-mlx-minilm-venv/bin/python")
    ).expanduser()
    mlx_script = staged_workspace / "bench/lanes/mlx-minilm/main.py"
    if not mlx_python.is_file() or not os.access(str(mlx_python), os.X_OK):
        references["mlx_python"] = {
            "status": "skipped",
            "reason": f"MLX_PYTHON is absent or not executable: {mlx_python}",
        }
    elif not mlx_script.is_file():
        references["mlx_python"] = {
            "status": "skipped",
            "reason": f"MLX reference script is absent: {mlx_script}",
        }
    else:
        output = output_root / "mlx-python-reference.json"
        vectors = output_root / "mlx-python-reference-vectors.jsonl"
        log = log_root / "mlx-python-reference.log"
        status = run_through_runner(
            runner,
            [
                *environment,
                str(mlx_python),
                str(mlx_script),
                "--model",
                str(model),
                "--corpus",
                str(corpus),
                "--out",
                str(output),
                "--vectors-out",
                str(vectors),
                "--model-label",
                "gte-modernbert-mlx-python-reference",
            ],
            log,
        )
        if status != 0:
            references["mlx_python"] = {
                "status": "skipped",
                "reason": f"MLX reference cell exited {status}: {runner_output(log)[-2048:]}",
            }
        else:
            try:
                payload = json.loads(output.read_text())
                if not isinstance(payload, dict) or not finite_number(payload.get("tok_per_s")):
                    raise ValueError("reference output lacks a finite tok_per_s")
                references["mlx_python"] = {
                    "status": "measured",
                    "tok_per_s": float(payload["tok_per_s"]),
                    "infer_wall_s": payload.get("infer_wall_s"),
                    "items": payload.get("items"),
                    "input_tokens": payload.get("input_tokens"),
                }
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                references["mlx_python"] = {
                    "status": "skipped",
                    "reason": f"MLX reference output was invalid: {error}",
                }

    llama_binary = os.environ.get("SYNAPSE_CAMPAIGN_LLAMA_LANE")
    llama_server = os.environ.get("LLAMA_SERVER_BIN")
    llama_model = os.environ.get("SYNAPSE_CAMPAIGN_GTE_GGUF")
    if not llama_binary or not llama_server or not llama_model:
        references["llama_server"] = {
            "status": "skipped",
            "reason": "llama-server reference is not configured (optional cell)",
        }
    else:
        output = output_root / "llama-server-reference.json"
        vectors = output_root / "llama-server-reference-vectors.jsonl"
        log = log_root / "llama-server-reference.log"
        status = run_through_runner(
            runner,
            [
                *environment,
                llama_binary,
                "embed",
                "--server-binary",
                llama_server,
                "--model",
                llama_model,
                "--tokenizer",
                str(model / "tokenizer.json"),
                "--corpus",
                str(corpus),
                "--out",
                str(output),
                "--vectors-out",
                str(vectors),
                "--pooling",
                "cls",
                "--model-label",
                "gte-modernbert-llama-server-reference",
            ],
            log,
        )
        if status != 0:
            references["llama_server"] = {
                "status": "skipped",
                "reason": f"llama-server reference cell exited {status}: {runner_output(log)[-2048:]}",
            }
        else:
            try:
                payload = json.loads(output.read_text())
                if not isinstance(payload, dict) or not finite_number(payload.get("tok_per_s")):
                    raise ValueError("reference output lacks a finite tok_per_s")
                references["llama_server"] = {
                    "status": "measured",
                    "tok_per_s": float(payload["tok_per_s"]),
                    "infer_wall_s": payload.get("infer_wall_s"),
                    "items": payload.get("items"),
                    "input_tokens": payload.get("input_tokens"),
                }
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                references["llama_server"] = {
                    "status": "skipped",
                    "reason": f"llama-server reference output was invalid: {error}",
                }
    return references

def run_harness(workspace_arg: str, runner_arg: str, result_arg: str) -> int:
    workspace = Path(workspace_arg).expanduser().resolve()
    runner = Path(runner_arg).expanduser().resolve()
    result_path = Path(result_arg).expanduser().resolve()
    if not workspace.is_dir():
        raise HarnessError(f"candidate workspace is not a directory: {workspace}")
    if not runner.is_file() or not os.access(str(runner), os.X_OK):
        raise HarnessError(f"candidate runner is not executable: {runner}")
    baseline = configured_constants()
    writer = ResultWriter(result_path)
    temp_root = Path(tempfile.mkdtemp(prefix="synapse-vulkan-embed-campaign-", dir="/tmp"))
    temp_root.chmod(0o777)
    remote_info: Dict[str, str] = {}
    remote_model = os.environ.get("SYNAPSE_CAMPAIGN_REMOTE_MODEL", REMOTE_MODEL)
    baseline_label = "pending (run the idle-gated Ally baseline command)" if baseline is None else f"{baseline:.6f} tok/s"
    baseline_note = (
        f"gte-modernbert-base f16 Vulkan cooperative embedding baseline: {baseline_label}. "
        f"Protocol: {PROCESS_RUNS} paired fresh-process runs, {PASSES_PER_PROCESS - WARMUP_PASSES} "
        "timed steady passes after one discarded warmup; worse-of-two per run. "
        "The controller uses the explicitly configured remote target and refuses a busy benchmark tenant."
    )
    writer.write(initial_payload(baseline_note))
    try:
        corpus_path, _reference_path, corpus_rows, reference_vectors = extract_fixtures(
            workspace, temp_root / "fixtures"
        )
        verify_utf8_no_bom(corpus_path, "embedding corpus fixture")
        fixture_ids = [str(row["id"]) for row in corpus_rows]
        ranker = RankComputer(reference_vectors)

        # The first remote action is reachability plus the tenant check. No build or
        # benchmark command is admitted until both the box and its process list are idle.
        user_profile = remote_idle_probe(temp_root)
        remote_model_components = remote_model_hashes(remote_model, temp_root)
        remote_info = sync_candidate_to_ally(workspace, temp_root, user_profile, temp_root)
        remote_workspace = remote_info["workspace"]
        remote_fixture = remote_info["fixtures"]
        remote_root = remote_info["root"]
        remote_corpus = remote_fixture + "\\" + CORPUS_NAME
        remote_determinism = remote_fixture + "\\determinism-corpus.jsonl"
        upload_remote_fixture(corpus_path, remote_corpus, CORPUS_SHA256, temp_root, "embedding-corpus")

        # A no-BOM, eight-row fixture keeps the repeated-input determinism probe
        # short while using the same exact text bytes as the measured corpus.
        determinism_bytes = b"".join(
            (json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
            for row in corpus_rows[:8]
        )
        determinism_path = temp_root / "determinism-corpus.jsonl"
        determinism_path.write_bytes(determinism_bytes)
        determinism_digest = hashlib.sha256(determinism_bytes).hexdigest()
        upload_remote_fixture(
            determinism_path,
            remote_determinism,
            determinism_digest,
            temp_root,
            "determinism-corpus",
        )

        manifest = remote_workspace + "\\bench\\spikes\\unified-rt\\Cargo.toml"
        remote_cargo_target = os.environ.get(
            "SYNAPSE_CAMPAIGN_REMOTE_CARGO_TARGET", REMOTE_CARGO_TARGET
        )
        # Pass the target directory as a Cargo argument. OpenSSH's Windows
        # command bridge executes the first `cmd /c` token separately when a
        # compound `set ...&&cargo ...` command is sent as argv pieces, which
        # otherwise silently falls back to the workspace target directory.
        build_command = (
            f"cd /d {windows_quote(remote_workspace)}&&"
            f"{windows_quote(os.environ.get('SYNAPSE_CAMPAIGN_REMOTE_CARGO', REMOTE_CARGO))} "
            f"build --target-dir {windows_quote(remote_cargo_target)} --locked --offline --release --features vulkan "
            f"--manifest-path {windows_quote(manifest)} --bin spike-unified-rt"
        )
        if run_remote(build_command, temp_root / "ally-build.log") != 0:
            raise CandidateRejected(
                f"Ally Vulkan release build failed: {runner_output(temp_root / 'ally-build.log')[-4096:]}"
            )
        binary = remote_cargo_target + "\\release\\spike-unified-rt.exe"
        if run_remote(f"if not exist {windows_quote(binary)} exit /b 1", temp_root / "ally-binary.log") != 0:
            raise CandidateRejected("Ally Vulkan release binary was not produced")

        # Determinism is checked before timed cells and across two fresh remote
        # processes. The comparison is the exact emitted JSONL byte stream.
        deterministic_vectors: List[bytes] = []
        for repeat in (1, 2):
            output_remote = remote_root + f"\\determinism-{repeat}.json"
            vectors_remote = remote_root + f"\\determinism-{repeat}-vectors.jsonl"
            output_local = temp_root / f"determinism-{repeat}.json"
            vectors_local = temp_root / f"determinism-{repeat}-vectors.jsonl"
            command = remote_candidate_command(
                binary,
                remote_model,
                remote_determinism,
                output_remote,
                vectors_remote,
                remote_workspace,
                1,
                "gte-modernbert-vulkan-determinism",
                extra=("--limit", "8"),
            )
            run_remote_cell(
                command,
                temp_root / f"determinism-{repeat}.log",
                output_remote,
                vectors_remote,
                output_local,
                vectors_local,
            )
            deterministic_vectors.append(vectors_local.read_bytes())
        if deterministic_vectors[0] != deterministic_vectors[1]:
            raise CandidateRejected("repeated Ally embedding inputs were not byte-identical")

        all_samples: List[float] = []
        paired_runs: List[Dict[str, Any]] = []
        parity_metrics: List[Dict[str, Any]] = []
        for run_index in range(PROCESS_RUNS):
            process_samples: List[List[float]] = []
            for repeat in range(FRESH_PROCESSES_PER_RUN):
                remote_idle_probe(temp_root / f"idle-run-{run_index + 1}-{repeat + 1}")
                stem = f"run-{run_index + 1}-process-{repeat + 1}"
                output_remote = remote_root + f"\\{stem}.json"
                vectors_remote = remote_root + f"\\{stem}-vectors.jsonl"
                output_local = temp_root / f"{stem}.json"
                vectors_local = temp_root / f"{stem}-vectors.jsonl"
                command = remote_candidate_command(
                    binary,
                    remote_model,
                    remote_corpus,
                    output_remote,
                    vectors_remote,
                    remote_workspace,
                    PASSES_PER_PROCESS,
                    "gte-modernbert-vulkan-embed-campaign",
                )
                log = temp_root / f"{stem}.log"
                run_remote_cell(command, log, output_remote, vectors_remote, output_local, vectors_local)
                try:
                    payload = json.loads(output_local.read_text())
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CandidateRejected(f"{stem} result was not valid JSON: {error}") from error
                if not isinstance(payload, dict):
                    raise CandidateRejected(f"{stem} result was not a JSON object")
                validated = validate_vulkan_process_result(payload, vectors_local, fixture_ids, ranker)
                process_samples.append([float(value) for value in validated["samples"]])
                parity_metrics.append(validated["metrics"])
            worse = [
                min(process_samples[0][index], process_samples[1][index])
                for index in range(PASSES_PER_PROCESS - WARMUP_PASSES)
            ]
            all_samples.extend(worse)
            paired_runs.append(
                {
                    "run": run_index + 1,
                    "process_samples": process_samples,
                    "worse_of_two": worse,
                }
            )

        median = float(statistics.median(all_samples))
        mean_cosine = min(metric["mean_cosine"] for metric in parity_metrics)
        worst_overlap = min(metric["worst_decile_top10_rank_overlap"] for metric in parity_metrics)
        writer.write(
            {
                "gate_passed": True,
                "parity_passed": True,
                "determinism_passed": True,
                "samples": all_samples,
                "paired_runs": paired_runs,
                "median_tok_s": median,
                "baseline_note": (
                    f"{baseline_note} mean cosine floor={mean_cosine:.9f}; "
                    f"worst-decile top-10 overlap floor={worst_overlap:.5f}."
                ),
                "workspace_commit": remote_info["commit"],
                "model_components": remote_model_components,
                "fixture": {
                    "corpus_sha256": CORPUS_SHA256,
                    "reference_vectors_sha256": REFERENCE_VECTORS_SHA256,
                    "rows": EXPECTED_ROWS,
                    "dimensions": EXPECTED_DIMENSIONS,
                    "real_tokens": EXPECTED_REAL_TOKENS,
                    "rank_queries": parity_metrics[-1]["rank_queries"],
                    "determinism_corpus_sha256": determinism_digest,
                },
                "remote": {
                    "target": configured_remote_target(),
                    "workspace": remote_workspace,
                    "user_profile": user_profile,
                    "cargo_target_dir": os.environ.get(
                        "SYNAPSE_CAMPAIGN_REMOTE_CARGO_TARGET", REMOTE_CARGO_TARGET
                    ),
                    "worktree_status": remote_info["worktree_status"],
                },
            }
        )
        return 0
    except CandidateRejected as error:
        preserve_failure_scene(temp_root, result_path, remote_info.get("root", ""))
        writer.write({**initial_payload(f"{baseline_note} Candidate rejected: {error}"), "remote": remote_info})
        print(f"Vulkan embedding campaign candidate rejected: {error}", file=sys.stderr)
        return 3
    except HarnessError as error:
        preserve_failure_scene(temp_root, result_path, remote_info.get("root", ""))
        writer.write({**initial_payload(f"{baseline_note} Harness refused to run: {error}"), "remote": remote_info})
        print(f"Vulkan embedding campaign harness refused to run: {error}", file=sys.stderr)
        return 1
    finally:
        if remote_info.get("root"):
            try:
                run_remote(
                    f"if exist {windows_quote(remote_info['root'])} rmdir /s /q {windows_quote(remote_info['root'])}",
                    temp_root / "ally-cleanup.log",
                )
            except OSError:
                pass
        shutil.rmtree(temp_root, ignore_errors=True)
        writer.close()


def self_test() -> int:
    ac = parse_power_state(
        "Now drawing from 'AC Power'\n -InternalBattery-0\t42%; charging; present: true"
    )
    assert ac["power_source"] == "AC Power"
    battery = parse_power_state(
        "Now drawing from 'Battery Power'\n -InternalBattery-0\t19%; discharging; present: true"
    )
    assert battery["battery_percent"] == 19
    try:
        enforce_power_preflight(battery)
    except HarnessError:
        pass
    else:
        raise AssertionError("low battery preflight did not reject")
    try:
        parse_power_state("Battery Power\n19%")
    except HarnessError:
        pass
    else:
        raise AssertionError("malformed power output did not reject")

    reference = []
    candidate = []
    for index in range(12):
        row = [0.0] * EXPECTED_DIMENSIONS
        row[index] = 1.0
        reference.append(row)
        candidate.append(list(row))
    ranker = RankComputer(reference)
    metrics = ranker.evaluate(candidate)
    assert metrics["mean_cosine"] == 1.0
    assert metrics["worst_decile_top10_rank_overlap"] == 1.0
    assert metrics["parity_passed"] is True
    configured_fixture_dir = os.environ.get("SYNAPSE_CAMPAIGN_FIXTURES")
    if configured_fixture_dir:
        fixture_test_root = Path(tempfile.mkdtemp(prefix="metal-embed-fixture-test-"))
        try:
            _, reference_path, rows, vectors = extract_fixtures(Path.cwd(), fixture_test_root)
            assert len(rows) == EXPECTED_ROWS
            assert len(vectors) == EXPECTED_ROWS
            assert reference_path.is_file()
            fixture_metrics = RankComputer(vectors).evaluate(vectors)
            assert fixture_metrics["mean_cosine"] > 0.999999
            assert fixture_metrics["worst_decile_top10_rank_overlap"] == 1.0
        finally:
            shutil.rmtree(fixture_test_root, ignore_errors=True)
    print("vulkan-embed-harness self-test passed")
    return 0


def main() -> int:
    arguments = sys.argv[1:]
    if arguments == ["--self-test"]:
        return self_test()
    if len(arguments) != 3:
        print("usage: vulkan-embed-harness.sh {workspace} {candidate_runner} {result}", file=sys.stderr)
        return 2
    try:
        return run_harness(arguments[0], arguments[1], arguments[2])
    except (HarnessError, OSError) as error:
        print(f"Vulkan embedding campaign harness failed before result initialization: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
