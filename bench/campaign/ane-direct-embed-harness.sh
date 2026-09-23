#!/usr/bin/env bash
set -euo pipefail

# Trusted controller for the gte-modernbert direct-API embedding campaign.
# The port computes its fp32 reference in the same process because reproducing
# that comparator outside Rust would create a second numerical contract.
exec /usr/bin/python3 - "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BASELINE_TOK_S = 9_222.454033809034
ROW_SET_SHA256 = "f4889a38df77b9940ce973c4d9b82857d0c401987ae8e77b5ca25e6062808c39"
MODEL_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
MODEL_COMPONENT_SHA256 = {
    "config.json": "8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed",
    "model.safetensors": "3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba",
}
BINDING_COMMIT = "ec54af9501d4bfd0cf3a4b162e59022dee2118cb"
PROBE_MAIN_SHA256 = "6fb9f3e6666501d3d721c538d13634ea48e38dd4c7baa80d07ea41e975118a1a"
CARGO_TOML_SHA256 = "fb8879f32857ef99f7945a0ba728d9cd711d1455f513d7347a1a4a51ed7ddd47"
CARGO_LOCK_SHA256 = "c1d21e074012193f57b8c5bde32e36d564fa77bd79ebb15080ac6e062086b89d"
PROTECTED_RUST_SHA256 = "919afda47a643e3b49cc421c74325b01b4b6b213b08b37405663e1bd0b79dd0d"
EXPECTED_ANE_DEPENDENCY = (
    'ane = { path = "../../../../../../OSS/siliconswarm-at-ensue-plugin/ane_kernel/crates/ane" }'
)
PROBE_DIR = Path("bench/spikes/ane-direct-probe")
FULL_MODEL = PROBE_DIR / "src/bin/modernbert_full.rs"
ROWS = PROBE_DIR / "rows.jsonl"
SEQUENCES = (512, 1024, 2048)
LAYERS_PER_EXECUTABLE = 1
WARM_REPETITIONS = 7
EXPECTED_ROWS = 8
EXPECTED_DIMENSIONS = 768
MIN_COSINE = 0.999
DEFAULT_MAX_LOAD_1M = 16.0


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
            os.unlink(path)
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
        current = os.lstat(self.path)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self.identity:
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
        "aggregate_tok_s": None,
        "min_cosine": None,
        "sequence_1024_tok_s": None,
        "sequence_2048_tok_s": None,
        "baseline_note": note,
        "workspace_commit": "",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise HarnessError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_average_1m() -> float:
    try:
        value = os.getloadavg()[0]
    except (AttributeError, OSError) as error:
        raise HarnessError(f"LOAD_PREFLIGHT_FAILED: cannot read one-minute load average: {error}") from error
    if not math.isfinite(value) or value < 0.0:
        raise HarnessError("LOAD_PREFLIGHT_FAILED: one-minute load average is invalid")
    return value


def configured_max_load() -> float:
    raw = os.environ.get("SYNAPSE_CAMPAIGN_MAX_LOAD_1M", str(DEFAULT_MAX_LOAD_1M))
    try:
        value = float(raw)
    except ValueError as error:
        raise HarnessError("configured one-minute load threshold is not numeric") from error
    if not math.isfinite(value) or value <= 0.0:
        raise HarnessError("configured one-minute load threshold must be finite and positive")
    return value


# How long a load check waits for the one-minute average to come back under the
# threshold before refusing. The rig is shared, and a single burst from another
# tenant can lift the one-minute average past the ceiling for a minute or two.
# A one-shot check turned that burst into a failed campaign after the whole
# planning phase had already run; waiting a bounded time turns it into a delay.
# The measurement itself is never taken above the threshold.
LOAD_WAIT_SECONDS = 300.0
LOAD_POLL_SECONDS = 15.0


def enforce_load_preflight(label: str, maximum: float, wait_seconds: float = LOAD_WAIT_SECONDS) -> float:
    deadline = time.monotonic() + wait_seconds
    while True:
        observed = load_average_1m()
        if observed <= maximum:
            return observed
        if time.monotonic() >= deadline:
            raise HarnessError(
                f"LOAD_PREFLIGHT_REFUSED: {label} one-minute load {observed:.2f} exceeds configured threshold "
                f"{maximum:.2f} after waiting {wait_seconds:.0f}s for it to fall"
            )
        time.sleep(min(LOAD_POLL_SECONDS, max(0.0, deadline - time.monotonic())))


def configured_constants() -> None:
    baseline = os.environ.get("SYNAPSE_CAMPAIGN_BASELINE_TOK_S", str(BASELINE_TOK_S))
    if baseline != "pending":
        try:
            configured = float(baseline)
        except ValueError as error:
            raise HarnessError("configured baseline is not numeric") from error
        if configured != BASELINE_TOK_S:
            raise HarnessError("campaign registration baseline disagrees with the pinned harness")
    expected = {
        "SYNAPSE_CAMPAIGN_ROWS_SHA256": ROW_SET_SHA256,
        "SYNAPSE_CAMPAIGN_MODEL_SHA256": MODEL_COMPONENT_SHA256["model.safetensors"],
        "SYNAPSE_CAMPAIGN_ANE_BINDING_COMMIT": BINDING_COMMIT,
    }
    for name, pinned in expected.items():
        if os.environ.get(name, pinned) != pinned:
            raise HarnessError(f"campaign registration {name} disagrees with the pinned harness")
    # The load ceiling is a scoring constant, not an operational knob, so it is
    # validated with the digests rather than left to the environment. It was not,
    # and a registration setting 24 silently beat the documented 16 while the
    # emitted note went on asserting 16 -- measured consequence: the SAME
    # unmodified tree scored 7619 / 9049 / 7586 tok/s across three passes at
    # ambient load 27-30, a 19.3% spread that fails a 3% win threshold against
    # its own baseline in 3 of 3 runs. Above this ceiling the harness scores
    # ambient load, not the candidate.
    configured_load = os.environ.get(
        "SYNAPSE_CAMPAIGN_MAX_LOAD_1M", str(DEFAULT_MAX_LOAD_1M)
    )
    try:
        configured_load_value = float(configured_load)
    except ValueError as error:
        raise HarnessError("configured load threshold is not numeric") from error
    if configured_load_value != DEFAULT_MAX_LOAD_1M:
        raise HarnessError(
            "campaign registration SYNAPSE_CAMPAIGN_MAX_LOAD_1M disagrees with the "
            f"pinned harness ({configured_load_value:g} vs {DEFAULT_MAX_LOAD_1M:g})"
        )


def verify_regular_file(path: Path, label: str) -> None:
    # Only ENOENT means missing. This harness runs as the controller and inspects
    # a workspace the driver hands to the candidate, so EACCES is a real outcome
    # here: the file exists and the controller cannot see it. Reporting that as
    # "missing" sends the operator looking for a file that is present.
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise HarnessError(f"{label} is missing: {path}") from error
    except OSError as error:
        raise HarnessError(
            f"{label} cannot be inspected by the controller: {path}: {error}"
        ) from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise HarnessError(f"{label} is not a regular file: {path}")


def extract_braced_symbol(source: str, pattern: str, label: str) -> str:
    match = re.search(pattern, source, flags=re.MULTILINE)
    if match is None:
        raise CandidateRejected(f"protected CPU comparator symbol is missing: {label}")
    opening = source.find("{", match.start())
    if opening < 0:
        raise CandidateRejected(f"protected CPU comparator symbol has no body: {label}")
    depth = 0
    state = "code"
    index = opening
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == '"':
                state = "string"
            elif char == "/" and following == "/":
                state = "line-comment"
                index += 1
            elif char == "/" and following == "*":
                state = "block-comment"
                index += 1
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[match.start() : index + 1]
        elif state == "string":
            if char == "\\":
                index += 1
            elif char == '"':
                state = "code"
        elif state == "line-comment":
            if char == "\n":
                state = "code"
        elif state == "block-comment" and char == "*" and following == "/":
            state = "code"
            index += 1
        index += 1
    raise CandidateRejected(f"protected CPU comparator symbol is unterminated: {label}")


def protected_rust_digest(path: Path) -> str:
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise CandidateRejected(f"cannot read protected CPU comparator source: {error}") from error
    gate_lines = re.findall(r"^const GATE: f32 = .*;$", source, flags=re.MULTILINE)
    if gate_lines != ["const GATE: f32 = 0.999;"]:
        raise CandidateRejected("protected gate threshold changed; expected minimum cosine 0.999")
    patterns = [
        (r"^struct InputRow\b", "InputRow"),
        (r"^impl InputRow\b", "InputRow.ids"),
        (r"^struct RowMetric\b", "RowMetric"),
        (r"^struct Report\b", "Report"),
        (r"^fn tensor_values\b", "tensor_values"),
        (r"^fn load_linear\b", "load_linear"),
        (r"^fn load_vector\b", "load_vector"),
        (r"^fn load_model\b", "load_model"),
        (r"^fn l2_normalize\b", "l2_normalize"),
        (r"^fn matmul\b", "matmul"),
        (r"^fn linear_cpu\b", "linear_cpu"),
        (r"^fn layer_norm_cpu\b", "layer_norm_cpu"),
        (r"^fn apply_rope_cpu\b", "apply_rope_cpu"),
        (r"^fn softmax_cpu\b", "softmax_cpu"),
        (r"^fn attention_cpu\b", "attention_cpu"),
        (r"^fn cpu_reference\b", "cpu_reference"),
        (r"^fn cosine\b", "cosine"),
        (r"^fn max_abs\b", "max_abs"),
        (r"^fn byte_identical\b", "byte_identical"),
        (r"^fn main\(\) -> Result<\(\)>", "main"),
    ]
    pieces = [gate_lines[0]]
    pieces.extend(extract_braced_symbol(source, pattern, label) for pattern, label in patterns)
    return hashlib.sha256("\n\0\n".join(pieces).encode()).hexdigest()


def verify_candidate_contract(workspace: Path) -> None:
    probe = workspace / PROBE_DIR
    rows = workspace / ROWS
    model_source = workspace / FULL_MODEL
    files = [
        (rows, "protected row set", ROW_SET_SHA256),
        (probe / "src/main.rs", "private-API identity probe", PROBE_MAIN_SHA256),
        (probe / "Cargo.toml", "standalone probe manifest", CARGO_TOML_SHA256),
        (probe / "Cargo.lock", "standalone probe lockfile", CARGO_LOCK_SHA256),
    ]
    for path, label, expected in files:
        verify_regular_file(path, label)
        actual = sha256_file(path)
        if actual != expected:
            raise CandidateRejected(f"{label} digest mismatch: expected {expected}, got {actual}")
    verify_regular_file(model_source, "ModernBERT candidate source")
    actual_protected = protected_rust_digest(model_source)
    if actual_protected != PROTECTED_RUST_SHA256:
        raise CandidateRejected(
            "protected CPU comparator or gate wiring changed: "
            f"expected {PROTECTED_RUST_SHA256}, got {actual_protected}"
        )
    manifest = (probe / "Cargo.toml").read_text()
    if manifest.count(EXPECTED_ANE_DEPENDENCY) != 1:
        raise CandidateRejected("ANE path dependency changed; refusing an alternate binding location")


def run_command(argv: Sequence[str], log_path: Path, cwd: Optional[Path] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        completed = subprocess.run(
            list(argv), cwd=cwd, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    return completed.returncode


def runner_output(path: Path) -> str:
    # An empty string must keep meaning "the runner wrote nothing", because a
    # silent runner is a diagnosis in its own right (the step never became a
    # process). An unreadable log therefore reports itself instead of collapsing
    # into the same empty value.
    try:
        return path.read_text(errors="replace").strip()
    except FileNotFoundError:
        return ""
    except OSError as error:
        return f"<runner output unreadable: {path}: {error}>"


def run_through_runner(runner: Path, argv: Sequence[str], log_path: Path) -> int:
    return run_command([str(runner), *argv], log_path)


def runner_stdout(runner: Path, argv: Sequence[str], log_path: Path) -> Tuple[int, str, str]:
    # For commands whose stdout is compared exactly (a commit SHA). run_command
    # merges stderr into the same log, and anything the runner's shells print on
    # stderr (bash warns "shell-init: error retrieving current directory" when it
    # starts in a directory the candidate cannot read) would then sit beside the
    # value and fail an equality check on a correct answer. stdout is read on its
    # own; stderr is returned separately so a refusal can still show it.
    stdout_path = log_path.with_suffix(".stdout")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, log_path.open("ab") as stderr:
        completed = subprocess.run(
            [str(runner), *argv], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=False
        )
    return completed.returncode, runner_output(stdout_path), runner_output(log_path)


def runner_failure(status: int, log_path: Path) -> str:
    # The exit status is reported with the output because the output alone can
    # be empty for two different reasons: the step never became a process, or
    # the runner lost its stderr. The status is the only part of a failed runner
    # call that cannot go missing, so it is never dropped.
    output = runner_output(log_path)
    return f"exit status {status}; output: {output[-4096:] if output else '<runner wrote nothing>'}"


def git_output(repo: Path, args: Sequence[str], label: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        raise HarnessError(f"{label}: git {' '.join(args)} failed: {detail or '<empty>'}")
    return completed.stdout.strip()


def verify_binding_source(binding: Path) -> None:
    if not binding.is_dir():
        raise HarnessError(f"pinned ANE binding clone is missing: {binding}")
    head = git_output(binding, ["rev-parse", "HEAD"], "ANE binding clone")
    if head != BINDING_COMMIT:
        raise HarnessError(
            f"ANE binding clone commit mismatch: expected {BINDING_COMMIT}, got {head or '<empty>'}"
        )
    status = git_output(
        binding,
        ["status", "--porcelain", "--untracked-files=all", "--", "ane_kernel/crates/ane"],
        "ANE binding crate",
    )
    if status:
        raise HarnessError("ANE binding crate has local changes; refusing an unpinned dependency tree")


def copy_tree(runner: Path, source: Path, destination: Path, log_path: Path) -> None:
    status = run_through_runner(runner, ["/bin/cp", "-cR", str(source), str(destination)], log_path)
    if status != 0:
        if destination.exists() or destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        status = run_through_runner(runner, ["/bin/cp", "-R", str(source), str(destination)], log_path)
    if status != 0:
        raise HarnessError(f"could not stage {source}: {runner_output(log_path)[-4096:]}")
    if not destination.is_dir():
        raise HarnessError(f"candidate runner staged an invalid directory: {destination}")


def stage_sources(
    workspace: Path, binding: Path, temp_root: Path, runner: Path
) -> Tuple[Path, Path, Path]:
    stage_root = temp_root / "build"
    workspace_parent = stage_root / "Projects/CortexKit"
    binding_parent = stage_root / "OSS"
    mkdir_log = temp_root / "stage-mkdir.log"
    status = run_through_runner(
        runner, ["/bin/mkdir", "-p", str(workspace_parent), str(binding_parent)], mkdir_log
    )
    if status != 0:
        raise HarnessError(
            f"candidate runner could not create staging directories: {runner_failure(status, mkdir_log)}"
        )
    staged_workspace = workspace_parent / "synapse"
    staged_binding = binding_parent / "siliconswarm-at-ensue-plugin"
    copy_tree(runner, workspace, staged_workspace, temp_root / "workspace-copy.log")
    copy_tree(runner, binding, staged_binding, temp_root / "binding-copy.log")
    # The copies are made as the candidate, so they carry the candidate's modes
    # (the workspace root is 700) and none of the controller read grant, which
    # the driver writes only on the candidate's own workspace directories. The
    # contract check and the dependency-path check below read this staged tree as
    # the controller, so the candidate opens it for reading first. No candidate
    # code has run yet, and a later chmod back would make those checks refuse,
    # not pass.
    stage_chmod_log = temp_root / "stage-chmod.log"
    status = run_through_runner(runner, ["/bin/chmod", "-R", "a+rX", str(stage_root)], stage_chmod_log)
    if status != 0:
        raise HarnessError(
            f"candidate runner could not open the staged tree for reading: {runner_failure(status, stage_chmod_log)}"
        )
    output_root = temp_root / "candidate-output"
    target = output_root / "target"
    output_mkdir_log = temp_root / "output-mkdir.log"
    status = run_through_runner(runner, ["/bin/mkdir", "-p", str(target)], output_mkdir_log)
    if status != 0:
        raise HarnessError(
            f"candidate runner could not create candidate output directories: {runner_failure(status, output_mkdir_log)}"
        )
    output_chmod_log = temp_root / "output-chmod.log"
    status = run_through_runner(
        runner, ["/bin/chmod", "777", str(output_root), str(target)], output_chmod_log
    )
    if status != 0:
        raise HarnessError(
            f"candidate output directories are not writable: {runner_failure(status, output_chmod_log)}"
        )
    verify_candidate_contract(staged_workspace)
    status, head, head_stderr = runner_stdout(
        runner,
        ["/usr/bin/git", "-c", f"safe.directory={staged_binding}", "-C", str(staged_binding), "rev-parse", "HEAD"],
        temp_root / "staged-binding-head.log",
    )
    if status != 0 or head != BINDING_COMMIT:
        raise HarnessError(
            f"staged ANE binding clone is not at pinned commit {BINDING_COMMIT}: "
            f"exit status {status}; stdout: {head or '<empty>'}; stderr: {head_stderr[-2048:] or '<empty>'}"
        )
    resolved_dependency = (
        staged_workspace / PROBE_DIR / "../../../../../../OSS/siliconswarm-at-ensue-plugin/ane_kernel/crates/ane"
    ).resolve()
    expected_dependency = (staged_binding / "ane_kernel/crates/ane").resolve()
    if resolved_dependency != expected_dependency or not expected_dependency.is_dir():
        raise HarnessError("staged ANE dependency does not resolve inside the pinned binding clone")
    return staged_workspace, target, output_root


def verify_model_snapshot(model: Path) -> Dict[str, str]:
    if not model.is_dir() or model.name != MODEL_REVISION:
        raise HarnessError(f"pinned model snapshot {MODEL_REVISION} is missing: {model}")
    observed: Dict[str, str] = {}
    for name, expected in MODEL_COMPONENT_SHA256.items():
        path = model / name
        # Hugging Face snapshots are symlink forests into a content-addressed
        # blob store; the pinned content digest, not inode type, is the identity.
        if not path.is_file():
            raise HarnessError(f"model component {name} is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise HarnessError(f"model component digest mismatch for {name}: {actual}")
        observed[name] = actual
    return observed


def candidate_environment(target: Path) -> List[str]:
    cargo = Path(os.environ.get("SYNAPSE_CAMPAIGN_CARGO", shutil.which("cargo") or ""))
    if not cargo.is_file():
        raise HarnessError("cargo is not available to build the candidate")
    path_parts = [str(cargo.parent), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    values = [
        "/usr/bin/env",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "CARGO_NET_OFFLINE=true",
        "CARGO_TERM_COLOR=never",
        # An empty RUSTC_WRAPPER overrides any build.rustc-wrapper in the
        # forwarded CARGO_HOME config. A caching wrapper such as sccache talks to
        # its server over a local socket, and the candidate runs under a
        # network-denying sandbox that blocks local sockets too, so the wrapper
        # fails with EPERM before rustc ever runs. The candidate builds with rustc
        # directly.
        "RUSTC_WRAPPER=",
        f"PATH={':'.join(path_parts)}",
        f"CARGO_TARGET_DIR={target}",
    ]
    for name in ("RUSTUP_HOME", "CARGO_HOME"):
        if os.environ.get(name):
            values.append(f"{name}={os.environ[name]}")
    return values


def workspace_commit(runner: Path, workspace: Path, log_path: Path) -> str:
    status, value, stderr = runner_stdout(
        runner,
        ["/usr/bin/git", "-c", f"safe.directory={workspace}", "-C", str(workspace), "rev-parse", "HEAD"],
        log_path,
    )
    if status != 0 or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise CandidateRejected(
            "candidate workspace did not report one full Git commit SHA: "
            f"exit status {status}; stdout: {value or '<empty>'}; stderr: {stderr[-2048:] or '<empty>'}"
        )
    return value


def acquire_bench_lock() -> Path:
    lock = Path(os.environ.get("SYNAPSE_CAMPAIGN_BENCH_LOCK", str(Path(tempfile.gettempdir()) / "synapse-benchmark.lock")))
    measure_lock = Path(os.environ.get("SYNAPSE_CAMPAIGN_MEASURE_LOCK", "/tmp/aft-measure.lock"))
    if measure_lock.exists():
        raise HarnessError(f"measurement lock is already present: {measure_lock}")
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise HarnessError(f"benchmark lock is already present: {lock}") from error
    worker = subprocess.run(
        ["/usr/bin/pgrep", "-f", "Runner.Worker"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
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


def run_private_api_preflight(
    runner: Path, binary: Path, environment: Sequence[str], log_path: Path
) -> Dict[str, Any]:
    status = run_through_runner(runner, [*environment, str(binary)], log_path)
    output = runner_output(log_path)
    match = re.search(r"worst absolute error against the identity projection: ([^\s]+)", output)
    exact_zero = False
    if match is not None:
        try:
            exact_zero = float(match.group(1)) == 0.0
        except ValueError:
            exact_zero = False
    if status != 0 or not exact_zero or "VERDICT: private ANE API works on this chip and OS" not in output:
        raise HarnessError(
            "PRIVATE_ANE_API_PREFLIGHT_REFUSED: _ANEInMemoryModel private API failed to dlopen, execute, "
            f"or return exact-zero identity error (status {status}): {output[-2048:] or '<empty>'}"
        )
    return {"exact_identity_error": 0.0, "output": output}


def expected_row_protocol(rows_path: Path) -> Dict[int, List[Tuple[str, int]]]:
    try:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"protected row set is not valid JSONL: {error}") from error
    expected: Dict[int, List[Tuple[str, int]]] = {sequence: [] for sequence in SEQUENCES}
    if len(rows) != EXPECTED_ROWS:
        raise HarnessError(f"protected row set must contain exactly {EXPECTED_ROWS} rows")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise HarnessError(f"protected row {index} has no string identity")
        by_shape = row.get("input_ids_by_shape")
        if not isinstance(by_shape, dict):
            raise HarnessError(f"protected row {index} has no shape-indexed token IDs")
        for sequence in SEQUENCES:
            ids = by_shape.get(str(sequence))
            if not isinstance(ids, list) or not ids or len(ids) > sequence or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in ids
            ):
                raise HarnessError(f"protected row {index} has invalid token IDs for shape {sequence}")
            expected[sequence].append((row["id"], len(ids)))
    return expected


def validate_report(
    payload: Mapping[str, Any], sequence: int, expected_rows: Sequence[Tuple[str, int]]
) -> Dict[str, Any]:
    expected_top = {
        "model", "sequence_length", "layers_per_executable", "pooling", "cpu_reference",
        "row_set_sha256", "rows", "min_cosine", "mean_cosine", "deterministic",
        "gate_min_cosine", "gate_passed", "first_divergence", "checkpoint_metrics",
        "one_minute_load_average", "timing_note", "vectors",
    }
    if set(payload) != expected_top:
        raise CandidateRejected("candidate report fields changed")
    if payload.get("sequence_length") != sequence or payload.get("layers_per_executable") != LAYERS_PER_EXECUTABLE:
        raise CandidateRejected(f"candidate changed the sequence-{sequence} graph protocol")
    if payload.get("row_set_sha256") != ROW_SET_SHA256:
        raise CandidateRejected("candidate report used a different protected row set")
    model = payload.get("model")
    if not isinstance(model, dict) or model != {
        "model_id": "Alibaba-NLP/gte-modernbert-base",
        "snapshot_hash": MODEL_REVISION,
        "config_sha256": MODEL_COMPONENT_SHA256["config.json"],
        "model_safetensors_sha256": MODEL_COMPONENT_SHA256["model.safetensors"],
    }:
        raise CandidateRejected("candidate report changed the pinned model identity")
    if payload.get("pooling") != "first token (CLS), then L2 normalize; the model card defines embedding pooling, while config classifier_pooling=mean is classifier-head metadata":
        raise CandidateRejected("candidate report changed the pooling contract")
    if payload.get("cpu_reference") != "fp32, exact erf GELU, full permitted attention":
        raise CandidateRejected("candidate report changed the CPU comparator contract")
    if payload.get("gate_min_cosine") != MIN_COSINE:
        raise CandidateRejected("candidate report changed the minimum cosine gate")
    rows = payload.get("rows")
    vectors = payload.get("vectors")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise CandidateRejected("candidate emitted the wrong number of row metrics")
    if not isinstance(vectors, list) or len(vectors) != EXPECTED_ROWS:
        raise CandidateRejected("candidate emitted the wrong number of vectors")
    row_rates: List[float] = []
    row_cosines: List[float] = []
    row_measurements: List[Dict[str, Any]] = []
    ids = set()
    for index, (row, vector, expected_row) in enumerate(zip(rows, vectors, expected_rows), start=1):
        if not isinstance(row, dict) or set(row) != {
            "id", "active_tokens", "cosine", "warm_wall_ms_median", "deterministic"
        }:
            raise CandidateRejected(f"candidate row metric {index} has the wrong fields")
        identifier = row.get("id")
        active = row.get("active_tokens")
        wall_ms = row.get("warm_wall_ms_median")
        cosine = row.get("cosine")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise CandidateRejected(f"candidate row metric {index} has an invalid identity")
        ids.add(identifier)
        if (identifier, active) != expected_row:
            raise CandidateRejected(
                f"candidate row metric {index} changed the protected identity or active-token accounting"
            )
        if not finite_number(wall_ms) or float(wall_ms) <= 0.0 or not finite_number(cosine):
            raise CandidateRejected(f"candidate row metric {index} has invalid timing or cosine")
        if row.get("deterministic") is not True:
            raise CandidateRejected(f"candidate row metric {index} was not byte-identical on repeat")
        if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSIONS or any(
            not finite_number(value) for value in vector
        ):
            raise CandidateRejected(f"candidate vector {index} is not a finite {EXPECTED_DIMENSIONS}-vector")
        rate = active * 1000.0 / float(wall_ms)
        row_rates.append(rate)
        row_cosines.append(float(cosine))
        row_measurements.append({
            "id": identifier,
            "active_tokens": active,
            "warm_wall_ms_median": float(wall_ms),
            "tok_s": rate,
            "min_cosine": float(cosine),
        })
    reported_min = payload.get("min_cosine")
    reported_mean = payload.get("mean_cosine")
    if not finite_number(reported_min) or not math.isclose(float(reported_min), min(row_cosines), rel_tol=1e-7, abs_tol=1e-7):
        raise CandidateRejected("candidate min cosine does not match its row metrics")
    if not finite_number(reported_mean) or not math.isclose(
        float(reported_mean), statistics.fmean(row_cosines), rel_tol=1e-6, abs_tol=1e-6
    ):
        raise CandidateRejected("candidate mean cosine does not match its row metrics")
    if float(reported_min) < MIN_COSINE or payload.get("deterministic") is not True or payload.get("gate_passed") is not True:
        raise CandidateRejected(
            f"embedding correctness gate failed at sequence {sequence}: min cosine {float(reported_min):.9f}, "
            f"deterministic={payload.get('deterministic')!r}"
        )
    report_load = payload.get("one_minute_load_average")
    if not finite_number(report_load) or float(report_load) < 0.0:
        raise CandidateRejected("candidate report omitted a valid one-minute load average")
    # The row set is half 12-38-token rows and half near-512-token rows. A median
    # of per-row rates lands between those two regimes and rewards shaving the
    # per-row fixed cost that dominates the short rows. Aggregate throughput —
    # every active token over every millisecond spent — weights the rows by the
    # work they carry, which is what the lane exists to serve.
    total_active = sum(int(row["active_tokens"]) for row in row_measurements)
    total_wall_ms = sum(float(row["warm_wall_ms_median"]) for row in row_measurements)
    aggregate = total_active * 1000.0 / total_wall_ms
    return {
        # The campaign runner reads the objective from the field every embed
        # harness in this repository names `median_tok_s`; the name is a wire
        # contract, so it stays, and `aggregate_tok_s` carries the same value
        # under the name that describes it.
        "median_tok_s": aggregate,
        "aggregate_tok_s": aggregate,
        "row_tok_s": row_rates,
        "rows": row_measurements,
        "min_cosine": float(reported_min),
        "mean_cosine": float(reported_mean),
        "reported_load_1m": float(report_load),
    }


def run_harness(workspace_arg: str, runner_arg: str, result_arg: str) -> int:
    workspace = Path(workspace_arg).expanduser().resolve()
    runner = Path(runner_arg).expanduser().resolve()
    result_path = Path(result_arg).expanduser().resolve()
    if not workspace.is_dir():
        raise HarnessError(f"candidate workspace is not a directory: {workspace}")
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise HarnessError(f"candidate runner is not executable: {runner}")
    configured_constants()
    writer = ResultWriter(result_path)
    baseline_note = (
        f"gte-modernbert-base private _ANEInMemoryModel baseline: {BASELINE_TOK_S:.6f} tok/s at sequence 512. "
        f"Protocol: one layer per executable; each row has correctness and repeat executions before "
        f"{WARM_REPETITIONS} warm timed repetitions; score is the median active-token tok/s across {EXPECTED_ROWS} rows. "
        "The in-port reporter exposes the median rather than individual warm samples, so unlike the Metal harness "
        "the already-discarded correctness/repeat executions replace its first-pass warmup."
    )
    writer.write(initial_payload(baseline_note))
    temp_root = Path(tempfile.mkdtemp(prefix="synapse-ane-direct-embed-campaign-", dir="/tmp"))
    temp_root.chmod(0o777)
    lock: Optional[Path] = None
    try:
        verify_candidate_contract(workspace)
        row_protocol = expected_row_protocol(workspace / ROWS)
        model = Path(
            os.environ.get(
                "SYNAPSE_CAMPAIGN_MODEL",
                str(Path.home() / ".cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots" / MODEL_REVISION),
            )
        ).expanduser().resolve()
        binding_value = os.environ.get("SYNAPSE_CAMPAIGN_ANE_BINDING")
        if not binding_value:
            raise HarnessError("SYNAPSE_CAMPAIGN_ANE_BINDING is required for the pinned private-API clone")
        binding = Path(binding_value).expanduser().resolve()
        verify_binding_source(binding)
        model_components = verify_model_snapshot(model)
        staged_workspace, target, output_root = stage_sources(workspace, binding, temp_root, runner)
        commit = workspace_commit(runner, staged_workspace, temp_root / "workspace-commit.log")
        environment = candidate_environment(target)
        cargo = os.environ.get("SYNAPSE_CAMPAIGN_CARGO") or shutil.which("cargo")
        manifest = staged_workspace / PROBE_DIR / "Cargo.toml"
        build_log = temp_root / "build.log"
        build_status = run_through_runner(
            runner,
            [*environment, str(cargo), "build", "--locked", "--offline", "--release", "--manifest-path", str(manifest), "--bins"],
            build_log,
        )
        if build_status != 0:
            raise CandidateRejected(
                f"candidate ANE direct release build failed with status {build_status}: {runner_output(build_log)[-4096:]}"
            )
        preflight_binary = target / "release/ane-direct-probe"
        model_binary = target / "release/modernbert_full"
        if not preflight_binary.is_file() or not os.access(preflight_binary, os.X_OK):
            raise CandidateRejected("candidate private-API identity binary was not executable")
        if not model_binary.is_file() or not os.access(model_binary, os.X_OK):
            raise CandidateRejected("candidate ModernBERT binary was not executable")
        lock = acquire_bench_lock()
        maximum_load = configured_max_load()
        preflight_load = enforce_load_preflight("private-API preflight", maximum_load)
        private_api = run_private_api_preflight(
            runner, preflight_binary, environment, temp_root / "private-api-preflight.log"
        )
        measurements: List[Dict[str, Any]] = []
        for sequence in SEQUENCES:
            admission_load = enforce_load_preflight(f"sequence {sequence}", maximum_load)
            report_path = output_root / f"sequence-{sequence}.json"
            log_path = temp_root / f"sequence-{sequence}.log"
            status = run_through_runner(
                runner,
                [
                    *environment,
                    str(model_binary),
                    str(model),
                    str(staged_workspace / ROWS),
                    "--seq", str(sequence),
                    "--layers-per-executable", str(LAYERS_PER_EXECUTABLE),
                    "--warm-repetitions", str(WARM_REPETITIONS),
                    "--report", str(report_path),
                ],
                log_path,
            )
            if status != 0:
                raise CandidateRejected(
                    f"sequence-{sequence} candidate failed with status {status}: {runner_output(log_path)[-4096:]}"
                )
            try:
                payload = json.loads(report_path.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CandidateRejected(f"sequence-{sequence} did not produce valid report JSON: {error}") from error
            if not isinstance(payload, dict):
                raise CandidateRejected(f"sequence-{sequence} report was not a JSON object")
            validated = validate_report(payload, sequence, row_protocol[sequence])
            validated.update({
                "sequence_length": sequence,
                "admission_load_1m": admission_load,
                "completion_load_1m": load_average_1m(),
            })
            measurements.append(validated)
        objective = measurements[0]
        min_cosine = min(item["min_cosine"] for item in measurements)
        note = (
            f"{baseline_note} Row set SHA-256={ROW_SET_SHA256}; binding commit={BINDING_COMMIT}; "
            f"minimum cosine across shapes={min_cosine:.9f}; one-minute load threshold={maximum_load:.2f}. "
            f"A threshold of {DEFAULT_MAX_LOAD_1M:g} admits this shared workstation's normal background load and therefore carries more "
            "absolute variance than a dedicated 2.5-load rig; every shape records admission, in-report, and completion load."
        )
        writer.write({
            "gate_passed": True,
            "parity_passed": True,
            "determinism_passed": True,
            "samples": objective["row_tok_s"],
            "paired_runs": measurements,
            "median_tok_s": objective["median_tok_s"],
            "aggregate_tok_s": objective["aggregate_tok_s"],
            "min_cosine": min_cosine,
            "sequence_1024_tok_s": measurements[1]["median_tok_s"],
            "sequence_2048_tok_s": measurements[2]["median_tok_s"],
            "baseline_note": note,
            "workspace_commit": commit,
            "model_components": model_components,
            "row_set_sha256": ROW_SET_SHA256,
            "private_api_preflight": private_api,
            "preflight_load_1m": preflight_load,
            "load_threshold_1m": maximum_load,
            "protocol": {
                "objective_sequence": 512,
                "layers_per_executable": LAYERS_PER_EXECUTABLE,
                "warm_repetitions": WARM_REPETITIONS,
                "unmeasured_per_row_before_timing": ["correctness", "byte-identical repeat"],
                "aggregation": "median of active_tokens / row warm-wall median",
            },
        })
        return 0
    except CandidateRejected as error:
        writer.write(initial_payload(f"{baseline_note} Candidate rejected: {error}"))
        print(f"ANE direct embedding campaign candidate rejected: {error}", file=sys.stderr)
        return 3
    except HarnessError as error:
        writer.write(initial_payload(f"{baseline_note} Harness refused to run: {error}"))
        print(f"ANE direct embedding campaign harness refused to run: {error}", file=sys.stderr)
        return 1
    finally:
        release_bench_lock(lock)
        try:
            run_through_runner(runner, ["/bin/rm", "-rf", str(temp_root / "build")], temp_root / "cleanup.log")
        except OSError:
            pass
        shutil.rmtree(temp_root, ignore_errors=True)
        writer.close()


def self_test(workspace_arg: Optional[str]) -> int:
    workspace = Path(workspace_arg or ".").expanduser().resolve()
    verify_candidate_contract(workspace)
    assert protected_rust_digest(workspace / FULL_MODEL) == PROTECTED_RUST_SHA256
    assert validate_report
    assert math.isclose(statistics.median([1.0, 2.0, 3.0]), 2.0)
    try:
        enforce_load_preflight("self-test", 0.000001, wait_seconds=0.0)
    except HarnessError as error:
        assert "LOAD_PREFLIGHT_REFUSED" in str(error)
    else:
        raise AssertionError("load preflight did not refuse")
    print("ane-direct-embed-harness self-test passed")
    return 0


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--self-test":
        if len(arguments) > 2:
            print("usage: ane-direct-embed-harness.sh --self-test [workspace]", file=sys.stderr)
            return 2
        try:
            return self_test(arguments[1] if len(arguments) == 2 else None)
        except HarnessError as error:
            print(f"ANE direct embedding campaign self-test refused: {error}", file=sys.stderr)
            return 1
    if len(arguments) != 3:
        print("usage: ane-direct-embed-harness.sh {workspace} {candidate_runner} {result}", file=sys.stderr)
        return 2
    try:
        return run_harness(arguments[0], arguments[1], arguments[2])
    except (HarnessError, OSError) as error:
        print(f"ANE direct embedding campaign harness failed before result initialization: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
