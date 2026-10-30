#!/usr/bin/env python3
"""Run push CI before pushing: quick guards first, native suites in parallel."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import hashlib
import io
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib.remote_gate import fingerprint, motor_command


@dataclass
class Step:
    name: str
    script: str
    shell: str = "bash"
    cwd: str = "."


@dataclass
class Job:
    name: str
    platform: str
    vcpus: int
    steps: list
    env: dict


def render(text, platform):
    """Resolve only the expressions used by push steps; refuse unknown CI syntax."""
    for key, value in {
        "inputs.platform": platform,
        "matrix.partition": "1/1",
        "github.event_name": "push",
    }.items():
        text = text.replace("${{ " + key + " }}", value)
    if "${{" in text:
        raise ValueError(f"unmapped CI expression: {text}")
    return text


def ci_step(step, platform, guards=False):
    script = render(step["run"], platform)
    # CI writes build evidence beside source. MOTOR mounts Git metadata read-only,
    # so use ignored output paths rather than changing its exclude rules.
    script = script.replace("build-gates/", "target/gate-build-gates/")
    script = script.replace("mkdir -p build-gates", "mkdir -p target/gate-build-gates")
    script = script.replace("-Force build-gates", "-Force target/gate-build-gates")
    if platform == "linux":
        script = script.replace("binary='target/release/ck-synapse-worker-llama'",
                                'binary="${CARGO_TARGET_DIR:-target}/release/ck-synapse-worker-llama"')
    else:
        script = script.replace("$binary = 'target/release/ck-synapse-worker-llama.exe'",
                                "$binary = Join-Path $env:CARGO_TARGET_DIR 'release/ck-synapse-worker-llama.exe'")
        # Windows MOTOR cannot enable job networking; it prefetches Cargo.lock
        # dependencies before start. Keep execution locked to that offline cache.
        for original, offline in (
            ("cargo tree --workspace", "cargo tree --locked --offline --workspace"),
            ("cargo clippy --locked", "cargo clippy --locked --offline"),
            ("cargo nextest run --locked", "cargo nextest run --locked --offline"),
            ("cargo build -p synapse-worker-llama", "cargo build --locked --offline -p synapse-worker-llama"),
        ):
            script = script.replace(original, offline)
    cwd = step.get("working-directory", ".")
    if guards:
        # The guard job starts outside its checkout; MOTOR starts at its root.
        script = script.replace("synapse/scripts/", "scripts/").replace("cd synapse\n", "")
    if cwd == "synapse":
        cwd = "."
    elif cwd.startswith("synapse/"):
        cwd = cwd[len("synapse/"):]
    if "env" in step or step.get("continue-on-error"):
        raise ValueError(f"unmapped step semantics: {step['name']}")
    return Step(step["name"], script, step.get("shell", "bash"), cwd)


def plan():
    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/tests.yml").read_text())
    portable = yaml.safe_load(Path(".github/workflows/portable.yml").read_text())
    allowed = {"linux-checks", "windows-checks", "test", "linux", "windows",
               "e2e", "linux-llama-cuda-manual", "windows-llama-vulkan-manual",
               "windows-owned-cuda-manual"}
    if set(workflow["jobs"]) != allowed or set(portable["jobs"]) != {"clippy", "nextest", "llama"}:
        raise ValueError("CI job set changed; map new push jobs before running the gate")
    env = {**workflow.get("env", {}), **portable.get("env", {}),
           "CI": "true", "GITHUB_EVENT_NAME": "push", "RUSTC_WRAPPER": "",
           "CARGO_BUILD_RUSTC_WRAPPER": ""}
    quick, parity = [], []
    for job in [workflow["jobs"]["test"], *portable["jobs"].values()]:
        for step in job["steps"]:
            if "uses" in step and step["uses"] not in {
                "actions/checkout@v4", "./synapse/.github/actions/ci-rust",
                "dtolnay/rust-toolchain@stable", "Swatinem/rust-cache@v2",
                "actions/upload-artifact@v4",
            }:
                raise ValueError(f"unmapped CI action: {step['uses']}")
    for step in workflow["jobs"]["test"]["steps"]:
        if "run" in step:
            if step.get("if") not in (None, "matrix.name == 'linux'"):
                raise ValueError(f"unmapped guard condition: {step['name']}")
            converted = ci_step(step, "linux", guards=True)
            (parity if converted.cwd == "bench/parity" else quick).append(converted)

    lanes = {}
    for platform in ("linux", "windows"):
        steps = []
        for lane in ("clippy", "nextest", "llama"):
            for step in portable["jobs"][lane]["steps"]:
                if "run" not in step:
                    continue
                condition = step.get("if")
                if condition not in (None, "inputs.platform == 'linux'", "inputs.platform == 'windows'"):
                    raise ValueError(f"unmapped portable condition: {step['name']}")
                if condition and condition != f"inputs.platform == '{platform}'":
                    continue
                converted = ci_step(step, platform)
                if lane == "clippy" and step["name"] == "fmt" and platform == "linux":
                    quick.append(converted)
                elif lane == "llama" and platform == "linux":
                    parity.insert(0, converted)
                else:
                    steps.append(converted)
        environment = dict(env, CARGO_NET_OFFLINE="true") if platform == "windows" else env
        # Ask for the cores a full workspace build keeps busy, not more: a
        # build-server job holds every requested slot (and memory per slot)
        # for its whole run.
        lanes[platform] = Job(platform + "-suite", platform, 8 if platform == "linux" else 2, steps, environment)
    phases = [Job("quick", "linux", 4, quick, env)], [
        lanes["linux"], lanes["windows"], Job("linux-llama-parity", "linux", 4, parity, env)]
    for jobs in phases:
        for job in jobs:
            job.steps.insert(0, Step("runner toolchain prerequisites",
                                    "cargo --version; rustc -vV; cargo clippy --version; cargo fmt --version; cargo nextest --version"))
    return phases


def shell_command(step):
    if step.shell == "bash":
        return ["bash", "-e", "-o", "pipefail", "-c", step.script]
    if step.shell == "pwsh":
        return ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                "$ErrorActionPreference = 'Stop'\n" + step.script + "\nif ($LASTEXITCODE) { exit $LASTEXITCODE }"]
    raise ValueError(f"unmapped CI shell: {step.shell}")


def windows_command(job, expected):
    """Render Windows work on the submitter; the runner needs neither Python nor Git."""
    def powershell(script):
        # Windows PowerShell 5 treats redirected native stderr as error records.
        # Cargo's exit code, not its progress output, decides build success.
        lines = []
        for line in script.splitlines():
            if line.lstrip().startswith("& cargo "):
                lines.extend(["$ErrorActionPreference = 'Continue'", line, "$ErrorActionPreference = 'Stop'"])
            else:
                lines.append(line)
        body = "$ErrorActionPreference = 'Stop'\n" + "\n".join(lines) + "\nif ($LASTEXITCODE) { exit $LASTEXITCODE }"
        return "gate_ps " + base64.b64encode(body.encode("utf-16le")).decode("ascii")

    evidence = powershell(r"""
Add-Type -AssemblyName System.IO.Compression.FileSystem
$memory = New-Object System.IO.MemoryStream
$archive = [System.IO.Compression.ZipArchive]::new($memory, [System.IO.Compression.ZipArchiveMode]::Create, $true)
try {
  foreach ($directory in @('target/gate-build-gates', (Join-Path $env:CARGO_TARGET_DIR 'cargo-timings'))) {
    if (Test-Path -LiteralPath $directory -PathType Container) {
      $root = Get-Item -LiteralPath $directory
      foreach ($file in Get-ChildItem -LiteralPath $directory -Recurse -File) {
        $relative = $file.FullName.Substring($root.FullName.Length + 1).Replace('\', '/')
        $name = $root.Name + '/' + $relative
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $name) | Out-Null
      }
    }
  }
} finally { $archive.Dispose() }
[Console]::WriteLine('GATE_EVIDENCE ' + [Convert]::ToBase64String($memory.ToArray()))
$memory.Dispose()
""")
    lines = ["set -euo pipefail", "gate_job=" + shlex.quote(job.name), "gate_root=$PWD",
             "gate_step='shell prerequisites'", "gate_powershell=",
             "gate_ps() { \"$gate_powershell\" -NoLogo -NoProfile -NonInteractive -EncodedCommand \"$1\"; }",
             "gate_finish() {", "  gate_status=$?", "  trap - EXIT", "  set +e",
             "  if [ \"$gate_status\" -ne 0 ]; then printf 'FAIL %s / %s: exit=%s\\n' \"$gate_job\" \"$gate_step\" \"$gate_status\"; fi",
             "  if [ -n \"$gate_powershell\" ]; then", "    " + evidence, "    evidence_status=$?",
             "    if [ \"$gate_status\" -eq 0 ] && [ \"$evidence_status\" -ne 0 ]; then",
             "      gate_status=$evidence_status; printf 'FAIL %s / evidence: exit=%s\\n' \"$gate_job\" \"$gate_status\"; fi",
             "  fi", "  printf 'RESULT %s exit=%s\\n' \"$gate_job\" \"$gate_status\"", "  exit \"$gate_status\"", "}",
             "trap gate_finish EXIT"]
    lines += ["export " + key + "=" + shlex.quote(str(value)) for key, value in sorted(job.env.items())]
    lines += ['export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(cygpath -m "$PWD/target")}"',
              'if [ -z "${LIBCLANG_PATH:-}" ]; then export LIBCLANG_PATH=' + shlex.quote(r"C:\Program Files\LLVM\bin") + '; fi',
              "if command -v pwsh >/dev/null 2>&1; then gate_powershell=pwsh;",
              "elif command -v powershell.exe >/dev/null 2>&1; then gate_powershell=powershell.exe;",
              "else echo 'FAIL Windows shell prerequisites: PowerShell is required'; exit 127; fi",
              "printf '%s\\n' " + shlex.quote(f"TREE {job.name} submitted {expected}")]
    native = Step("native MSVC and LLVM prerequisites", r"""
$hostInfo = (& rustc -vV) -join "`n"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($hostInfo -notmatch 'host: x86_64-pc-windows-msvc') { throw 'native MSVC toolchain is required' }
if (!(Test-Path -LiteralPath (Join-Path $env:LIBCLANG_PATH 'libclang.dll'))) { throw 'provision LIBCLANG_PATH on the runner' }
""", "pwsh")
    steps = [Step(**raw) if isinstance(raw, dict) else raw for raw in job.steps]
    steps = [*steps[:1], native, *steps[1:]]
    for step in steps:
        command = powershell(step.script) if step.shell == "pwsh" else shlex.join(shell_command(step))
        lines += ["gate_step=" + shlex.quote(step.name), "gate_started=$SECONDS",
                  "printf 'STEP %s / %s\\n' \"$gate_job\" \"$gate_step\"", "gate_step_status=0",
                  "(cd \"$gate_root\"/" + shlex.quote(step.cwd) + " && " + command + ") || gate_step_status=$?",
                  "printf 'TIMING %s / %s: %ss exit=%s\\n' \"$gate_job\" \"$gate_step\" \"$((SECONDS - gate_started))\" \"$gate_step_status\"",
                  '[ "$gate_step_status" -eq 0 ] || exit "$gate_step_status"']
    lines.append("exit 0")
    return ["bash", "-e", "-o", "pipefail", "-c", "\n".join(lines)]


def python_dependencies():
    """Use pinned pure-Python PyYAML without pip or a system-package mutation."""
    directory = Path("target/gate-python").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "pyyaml-6.0.3.tar.gz"
    digest = "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
    if not archive.exists():
        url = ("https://files.pythonhosted.org/packages/05/8e/"
               "961c0007c59b8dd7729d542c61a4d537767a59645b82a0b521206e1e25c2/pyyaml-6.0.3.tar.gz")
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("PyYAML archive digest mismatch")
        archive.write_bytes(content)
    content = archive.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("cached PyYAML archive digest mismatch")
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as source:
        for member in source.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 4 and parts[1:3] == ("lib", "yaml") and parts[3].endswith(".py") and member.isfile():
                target = directory / "yaml" / parts[3]
                target.parent.mkdir(exist_ok=True)
                stream = source.extractfile(member)
                if stream is not None:
                    target.write_bytes(stream.read())
    print("Python prerequisite: PyYAML 6.0.3 (SHA-256 verified, pure Python)", flush=True)
    return str(directory)


def worker(job, expected):
    """Execute Linux work on the build runner; Windows uses submitter-rendered shell."""
    if job.platform != "linux":
        print(f"FAIL {job.name}: Windows requires inline shell mode", flush=True)
        return 2
    root = Path.cwd()
    if fingerprint() != expected:
        print(f"FAIL {job.name}: snapshot differs from {expected}", flush=True)
        return 1
    environment = dict(os.environ, **job.env)
    environment.setdefault("CARGO_TARGET_DIR", str(root / "target"))
    if job.name == "quick":
        try:
            environment["PYTHONPATH"] = python_dependencies() + os.pathsep + environment.get("PYTHONPATH", "")
        except (OSError, ValueError) as error:
            print(f"FAIL {job.name} / Python prerequisites: {error}", flush=True)
            return 1
    print(f"TREE {job.name} {expected}", flush=True)
    status = 0
    try:
        for raw in job.steps:
            step = Step(**raw) if isinstance(raw, dict) else raw
            start = time.monotonic()
            print(f"STEP {job.name} / {step.name}", flush=True)
            print(step.script, flush=True)
            try:
                status = subprocess.run(shell_command(step), cwd=root / step.cwd, env=environment).returncode
            except OSError as error:
                print(error, flush=True)
                status = 127
            print(f"TIMING {job.name} / {step.name}: {time.monotonic() - start:.2f}s exit={status}", flush=True)
            if status:
                print(f"FAIL {job.name} / {step.name}: exit={status}", flush=True)
                break
    finally:
        if fingerprint() != expected:
            print(f"FAIL {job.name}: source changed during checks", flush=True)
            status = status or 1
        evidence = root / "target/gate-evidence.zip"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(evidence, "w", zipfile.ZIP_DEFLATED) as archive:
            for directory in (root / "target/gate-build-gates", Path(environment["CARGO_TARGET_DIR"]) / "cargo-timings"):
                if directory.exists():
                    for path in directory.rglob("*"):
                        if path.is_file():
                            archive.write(path, Path(directory.name) / path.relative_to(directory))
    return status


# Windows MOTOR flattens argv into one command line, which mangles a multi-line
# `bash -c` script's quoting ("unexpected EOF while looking for matching `''").
# The Windows job therefore runs its script as a file committed into its snapshot.
WINDOWS_SCRIPT = ".gate/windows-job.sh"


def snapshot(destination, allowed_roots, extra_files=None):
    """Overlay current source inputs onto a detached worktree of this repository.

    MOTOR ships a checkout to the runner as a git bundle against history the
    runner already has, so the snapshot needs full history. A shallow clone
    fails there with `bundle_rejected` before any check runs. A worktree shares
    this repository's object store, costs no copy, and the caller's
    `git worktree prune` cleans it up after `rmtree`.
    """
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]).split(b"\0")
    destination = validate_snapshot_path(destination, destination.name, allowed_roots)
    subprocess.run(["git", "worktree", "add", "--quiet", "--detach", "--no-checkout",
                    str(destination), "HEAD"], check=True)
    # Seed the index without checking out files, so deleted source inputs stay
    # deleted and tracked-but-ignored files retain their tracked status.
    subprocess.run(["git", "-C", str(destination), "reset", "--mixed", "--quiet", "HEAD"], check=True)
    for raw in sorted(set(names) - {b""}):
        source = Path(os.fsdecode(raw))
        if not source.exists() and not source.is_symlink():
            continue
        target = destination / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)
    for relative, text in (extra_files or {}).items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as output:
            output.write(text)
    if extra_files:
        subprocess.run(["git", "-C", str(destination), "add", "--force", "--", *extra_files], check=True)
    subprocess.run(["git", "-C", str(destination), "add", "--all"], check=True)
    # Commit even a clean source snapshot without changing the submitting branch
    # or inheriting signing/hooks configuration.
    subprocess.run(["git", "-C", str(destination), "-c", "user.name=Synapse gate",
                    "-c", "user.email=gate@localhost", "-c", "commit.gpgsign=false",
                    "-c", "core.hooksPath=/dev/null", "commit", "--allow-empty", "-qm", "gate source snapshot"], check=True)


class Runner:
    def __init__(self, expected, scratch, evidence, allowed_roots):
        self.expected, self.scratch, self.evidence = expected, scratch, evidence
        self.allowed_roots = tuple(allowed_roots)
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.active = {}
        self.capacity_refusals = set()

    def cancel(self):
        self.stopped.set()
        with self.lock:
            for child in self.active.values():
                if child.poll() is None:
                    # ck translates SIGTERM into cancellation and waits for the
                    # remote outcome. Join it rather than abandon a compiler.
                    child.terminate()

    def run(self, job):
        start = time.monotonic()
        checkout = validate_snapshot_path(self.scratch / job.name, job.name, self.allowed_roots)
        remote = (["bash", WINDOWS_SCRIPT] if job.platform == "windows" else
                  ["bash", "scripts/gate.sh", "--worker", json.dumps(asdict(job)), self.expected])
        command = motor_command(
            remote,
            cwd=checkout, platform=job.platform, vcpus=job.vcpus, timeout=7200, net=job.platform != "windows",
            return_paths=[checkout / "target/gate-evidence.zip"] if job.platform == "linux" else [],
        )
        print(f"JOB {job.name}: {shlex.join(command[:command.index('--')])}", flush=True)
        capacity = False
        try:
            with self.lock:
                if self.stopped.is_set():
                    return 203, 0.0
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.active[job.name] = child
            assert child.stdout is not None
            with child.stdout, (self.evidence / (job.name + ".log")).open("w") as log:
                for line in child.stdout:
                    capacity = capacity or any(reason in line for reason in (
                        "runner_full", "hosted_capacity", "queue_wait_exceeded"))
                    if line.startswith("GATE_EVIDENCE "):
                        content = base64.b64decode(line.split(" ", 1)[1].strip(), validate=True)
                        (self.evidence / (job.name + ".zip")).write_bytes(content)
                        line = f"Retained {len(content)} bytes of Windows build evidence\n"
                    log.write(line)
                    print(f"[{job.name}] {line}", end="", flush=True)
            status = child.wait()
            artifact = checkout / "target/gate-evidence.zip"
            if artifact.exists():
                shutil.copy2(artifact, self.evidence / (job.name + ".zip"))
        except OSError as error:
            print(f"FAIL {job.name}: {error}", flush=True)
            status = 127
        finally:
            with self.lock:
                self.active.pop(job.name, None)
        if status:
            if status == 200 and capacity:
                with self.lock:
                    self.capacity_refusals.add(job.name)
                print(f"CAPACITY job {job.name}: capacity, retry later (no checks ran)", flush=True)
            elif status in (203, -signal.SIGTERM):
                print(f"CANCELLED job {job.name}: exit={status}", flush=True)
            else:
                reason = "refused before start (no checks ran)" if status == 200 else f"exit={status}"
                print(f"FAIL job {job.name}: {reason}", flush=True)
        return status, time.monotonic() - start


def parallel(runner, jobs):
    results, failure = {}, 0
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(runner.run, job): job.name for job in jobs}
        try:
            for future in as_completed(futures):
                status, duration = future.result()
                name = futures[future]
                results[name] = (status, duration)
                if status and not failure:
                    failure = status
                    reason = "CAPACITY, retry later" if name in runner.capacity_refusals else "FIRST FAILURE"
                    print(f"{reason}: {name}; cancelling other jobs", flush=True)
                    runner.cancel()
        except BaseException:
            runner.cancel()
            raise
        finally:
            if failure:
                runner.cancel()
    return failure, results


def validate_snapshot_path(path, label, allowed_roots):
    """Refuse side checkouts outside the build runner's allowed roots or inside the source."""
    checkout = Path.cwd().resolve()
    path = Path(path).resolve()
    if path == checkout or checkout in path.parents:
        raise ValueError(f"{label}: snapshot_path_rejected: destination is under the checkout: {path}")
    roots = [Path(root).resolve() for root in allowed_roots]
    if roots and not any(path == root or root in path.parents for root in roots):
        names = ", ".join(str(root) for root in roots)
        raise ValueError(f"{label}: workspace_key_rejected: destination outside runner-owned roots [{names}]: {path}")
    return path


def snapshot_directory(allowed_roots):
    """Use the side-checkout area under the user's home, never the system temp root."""
    base = Path.home() / "Work/Projects/CortexKit-scratch/synapse-gate"
    return validate_snapshot_path(base / uuid.uuid4().hex, "snapshot root", allowed_roots)


def run_gate(phases, allowed_roots):
    start = time.monotonic()
    expected = fingerprint()
    print(f"TREE requested {expected}", flush=True)
    allowed_roots = tuple(allowed_roots)
    scratch = snapshot_directory(allowed_roots)
    evidence = Path(".cortexkit/internal-docs/investigations/gate") / scratch.name
    runner = Runner(expected, scratch, evidence, allowed_roots)
    timings, status = [], 1
    old_handlers = {}

    def interrupt(signum, frame):
        runner.cancel()
        raise KeyboardInterrupt

    try:
        for jobs in phases:
            for job in jobs:
                validate_snapshot_path(scratch / job.name, job.name, allowed_roots)
        scratch.mkdir(parents=True)
        evidence.mkdir(parents=True, exist_ok=True)
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupt)
        # Prepare independent identical checkouts before any compiler starts.
        for jobs in phases:
            for job in jobs:
                extra = ({WINDOWS_SCRIPT: windows_command(job, expected)[-1] + "\n"}
                         if job.platform == "windows" else None)
                snapshot(scratch / job.name, allowed_roots, extra)
        if fingerprint() != expected:
            raise ValueError("source changed while preparing snapshots")
        status = 0
        for number, jobs in enumerate(phases, 1):
            phase_start = time.monotonic()
            status, results = parallel(runner, jobs)
            timings.append((number, time.monotonic() - phase_start, results))
            if status:
                break
    except KeyboardInterrupt:
        status = 130
    except Exception:
        status = 1
        raise
    finally:
        runner.cancel()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        # All MOTOR clients have joined, including cancelled jobs, before their
        # temporary source checkouts are removed. Retained evidence lives elsewhere.
        try:
            if scratch.exists():
                shutil.rmtree(scratch)
        finally:
            subprocess.run(["git", "worktree", "prune"], check=True)
        if fingerprint() != expected:
            print("FAIL: submitting source changed during verification", flush=True)
            status = status or 1
        print("\nPHASE TIMING | seconds | result", flush=True)
        for number, duration, results in timings:
            outcomes = [job_outcome(name, code, runner.capacity_refusals) for name, (code, _) in results.items()]
            if "FAIL" in outcomes:
                phase = "FAIL"
            elif "CAPACITY, retry later" in outcomes:
                phase = "CAPACITY, retry later"
            else:
                phase = "PASS" if all(s == 0 for s, _ in results.values()) else "CANCELLED"
            print(f"Phase {number} | {duration:.2f} | {phase}")
            for name, (code, elapsed) in results.items():
                print(f"  {name} | {elapsed:.2f} | {job_outcome(name, code, runner.capacity_refusals)} exit={code}")
        print(f"Total (including snapshots) | {time.monotonic() - start:.2f} | exit={status}", flush=True)
        print(f"Evidence: {evidence}", flush=True)
    return status


def job_outcome(name, status, capacity_refusals):
    if name in capacity_refusals:
        return "CAPACITY, retry later"
    if status == 0:
        return "PASS"
    if status in (203, -signal.SIGTERM):
        return "CANCELLED"
    return "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="legacy preflight only; not push-CI parity")
    parser.add_argument("--plan", action="store_true", help="print the workflow-derived jobs without running")
    parser.add_argument("--worker", nargs=2, metavar=("JOB_JSON", "TREE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(Job(**json.loads(args.worker[0])), args.worker[1])
    if args.fast:
        print("FAST ONLY: full push-CI parity was not requested", flush=True)
        return subprocess.run([sys.executable, "scripts/train-preflight.py"]).returncode
    try:
        phases = plan()
        if args.plan:
            print(json.dumps([[asdict(job) for job in jobs] for jobs in phases], indent=2))
            return 0
        allowed_roots = (Path("/Users"),) if sys.platform == "darwin" else ()
        return run_gate(phases, allowed_roots)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
