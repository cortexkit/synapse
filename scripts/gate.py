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
import tempfile
import tarfile
import threading
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib.remote_gate import fingerprint


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
        lanes[platform] = Job(platform + "-suite", platform, 16 if platform == "linux" else 2, steps, env)
    phases = [Job("quick", "linux", 4, quick, env)], [
        lanes["linux"], lanes["windows"], Job("linux-llama-parity", "linux", 16, parity, env)]
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
    """Execute only on MOTOR; never compile in the submitting shell."""
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
    if job.platform == "windows":
        environment.setdefault("LIBCLANG_PATH", r"C:\Program Files\LLVM\bin")
        job.steps.insert(1, Step("native MSVC and LLVM prerequisites",
                                "python3 -c " + shlex.quote(
                                    "import os,subprocess; from pathlib import Path; "
                                    "assert 'host: x86_64-pc-windows-msvc' in subprocess.check_output(['rustc','-vV'],text=True); "
                                    "assert (Path(os.environ['LIBCLANG_PATH'])/'libclang.dll').is_file(), 'provision LIBCLANG_PATH on the runner'")))
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
        if job.platform == "windows":
            # Windows MOTOR has no return-path transport. Carry the same small
            # evidence archive over stdout; the submitter removes it from logs.
            print("GATE_EVIDENCE " + base64.b64encode(evidence.read_bytes()).decode("ascii"), flush=True)
    return status


def snapshot(destination):
    """Copy source inputs, including dirty files, to an independent MOTOR workspace."""
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]).split(b"\0")
    destination.mkdir(parents=True)
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
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "config", "core.autocrlf", "false"], check=True)
    subprocess.run(["git", "-C", str(destination), "add", "--all"], check=True)
    # MOTOR snapshots a Git checkout, so give this isolated index a HEAD without
    # changing the submitting branch or inheriting signing/hooks configuration.
    subprocess.run(["git", "-C", str(destination), "-c", "user.name=Synapse gate",
                    "-c", "user.email=gate@localhost", "-c", "commit.gpgsign=false",
                    "-c", "core.hooksPath=/dev/null", "commit", "-qm", "gate source snapshot"], check=True)


class Runner:
    def __init__(self, expected, scratch, evidence):
        self.expected, self.scratch, self.evidence = expected, scratch, evidence
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.active = {}

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
        checkout = self.scratch / job.name
        command = ["ck", "motor", "run", "--platform", job.platform,
                   "--timeout", "7200", "--net", "--cwd", str(checkout)]
        if job.platform == "linux":
            command += ["--vcpus", str(job.vcpus), "--return-path", str(checkout / "target/gate-evidence.zip")]
        command += ["--", "bash", "scripts/gate.sh", "--worker",
                    json.dumps(asdict(job)), self.expected]
        print(f"JOB {job.name}: {shlex.join(command[:command.index('--')])}", flush=True)
        try:
            with self.lock:
                if self.stopped.is_set():
                    return 203, 0.0
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.active[job.name] = child
            assert child.stdout is not None
            with child.stdout, (self.evidence / (job.name + ".log")).open("w") as log:
                for line in child.stdout:
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
                    print(f"FIRST FAILURE: {name}; cancelling other jobs", flush=True)
                    runner.cancel()
        except BaseException:
            runner.cancel()
            raise
        finally:
            if failure:
                runner.cancel()
    return failure, results


def run_gate(phases):
    start = time.monotonic()
    expected = fingerprint()
    print(f"TREE requested {expected}", flush=True)
    scratch_root = Path("target/scratch").resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="gate-", dir=scratch_root))
    evidence = Path(".cortexkit/internal-docs/investigations/gate") / scratch.name
    evidence.mkdir(parents=True, exist_ok=True)
    runner = Runner(expected, scratch, evidence)
    timings, status = [], 1
    old_handlers = {}

    def interrupt(signum, frame):
        runner.cancel()
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupt)
        # Prepare independent identical checkouts before any compiler starts.
        for jobs in phases:
            for job in jobs:
                snapshot(scratch / job.name)
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
    finally:
        runner.cancel()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if fingerprint() != expected:
            print("FAIL: submitting source changed during verification", flush=True)
            status = status or 1
        print("\nPHASE TIMING | seconds | result", flush=True)
        for number, duration, results in timings:
            print(f"Phase {number} | {duration:.2f} | " + ("PASS" if all(s == 0 for s, _ in results.values()) else "FAIL"))
            for name, (code, elapsed) in results.items():
                print(f"  {name} | {elapsed:.2f} | exit={code}")
        print(f"Total (including snapshots) | {time.monotonic() - start:.2f} | exit={status}", flush=True)
        print(f"Evidence: {evidence}", flush=True)
    return status


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
        return run_gate(phases)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
