"""Push parity selection and fail-fast fan-out without submitting real builds."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

GATE = runpy.run_path(str(Path(__file__).with_name("gate.py")))
ROOT = Path(__file__).resolve().parent.parent
Job, Step = GATE["Job"], GATE["Step"]


class PlanTests(unittest.TestCase):
    def test_snapshot_preserves_dirty_untracked_deleted_and_executable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, copy = root / "source", root / "copy"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "tracked").write_text("original")
            (source / "deleted").write_text("remove me")
            (source / ".gitignore").write_text("ignored\n")
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            (source / "tracked").write_text("dirty")
            (source / "deleted").unlink()
            (source / "new").write_text("untracked")
            (source / "new").chmod(0o755)
            (source / "ignored").write_text("output")
            previous = Path.cwd()
            try:
                os.chdir(source)
                before = GATE["fingerprint"]()
                GATE["snapshot"](copy)
                os.chdir(copy)
                self.assertEqual(GATE["fingerprint"](), before)
                subprocess.run(["git", "rev-parse", "--verify", "HEAD"], check=True, capture_output=True)
                self.assertFalse((copy / "ignored").exists())
                self.assertFalse((copy / "deleted").exists())
            finally:
                os.chdir(previous)

    def test_ci_commands_are_read_not_reimplemented(self):
        first, second = GATE["plan"]()
        self.assertEqual([(j.name, j.platform, j.vcpus) for j in first], [("quick", "linux", 4)])
        self.assertEqual([(j.name, j.platform, j.vcpus) for j in second], [
            ("linux-suite", "linux", 16), ("windows-suite", "windows", 2),
            ("linux-llama-parity", "linux", 16)])
        for job in second[:2]:
            scripts = "\n".join(s.script for s in job.steps)
            self.assertIn("cargo clippy --locked $crates --all-targets -- -D warnings", scripts)
            self.assertIn("--features synapse-module/test-support", scripts)
            self.assertIn("--partition 'count:1/1'", scripts)
            self.assertIn("not binary(skeleton_e2e)", scripts)
            self.assertNotIn("${{", scripts)
        self.assertEqual(second[1].steps[-1].shell, "pwsh")
        self.assertEqual(second[2].steps[-1].cwd, "bench/parity")
        self.assertEqual(second[2].steps[-1].script, "cargo test --locked")
        guards = "\n".join(s.script for s in first[0].steps)
        for command in ("check-train-preconditions.sh", "test-ci-e2e-output.py",
                        "scripts.test_ci_parallel", "check-release-candidate.py --self-test",
                        "crates/synapse-release-checks", "test_release_matrix.py",
                        "public-banlist-check.sh", "check-cortexkit-layout.sh",
                        "check-no-external-path-deps.sh --self-test", "scripts.test_module_runtime_deps",
                        "ane-direct-embed-harness.sh --self-test ."):
            self.assertIn(command, guards)
        self.assertNotIn("synapse/scripts/", guards)
        self.assertNotIn("cd synapse", guards)

    def test_workflow_changes_flow_into_plan_and_unknown_expressions_refuse(self):
        import yaml
        workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
        portable = yaml.safe_load((ROOT / ".github/workflows/portable.yml").read_text())
        portable["env"]["SYNAPSE_CRATES"] += " -p new-production-crate"
        portable["jobs"]["clippy"]["steps"].append({"name": "new guard", "run": "echo new-guard"})
        with patch.object(Path, "read_text", side_effect=[yaml.safe_dump(workflow), yaml.safe_dump(portable)]):
            _, jobs = GATE["plan"]()
        self.assertIn("new-production-crate", jobs[0].env["SYNAPSE_CRATES"])
        self.assertIn("echo new-guard", [s.script for s in jobs[0].steps])
        with self.assertRaisesRegex(ValueError, "unmapped CI expression"):
            GATE["render"]("echo ${{ inputs.unhandled }}", "linux")

    def test_train_hook_defaults_to_full_with_explicit_fast_escape(self):
        hook = (ROOT / "scripts/train-push.local.sh").read_text()
        self.assertIn("scripts/gate.sh", hook)
        self.assertIn("TRAIN_GATE_FAST", hook)
        self.assertNotIn("python3 scripts/train-preflight.py", hook)
        gate = (ROOT / "scripts/gate.sh").read_text()
        self.assertIn('"$job" != quick', gate)
        self.assertIn("cmake-3.31.6", gate)
        self.assertIn("1c8b05df0602365da91ee6a3336fe57525b137706c4ab5675498f662ae1dbcec", gate)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/gate.sh").write_text('printf "%s\\n" "$@" > "$GATE_ARGS"\nexit "$GATE_STATUS"\n')
            tools = root / "bin"
            tools.mkdir()
            (tools / "gh").write_text("#!/bin/sh\nexit 0\n")
            (tools / "gh").chmod(0o755)
            for fast, status in (("0", "0"), ("1", "0"), ("0", "7"), ("1", "7")):
                result = subprocess.run(["bash", "-uc", 'refuse() { echo "$*"; exit 1; }; say() { :; }; source "$1"',
                                         "hook-test", str(ROOT / "scripts/train-push.local.sh")], cwd=root,
                                        env=dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                                                 TRAIN_GATE_FAST=fast, GATE_STATUS=status, GATE_ARGS=str(root / "args")),
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0 if status == "0" else 1, result.stdout + result.stderr)
                self.assertEqual((root / "args").read_text().strip(), "--fast" if fast == "1" else "")


class FanoutTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux worker bootstrap")
    def test_linux_worker_uses_writable_cargo_home_only_when_resolution_needs_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "bin"
            tools.mkdir()
            python = tools / "python3"
            python.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
if sys.argv[1] == '-c':
    print('quick')
else:
    print(json.dumps({key: os.environ.get(key) for key in ('CARGO_HOME', 'CARGO_NET_OFFLINE')}))
''')
            python.chmod(0o755)
            cargo = tools / "cargo"
            cargo.write_text('#!/bin/sh\nexit "$FAKE_CARGO_STATUS"\n')
            cargo.chmod(0o755)
            for status in (0, 1):
                result = subprocess.run(["bash", str(ROOT / "scripts/gate.sh"), "--worker", '{"name":"quick"}', "tree"],
                                        env=dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                                                 TMPDIR=str(root), CARGO_HOME="/immutable-cache", CARGO_NET_OFFLINE="true",
                                                 FAKE_CARGO_STATUS=str(status)), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                environment = json.loads(result.stdout.splitlines()[-1])
                if status:
                    self.assertTrue(environment["CARGO_HOME"].startswith(str(root)))
                    self.assertEqual(environment["CARGO_NET_OFFLINE"], "false")
                else:
                    self.assertEqual(environment["CARGO_HOME"], "/immutable-cache")
                    self.assertEqual(environment["CARGO_NET_OFFLINE"], "true")

    def invoke(self, statuses, two_phases=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools, events = root / "bin", root / "events"
            tools.mkdir()
            events.mkdir()
            ck = tools / "ck"
            ck.write_text(f"#!{sys.executable}\n" + r'''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ['motor', 'run']
job = json.loads(args[-2])
name = job['name']
events = Path(os.environ['FAKE_EVENTS'])
(events / (name + '.json')).write_text(json.dumps(args))
def cancelled(signum, frame):
    (events / (name + '.cancelled')).touch()
    sys.exit(203)
signal.signal(signal.SIGTERM, cancelled)
if name != 'quick':
    deadline = time.monotonic() + 5
    while len(list(events.glob('*suite.json'))) != 2 or not (events / 'linux-llama-parity.json').exists():
        if time.monotonic() > deadline:
            sys.exit(90)
        time.sleep(.01)
status = json.loads(os.environ['FAKE_STATUSES']).get(name, 0)
if name == 'windows-suite':
    import base64
    print('GATE_EVIDENCE ' + base64.b64encode(b'evidence').decode(), flush=True)
if status:
    print('FAIL ' + name + ' / clippy: exit=' + str(status), flush=True)
elif name != 'quick' and os.getenv('FAKE_HOLD') == '1':
    time.sleep(10)
(events / (name + '.done')).touch()
sys.exit(status)
''')
            ck.chmod(0o755)
            first, second = GATE["plan"]()
            phases = (first, second) if two_phases else (first,)
            evidence = root / "evidence"
            evidence.mkdir()
            runner = GATE["Runner"]("tree", root / "snapshots", evidence)
            output = io.StringIO()
            status = 0
            with patch.dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                            FAKE_EVENTS=str(events), FAKE_STATUSES=json.dumps(statuses),
                            FAKE_HOLD="1" if any(statuses.get(j.name, 0) for j in second) else "0"), \
                    redirect_stdout(output):
                results = []
                for jobs in phases:
                    status, result = GATE["parallel"](runner, jobs)
                    results.append(result)
                    if status:
                        break
            recorded = {p.stem: json.loads(p.read_text()) for p in events.glob("*.json")}
            completed = {p.name for p in events.iterdir()}
            return status, results, recorded, completed, output.getvalue()

    def test_parallel_launch_platforms_resources_and_snapshot_isolation(self):
        status, results, recorded, completed, output = self.invoke({})
        self.assertEqual(status, 0, output)
        self.assertEqual(len(results[1]), 3)
        roots = set()
        for name, args in recorded.items():
            roots.add(args[args.index("--cwd") + 1])
            self.assertIn("--net", args)
            platform = args[args.index("--platform") + 1]
            self.assertEqual(platform, "windows" if name == "windows-suite" else "linux")
            if platform == "linux":
                self.assertEqual(args[args.index("--vcpus") + 1], "4" if name == "quick" else "16")
            else:
                self.assertNotIn("--vcpus", args)
                self.assertNotIn("--return-path", args)
        self.assertEqual(len(roots), 4)
        self.assertIn("Retained 8 bytes of Windows build evidence", output)
        self.assertNotIn("GATE_EVIDENCE", output)

    def test_first_failure_names_job_and_step_cancels_and_joins_siblings(self):
        status, results, recorded, completed, output = self.invoke({"windows-suite": 7})
        self.assertEqual(status, 7, output)
        self.assertIn("FIRST FAILURE: windows-suite", output)
        self.assertIn("FAIL windows-suite / clippy", output)
        self.assertIn("linux-suite.cancelled", completed)
        self.assertIn("linux-llama-parity.cancelled", completed)
        self.assertEqual(len(results[1]), 3)

    def test_refused_before_start_never_counts_as_pass_or_launches_phase_two(self):
        status, results, recorded, completed, output = self.invoke({"quick": 200})
        self.assertEqual(status, 200)
        self.assertEqual(set(recorded), {"quick"})
        self.assertIn("refused before start (no checks ran)", output)

    def test_timing_table_on_success_and_failure(self):
        first, second = GATE["plan"]()
        for status in (0, 200):
            output = io.StringIO()
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(output), \
                    patch.dict(GATE["run_gate"].__globals__, fingerprint=lambda: "tree", snapshot=lambda _: None,
                               parallel=lambda runner, jobs: (status, {jobs[0].name: (status, .25)})), \
                    patch.object(Path, "mkdir"), \
                    patch.object(tempfile, "mkdtemp", return_value=directory):
                self.assertEqual(GATE["run_gate"]((first, second)), status)
            self.assertIn("PHASE TIMING | seconds | result", output.getvalue())
            self.assertIn("Phase 1", output.getvalue())
            self.assertIn("PASS" if status == 0 else "FAIL", output.getvalue())
            self.assertEqual("Phase 2" in output.getvalue(), status == 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
