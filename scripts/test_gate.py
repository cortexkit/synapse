"""Push parity selection and fail-fast fan-out without submitting real builds."""

from contextlib import redirect_stdout
from dataclasses import asdict
import base64
import io
import json
import os
from pathlib import Path
import runpy
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

GATE = runpy.run_path(str(Path(__file__).with_name("gate.py")))
ROOT = Path(__file__).resolve().parent.parent
Job, Step = GATE["Job"], GATE["Step"]


def commit_fixture(checkout):
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Gate test", "-c", "user.email=gate@localhost",
                    "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture"], check=True)


class PlanTests(unittest.TestCase):
    def test_macos_side_checkouts_resolve_under_users_not_tmpdir(self):
        home = Path("/Users/synapse-gate-test")
        with patch.object(sys, "platform", "darwin"), patch.object(Path, "home", return_value=home), \
                patch.dict(os.environ, TMPDIR="/private/var/folders/not-a-workspace"):
            directory = GATE["snapshot_directory"]((Path("/Users"),))
            self.assertEqual(directory.parent, home / "Work/Projects/CortexKit-scratch/synapse-gate")
            for jobs in GATE["plan"]():
                for job in jobs:
                    destination = GATE["validate_snapshot_path"](directory / job.name, job.name, (Path("/Users"),))
                    self.assertIn(Path("/Users"), destination.parents)
                    self.assertNotIn(Path.cwd().resolve(), destination.parents)

    def test_invalid_macos_or_nested_destinations_refuse_up_front_by_name(self):
        with patch.object(sys, "platform", "darwin"), patch.object(Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError, "quick: workspace_key_rejected.*outside runner-owned roots"):
                GATE["validate_snapshot_path"](Path("/private/tmp/synapse-gate/quick"), "quick", (Path("/Users"),))
            with self.assertRaisesRegex(ValueError, "quick: snapshot_path_rejected.*under the checkout"):
                # A path that is not a symlink: a landing checkout's `target`
                # can link outside /Users, which trips the roots check first.
                GATE["validate_snapshot_path"](Path.cwd() / "scripts/quick", "quick", (Path("/Users"),))
            with patch.object(Path, "home", return_value=Path("/private/tmp/home")):
                with self.assertRaisesRegex(ValueError, "snapshot root: workspace_key_rejected"):
                    GATE["snapshot_directory"]((Path("/Users"),))
            mkdir.assert_not_called()

    def test_explicit_allowed_roots_are_independent_of_host_and_tmpdir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            source.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(source)
                for host in ("darwin", "linux"):
                    with patch.object(sys, "platform", host):
                        destination = root / "run/quick"
                        self.assertEqual(GATE["validate_snapshot_path"](destination, "quick", (root,)), destination)
                        with self.assertRaisesRegex(ValueError, "quick: workspace_key_rejected"):
                            GATE["validate_snapshot_path"](destination, "quick", (root / "other",))
            finally:
                os.chdir(previous)

    def test_entry_point_selects_native_default_allowed_roots(self):
        phases = ([], [])
        for host, expected in (("darwin", (Path("/Users"),)), ("linux", ())):
            with patch.object(sys, "platform", host), patch.object(sys, "argv", ["gate.py"]), \
                    patch.dict(GATE["main"].__globals__, plan=lambda: phases), \
                    patch.dict(GATE["main"].__globals__, run_gate=lambda actual, roots: (actual, roots)):
                self.assertEqual(GATE["main"](), (phases, expected))

    def test_side_checkout_base_resolves_symlinks_before_accepting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            checkout.mkdir()
            alias = root / "home-alias"
            alias.symlink_to(checkout, target_is_directory=True)
            previous = Path.cwd()
            try:
                os.chdir(checkout)
                with patch.object(Path, "home", return_value=alias):
                    with self.assertRaisesRegex(ValueError, "snapshot root: snapshot_path_rejected"):
                        GATE["snapshot_directory"]((root,))
            finally:
                os.chdir(previous)

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
            commit_fixture(source)
            (source / "tracked").write_text("dirty")
            (source / "deleted").unlink()
            (source / "new").write_text("untracked")
            (source / "new").chmod(0o755)
            (source / "ignored").write_text("output")
            previous = Path.cwd()
            try:
                os.chdir(source)
                before = GATE["fingerprint"]()
                GATE["snapshot"](copy, (root,))
                os.chdir(copy)
                self.assertEqual(GATE["fingerprint"](), before)
                subprocess.run(["git", "rev-parse", "--verify", "HEAD"], check=True, capture_output=True)
                self.assertFalse((copy / "ignored").exists())
                self.assertFalse((copy / "deleted").exists())
                self.assertTrue((copy / ".git").is_file())
                common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()
                self.assertEqual((copy / common).resolve(), (source / ".git").resolve())
                self.assertFalse((source / ".git/shallow").exists())
                self.assertNotEqual(subprocess.run(["git", "symbolic-ref", "-q", "HEAD"]).returncode, 0)
            finally:
                os.chdir(previous)

    def test_windows_script_is_committed_in_snapshot_not_in_submitting_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, copy = root / "source", root / "copy"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / ".gitignore").write_text(".gate/\n")
            (source / "input").write_text("source")
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            commit_fixture(source)
            script = "#!/bin/bash\nprintf '%s\\n' \"a'b\"\n"
            previous = Path.cwd()
            try:
                os.chdir(source)
                expected = GATE["fingerprint"]()
                GATE["snapshot"](copy, (root,), {GATE["WINDOWS_SCRIPT"]: script})
                self.assertEqual((copy / GATE["WINDOWS_SCRIPT"]).read_bytes(), script.encode("utf-8"))
                committed = subprocess.check_output(["git", "-C", str(copy), "show", "HEAD:" + GATE["WINDOWS_SCRIPT"]])
                self.assertEqual(committed, script.encode("utf-8"))
                self.assertFalse((source / GATE["WINDOWS_SCRIPT"]).exists())
                self.assertEqual(GATE["fingerprint"](), expected)
            finally:
                os.chdir(previous)

    def test_ci_commands_are_read_not_reimplemented(self):
        first, second = GATE["plan"]()
        self.assertEqual([(j.name, j.platform, j.vcpus) for j in first], [("quick", "linux", 4)])
        self.assertEqual([(j.name, j.platform, j.vcpus) for j in second], [
            ("linux-suite", "linux", 8), ("windows-suite", "windows", 2),
            ("linux-llama-parity", "linux", 4)])
        for job in second[:2]:
            scripts = "\n".join(s.script for s in job.steps)
            offline = " --offline" if job.platform == "windows" else ""
            self.assertIn(f"cargo clippy --locked{offline} $crates --all-targets -- -D warnings", scripts)
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

    def test_phase_two_linux_vcpu_sum_stays_within_motor_budget(self):
        phases = GATE["plan"]()
        self.assertLessEqual(sum(job.vcpus for job in phases[1] if job.platform == "linux"), 16)
        for jobs in phases:
            self.assertLessEqual(sum(job.vcpus for job in jobs if job.platform == "linux"), 16)

    def test_windows_cargo_uses_only_locked_prefetched_dependencies(self):
        _, jobs = GATE["plan"]()
        windows = next(job for job in jobs if job.platform == "windows")
        self.assertEqual(windows.env["CARGO_NET_OFFLINE"], "true")
        for name, expected in {
            "fmt": "cargo tree --locked --offline --workspace",
            "clippy": "cargo clippy --locked --offline $crates --all-targets -- -D warnings",
            "nextest": "cargo nextest run --locked --offline $crates --features synapse-module/test-support",
            "Windows llama CPU build gate": "cargo build --locked --offline -p synapse-worker-llama --no-default-features --features cpu --release",
        }.items():
            self.assertIn(expected, next(step.script for step in windows.steps if step.name == name))
        llama = next(step.script for step in windows.steps if step.name == "Windows llama CPU build gate")
        self.assertEqual(llama.count("cargo build --locked --offline -p synapse-worker-llama"), 2)
        self.assertTrue(all("CARGO_NET_OFFLINE" not in job.env for job in jobs if job.platform == "linux"))

    def test_windows_shell_keeps_workflow_commands_without_runner_python_or_git(self):
        job = next(job for job in GATE["plan"]()[1] if job.platform == "windows")
        script = GATE["windows_command"](job, "submitted-tree")[-1]
        tokens = shlex.split(script, comments=True)
        sources = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "-c"]
        sources += [base64.b64decode(value).decode("utf-16le") for value in re.findall(r"\bgate_ps ([A-Za-z0-9+/=]+)", script)]
        text = "\n".join(sources)
        for expected in ("cargo clippy --locked --offline $crates --all-targets -- -D warnings",
                         "cargo nextest run --locked --offline $crates --features synapse-module/test-support",
                         "--cargo-profile ci-test --timings --cargo-verbose", "--retries 0", "count:1/1",
                         "not binary(skeleton_e2e)", "--features cpu --release", "x86_64-pc-windows-msvc",
                         "libclang.dll", "GATE_EVIDENCE", "CreateEntryFromFile"):
            self.assertIn(expected, text)
        self.assertNotIn("scripts/gate.sh", script)
        self.assertNotIn("--worker", script)
        self.assertNotRegex(text, r"(?m)^\s*(?:python(?:3)?|git)\s")
        self.assertIn("powershell.exe", script)

    def test_windows_shell_executes_in_a_gitless_directory_and_stops_by_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "bin"
            tools.mkdir()
            cargo = tools / "cargo"
            cargo.write_text('#!/bin/sh\nprintf "%s|offline=%s\\n" "$*" "$CARGO_NET_OFFLINE" >> "$CALLS"\n'
                             'if [ "$1" = tree ]; then printf "synapse-core\\nsynapse-module\\n"; fi\n')
            cargo.chmod(0o755)
            for name, body in (("rustc", 'echo "host: x86_64-pc-windows-msvc"'), ("pwsh", "exit 0"),
                               ("git", "exit 99"), ("python", "exit 99"), ("python3", "exit 99"), ("py", "exit 99")):
                tool = tools / name
                tool.write_text("#!/bin/sh\n" + body + "\n")
                tool.chmod(0o755)
            job = next(job for job in GATE["plan"]()[1] if job.platform == "windows")
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                       CARGO_TARGET_DIR=str(root / "target"), CALLS=str(root / "calls"))
            script = root / GATE["WINDOWS_SCRIPT"]
            script.parent.mkdir()
            script.write_text(GATE["windows_command"](job, "tree")[-1])
            command = ["bash", GATE["WINDOWS_SCRIPT"]]
            subprocess.run(["bash", "-n", str(script)], check=True)
            success = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
            calls = (root / "calls").read_text()
            self.assertIn("clippy --locked --offline", calls)
            self.assertIn("--cargo-profile ci-test --timings --cargo-verbose", calls)
            self.assertIn("-E not binary(skeleton_e2e)|offline=true", calls)
            self.assertIn("RESULT windows-suite exit=0", success.stdout)
            self.assertFalse((root / ".git").exists())
            job.steps = [Step("versions", "true"), Step("bad", "exit 7"), Step("unreached", "touch should-not-exist")]
            script.write_text(GATE["windows_command"](job, "tree")[-1])
            failure = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(failure.returncode, 7, failure.stdout + failure.stderr)
            self.assertIn("FAIL windows-suite / bad: exit=7", failure.stdout)
            self.assertIn("TIMING windows-suite / bad", failure.stdout)
            self.assertFalse((root / "should-not-exist").exists())

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

    def test_train_hook_defaults_to_fast_with_explicit_full_opt_in(self):
        hook = (ROOT / "scripts/train-push.local.sh").read_text()
        self.assertIn("scripts/gate.sh", hook)
        self.assertIn("TRAIN_GATE_FULL", hook)
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
            for full, status in (("0", "0"), ("1", "0"), ("0", "7"), ("1", "7")):
                result = subprocess.run(["bash", "-uc", 'refuse() { echo "$*"; exit 1; }; say() { :; }; source "$1"',
                                         "hook-test", str(ROOT / "scripts/train-push.local.sh")], cwd=root,
                                        env=dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                                                 TRAIN_GATE_FULL=full, GATE_STATUS=status, GATE_ARGS=str(root / "args")),
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0 if status == "0" else 1, result.stdout + result.stderr)
                self.assertEqual((root / "args").read_text().strip(), "" if full == "1" else "--fast")


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

    def invoke(self, statuses, two_phases=True, through_gate=False, interrupt=False, outside_roots=False, capacity_jobs=(),
               capacity_reason="runner_full: hosted_capacity", windows_script_override=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools, events = root / "bin", root / "events"
            tools.mkdir()
            events.mkdir()
            checkout = root / "checkout"
            checkout.mkdir()
            (checkout / ".gitignore").write_text("target/\n.cortexkit/\n")
            (checkout / "source").write_text("snapshot input")
            (checkout / ".github/workflows").mkdir(parents=True)
            for name in ("tests.yml", "portable.yml"):
                (checkout / ".github/workflows" / name).write_text((ROOT / ".github/workflows" / name).read_text())
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
            commit_fixture(checkout)
            home = root / "home"
            allowed_root = home / "Work/Projects/CortexKit-scratch/synapse-gate"
            allowed_root.mkdir(parents=True)
            nested_temp = checkout / "target/custom-tmp"
            nested_temp.mkdir(parents=True)
            ck = tools / "ck"
            ck.write_text(f"#!{sys.executable}\n" + r'''
import base64, json, os, re, shlex, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ['motor', 'run']
remote = args[args.index('--') + 1:]
cwd = Path(args[args.index('--cwd') + 1]).resolve()
windows = args[args.index('--platform') + 1] == 'windows'
unsupported = False
if windows:
    file_form = remote == ['bash', '.gate/windows-job.sh']
    script_path = cwd / remote[1] if file_form else None
    script = script_path.read_text() if script_path and script_path.is_file() else ''
    match = re.search(r'^gate_job=(.+)$', script, re.M)
    name = shlex.split(match[1])[0] if match else 'windows-suite'
    unsupported = not file_form or not script
    scripts = [script]
    scripts.extend(base64.b64decode(value).decode('utf-16le') for value in re.findall(r'\bgate_ps ([A-Za-z0-9+/=]+)', script))
    for text in scripts:
        tokens = shlex.split(text, comments=True)
        if any(re.fullmatch(r'(?:git|python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?', token.replace('\\', '/').rsplit('/', 1)[-1], re.I) for token in tokens):
            unsupported = True
else:
    name = json.loads(args[-2])['name']
events = Path(os.environ['FAKE_EVENTS'])
(events / (name + '.json')).write_text(json.dumps(args))
if args[args.index('--platform') + 1] == 'windows' and '--net' in args:
    print('network_unsupported: Windows job network is unsupported', flush=True)
    sys.exit(200)
if unsupported:
    print('runner_prerequisite_unsupported: Windows command invokes Python/git or a worker wrapper', flush=True)
    sys.exit(200)
checkout = Path(os.environ['FAKE_CHECKOUT']).resolve()
if checkout in cwd.parents:
    print('lease_mismatch: --cwd overlaps the submitting checkout', flush=True)
    sys.exit(200)
allowed_roots = [Path(value).resolve() for value in json.loads(os.environ['FAKE_ALLOWED_ROOTS'])]
if not any(cwd == allowed or allowed in cwd.parents for allowed in allowed_roots):
    print('workspace_key_rejected: destination outside runner-owned roots: path=' + str(cwd), flush=True)
    sys.exit(200)
def cancelled(signum, frame):
    (events / (name + '.cancelled')).touch()
    sys.exit(203)
signal.signal(signal.SIGTERM, cancelled)
(events / (name + '.ready')).touch()
if name != 'quick':
    deadline = time.monotonic() + 5
    while len(list(events.glob('*suite.json'))) != 2 or not (events / 'linux-llama-parity.json').exists():
        if time.monotonic() > deadline:
            sys.exit(90)
        time.sleep(.01)
status = json.loads(os.environ['FAKE_STATUSES']).get(name, 0)
if name in json.loads(os.environ['FAKE_CAPACITY_JOBS']):
    print(os.environ['FAKE_CAPACITY_REASON'], flush=True)
    if status == 200:
        sys.exit(status)
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
            runner = GATE["Runner"]("tree", root / "unowned" if outside_roots else allowed_root / "test-run", evidence, (root,))
            if not through_gate and not interrupt:
                windows = next(job for job in second if job.platform == "windows")
                script = runner.scratch / windows.name / GATE["WINDOWS_SCRIPT"]
                script.parent.mkdir(parents=True)
                script.write_text(windows_script_override if windows_script_override is not None else
                                  GATE["windows_command"](windows, "tree")[-1])
            output = io.StringIO()
            status = 0
            with patch.dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                            FAKE_EVENTS=str(events), FAKE_STATUSES=json.dumps(statuses),
                            FAKE_CAPACITY_JOBS=json.dumps(list(capacity_jobs)),
                            FAKE_CAPACITY_REASON=capacity_reason,
                            FAKE_CHECKOUT=str(checkout), FAKE_ALLOWED_ROOTS=json.dumps([str(allowed_root)]),
                            HOME=str(home), TMPDIR=str(nested_temp),
                            FAKE_HOLD="1" if interrupt or any(statuses.get(j.name, 0) for j in second) else "0"), \
                    redirect_stdout(output):
                results = []
                if interrupt:
                    # The fixture changes HOME. Pass the parent's plan so the
                    # child does not need a YAML package installed in that home.
                    child = subprocess.Popen([sys.executable, "-c",
                                              "import json,runpy,sys; from pathlib import Path; gate=runpy.run_path(sys.argv[1]); "
                                              "phases=[[gate['Job'](**job) for job in phase] for phase in json.loads(sys.argv[3])]; "
                                              "raise SystemExit(gate['run_gate'](phases, (Path(sys.argv[2]),)))",
                                              str(ROOT / "scripts/gate.py"), str(root),
                                              json.dumps([[asdict(job) for job in jobs] for jobs in phases])], cwd=checkout,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    try:
                        # Real Git worktree preparation can be slow on macOS.
                        deadline = time.monotonic() + 60
                        while not all((events / (j.name + ".ready")).exists() for j in second):
                            if child.poll() is not None or time.monotonic() > deadline:
                                if child.poll() is None:
                                    child.terminate()
                                text, _ = child.communicate(timeout=10)
                                self.fail("gate did not launch every parallel job before interrupt:\n" + text)
                            time.sleep(.01)
                        child.send_signal(signal.SIGINT)
                        text, _ = child.communicate(timeout=10)
                        output.write(text)
                        status = child.returncode
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.communicate(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.communicate()
                elif through_gate:
                    previous = Path.cwd()
                    try:
                        os.chdir(checkout)
                        status = GATE["run_gate"](phases, (root,))
                    finally:
                        os.chdir(previous)
                else:
                    for jobs in phases:
                        status, result = GATE["parallel"](runner, jobs)
                        results.append(result)
                        if status:
                            break
            recorded = {p.stem: json.loads(p.read_text()) for p in events.glob("*.json")}
            completed = {p.name for p in events.iterdir()}
            if through_gate or interrupt:
                for args in recorded.values():
                    cwd = Path(args[args.index("--cwd") + 1]).resolve()
                    self.assertNotIn(checkout.resolve(), cwd.parents,
                                     "MOTOR --cwd must not be beneath the checkout, including target/")
                    self.assertIn(allowed_root.resolve(), cwd.parents,
                                  "MOTOR --cwd must resolve under its runner-owned roots")
                    self.assertFalse(cwd.parent.exists(), "temporary snapshots must be removed after joining jobs")
            return status, results, recorded, completed, output.getvalue()

    def test_motor_cwd_never_uses_checkout_target_even_with_nested_tmpdir(self):
        status, _, recorded, _, output = self.invoke({}, through_gate=True)
        self.assertEqual(status, 0, output)
        self.assertEqual(len(recorded), 4)
        self.assertIn("Phase 2", output)

    def test_fake_motor_rejects_destinations_outside_allowed_roots(self):
        status, _, recorded, _, output = self.invoke({}, two_phases=False, outside_roots=True)
        self.assertEqual(status, 200, output)
        self.assertEqual(set(recorded), {"quick"})
        self.assertIn("workspace_key_rejected: destination outside runner-owned roots", output)

    def test_fake_motor_rejects_windows_python_and_git_commands(self):
        for program in ("python3 --version", "git status"):
            status, _, _, _, output = self.invoke({}, windows_script_override="gate_job=windows-suite\n" + program)
            self.assertEqual(status, 200, output)
            self.assertIn("runner_prerequisite_unsupported", output)

    def test_capacity_refusal_is_retryable_not_a_check_failure(self):
        for reason in ("runner_full: hosted_capacity", "queue_wait_exceeded: waiting_on=permits"):
            status, _, _, _, output = self.invoke({"linux-suite": 200}, through_gate=True,
                                                capacity_jobs=("linux-suite",), capacity_reason=reason)
            self.assertEqual(status, 200, output)
            self.assertIn("CAPACITY job linux-suite: capacity, retry later", output)
            self.assertIn("Phase 2", output)
            self.assertIn("CAPACITY, retry later", output)
            self.assertNotIn("FAIL job linux-suite", output)
            self.assertNotIn("FIRST FAILURE", output)
            self.assertNotRegex(output, r"\bFAIL\b")

    def test_capacity_words_in_a_failed_check_are_not_an_admission_refusal(self):
        status, _, _, _, output = self.invoke({"linux-suite": 1}, capacity_jobs=("linux-suite",))
        self.assertEqual(status, 1, output)
        self.assertIn("FIRST FAILURE: linux-suite", output)
        self.assertNotIn("CAPACITY job linux-suite", output)

    def test_snapshot_cleanup_after_job_failure(self):
        status, _, _, completed, output = self.invoke({"windows-suite": 7}, through_gate=True)
        self.assertEqual(status, 7, output)
        self.assertIn("linux-suite.cancelled", completed)
        self.assertIn("linux-llama-parity.cancelled", completed)

    def test_snapshot_cleanup_after_ctrl_c_joins_cancelled_jobs(self):
        status, _, _, completed, output = self.invoke({}, interrupt=True)
        self.assertEqual(status, 130, output)
        for name in ("linux-suite", "windows-suite", "linux-llama-parity"):
            self.assertIn(name + ".cancelled", completed)

    def test_snapshot_cleanup_when_evidence_setup_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "snapshot"
            mkdir = Path.mkdir

            def setup(path, *args, **kwargs):
                if path.parts[0] == ".cortexkit":
                    raise PermissionError("evidence directory is not writable")
                return mkdir(path, *args, **kwargs)

            def prune(command, **kwargs):
                self.assertEqual(command, ["git", "worktree", "prune"])
                self.assertFalse(scratch.exists())
                return subprocess.CompletedProcess(command, 0)

            with redirect_stdout(io.StringIO()), \
                    patch.dict(GATE["run_gate"].__globals__, fingerprint=lambda: "tree", snapshot_directory=lambda roots: scratch), \
                    patch.object(Path, "mkdir", setup), patch.object(subprocess, "run", side_effect=prune) as run:
                with self.assertRaises(PermissionError):
                    GATE["run_gate"](([], []), (Path(directory),))
                run.assert_called_once()
            self.assertFalse(scratch.exists())

    def test_partial_worktree_is_removed_and_pruned_before_any_job_starts(self):
        phases = GATE["plan"]()
        with tempfile.TemporaryDirectory() as directory:
            source, scratch = Path(directory) / "source", Path(directory) / "run"
            source.mkdir()
            (source / "input").write_text("fixture")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            commit_fixture(source)
            previous = Path.cwd()
            git = subprocess.run

            def incomplete_worktree(destination, allowed_roots, extra_files=None):
                command = ["git", "worktree", "add", "--quiet", "--detach", "--no-checkout", str(destination), "HEAD"]
                git(command, check=True)
                raise subprocess.CalledProcessError(1, command)

            def prune(command, **kwargs):
                self.assertEqual(command, ["git", "worktree", "prune"])
                self.assertFalse(scratch.exists())
                return git(command, check=True)

            try:
                os.chdir(source)
                with redirect_stdout(io.StringIO()), \
                        patch.dict(GATE["run_gate"].__globals__, fingerprint=lambda: "tree", snapshot_directory=lambda roots: scratch,
                                   snapshot=incomplete_worktree), \
                        patch.object(subprocess, "run", side_effect=prune) as run, \
                        patch.object(GATE["Runner"], "run") as submit:
                    with self.assertRaises(subprocess.CalledProcessError):
                        GATE["run_gate"](phases, (Path(directory),))
                    run.assert_called_once()
                    submit.assert_not_called()
                listing = git(["git", "worktree", "list", "--porcelain"], capture_output=True, text=True, check=True).stdout
                self.assertNotIn(str(scratch), listing)
            finally:
                os.chdir(previous)

    def test_parallel_launch_platforms_resources_and_snapshot_isolation(self):
        status, results, recorded, completed, output = self.invoke({})
        self.assertEqual(status, 0, output)
        self.assertEqual(len(results[1]), 3)
        roots = set()
        for name, args in recorded.items():
            roots.add(args[args.index("--cwd") + 1])
            platform = args[args.index("--platform") + 1]
            self.assertEqual(platform, "windows" if name == "windows-suite" else "linux")
            if platform == "linux":
                self.assertIn("--net", args)
                self.assertEqual(args[args.index("--vcpus") + 1], "8" if name == "linux-suite" else "4")
            else:
                self.assertNotIn("--net", args)
                self.assertNotIn("--vcpus", args)
                self.assertNotIn("--return-path", args)
                remote = args[args.index("--") + 1:]
                self.assertEqual(remote, ["bash", GATE["WINDOWS_SCRIPT"]])
        self.assertEqual(len(roots), 4)
        self.assertIn("Retained 8 bytes of Windows build evidence", output)
        self.assertNotIn("GATE_EVIDENCE", output)

    def test_windows_remote_argv_is_flattening_safe(self):
        status, _, recorded, _, output = self.invoke({})
        args = recorded["windows-suite"]
        remote = args[args.index("--") + 1:]
        for argument in remote:
            self.assertFalse(any(char in argument for char in ("\n", "\r", "'", '"')),
                             "Windows remote argv must contain no newline or quote characters")
        self.assertEqual(remote, ["bash", ".gate/windows-job.sh"])
        self.assertEqual(status, 0, output)

    def test_linux_worker_json_round_trips_through_posix_argv_quoting(self):
        status, _, recorded, _, output = self.invoke({})
        self.assertEqual(status, 0, output)
        for name, args in recorded.items():
            if args[args.index("--platform") + 1] != "linux":
                continue
            remote = args[args.index("--") + 1:]
            self.assertEqual(remote[:3], ["bash", "scripts/gate.sh", "--worker"])
            self.assertIn('"', remote[3])
            self.assertNotIn("\n", remote[3])
            probe = subprocess.run(["bash", "-c", "printf '%s\\0' " + shlex.join(remote)], capture_output=True, check=True)
            received = probe.stdout.decode().split("\0")[:-1]
            self.assertEqual(received, remote)
            self.assertEqual(json.loads(received[3])["name"], name)

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
                    patch.dict(GATE["run_gate"].__globals__, fingerprint=lambda: "tree", snapshot=lambda destination, roots, extra=None: None,
                               snapshot_directory=lambda roots: Path(directory),
                               parallel=lambda runner, jobs: (status, {jobs[0].name: (status, .25)})), \
                    patch.object(Path, "mkdir"), \
                    patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                self.assertEqual(GATE["run_gate"]((first, second), (Path(directory),)), status)
            self.assertIn("PHASE TIMING | seconds | result", output.getvalue())
            self.assertIn("Phase 1", output.getvalue())
            self.assertIn("PASS" if status == 0 else "FAIL", output.getvalue())
            self.assertEqual("Phase 2" in output.getvalue(), status == 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
