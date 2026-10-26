"""Contracts for the parallel CI lanes and the repository-local preflight."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib
import unittest

import yaml


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
PORTABLE = yaml.safe_load((ROOT / ".github/workflows/portable.yml").read_text())
SETUP = yaml.safe_load((ROOT / ".github/actions/ci-rust/action.yml").read_text())


class ParallelCiTests(unittest.TestCase):
    def test_windows_defender_tuning_precedes_cache_restore(self):
        steps = SETUP["runs"]["steps"]
        defender = next(s for s in steps if s.get("name") == "Configure Windows Defender for hosted CI builds")
        self.assertEqual(defender["if"], "runner.os == 'Windows'")
        self.assertEqual(defender["shell"], "pwsh")
        self.assertNotIn("continue-on-error", defender)
        self.assertLess(steps.index(defender), next(i for i, s in enumerate(steps) if s.get("uses") == "Swatinem/rust-cache@v2"))
        for path in ("$env:GITHUB_WORKSPACE", "$env:TEMP", "$env:RUNNER_TEMP", "$env:USERPROFILE\\.cargo"):
            self.assertIn(path, defender["run"])
        self.assertIn("Add-MpPreference -ExclusionPath $paths", defender["run"])
        self.assertIn("Set-MpPreference -DisableRealtimeMonitoring $true", defender["run"])
        self.assertIn("Get-MpComputerStatus", defender["run"])

    def test_aggregates_refuse_failed_skipped_and_cancelled_children(self):
        for platform in ("linux", "windows"):
            gate = WORKFLOW["jobs"][platform]
            self.assertEqual(gate["name"], platform)
            self.assertEqual(gate["if"], "${{ always() }}")
            expected_needs = [f"{platform}-checks"] + (["test"] if platform == "linux" else [])
            self.assertEqual(gate["needs"], expected_needs)
            script = gate["steps"][0]["run"]
            statuses = ("success", "failure", "cancelled", "skipped")
            for native, guards in itertools.product(statuses, repeat=2):
                with self.subTest(platform=platform, native=native, guards=guards):
                    result = subprocess.run(
                        ["bash", "-e", "-c", script],
                        env=dict(os.environ, NATIVE_RESULT=native, GUARDS_RESULT=guards),
                        capture_output=True,
                    )
                    expected = native == "success" and (platform == "windows" or guards == "success")
                    self.assertEqual(result.returncode == 0, expected)

    def test_native_lanes_start_independently_and_shards_cover_the_suite(self):
        jobs = PORTABLE["jobs"]
        self.assertEqual(set(jobs), {"clippy", "nextest", "llama"})
        for job in jobs.values():
            self.assertNotIn("needs", job)
            self.assertNotIn("if", job)
            self.assertNotIn("continue-on-error", job)
        for platform, count in (("linux", 4), ("windows", 1)):
            caller = WORKFLOW["jobs"][f"{platform}-checks"]
            self.assertEqual(caller["uses"], "./.github/workflows/portable.yml")
            self.assertEqual(caller["with"]["platform"], platform)
            self.assertEqual(json.loads(caller["with"]["partitions"]),
                             [f"{i}/{count}" for i in range(1, count + 1)])
        self.assertEqual(jobs["nextest"]["strategy"], {
            "fail-fast": False, "matrix": {"partition": "${{ fromJSON(inputs.partitions) }}"},
        })

    def test_cache_lanes_are_isolated_and_only_one_shard_writes(self):
        cache = next(s["with"] for s in SETUP["runs"]["steps"] if s.get("uses") == "Swatinem/rust-cache@v2")
        self.assertEqual(cache["shared-key"], "ci-${{ inputs.lane }}")
        self.assertTrue(cache["cache-workspace-crates"])
        self.assertEqual(cache["save-if"], "${{ inputs.save-cache == 'true' }}")
        lanes = []
        for name, job in PORTABLE["jobs"].items():
            setup = next(s["with"] for s in job["steps"] if s.get("uses") == "./synapse/.github/actions/ci-rust")
            lanes.append(setup["lane"])
            if name == "nextest":
                self.assertEqual(setup["lane"],
                                 "${{ inputs.platform == 'windows' && 'nextest-ci-test-v1' || 'nextest' }}")
                self.assertEqual(setup["save-cache"], "${{ startsWith(matrix.partition, '1/') }}")
            else:
                self.assertNotIn("save-cache", setup)
        self.assertEqual(len(set(lanes)), len(lanes))

    def test_windows_profile_keeps_release_semantics_and_optimized_dependencies(self):
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        profile = manifest["profile"]["ci-test"]
        self.assertEqual(profile["inherits"], "release")
        self.assertEqual(profile["opt-level"], 1)
        self.assertEqual(profile["codegen-units"], 256)
        self.assertIs(profile["debug-assertions"], False)
        self.assertIs(profile["overflow-checks"], False)
        self.assertEqual(profile["package"], {"*": {"opt-level": 3, "codegen-units": 16}})

    def test_nextest_profile_changes_only_windows_build_not_test_selection(self):
        job = PORTABLE["jobs"]["nextest"]
        step = next(s for s in job["steps"] if s.get("name") == "nextest")
        packages = ["synapse-core", "synapse-native-probe", "synapse-module", "synapse-certify",
                    "synapse-certify-runner", "synapse-worker-llama", "owned-decode-worker",
                    "synapse-opctl", "synapse-engine-cuda",
                    "synapse-worker-cuda", "synapse-worker-vulkan"]
        expected_crates = list(itertools.chain.from_iterable(("-p", p) for p in packages))
        self.assertEqual(PORTABLE["env"]["SYNAPSE_TEST_CRATES"].split(), expected_crates)
        # Clippy still covers the macOS-only decode worker, which has no tests off macOS.
        self.assertEqual(PORTABLE["env"]["SYNAPSE_CRATES"].split(),
                         expected_crates[:14] + ["-p", "synapse-worker-decode"] + expected_crates[14:])
        for platform, partition in (("linux", "2/4"), ("windows", "1/1")):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cargo = root / "cargo"
                cargo.write_text("#!/usr/bin/env python3\nimport json, sys\n"
                                 "from pathlib import Path\n"
                                 "Path('args.json').write_text(json.dumps(sys.argv[1:]))\n")
                cargo.chmod(0o755)
                script = (step["run"].replace("${{ inputs.platform }}", platform)
                          .replace("${{ matrix.partition }}", partition))
                result = subprocess.run(
                    ["bash", "-e", "-o", "pipefail", "-c", script], cwd=root,
                    env=dict(os.environ, PATH=f"{root}{os.pathsep}{os.environ['PATH']}",
                             SYNAPSE_TEST_CRATES=PORTABLE["env"]["SYNAPSE_TEST_CRATES"]),
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                build = ["--cargo-profile", "ci-test", "--timings", "--cargo-verbose"] if platform == "windows" else []
                self.assertEqual(json.loads((root / "args.json").read_text()),
                                 ["nextest", "run", "--locked"] + expected_crates +
                                 ["--features", "synapse-module/test-support"] + build +
                                 ["--retries", "0", "--partition", f"count:{partition}",
                                  "-E", "not binary(skeleton_e2e)"])


class LocalPreflightTests(unittest.TestCase):
    def run_preflight(self, windows_status=0, native_status=0):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            tools = root / "bin"
            tools.mkdir()
            local, remote = root / "local", root / "remote"
            for checkout in (local, remote):
                (checkout / "scripts/lib").mkdir(parents=True)
                for name in ("train-preflight.py", "lib/remote_gate.py"):
                    (checkout / "scripts" / name).write_text((ROOT / "scripts" / name).read_text())
                for name in ("check-train-preconditions.sh", "check-no-external-path-deps.sh",
                             "check-module-runtime-deps.py"):
                    (checkout / "scripts" / name).write_text("")
                # Mirror the repository's ignore rule: importing remote_gate writes a
                # .pyc that embeds the source mtime, which differs between the two
                # copies and would otherwise change the tree fingerprint.
                (checkout / ".gitignore").write_text("__pycache__/\n")
                subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            import sys
            cargo = tools / "cargo"
            cargo.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
remote = os.getenv('REMOTE') == '1'
name = 'windows' if remote else 'native'
events = Path(os.environ['EVENTS'])
with (events / (name + '.commands')).open('a') as log:
    log.write(json.dumps(args) + '\\n')
if args[0] != 'clippy' or '--version' in args:
    sys.exit(0)
assert ('--target' in args) == remote
Path(events / (name + '.json')).write_text(json.dumps({'args': args,
    'target': os.getenv('CARGO_TARGET_DIR'), 'cc': os.getenv('CC_x86_64_pc_windows_gnu'),
    'ar': os.getenv('AR_x86_64_pc_windows_gnu')}))
other = 'native' if remote else 'windows'
deadline = time.monotonic() + 5
while not (events / (other + '.json')).exists():
    if time.monotonic() > deadline:
        sys.exit(90)
    time.sleep(0.01)
(events / (name + '.joined')).touch()
sys.exit(int(os.environ['WINDOWS_STATUS' if remote else 'NATIVE_STATUS']))
''')
            cargo.chmod(0o755)
            ck = tools / "ck"
            ck.write_text(f"#!{sys.executable}\n" + '''
import os, subprocess, sys
args = sys.argv[1:]
assert args[:2] == ['motor', 'run']
assert args[args.index('--vcpus') + 1] == '4'
sys.exit(subprocess.run(args[args.index('--') + 1:], cwd=os.environ['REMOTE_ROOT'],
    env=dict(os.environ, REMOTE='1')).returncode)
''')
            ck.chmod(0o755)
            for name in ("gh", "xcrun"):
                tool = tools / name
                tool.write_text("#!/bin/sh\nexit 0\n")
                tool.chmod(0o755)
            result = subprocess.run(
                [sys.executable, "-c", "import runpy; from unittest.mock import patch; "
                 "p=patch('platform.system', return_value='Darwin'); p.start(); "
                 "runpy.run_path('scripts/train-preflight.py', run_name='__main__')"],
                cwd=local,
                env=dict(os.environ, PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
                         REMOTE_ROOT=str(remote), EVENTS=str(root),
                         WINDOWS_STATUS=str(windows_status), NATIVE_STATUS=str(native_status)),
                text=True, capture_output=True, timeout=15,
            )
            windows = json.loads((root / "windows.json").read_text())
            self.assertEqual(windows["target"], "target/train-windows-clippy")
            self.assertEqual(windows["cc"], "x86_64-w64-mingw32-gcc")
            self.assertEqual(windows["ar"], "x86_64-w64-mingw32-ar")
            self.assertTrue((root / "windows.joined").exists(), result.stdout + result.stderr)
            self.assertTrue((root / "native.joined").exists(), result.stdout + result.stderr)
            self.assertIn("TREE linux ", result.stdout)
            self.assertIn("TREE macos ", result.stdout)
            self.assertIn("TIMING preflight wall", result.stdout)
            return result

    def test_windows_cross_clippy_overlaps_native_preflight(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_either_failure_refuses_after_both_children_join(self):
        for windows, native in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(windows=windows, native=native):
                result = self.run_preflight(windows, native)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
