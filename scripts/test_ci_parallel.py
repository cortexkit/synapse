"""Contracts for the parallel CI lanes and the repository-local preflight."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
PORTABLE = yaml.safe_load((ROOT / ".github/workflows/portable.yml").read_text())
SETUP = yaml.safe_load((ROOT / ".github/actions/ci-rust/action.yml").read_text())


class ParallelCiTests(unittest.TestCase):
    def test_aggregates_refuse_failed_skipped_and_cancelled_children(self):
        for platform in ("linux", "windows"):
            gate = WORKFLOW["jobs"][platform]
            self.assertEqual(gate["name"], platform)
            self.assertEqual(gate["if"], "${{ always() }}")
            expected_needs = [f"{platform}-checks", f"{platform}-catalogue"] + (["test"] if platform == "linux" else [])
            self.assertEqual(gate["needs"], expected_needs)
            script = gate["steps"][0]["run"]
            statuses = ("success", "failure", "cancelled", "skipped")
            for native, catalogue, guards in itertools.product(statuses, repeat=3):
                with self.subTest(platform=platform, native=native, catalogue=catalogue, guards=guards):
                    result = subprocess.run(
                        ["bash", "-e", "-c", script],
                        env=dict(os.environ, NATIVE_RESULT=native, CATALOGUE_RESULT=catalogue, GUARDS_RESULT=guards),
                        capture_output=True,
                    )
                    expected = native == catalogue == "success" and (platform == "windows" or guards == "success")
                    self.assertEqual(result.returncode == 0, expected)

    def test_native_lanes_start_independently_and_shards_cover_the_suite(self):
        jobs = PORTABLE["jobs"]
        self.assertEqual(set(jobs), {"clippy", "nextest", "llama"})
        for job in list(jobs.values()) + [WORKFLOW["jobs"][f"{p}-catalogue"] for p in ("linux", "windows")]:
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
        jobs = dict(PORTABLE["jobs"], catalogue=WORKFLOW["jobs"]["linux-catalogue"])
        for name, job in jobs.items():
            setup = next(s["with"] for s in job["steps"] if s.get("uses") == "./synapse/.github/actions/ci-rust")
            lanes.append(setup["lane"])
            if name == "nextest":
                self.assertEqual(setup["save-cache"], "${{ startsWith(matrix.partition, '1/') }}")
            else:
                self.assertNotIn("save-cache", setup)
        self.assertEqual(len(set(lanes)), len(lanes))


class LocalPreflightTests(unittest.TestCase):
    def run_preflight(self, windows_status=0, native_status=0):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = root / "bin"
            tools.mkdir()
            # The two clippy invocations rendezvous before finishing. A serial
            # implementation times out instead of passing a timing-based guess.
            cargo = tools / "cargo"
            cargo.write_text("""#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[0] != 'clippy':
    sys.exit(0)
windows = '--target' in args
name = 'windows' if windows else 'native'
other = 'native' if windows else 'windows'
Path(name + '.json').write_text(json.dumps({'args': args, 'target': os.getenv('CARGO_TARGET_DIR'),
    'cc': os.getenv('CC_x86_64_pc_windows_gnu'), 'ar': os.getenv('AR_x86_64_pc_windows_gnu')}))
deadline = time.monotonic() + 5
while not Path(other + '.json').exists():
    if time.monotonic() > deadline:
        sys.exit(90)
    time.sleep(0.01)
Path(name + '.joined').touch()
sys.exit(int(os.environ['WINDOWS_STATUS' if windows else 'NATIVE_STATUS']))
""")
            cargo.chmod(0o755)
            for name, body in {
                "uname": "echo Darwin",
                "bash": "exit 0",
                "python3": "exit 0",
                "gh": "exit 0",
            }.items():
                tool = tools / name
                tool.write_text(f"#!/bin/sh\n{body}\n")
                tool.chmod(0o755)
            # The cargo fixture's interpreter must be real, not the checker stub.
            import sys
            cargo.write_text(cargo.read_text().replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
            result = subprocess.run(
                ["/bin/bash", "-e", "-c",
                 'refuse() { echo "$*"; exit 1; }; say() { :; }; source "$PREFLIGHT"'],
                cwd=root,
                env=dict(os.environ, PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
                         PREFLIGHT=str(ROOT / "scripts/train-push.local.sh"),
                         WINDOWS_STATUS=str(windows_status), NATIVE_STATUS=str(native_status)),
                text=True, capture_output=True, timeout=15,
            )
            windows = json.loads((root / "windows.json").read_text())
            self.assertEqual(windows["args"], [
                "clippy", "--locked", "-p", "synapse-module", "--features", "test-support",
                "--all-targets", "--target", "x86_64-pc-windows-gnu", "--", "-D", "warnings",
            ])
            self.assertEqual(windows["target"], str(root / "target/train-windows-clippy"))
            self.assertEqual(windows["cc"], str(root / "scripts/lib/zig-cc-windows.sh"))
            self.assertEqual(windows["ar"], str(root / "scripts/lib/zig-ar.sh"))
            self.assertTrue((root / "windows.joined").exists())
            self.assertTrue((root / "native.joined").exists())
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
