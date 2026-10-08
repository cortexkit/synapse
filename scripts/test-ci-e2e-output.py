#!/usr/bin/env python3
"""Exercise CI's actual shell steps without compiling or loading model fixtures."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
STEPS = WORKFLOW["jobs"]["test"]["steps"]


def step(name):
    return next(s for s in STEPS if s.get("name") == name)


class E2eOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cargo = self.root / "cargo"
        cargo.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with Path('calls.jsonl').open('a') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "selection = sys.argv[sys.argv.index('-E') + 1]\n"
            "if selection == 'not binary(skeleton_e2e)':\n"
            "    print('skipping optional worker checkpoint')\n"
            "else:\n"
            "    print('skipping missing e2e fixture' if os.getenv('FAKE_SKIP') else 'e2e fixture ran')\n"
            "    sys.exit(int(os.getenv('FAKE_STATUS', '0')))\n"
        )
        cargo.chmod(0o755)
        self.env = dict(os.environ)
        self.env.update(
            PATH=f"{self.root}{os.pathsep}{os.environ['PATH']}",
            SYNAPSE_CRATES="-p synapse-module -p synapse-worker-llama",
        )
        self.env.pop("FAKE_SKIP", None)
        self.env.pop("FAKE_STATUS", None)

    def run_step(self, name, platform="linux"):
        script = step(name)["run"].replace("${{ matrix.name }}", platform)
        return subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
        )

    def test_suite_partitions_run_once_and_guard_reuses_output(self):
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform):
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                run = self.run_step("nextest", platform)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("skipping optional worker checkpoint", run.stdout)
                guard = self.run_step("Assert no silently-skipped e2e", platform)
                self.assertEqual(guard.returncode, 0, guard.stderr)
                calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
                self.assertEqual(len(calls), 2, "the guard must not rerun tests")
                self.assertEqual([call[call.index("-E") + 1] for call in calls],
                                 ["not binary(skeleton_e2e)", "binary(skeleton_e2e)"])
                for call in calls:
                    self.assertEqual("--release" in call, platform == "windows")
                self.assertIn("--no-capture", calls[1])

    def test_fixture_skip_is_refused(self):
        self.env["FAKE_SKIP"] = "1"
        run = self.run_step("nextest")
        self.assertEqual(run.returncode, 0, run.stderr)
        guard = self.run_step("Assert no silently-skipped e2e")
        self.assertNotEqual(guard.returncode, 0, "a skipping first run must fail the guard")
        self.assertIn("fixture e2e silently skipped", guard.stdout)

    def test_nextest_failure_survives_tee(self):
        self.env["FAKE_STATUS"] = "42"
        run = self.run_step("nextest")
        self.assertEqual(run.returncode, 42, run.stderr)

    def test_missing_transcript_is_refused(self):
        guard = self.run_step("Assert no silently-skipped e2e")
        self.assertNotEqual(guard.returncode, 0)

    def test_parity_cache_precedes_tests_and_covers_its_target(self):
        cache = step("Restore parity Rust cache")
        self.assertLess(STEPS.index(cache), STEPS.index(step("Parity manifest and evaluator tests")))
        self.assertEqual(cache["with"]["workspaces"], "synapse/bench/parity -> target")
        root_cache = next(s for s in STEPS if s.get("uses") == "Swatinem/rust-cache@v2"
                          and s["with"].get("workspaces") == "synapse")
        self.assertNotEqual(cache["with"]["prefix-key"], root_cache["with"]["prefix-key"])


class ReleaseTargetsTests(unittest.TestCase):
    def test_release_build_selects_only_shipped_rust_binaries(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/release-candidate.yml").read_text())
        job = workflow["jobs"]["build"]
        build = next(s for s in job["steps"] if s.get("name") == "Build every release asset once")
        for matrix in job["strategy"]["matrix"]["include"]:
            with self.subTest(platform=matrix["os_arch"]), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cargo = root / "cargo"
                cargo.write_text("#!/usr/bin/env python3\nimport json, sys\n"
                                 "from pathlib import Path\n"
                                 "Path('args.json').write_text(json.dumps(sys.argv[1:]))\n")
                cargo.chmod(0o755)
                script = build["run"]
                for key in ("packages", "features", "binaries"):
                    script = script.replace("${{ matrix." + key + " }}", matrix[key])
                env = dict(os.environ, PATH=f"{root}{os.pathsep}{os.environ['PATH']}")
                result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script],
                                        cwd=root, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads((root / "args.json").read_text())
                selected = {args[i + 1] for i, arg in enumerate(args) if arg == "--bin"}
                self.assertIn("ck-synapse", selected)
                self.assertIn("--release", args)
                self.assertIn("--locked", args)
                self.assertEqual(selected, set(matrix["binaries"].split()) - {"ck-synapse-worker-ane-swift"})
                self.assertTrue(all(name.startswith("ck-") for name in selected))
                self.assertTrue(selected.isdisjoint({"inline_embed_throughput", "subc_call",
                                                      "synapse-worker-timeout-mock"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
