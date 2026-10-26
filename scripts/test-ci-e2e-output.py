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
PORTABLE = yaml.safe_load((ROOT / ".github/workflows/portable.yml").read_text())
SETUP = yaml.safe_load((ROOT / ".github/actions/ci-rust/action.yml").read_text())
GUARDS = WORKFLOW["jobs"]["test"]["steps"]
STEPS = GUARDS + [s for job in PORTABLE["jobs"].values() for s in job["steps"]]
E2E_STEPS = WORKFLOW["jobs"]["e2e"]["steps"]


def step(name):
    return next(s for s in STEPS + E2E_STEPS if s.get("name") == name)


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
            "if selection != 'not binary(skeleton_e2e)':\n"
            "    sys.exit(int(os.getenv('FAKE_STATUS', '0')))\n"
        )
        cargo.chmod(0o755)
        self.env = dict(os.environ)
        self.env.update(
            PATH=f"{self.root}{os.pathsep}{os.environ['PATH']}",
            SYNAPSE_TEST_CRATES="-p synapse-module -p synapse-worker-llama",
        )
        self.env.pop("FAKE_STATUS", None)

    def run_step(self, name, platform="linux", partition="1/1"):
        script = (step(name)["run"].replace("${{ matrix.name }}", platform)
                  .replace("${{ inputs.platform }}", platform)
                  .replace("${{ matrix.partition }}", partition))
        return subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
        )

    def test_push_partitions_once_and_nightly_runs_only_the_e2e_binary(self):
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform):
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                partitions = json.loads(WORKFLOW["jobs"][f"{platform}-checks"]["with"]["partitions"])
                for partition in partitions:
                    run = self.run_step("nextest", platform, partition)
                    self.assertEqual(run.returncode, 0, run.stderr)
                e2e = self.run_step("skeleton e2e", platform)
                self.assertEqual(e2e.returncode, 0, e2e.stderr)
                calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
                self.assertEqual(len(calls), len(partitions) + 1)
                self.assertEqual([call[call.index("-E") + 1] for call in calls],
                                 ["not binary(skeleton_e2e)"] * len(partitions) + ["binary(skeleton_e2e)"])
                self.assertEqual([call[call.index("--partition") + 1] for call in calls[:-1]],
                                 [f"count:{partition}" for partition in partitions])
                for call in calls[:-1]:
                    self.assertEqual(call[call.index("--retries") + 1], "0")
                # Push tests use a cheaper release-derived Cargo profile; nightly
                # whole-process fixtures still exercise the actual release build.
                for call in calls[:-1]:
                    self.assertNotIn("--release", call)
                    self.assertEqual("--cargo-profile" in call, platform == "windows")
                    if platform == "windows":
                        self.assertEqual(call[call.index("--cargo-profile") + 1], "ci-test")
                self.assertEqual("--release" in calls[-1], platform == "windows")
                self.assertIn("--no-capture", calls[-1], "whole-process fixtures must remain serial")
                # The e2e targets require the e2e feature; only the nightly lane builds them.
                self.assertIn("e2e", calls[-1][calls[-1].index("--features") + 1].split(","))
                for call in calls[:-1]:
                    self.assertEqual(call[call.index("--features") + 1], "synapse-module/test-support")
                for call in calls:
                    self.assertIn("--locked", call)

    def test_nextest_failure_fails_the_e2e_step(self):
        self.env["FAKE_STATUS"] = "42"
        run = self.run_step("skeleton e2e")
        self.assertEqual(run.returncode, 42, run.stderr)

    def test_parity_cache_precedes_tests_and_covers_its_target(self):
        cache = step("Restore parity Rust cache")
        self.assertLess(STEPS.index(cache), STEPS.index(step("Parity manifest and evaluator tests")))
        self.assertEqual(cache["with"]["workspaces"], "synapse/bench/parity -> target")
        root_cache = next(s for s in SETUP["runs"]["steps"] if s.get("uses") == "Swatinem/rust-cache@v2"
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
