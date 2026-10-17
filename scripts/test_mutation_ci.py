"""Compile-only contracts that cannot be observed in a test-support build."""
import os
import subprocess
import sys
import unittest


class ProductionEngineGuardTests(unittest.TestCase):
    def test_production_allow_list_compiles(self):
        # Unit and integration builds enable test-support through a self dev
        # dependency. Only a normal library build exercises the const assertion
        # keeping the deterministic double out of the production allow-list.
        result = subprocess.run(
            ["cargo", "check", "--locked", "-p", "synapse-module", "--lib"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


class MutationCiTests(unittest.TestCase):
    def test_catalogue_check_is_required(self):
        # Run the checker against the real tests.yml to verify that removing
        # required catalogue validation is refused before merging. A fixture
        # alone would not test the live workflow; replay runs asynchronously.
        result = subprocess.run(
            ["bash", "scripts/check-train-preconditions.sh"],
            env=dict(os.environ, PYTHON=sys.executable),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


class MutationAuditTests(unittest.TestCase):
    def test_nightly_shards_have_separate_evidence(self):
        from pathlib import Path
        import yaml

        workflow = yaml.safe_load(Path(".github/workflows/mutations-audit.yml").read_text())
        audit = workflow["jobs"]["audit"]
        self.assertEqual(audit["strategy"]["matrix"]["shard"], ["1/4", "2/4", "3/4", "4/4"])
        run = next(s["run"] for s in audit["steps"] if s.get("name") == "Audit all package test targets")
        self.assertIn("--shard '${{ matrix.shard }}'", run)
        evidence = next(s["with"]["name"] for s in audit["steps"] if s.get("name") == "Retain breadth evidence")
        self.assertIn("strategy.job-index", evidence)

    def test_nightly_builds_feature_off_gpu_workers_before_audit(self):
        from pathlib import Path
        import yaml

        workflow = yaml.safe_load(Path(".github/workflows/mutations-audit.yml").read_text())
        steps = workflow["jobs"]["audit"]["steps"]
        audit_index = next(
            i for i, step in enumerate(steps)
            if step.get("name") == "Audit all package test targets"
        )
        fixture_command = (
            "cargo build --locked --no-default-features "
            "-p synapse-worker-cuda -p synapse-worker-vulkan"
        )
        self.assertTrue(
            any(
                step.get("run", "").strip() == fixture_command
                and step.get("shell") == "bash"
                and "if" not in step
                for step in steps[:audit_index]
            ),
            "every nightly platform needs feature-off worker binaries before the baseline",
        )

    def test_expected_selection_is_removed_without_changing_anchors(self):
        import runpy
        import tomllib

        prepare = runpy.run_path("scripts/prepare-mutation-audit.py")["prepare"]
        source = '''prebuild = [{ name = "fixture", command = ["cargo", "build", "--locked"] }]
[[control]]
id = "preserve-anchor"
runner = "cargo"
select = "expected"
old = """
select = "expected"
"""
new = "a different anchor"
expect_red = ["tests::guard"]
'''
        rendered = prepare(source)
        parsed = tomllib.loads(rendered)
        self.assertNotIn("select", parsed["control"][0], "nightly must observe the whole target")
        self.assertEqual(parsed["control"][0]["old"], 'select = "expected"\n')
        self.assertEqual(parsed["control"][0]["new"], "a different anchor")
        self.assertEqual(parsed["control"][0]["expect_red"], ["tests::guard"])
        self.assertEqual(parsed["prebuild"], [{"name": "fixture", "command": ["cargo", "build", "--locked"]}])

    def test_rows_running_the_same_package_suite_share_one_timeout(self):
        import runpy
        import tomllib
        from pathlib import Path

        # The breadth audit drops each row's expected-test selection, so every
        # cargo or nextest row runs its whole package test suite, and rows with
        # the same runner, package and features share a single baseline run.
        # ckdev-mutate runs that shared baseline under the timeout_s of the first
        # such row in the shard, and reports a timeout as a baseline failure on
        # every row sharing it. A row sized for its own few tests (60 s against a
        # module suite that takes about 315 s on GitHub's Linux runner) therefore
        # breaks the whole group whenever sharding makes it the first row.
        helpers = runpy.run_path("scripts/prepare-mutation-audit.py")
        catalogue = tomllib.loads(helpers["prepare"](helpers["load_source"](Path("mutations.d"))))
        timeouts = {}
        for row in catalogue["control"]:
            if row.get("runner") not in ("cargo", "nextest"):
                continue
            suite = (
                row["runner"],
                row.get("package"),
                repr(row.get("features")),
                repr(row.get("ignored")),
            )
            timeouts.setdefault(suite, {}).setdefault(row["timeout_s"], []).append(row["id"])
        mixed = {suite: rows for suite, rows in timeouts.items() if len(rows) > 1}
        self.assertEqual(mixed, {}, "rows that share a package suite need one timeout_s")

    def test_directory_fragments_are_loaded_in_filename_order(self):
        import runpy
        import tempfile
        import tomllib
        from pathlib import Path

        helpers = runpy.run_path("scripts/prepare-mutation-audit.py")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "last.toml").write_text(
                '[[control]]\nid = "last"\nselect = "expected"\n',
                encoding="utf-8",
            )
            (source / "00-prebuild.toml").write_text(
                'prebuild = [{ name = "fixture", command = ["cargo", "build", "--locked"] }]\n',
                encoding="utf-8",
            )
            (source / "first.toml").write_text(
                '[[control]]\nid = "first"\nselect = "expected"\n',
                encoding="utf-8",
            )
            (source / ".hidden.toml").write_text(
                '[[control]]\nid = "hidden"\n', encoding="utf-8"
            )
            catalogue = helpers["load_source"](source)
            parsed = tomllib.loads(helpers["prepare"](catalogue))

        self.assertEqual([row["id"] for row in parsed["control"]], ["first", "last"])
        self.assertEqual(
            parsed["prebuild"],
            [{"name": "fixture", "command": ["cargo", "build", "--locked"]}],
        )


if __name__ == "__main__":
    unittest.main()
