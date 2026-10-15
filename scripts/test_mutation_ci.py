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
    def test_touched_rows_are_required(self):
        # Run the real precondition checker over the real tests.yml. A fixture
        # alone would not prove that deleting the CI step that replays the
        # mutations touched by a change is refused before that change lands.
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
