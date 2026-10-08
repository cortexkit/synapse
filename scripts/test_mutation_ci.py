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
        # Run the real checker over the real workflow; a fixture-only self-test
        # would not prove that a removed CI replay is refused before landing.
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


if __name__ == "__main__":
    unittest.main()
