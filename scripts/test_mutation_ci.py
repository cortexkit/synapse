"""Compile-only contracts that cannot be observed in a test-support build."""
import subprocess
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
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
