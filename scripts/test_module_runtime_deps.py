"""Test the module dependency guard with synthetic inputs and the actual Cargo tree."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts/check-module-runtime-deps.py"
SPEC = importlib.util.spec_from_file_location("module_runtime_deps", CHECK)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class ModuleRuntimeDependencyTests(unittest.TestCase):
    def test_module_normal_tree_excludes_daemon_and_presence(self):
        result = subprocess.run([sys.executable, str(CHECK)], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"Inspected [1-9][0-9]* normal-edge packages")

    def test_each_forbidden_package_is_refused(self):
        for package in ("subc-daemon", "subc-presence"):
            with self.subTest(package=package):
                with self.assertRaisesRegex(ValueError, "forbidden packages"):
                    GUARD.inspect_packages(f"synapse-module v0.1.0\n{package} v0.1.0\n")

    def test_empty_or_wrong_root_tree_is_refused(self):
        for tree in ("", "synapse-module v0.1.0\n", "serde v1.0.0\nsha2 v0.10.0\n"):
            with self.subTest(tree=tree):
                with self.assertRaisesRegex(ValueError, "empty or incomplete"):
                    GUARD.inspect_packages(tree)

    def test_complete_normal_tree_is_counted(self):
        self.assertEqual(GUARD.inspect_packages("synapse-module v0.1.0\nserde v1.0.0\nserde v1.0.0 (*)\n"),
                         {"synapse-module", "serde"})


if __name__ == "__main__":
    unittest.main()
