"""Exercise case selection without launching a worker or querying hardware."""
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import Mock, patch


class StressCaseSelection(unittest.TestCase):
    def run_case(self, *arguments):
        driver = Path(__file__).with_name("run_stress.py")
        process = Mock(returncode=0)
        process.communicate.return_value = ("", None)
        with patch.object(sys, "argv", [str(driver), "--driver", "/unused/driver", *arguments]), \
                patch("subprocess.Popen", return_value=process) as spawn, \
                patch("subprocess.run", return_value=Mock(stdout="{ 1 2 3 }")):
            runpy.run_path(str(driver), run_name="__main__")
        return spawn.call_args.args[0]

    def test_default_is_residency_safety(self):
        command = self.run_case()
        self.assertIn("worker_host::ane_residency::hardware_tests::real_direct_ane_residency_safety", command)
        self.assertIn("--ignored", command)
        self.assertIn("--exact", command)

    def test_cold_throughput_is_a_separate_exact_ignored_case(self):
        command = self.run_case("--case", "cold-throughput")
        self.assertIn("worker_host::ane_residency::hardware_tests::real_direct_ane_cold_throughput", command)
        self.assertIn("--ignored", command)
        self.assertIn("--exact", command)


if __name__ == "__main__":
    unittest.main()
