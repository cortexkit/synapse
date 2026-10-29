"""Snapshot, routing and refusal contracts without compiling on the test host."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from pathlib import PosixPath
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch


from scripts.lib import remote_gate

PREFLIGHT = runpy.run_path(str(Path(__file__).with_name("train-preflight.py")))


class SnapshotTests(unittest.TestCase):
    def test_motor_command_defaults_to_current_checkout_root(self):
        self.assertEqual(remote_gate.motor_command(["python3", "worker.py"]), [
            "ck", "motor", "run", "--vcpus", "4", "--timeout", "3600", "--cwd", str(Path.cwd()),
            "--", "python3", "worker.py"])

    @unittest.skipIf(os.name == "nt", "simulates NTFS mode loss on a POSIX host")
    def test_windows_fingerprint_uses_index_modes_and_disk_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = PosixPath(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "script").write_text("source")
            (root / "script").chmod(0o755)
            subprocess.run(["git", "-C", str(root), "add", "script"], check=True)
            previous = Path.cwd()
            try:
                os.chdir(root)
                expected = remote_gate.fingerprint()
                (root / "script").chmod(0o644)
                with patch.object(remote_gate.os, "name", "nt"), \
                        patch.object(remote_gate, "Path", PosixPath):
                    self.assertEqual(remote_gate.fingerprint(), expected)
                    (root / "script").write_text("dirty")
                    self.assertNotEqual(remote_gate.fingerprint(), expected)
            finally:
                os.chdir(previous)

    def test_fingerprint_covers_dirty_untracked_deleted_modes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "tracked").write_text("original")
            subprocess.run(["git", "-C", str(root), "add", "tracked"], check=True)
            previous = Path.cwd()
            try:
                os.chdir(root)
                original = remote_gate.fingerprint()
                (root / "tracked").write_text("dirty")
                dirty = remote_gate.fingerprint()
                (root / "untracked").write_text("new")
                untracked = remote_gate.fingerprint()
                (root / "untracked").chmod(0o755)
                executable = remote_gate.fingerprint()
                (root / "link").symlink_to("tracked")
                symlink = remote_gate.fingerprint()
                (root / "tracked").unlink()
                deleted = remote_gate.fingerprint()
                self.assertEqual(len({original, dirty, untracked, executable, symlink, deleted}), 6)
                (root / ".git/info/exclude").write_text("ignored\n")
                (root / "ignored").write_text("not a snapshot input")
                self.assertEqual(remote_gate.fingerprint(), deleted)
            finally:
                os.chdir(previous)

    def test_worker_rejects_wrong_snapshot_before_running_commands(self):
        with patch.object(remote_gate, "fingerprint", return_value="different"), \
                patch.object(remote_gate, "steps") as steps:
            self.assertEqual(remote_gate.worker("linux", "requested", [["cargo", "test"]]), 1)
        steps.assert_not_called()

    def test_worker_rejects_tree_changed_during_commands(self):
        with patch.object(remote_gate, "fingerprint", side_effect=["original", "changed"]), \
                patch.object(remote_gate, "steps", return_value=0):
            self.assertEqual(remote_gate.worker("macos", "original", []), 1)

    def test_worker_preserves_failed_check_status(self):
        with patch.object(remote_gate, "fingerprint", return_value="original"), \
                patch.object(remote_gate, "steps", return_value=7):
            self.assertEqual(remote_gate.worker("linux", "original", []), 7)

    def test_refused_jobs_retry_but_other_statuses_never_do(self):
        for statuses in ([200, 200, 0], [201], [202], [203], [204], [1]):
            with self.subTest(statuses=statuses), \
                    patch.object(remote_gate.subprocess, "run", side_effect=[
                        subprocess.CompletedProcess([], status) for status in statuses]) as run, \
                    patch.object(remote_gate.time, "sleep") as sleep:
                self.assertEqual(remote_gate.motor(["ck"]), statuses[-1])
                self.assertEqual(run.call_count, len(statuses))
                self.assertEqual(sleep.call_count, statuses.count(200))
                for call in sleep.call_args_list:
                    self.assertEqual(call.args, (15,))

    def test_refused_jobs_stop_after_three_minutes(self):
        now = [0]

        def advance(seconds):
            now[0] += seconds

        with patch.object(remote_gate.subprocess, "run", return_value=subprocess.CompletedProcess([], 200)) as run, \
                patch.object(remote_gate.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(remote_gate.time, "sleep", side_effect=advance):
            self.assertEqual(remote_gate.motor(["ck"]), 200)
        self.assertEqual(now[0], 180)
        self.assertEqual(run.call_count, 13)

    def test_remote_failure_still_joins_mac_and_checks_current_tree(self):
        with patch.object(remote_gate, "fingerprint", return_value="tree"), \
                patch.object(remote_gate, "motor", return_value=201), \
                patch.object(remote_gate.subprocess, "Popen") as child:
            child.return_value.wait.return_value = 0
            self.assertEqual(remote_gate.run([["remote"]], [["local"]]), 201)
        child.return_value.wait.assert_called_once()

    def test_parent_rejects_change_even_if_both_halves_pass(self):
        with patch.object(remote_gate, "fingerprint", side_effect=["before", "after"]), \
                patch.object(remote_gate, "motor", return_value=0):
            self.assertEqual(remote_gate.run([]), 1)

    def test_step_timing_is_printed_on_failure_and_later_commands_do_not_run(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(remote_gate.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 9)) as run:
            self.assertEqual(remote_gate.steps([["first"], ["second"]]), 9)
        self.assertEqual(run.call_count, 1)
        self.assertIn("TIMING", output.getvalue())
        self.assertIn("exit=9: first", output.getvalue())
        self.assertEqual(run.call_args.kwargs["env"]["RUSTC_WRAPPER"], "")
        self.assertEqual(run.call_args.kwargs["env"]["CARGO_BUILD_RUSTC_WRAPPER"], "")

    def test_missing_tool_refuses_with_a_step_duration(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(remote_gate.subprocess, "run", side_effect=FileNotFoundError("tool missing")):
            self.assertEqual(remote_gate.steps([["missing"]]), 127)
        self.assertIn("TIMING", output.getvalue())
        self.assertIn("exit=127: missing", output.getvalue())


class PreflightPlanTests(unittest.TestCase):
    def test_portable_checks_only_run_remotely(self):
        remote = PREFLIGHT["remote_commands"]()
        native = PREFLIGHT["macos_commands"]()
        for command in (["cargo", "fmt", "--all", "--check"],
                        ["bash", "scripts/check-train-preconditions.sh"],
                        ["bash", "scripts/check-no-external-path-deps.sh"],
                        ["python3", "scripts/check-module-runtime-deps.py"]):
            self.assertIn(command, remote)
            self.assertNotIn(command, native)
        windows = next(command for command in remote if "x86_64-pc-windows-gnu" in command)
        self.assertEqual(windows[windows.index("cargo"):], [
            "cargo", "clippy", "--locked", "-p", "synapse-module", "--features", "test-support",
            "--all-targets", "--target", "x86_64-pc-windows-gnu", "--", "-D", "warnings"])
        self.assertIn("CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc", windows)

    def test_preflight_runs_nothing_locally_except_on_macos(self):
        self.assertEqual(PREFLIGHT["local_commands"]("Linux"), [])
        self.assertEqual(PREFLIGHT["local_commands"]("Darwin"), PREFLIGHT["macos_commands"]())

    def test_all_existing_mac_coverage_is_preserved(self):
        native = PREFLIGHT["macos_commands"]()
        self.assertIn(["xcrun", "--find", "metal"], native)
        self.assertIn(["cargo", "clippy", "--locked", "-p", "synapse-module", "-p", "synapse-engine-owned",
                       "-p", "synapse-worker-ane", "--all-targets", "--", "-D", "warnings"], native)
        self.assertIn(["cargo", "test", "--locked", "-p", "synapse-engine-owned", "--lib"], native)


if __name__ == "__main__":
    unittest.main(verbosity=2)
