#!/usr/bin/env python3
"""Repository preflight: portable builds on MOTOR; Apple builds locally."""

import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib.remote_gate import run


def remote_commands():
    return [
        ["cargo", "--version"],
        ["rustc", "--version"],
        ["cargo", "clippy", "--version"],
        ["cargo", "nextest", "--version"],
        ["bash", "scripts/check-train-preconditions.sh"],
        ["bash", "scripts/check-no-external-path-deps.sh"],
        ["python3", "scripts/check-module-runtime-deps.py"],
        ["cargo", "fmt", "--all", "--check"],
        ["env", "CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc",
         "AR_x86_64_pc_windows_gnu=x86_64-w64-mingw32-ar",
         "CARGO_TARGET_DIR=target/train-windows-clippy",
         "cargo", "clippy", "--locked", "-p", "synapse-module", "--features", "test-support",
         "--all-targets", "--target", "x86_64-pc-windows-gnu", "--", "-D", "warnings"],
    ]


def macos_commands():
    # These packages compile Metal/Objective-C or Swift, or lint the module's
    # macOS-only Metal integration. Linux clippy cannot replace this coverage.
    return [
        ["cargo", "--version"],
        ["rustc", "--version"],
        ["cargo", "clippy", "--version"],
        ["xcrun", "--find", "metal"],
        ["cargo", "clippy", "--locked", "-p", "synapse-module", "-p", "synapse-engine-owned",
         "-p", "synapse-worker-ane", "--all-targets", "--", "-D", "warnings"],
        ["cargo", "test", "--locked", "-p", "synapse-engine-owned", "--lib"],
    ]


def local_commands(system):
    return macos_commands() if system == "Darwin" else []


def main():
    system = platform.system()
    if system == "Darwin" and Path("/Applications/Xcode.app").is_dir():
        os.environ.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    return run(remote_commands(), local_commands(system), vcpus=4)


if __name__ == "__main__":
    raise SystemExit(main())
