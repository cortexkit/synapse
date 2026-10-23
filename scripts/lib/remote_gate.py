"""Run portable and Mac-only landing checks against the same source contents."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time


def fingerprint():
    """Hash snapshot inputs, including dirty/untracked files but not ignored output."""
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(set(names) - {b""}):
        path = Path(os.fsdecode(name))
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            kind, content = b"symlink", os.fsencode(os.readlink(path))
        elif stat.S_ISREG(mode):
            kind = b"executable" if mode & 0o111 else b"file"
            content = path.read_bytes()
        else:
            raise ValueError(f"unsupported snapshot input: {path}")
        for part in (name, kind, content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def steps(commands):
    environment = dict(os.environ, RUSTC_WRAPPER="", CARGO_BUILD_RUSTC_WRAPPER="")
    for command in commands:
        print("$ " + shlex.join(command), flush=True)
        start = time.monotonic()
        try:
            status = subprocess.run(command, env=environment).returncode
        except OSError as error:
            print(f"Cannot start {command[0]}: {error}", flush=True)
            status = 127
        print(f"TIMING {time.monotonic() - start:.2f}s exit={status}: "
              + shlex.join(command), flush=True)
        if status:
            return status
    print(f"Passed {len(commands)} landing stages", flush=True)
    return 0


def worker(half, expected, commands):
    initial = fingerprint()
    print(f"TREE {half} {initial}", flush=True)
    if initial != expected:
        print(f"REFUSE: {half} snapshot differs from requested tree {expected}", flush=True)
        return 1
    status = steps(commands)
    final = fingerprint()
    print(f"TREE {half} after {final}", flush=True)
    if final != expected:
        print(f"REFUSE: {half} source changed during checks", flush=True)
        return 1
    return status


def motor(command):
    """Only a refused-before-start job is safe to submit again."""
    deadline = time.monotonic() + 180
    while True:
        status = subprocess.run(command).returncode
        if status != 200 or time.monotonic() + 15 > deadline:
            return status
        print("MOTOR refused before start (200); retrying in 15s (180s budget)", flush=True)
        time.sleep(15)


def run(remote_commands, local_commands=(), vcpus=4):
    start = time.monotonic()
    expected = fingerprint()
    print(f"TREE requested {expected}", flush=True)
    helper = "scripts/lib/remote_gate.py"
    local = None
    # Start the Mac half before submitting the snapshot, and always join it even
    # when MOTOR refuses. No compiler should outlive a failed landing gate.
    if local_commands:
        local = subprocess.Popen([sys.executable, helper, "--worker", "macos", expected,
                                  json.dumps(local_commands)])
    remote_status = 1
    try:
        remote_status = motor([
            "ck", "motor", "run", "--vcpus", str(vcpus), "--timeout", "3600",
            "--cwd", str(Path.cwd()), "--", "python3", helper, "--worker", "linux",
            expected, json.dumps(remote_commands),
        ])
    finally:
        local_status = local.wait() if local else 0
    current = fingerprint()
    print(f"TREE joined {current}", flush=True)
    print(f"TIMING preflight wall {time.monotonic() - start:.2f}s "
          f"remote_exit={remote_status} macos_exit={local_status}", flush=True)
    if current != expected:
        print("REFUSE: source changed while landing checks ran", flush=True)
        return 1
    return remote_status or local_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("HALF", "TREE", "COMMANDS"))
    parser.add_argument("--commands", help="JSON argv lists to run remotely")
    args = parser.parse_args()
    if args.worker:
        half, expected, commands = args.worker
        return worker(half, expected, json.loads(commands))
    if args.commands:
        return run(json.loads(args.commands))
    parser.error("--worker or --commands is required")


if __name__ == "__main__":
    raise SystemExit(main())
