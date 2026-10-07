#!/usr/bin/env python3
"""Sample only a command's descendants; never signal ambient fleet processes."""

import json
import os
import subprocess
import sys
import time
import unittest


def processes():
    rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    images = {}
    for row in rows.splitlines():
        pid, name = row.strip().split(maxsplit=1)
        images[int(pid)] = name
    parents = subprocess.check_output(["ps", "-axo", "pid=,ppid=,lstart="], text=True)
    identities = {}
    for row in parents.splitlines():
        pid, parent, birth = row.split(maxsplit=2)
        identities[int(pid)] = (int(parent), " ".join(birth.split()))
    return images, identities


def descendants(owned, identities):
    # PIDs are recycled on long builds. A retained PID alone can silently
    # attribute another worktree's later process to this command.
    changed = True
    while changed:
        changed = False
        for pid, (parent, birth) in identities.items():
            if (parent in owned and parent in identities
                    and owned[parent] == identities[parent][1]
                    and owned.get(pid) != birth):
                owned[pid] = birth
                changed = True
    return {pid for pid, (_, birth) in identities.items() if owned.get(pid) == birth}


class SamplingTests(unittest.TestCase):
    def test_recycled_pid_is_not_a_descendant(self):
        owned = {10: "cargo birth", 11: "compiler birth"}
        identities = {10: (1, "cargo birth"), 11: (99, "unrelated birth"),
                      12: (11, "other worker birth"), 13: (10, "test birth")}
        self.assertEqual(descendants(owned, identities), {10, 13})

    def test_orphan_with_same_birth_remains_owned(self):
        owned = {10: "cargo birth", 11: "module birth"}
        self.assertEqual(descendants(owned, {11: (1, "module birth")}), {11})


def main():
    command = sys.argv[1:]
    if not command:
        raise SystemExit("usage: sample-dev-images.py <command> [args...]")
    print("Python", sys.version.split()[0], "sampling ps -axo pid=,comm= every 0.5 s", flush=True)
    child = subprocess.Popen(command)
    _, identities = processes()
    owned = {child.pid: identities[child.pid][1]}
    seen = {"ck": {}, "ckdev": {}}
    samples = 0
    observations = {"ck": 0, "ckdev": 0}

    def sample():
        nonlocal samples
        images, identities = processes()
        live_owned = descendants(owned, identities)
        samples += 1
        for pid in live_owned & images.keys():
            name = images[pid]
            basename = os.path.basename(name)
            category = "ckdev" if basename.startswith("ckdev-") else "ck" if basename.startswith("ck-") else None
            if category:
                seen[category][f"{pid}:{owned[pid]}"] = name
                observations[category] += 1
        return images, live_owned

    deadline = time.monotonic()
    while child.poll() is None:
        sample()
        deadline += 0.5
        time.sleep(max(0, deadline - time.monotonic()))
    # Allow the test harness's kill-on-drop children to be reaped, but inspect
    # rather than killing any process if cleanup failed.
    for _ in range(4):
        time.sleep(0.5)
        images, live_owned = sample()
    remaining = {pid: images[pid] for pid in live_owned & images.keys()
                 if os.path.basename(images[pid]).startswith(("ck-", "ckdev-"))}
    report = {"command": command, "interval_seconds": 0.5, "samples": samples,
              "unique_ck_images": len(seen["ck"]), "unique_ckdev_images": len(seen["ckdev"]),
              "observations": observations, "remaining": remaining, "test_exit": child.returncode}
    print("DEV_IMAGE_MEASUREMENT " + json.dumps(report, sort_keys=True), flush=True)
    if seen["ck"]:
        print("production-named descendants:", json.dumps(seen["ck"], sort_keys=True))
    return child.returncode or int(bool(seen["ck"] or remaining or not seen["ckdev"]))


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        raise SystemExit(main())
