#!/usr/bin/env python3
"""Sample only a command's descendants; never signal ambient fleet processes."""

import json
import os
import subprocess
import sys
import time


def processes():
    rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    images = {}
    for row in rows.splitlines():
        pid, name = row.strip().split(maxsplit=1)
        images[int(pid)] = name
    parents = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    return images, {int(pid): int(parent) for pid, parent in (row.split() for row in parents.splitlines())}


def main():
    command = sys.argv[1:]
    if not command:
        raise SystemExit("usage: sample-dev-images.py <command> [args...]")
    print("Python", sys.version.split()[0], "sampling ps -axo pid=,comm= every 0.5 s", flush=True)
    baseline, _ = processes()
    child = subprocess.Popen(command)
    owned = {child.pid}
    seen = {"ck": {}, "ckdev": {}}
    samples = 0
    observations = {"ck": 0, "ckdev": 0}

    def sample():
        nonlocal samples
        images, parents = processes()
        # Find grandchildren regardless of ps's ordering; retain observed PIDs
        # so a child orphaned during shutdown remains attributable to this run.
        changed = True
        while changed:
            changed = False
            for pid, parent in parents.items():
                if parent in owned and pid not in owned and pid not in baseline:
                    owned.add(pid)
                    changed = True
        samples += 1
        for pid in owned & images.keys():
            name = images[pid]
            basename = os.path.basename(name)
            category = "ckdev" if basename.startswith("ckdev-") else "ck" if basename.startswith("ck-") else None
            if category:
                seen[category][pid] = name
                observations[category] += 1
        return images

    deadline = time.monotonic()
    while child.poll() is None:
        sample()
        deadline += 0.5
        time.sleep(max(0, deadline - time.monotonic()))
    # Allow the test harness's kill-on-drop children to be reaped, but inspect
    # rather than killing any process if cleanup failed.
    for _ in range(4):
        time.sleep(0.5)
        images = sample()
    remaining = {pid: images[pid] for pid in owned & images.keys()
                 if os.path.basename(images[pid]).startswith(("ck-", "ckdev-"))}
    report = {"command": command, "interval_seconds": 0.5, "samples": samples,
              "unique_ck_images": len(seen["ck"]), "unique_ckdev_images": len(seen["ckdev"]),
              "observations": observations, "remaining": remaining, "test_exit": child.returncode}
    print("DEV_IMAGE_MEASUREMENT " + json.dumps(report, sort_keys=True), flush=True)
    if seen["ck"]:
        print("production-named descendants:", json.dumps(seen["ck"], sort_keys=True))
    return child.returncode or int(bool(seen["ck"] or remaining or not seen["ckdev"]))


if __name__ == "__main__":
    raise SystemExit(main())
