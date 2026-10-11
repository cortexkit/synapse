#!/usr/bin/env python3
"""Sample macOS memory and ckdev serving processes without allocating test pressure."""
import argparse
import json
import re
import subprocess
import time
import unittest
from pathlib import Path


COMMANDS = {
    "processes": ["ps", "-axo", "pid=,ppid=,rss=,comm="],
    "vm_stat": ["vm_stat"],
    "memory_pressure": ["memory_pressure", "-Q"],
}


def processes(text):
    records = []
    for line in text.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4 or "ckdev-ane-packing" not in fields[3]:
            continue
        arm = re.search(r"private-\d+-(serial|packed)-(\d+)", fields[3])
        records.append({
            "pid": int(fields[0]), "ppid": int(fields[1]), "rss_kib": int(fields[2]),
            "name": Path(fields[3]).name,
            "arm": f"{arm[1]}-{arm[2]}" if arm else None,
        })
    return records


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def sample():
    record = {"unix_us": time.time_ns() // 1000}
    process_result = command(COMMANDS["processes"])
    record["processes"] = processes(process_result.get("stdout", ""))
    if process_result.get("exit_code") != 0:
        record["process_error"] = process_result
    record["vm_stat"] = command(COMMANDS["vm_stat"])
    record["memory_pressure"] = command(COMMANDS["memory_pressure"])
    return record


class Tests(unittest.TestCase):
    def test_rss_records_only_private_serving_processes_and_strip_paths(self):
        values = processes("1 0 123 /unrelated/daemon\n2 1 4096 /private/private-7-packed-2/ckdev-ane-packing-worker\n3 1 2048 /private/private-7-packed-2/ckdev-ane-packing-module\n")
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0], {"pid": 2, "ppid": 1, "rss_kib": 4096, "name": "ckdev-ane-packing-worker", "arm": "packed-2"})
        self.assertNotIn("/private", json.dumps(values))

    def test_memory_pressure_command_is_query_only(self):
        self.assertEqual(COMMANDS["memory_pressure"], ["memory_pressure", "-Q"])
        self.assertEqual(COMMANDS["vm_stat"], ["vm_stat"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], exit=True)
    if not args.out or not args.stop_file or args.interval_seconds < 1:
        parser.error("set --out and --stop-file, with interval at least one second")
    with args.out.open("x") as output:
        while not args.stop_file.exists():
            start = time.monotonic()
            output.write(json.dumps(sample()) + "\n")
            output.flush()
            time.sleep(max(0, args.interval_seconds - (time.monotonic() - start)))
