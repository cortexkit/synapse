#!/usr/bin/env python3
"""Sample macOS memory and ckdev serving processes without allocating test pressure."""
import argparse
import hashlib
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
    record: dict[str, object] = {"unix_us": time.time_ns() // 1000}
    process_result = command(COMMANDS["processes"])
    record["processes"] = processes(process_result.get("stdout", ""))
    if process_result.get("exit_code") != 0:
        record["process_error"] = process_result
    record["vm_stat"] = command(COMMANDS["vm_stat"])
    record["memory_pressure"] = command(COMMANDS["memory_pressure"])
    return record


def summarize_memory(data):
    records = [json.loads(line) for line in data.splitlines()]
    if not records:
        raise ValueError("empty memory capture")
    values = {}
    pressure = []
    errors = []
    vm = []
    physical = set()
    page_sizes = set()
    overlapping = 0
    for record in records:
        arms = {p["arm"] for p in record["processes"] if p["arm"] is not None}
        overlapping += len(arms) > 1
        for process in record["processes"]:
            values.setdefault((process["arm"], process["name"]), []).append(process["rss_kib"])
        pressure_text = record["memory_pressure"].get("stdout", "")
        match = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure_text)
        if match:
            pressure.append(int(match[1]))
        match = re.search(r"The system has (\d+)", pressure_text)
        if match:
            physical.add(int(match[1]))
        text = record["vm_stat"].get("stdout", "")
        match = re.search(r"page size of (\d+) bytes", text)
        if match:
            page_sizes.add(int(match[1]))
        counters = {match[1].strip('"'): int(match[2]) for line in text.splitlines()
                    if (match := re.match(r"^(.+?):\s+(\d+)\.$", line))}
        vm.append(counters)
        if record["memory_pressure"].get("exit_code") != 0 or record["vm_stat"].get("exit_code") != 0 or "process_error" in record:
            errors.append(record["unix_us"])
    if errors or len(pressure) != len(records) or len(physical) != 1 or len(page_sizes) != 1:
        raise ValueError("memory sampling incomplete or inconsistent")
    keys = ["Swapins", "Swapouts", "Compressions", "Decompressions", "Pageins", "Pageouts"]
    return {
        "schema": 1, "physical_memory_bytes": physical.pop(), "page_size_bytes": page_sizes.pop(),
        "samples": len(records), "sample_errors": errors,
        "start_unix_us": records[0]["unix_us"], "end_unix_us": records[-1]["unix_us"],
        "memory_pressure_free_pct_min": min(pressure), "memory_pressure_free_pct_max": max(pressure),
        "memory_pressure_free_pct_start": pressure[0], "memory_pressure_free_pct_end": pressure[-1],
        "raw_memory_bytes": len(data), "raw_memory_sha256": hashlib.sha256(data).hexdigest(),
        "global_vm_counter_deltas_pages": {key: vm[-1][key] - vm[0][key] for key in keys},
        "compressor_occupied_pages_min": min(v["Pages occupied by compressor"] for v in vm),
        "compressor_occupied_pages_max": max(v["Pages occupied by compressor"] for v in vm),
        "free_pages_min": min(v["Pages free"] for v in vm), "free_pages_max": max(v["Pages free"] for v in vm),
        "samples_with_multiple_active_arms": overlapping,
        "peak_rss_kib_by_arm": [{"arm": arm, "name": name, "peak_rss_kib": max(v)} for (arm, name), v in values.items()],
    }


class Tests(unittest.TestCase):
    def test_memory_reduction_uses_counter_deltas_and_sampled_rss(self):
        def record(t, swap):
            vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            vm += "\n".join(f"{key}: {swap}." for key in ["Swapins", "Swapouts", "Compressions", "Decompressions", "Pageins", "Pageouts", "Pages occupied by compressor", "Pages free"])
            return {"unix_us": t, "processes": [{"arm": "packed-0", "name": "ckdev-ane-packing-worker", "rss_kib": swap}],
                    "vm_stat": {"exit_code": 0, "stdout": vm},
                    "memory_pressure": {"exit_code": 0, "stdout": "The system has 17179869184\nSystem-wide memory free percentage: 14%"}}
        data = (json.dumps(record(100, 20)) + "\n" + json.dumps(record(200, 25)) + "\n").encode()
        summary = summarize_memory(data)
        self.assertEqual(summary["global_vm_counter_deltas_pages"]["Swapouts"], 5)
        self.assertEqual(summary["peak_rss_kib_by_arm"][0]["peak_rss_kib"], 25)
        self.assertEqual(summary["memory_pressure_free_pct_min"], 14)
        self.assertEqual(summary["samples_with_multiple_active_arms"], 0)
        with self.assertRaises(ValueError):
            summarize_memory(b"")

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
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], exit=True)
    if args.summarize:
        if not args.summary_out:
            parser.error("set --summary-out for reduction")
        args.summary_out.write_text(json.dumps(summarize_memory(args.summarize.read_bytes()), indent=2, sort_keys=True) + "\n")
        parser.exit()
    if not args.out or not args.stop_file or args.interval_seconds < 1:
        parser.error("set --out and --stop-file, with interval at least one second")
    with args.out.open("x") as output:
        while not args.stop_file.exists():
            start = time.monotonic()
            output.write(json.dumps(sample()) + "\n")
            output.flush()
            time.sleep(max(0, args.interval_seconds - (time.monotonic() - start)))
