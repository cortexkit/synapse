#!/usr/bin/env python3
"""Measure resource recovery after unload/owner exit and concurrent compile failures (development)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--driver", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=1800, help="seconds per experiment")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
env = dict(os.environ)
env.pop("TMPDIR", None)
env.pop("ANE_DIAGNOSTICS", None)
env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
env["ANE_TEST_PACKAGES"] = str(root / "target/ane-direct-packages")
env["ANE_UNLOAD_DIAGNOSTICS"] = "1"
result = {"classification": "development", "machine_model": subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip(), "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(), "production_coreml_lane": "Resident per run context, untouched; shared pool consumption remains unproven.", "runs": []}
args.out.parent.mkdir(parents=True, exist_ok=True)


def command(test):
    return [str(args.driver.resolve()), f"backend::fresh_process_hardware::{test}", "--exact", "--ignored", "--nocapture", "--test-threads=1"]


def unload_summary(log):
    lines = [line for line in log.splitlines() if line.startswith("ANE_UNLOAD ")]
    return {"calls": len(lines), "success_true": sum("success=true " in line for line in lines), "success_false": sum("success=false " in line for line in lines), "non_null_errors": sorted(set(line for line in lines if "error=None" not in line))}


def record(data, before, logs):
    data["load_before_1_5_15"] = before
    data["load_after_1_5_15"] = list(os.getloadavg())
    data["unloads"] = [unload_summary(log) for log in logs]
    result["runs"].append(data)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


def run(test, extra=None):
    before = list(os.getloadavg())
    print(test, extra, "load1/5/15", before, flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        child_env = dict(env, ANE_CAPACITY_OUT=str(report), **(extra or {}))
        child = subprocess.run(command(test), cwd=root, env=child_env, text=True, capture_output=True, timeout=args.timeout)
        if child.returncode:
            print(child.stdout, child.stderr)
            raise RuntimeError(f"{test} failed: {child.returncode}")
        record(json.loads(report.read_text()), before, [child.stderr])


run("fresh_process_reclaim")
before = list(os.getloadavg())
print("owner-exit-reclaim", "load1/5/15", before, flush=True)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    release = tmp / "release"
    exited = tmp / "exited"
    ready = tmp / "ready"
    processes = []
    logs = []
    deadline = time.monotonic() + args.timeout
    try:
        for index, test in enumerate(["fresh_process_residency_capacity", "fresh_process_reclaim"]):
            log = (tmp / f"{index}.log").open("w+")
            logs.append(log)
            child_env = dict(env, ANE_CAPACITY_OUT=str(tmp / f"{index}.json"))
            if index == 0:
                child_env.update(ANE_CAPACITY_MIX="gte", ANE_CAPACITY_RELEASE=str(release))
            else:
                child_env.update(ANE_RECLAIM_OWNER_EXITED=str(exited), ANE_RECLAIM_READY=str(ready))
            processes.append(subprocess.Popen(command(test), cwd=root, env=child_env, stdout=log, stderr=subprocess.STDOUT))
            target = tmp / "0.json" if index == 0 else ready
            while not target.exists():
                if time.monotonic() >= deadline or processes[-1].poll() is not None:
                    raise RuntimeError("paired reclaim process did not report")
                time.sleep(0.1)
        release.touch()
        assert processes[0].wait(timeout=max(1, deadline-time.monotonic())) == 0
        exited.touch()
        assert processes[1].wait(timeout=max(1, deadline-time.monotonic())) == 0
        for log in logs:
            log.seek(0)
        record({"experiment": "owner-exit-reclaim", "processes": [json.loads((tmp / f"{i}.json").read_text()) for i in range(2)]}, before, [log.read() for log in logs])
    finally:
        release.touch()
        exited.touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for log in logs:
            log.close()

for count in [2, 4]:
    run("fresh_process_concurrent_compiles", {"ANE_COMPILE_COUNT": str(count), "ANE_COMPILE_FULL_SHAPE": "1"})
for count in [2, 4, 8, 16, 32]:
    run("fresh_process_concurrent_compiles", {"ANE_COMPILE_COUNT": str(count)})
