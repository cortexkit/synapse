#!/usr/bin/env python3
"""Development-only paired-process and large-width ANE capacity probes."""
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
parser.add_argument("--timeout", type=int, default=1800)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
env = dict(os.environ)
env.pop("TMPDIR", None)
env.pop("ANE_DIAGNOSTICS", None)
env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
env["ANE_TEST_PACKAGES"] = str(root / "target/ane-direct-packages")
command = [str(args.driver.resolve()), "backend::fresh_process_hardware::fresh_process_residency_capacity",
           "--exact", "--ignored", "--nocapture", "--test-threads=1"]
result = {"classification": "development", "machine_model": subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip(),
          "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
          "production_coreml_lane": "Resident per task context; not stopped. Pool sharing cannot be inferred without a no-Core-ML control.", "runs": []}
args.out.parent.mkdir(parents=True, exist_ok=True)
for experiment in ["paired-gte", "gte-large"]:
    before = list(os.getloadavg())
    print(experiment, "load1/5/15", before, flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        count = 2 if experiment == "paired-gte" else 1
        release = tmp / "release"
        processes = []
        logs = []
        try:
            for index in range(count):
                child_env = dict(env, ANE_CAPACITY_MIX="gte" if count == 2 else "gte-large", ANE_CAPACITY_OUT=str(tmp / f"{index}.json"))
                if count == 2:
                    child_env["ANE_CAPACITY_RELEASE"] = str(release)
                log = (tmp / f"{index}.log").open("w+")
                logs.append(log)
                processes.append(subprocess.Popen(command, cwd=root, env=child_env, stdout=log, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + args.timeout
            while not all((tmp / f"{i}.json").exists() for i in range(count)):
                if time.monotonic() >= deadline or any(p.poll() is not None for p in processes):
                    raise RuntimeError("capacity process failed or timed out before report")
                time.sleep(0.1)
            release.touch()
            for process in processes:
                assert process.wait(timeout=max(1, deadline - time.monotonic())) == 0
            runs = [json.loads((tmp / f"{i}.json").read_text()) for i in range(count)]
            result["runs"].append({"experiment": experiment, "load_before_1_5_15": before, "load_after_1_5_15": list(os.getloadavg()), "processes": runs})
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(experiment, [(r["resident_executables"], r["exhausted"]) for r in runs], flush=True)
        finally:
            release.touch()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
            for log in logs:
                log.seek(0)
                if not all((tmp / f"{i}.json").exists() for i in range(count)):
                    print(log.read())
                log.close()
