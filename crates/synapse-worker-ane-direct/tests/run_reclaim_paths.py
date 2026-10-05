#!/usr/bin/env python3
"""Limit development probes of in-process ANE resource-release methods to one hour."""
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
parser.add_argument("--timeout", type=int, default=3600)
parser.add_argument("--methods", nargs="+", choices=["drop-client-references", "fresh-client", "fresh-allocated-client", "model-purge", "client-purge", "release-all", "autorelease-all", "scoped-single-evict"], default=["drop-client-references", "fresh-client", "fresh-allocated-client", "model-purge", "client-purge", "release-all", "autorelease-all"])
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
env = dict(os.environ)
env.pop("TMPDIR", None)
env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
env["ANE_TEST_PACKAGES"] = str(root / "target/ane-direct-packages")
env["ANE_UNLOAD_DIAGNOSTICS"] = "1"
selectors = json.loads(subprocess.check_output(["python3", str(root / "crates/synapse-worker-ane-direct/tests/list_reclaim_selectors.py")], env=env, text=True))
report = {"classification": "development", "machine_model": subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip(), "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(), "production_coreml_lane": "Resident per task context; untouched", "available_selectors": selectors, "runs": []}
args.out.parent.mkdir(parents=True, exist_ok=True)
deadline = time.monotonic() + args.timeout
for method in args.methods:
    before = list(os.getloadavg())
    print(method, "load1/5/15", before, flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "result.json"
        child_env = dict(env, ANE_RECLAIM_PATH=method, ANE_CAPACITY_OUT=str(result))
        test = {"autorelease-all": "fresh_process_autorelease_reclaim", "scoped-single-evict": "fresh_process_scoped_pools_reclaim"}.get(method, "fresh_process_reclaim_paths")
        command = [str(args.driver.resolve()), "backend::fresh_process_hardware::" + test, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
        try:
            child = subprocess.run(command, env=child_env, cwd=root, text=True, capture_output=True, timeout=max(1, deadline-time.monotonic()))
            data = json.loads(result.read_text()) if result.exists() else {"method": method, "result": "no report; isolated probe failed", "exit_code": child.returncode}
            lines = [line.split("ANE_UNLOAD ", 1)[1] for line in (child.stdout + child.stderr).splitlines() if "ANE_UNLOAD " in line]
            data["unloads"] = {"calls": len(lines), "success_true": sum(line.startswith("success=true ") for line in lines), "success_false": sum(line.startswith("success=false ") for line in lines), "non_null_errors": sorted(set(line for line in lines if "error=None" not in line))}
        except subprocess.TimeoutExpired:
            data = {"method": method, "result": "one-hour hardware deadline reached"}
        data["load_before_1_5_15"] = before
        data["load_after_1_5_15"] = list(os.getloadavg())
        report["runs"].append(data)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(method, data.get("attempts", data.get("result")), flush=True)
        if time.monotonic() >= deadline:
            break
