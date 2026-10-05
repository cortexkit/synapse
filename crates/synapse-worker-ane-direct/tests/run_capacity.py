#!/usr/bin/env python3
"""Development-only sequential ANE capacity runs, each in a fresh test process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--driver", required=True, type=Path)
parser.add_argument("--batch", required=True, type=int, choices=[1, 2])
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--timeout", type=int, default=1800, help="seconds per fresh process")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
manifest = json.loads((root / "bench/parity/models.json").read_text())
packages = root / "target/ane-direct-packages"
digests = {}
for slug in manifest["models"]:
    path = packages / f"{slug}.safetensors"
    digest = "sha256:" + sha256_file(path)
    assert digest == manifest["profiles"][f"{slug}.ane-direct-worker"]["converted_package_digest"]
    digests[slug] = digest

env = dict(os.environ)
env.pop("TMPDIR", None)
env.pop("ANE_DIAGNOSTICS", None)
env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
env["ANE_TEST_PACKAGES"] = str(packages)
metadata = {
    "classification": "development",
    "batch": args.batch,
    "machine_model": subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip(),
    "os_version": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
    "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
    "package_digests": digests,
    "driver_sha256": sha256_file(args.driver),
    "production_coreml_lane": "Resident per task context; ck-synapse-worker-ane-swift process observed. Not stopped or modified; private residency not independently introspectable.",
    "runs": [],
}
args.out.parent.mkdir(parents=True, exist_ok=True)
for mix in ["gte", "qwen", "alternating", "all-four"]:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        env["ANE_CAPACITY_MIX"] = mix
        env["ANE_CAPACITY_OUT"] = str(report)
        before = list(os.getloadavg())
        print(f"batch={args.batch} mix={mix} load1/5/15={before}", flush=True)
        result = subprocess.run(
            [str(args.driver.resolve()), "backend::fresh_process_hardware::fresh_process_residency_capacity",
             "--exact", "--ignored", "--nocapture", "--test-threads=1"],
            cwd=root, env=env, text=True, capture_output=True, timeout=args.timeout,
        )
        if result.returncode or not report.exists():
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"capacity process failed: mix={mix}, exit={result.returncode}")
        run = json.loads(report.read_text())
        run["load_before_1_5_15"] = before
        run["load_after_1_5_15"] = list(os.getloadavg())
        run["tests_passed"] = 1
        metadata["runs"].append(run)
        args.out.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"mix={mix} exhausted={run['exhausted']} resident_executables={run['resident_executables']} shapes={len(run['resident_shapes'])}", flush=True)
