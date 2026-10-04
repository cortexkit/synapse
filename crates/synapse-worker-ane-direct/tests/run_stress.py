"""Run the development-only hardware stress test; never writes release evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, help="JSON output path; defaults to a temporary file")
parser.add_argument("--timeout", type=int, default=3600, help="hard timeout in seconds, including any load wait")
parser.add_argument("--wait-for-load", action="store_true", help="queue until load1<16 and load5<20")
parser.add_argument("--driver", type=Path, help="prebuilt supervisor test executable; skips Cargo")
args = parser.parse_args()
env = os.environ.copy()
env.pop("SUBC_LAUNCH_NONCE", None)
env.pop("SUBC_LAUNCH_NONCE_FD", None)
# The private compiler requires macOS's per-user temporary directory.
env.pop("TMPDIR", None)
if args.out:
    env["ANE_STRESS_OUT"] = str(args.out.resolve())
env.setdefault("ANE_TEST_WORKER", str(ROOT / "target/debug/ck-synapse-worker-ane-direct"))
env.setdefault("ANE_TEST_PACKAGES", str(ROOT / "target/ane-direct-packages"))
deadline = time.monotonic() + args.timeout

def run(command, cwd, capture=False):
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True,
                               stdout=subprocess.PIPE if capture else None, text=True)
    try:
        stdout, _ = process.communicate(timeout=max(1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise SystemExit("stress timed out; terminated the test and its workers")
    if process.returncode:
        raise SystemExit(process.returncode)
    return stdout

# Build before waiting: compilation can push load back over the measurement bar.
if args.driver:
    executable = str(args.driver.resolve())
else:
    build = run(["cargo", "test", "--locked", "-p", "synapse-module", "--lib", "--no-run", "--message-format=json"], ROOT, capture=True)
    artifacts = [json.loads(line) for line in build.splitlines() if line.startswith("{")]
    executables = [a["executable"] for a in artifacts if a.get("reason") == "compiler-artifact"
                   and a.get("target", {}).get("name") == "synapse_module"
                   and a.get("profile", {}).get("test") and a.get("executable")]
    if len(executables) != 1:
        raise SystemExit("expected exactly one compiled supervisor test executable")
    executable = executables[0]

def load_averages():
    result = subprocess.run(["sysctl", "-n", "vm.loadavg"], check=True, capture_output=True, text=True)
    return [float(part) for part in result.stdout.split() if part not in ("{", "}")]

load = load_averages()
while load[0] >= 16 or load[1] >= 20:
    if not args.wait_for_load:
        raise SystemExit(f"stress deferred: load1<16 and load5<20 required; load={load}")
    print(f"Stress queued for load1<16/load5<20; current1/5/15={load}", flush=True)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SystemExit("stress timed out waiting for quiet window; no model requests submitted")
    threading.Event().wait(min(60, remaining))
    load = load_averages()
print(f"STRESS_LOAD_BEFORE={load}", flush=True)
try:
    run([executable, "--ignored", "--exact", "worker_host::ane_residency::hardware_tests::real_direct_ane_residency_stress", "--nocapture"], ROOT / "crates/synapse-module")
finally:
    print(f"STRESS_LOAD_AFTER={load_averages()}", flush=True)
