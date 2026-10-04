"""Run the development-only hardware stress test; never writes release evidence."""
import argparse
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
parser.add_argument("--wait-for-load", action="store_true", help="queue until the 1-minute load is below16")
args = parser.parse_args()
env = os.environ.copy()
# The private compiler requires macOS's per-user temporary directory.
env.pop("TMPDIR", None)
if args.out:
    env["ANE_STRESS_OUT"] = str(args.out.resolve())
env.setdefault("ANE_TEST_WORKER", str(ROOT / "target/release/ck-synapse-worker-ane-direct"))
env.setdefault("ANE_TEST_PACKAGES", str(ROOT / "target/ane-direct-packages"))
command = ["cargo", "test", "--release", "--locked", "-p", "synapse-module", "worker_host::ane_residency::hardware_tests::real_direct_ane_residency_stress", "--", "--ignored", "--exact", "--nocapture"]
deadline = time.monotonic() + args.timeout
if os.getloadavg()[0] >= 16:
    if not args.wait_for_load:
        raise SystemExit("stress deferred: 1-minute load must be below16")
    print(f"Stress queued for load<16; current1/5/15={os.getloadavg()}", flush=True)
    while os.getloadavg()[0] >= 16:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit("stress timed out waiting for load<16; no model requests submitted")
        threading.Event().wait(min(30, remaining))
process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
try:
    result = process.wait(timeout=max(1, deadline - time.monotonic()))
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    raise SystemExit("stress timed out; terminated the test and its workers")
raise SystemExit(result)
