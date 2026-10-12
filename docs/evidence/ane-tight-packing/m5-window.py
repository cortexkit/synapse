#!/usr/bin/env python3
"""Supervise the approved 2026-10-13 M5 window; raw evidence stays outside Git.

--dry-run only reads production metadata/logs and verifies assets. --self-test
uses synthetic logs/processes, never a model. --run is the only mutating mode.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
START = dt.datetime(2026, 10, 13, 2, tzinfo=dt.timezone.utc).timestamp()
QUIET_END = START + 15 * 60
END = START + 50 * 60
LANES = {"qwen3-embedding-0.6b-ane": ("5f7ac7cd", 1024),
         "gte-modernbert-base-ane-fp16": ("5a2374bc", 768)}
WORKERS = {"ck-synapse-worker-ane-direct", "ck-synapse-worker-ane-swift"}
CALL = Path.home() / ".local/share/cortexkit/synapse/tools-target/release/subc_call"
LOGS = Path.home() / ".local/share/cortexkit/synapse/logs"


def utc(value=None):
    return dt.datetime.fromtimestamp(time.time() if value is None else value,
                                     dt.timezone.utc).isoformat(timespec="milliseconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(argv):
    return subprocess.run([str(a) for a in argv], check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def processes():
    rows = {}
    for line in command(["ps", "-axo", "pid=,ppid=,lstart=,comm="]).splitlines():
        fields = line.split(maxsplit=7)
        if len(fields) == 8:
            rows[int(fields[0])] = {"ppid": int(fields[1]),
                                   "identity": " ".join(fields[2:7]),
                                   "exe": fields[7]}
    return rows


def descendants(rows, root):
    owned = {root}
    while True:
        more = {pid for pid, row in rows.items() if row["ppid"] in owned}
        if more <= owned:
            return owned
        owned |= more


def external_workers(rows, owned=()):
    return [pid for pid, row in rows.items()
            if (Path(row["exe"]).name in WORKERS or "-worker-ane" in Path(row["exe"]).name)
            and pid not in owned]


def lane_event(line):
    # Admission, completion, refusal, and cold-load events all reset quietness.
    # Management reads/unloads are not embedding traffic. Production does not
    # log every arrival, so worker absence and unloaded states are also required.
    return any("model_id=" + lane in line for lane in LANES) and (
        "method=embed." in line or "job " in line or "cold_load" in line)


class LogTail:
    def __init__(self, root):
        self.root = root
        self.positions = {}
        self.pending = {}

    def read(self):
        files = sorted(self.root.glob("*.log"))
        if not files:
            raise RuntimeError("no readable production logs; cannot prove quietness")
        events = []
        for path in files:
            stat = path.stat()
            key = (str(path), stat.st_ino)
            position = self.positions.get(key, max(0, stat.st_size - 256 * 1024))
            initial_fragment = key not in self.positions and position > 0
            if stat.st_size < position:
                position = 0
                self.pending.pop(key, None)
            with path.open("rb") as f:
                f.seek(position)
                data = self.pending.get(key, b"") + f.read()
                self.positions[key] = f.tell()
            if initial_fragment:
                # A bounded historical tail may begin in the middle of a record.
                # Only subsequent complete records can establish event times.
                data = data.split(b"\n", 1)[1] if b"\n" in data else b""
            parts = data.split(b"\n")
            self.pending[key] = parts.pop()
            for raw in parts:
                line = raw.decode(errors="replace")
                if lane_event(line):
                    try:
                        stamp = dt.datetime.fromisoformat(line.split()[0].replace("Z", "+00:00")).timestamp()
                    except (ValueError, IndexError):
                        raise RuntimeError("unparseable production lane event")
                    events.append(stamp)
        return events


def restored_vector(result, lane):
    prefix, dims = LANES[lane]
    vectors = result.get("vectors", [])
    vector = vectors[0].get("vector", []) if len(vectors) == 1 else []
    return (str(result.get("fingerprint", "")).startswith(prefix)
            and len(vector) == dims
            and all(isinstance(v, (float, int)) and math.isfinite(v) for v in vector))


class Window:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.tail = LogTail(LOGS)
        self.owned = {}
        self.prod = None
        self.prod_identity = None
        self.touched = False
        self.record = {"schema": 1, "window_start": utc(START), "window_end": utc(END),
                       "source_commit": command(["git", "rev-parse", "HEAD"]),
                       "events": [], "stages": {}, "restoration": {}}

    def event(self, kind, **fields):
        item = {"time": utc(), "kind": kind, **fields}
        self.record["events"].append(item)
        print(json.dumps(item), flush=True)
        self.save()

    def save(self):
        path = self.out / "window.json"
        path.with_suffix(".tmp").write_text(json.dumps(self.record, indent=2) + "\n")
        path.with_suffix(".tmp").replace(path)

    def rpc(self, method, params):
        process = subprocess.run([str(CALL), "--module", "synapse", "--method", method,
                                  "--params", json.dumps(params)], capture_output=True,
                                 text=True, timeout=30)
        value = json.loads(process.stdout)
        if process.returncode or "error" in value:
            raise RuntimeError(json.dumps(value.get("error", value)))
        return value["result"]

    def states(self):
        value = self.rpc("models.list", {})
        return {m["model_id"]: m["state"] for m in value["models"] if m["model_id"] in LANES}

    def check_production(self, rows):
        if rows.get(self.prod) != self.prod_identity:
            raise RuntimeError("production PID changed; never allowlist its replacement")

    def preflight(self):
        if self.out == ROOT or ROOT in self.out.parents:
            raise RuntimeError("raw output must be outside repository")
        self.out.mkdir(parents=True, exist_ok=False)
        rows = processes()
        prod = [pid for pid, row in rows.items() if Path(row["exe"]).name == "ck-synapse"]
        if len(prod) != 1:
            raise RuntimeError("require exactly one production ck-synapse PID")
        self.prod = prod[0]
        self.prod_identity = rows[self.prod]
        self.record["production_pid"] = self.prod
        pins = json.loads((ROOT / "docs/evidence/ane-tight-packing/module-serving-rerun.json").read_text())
        paths = {"input_sha256": self.args.input,
                 "meta_sha256": Path(str(self.args.input) + ".meta.json"),
                 "checkpoint_sha256": self.args.weights / "model.safetensors",
                 "tokenizer_sha256": self.args.weights / "tokenizer.json"}
        hashes = {}
        for key, path in paths.items():
            hashes[key] = sha(path)
            if hashes[key] != pins[key]:
                raise RuntimeError("M4 workload/checkpoint pin mismatch: " + key)
        manifest = json.loads((ROOT / "bench/parity/models.json").read_text())
        for profile_id, profile in manifest["profiles"].items():
            if profile_id.endswith(".ane-direct-worker"):
                slug = profile_id.removesuffix(".ane-direct-worker")
                package = self.args.packages / (slug + ".safetensors")
                digest = sha(package)
                if "sha256:" + digest != profile["converted_package_digest"]:
                    raise RuntimeError("stress package pin mismatch: " + slug)
                hashes[package.name] = digest
        alias = self.out / "ckdev-ane-packing-serving"
        shutil.copy2(self.args.assets / "examples/ane_packing_serving", alias)
        command(["codesign", "--force", "--sign", "-", alias])
        for path in [alias, self.args.driver, self.args.assets / "ck-synapse",
                     self.args.assets / "ck-synapse-worker-ane-direct"]:
            command(["codesign", "--verify", "--strict", path])
            hashes[path.name] = sha(path)
        self.record["asset_sha256"] = hashes
        self.record["environment"] = {"chip": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                                      "load": command(["sysctl", "-n", "vm.loadavg"]),
                                      "rustc": command(["rustc", "--version"]),
                                      "cargo": command(["cargo", "--version"]),
                                      "python": sys.version.split()[0]}
        self.event("preflight", states=self.states(), workers=external_workers(rows),
                   log_events_seen=len(self.tail.read()), dry_run=self.args.dry_run)

    def quiet(self):
        last_activity = time.time()
        while time.time() < QUIET_END:
            rows = processes()
            self.check_production(rows)
            events = self.tail.read()
            if events:
                last_activity = max(last_activity, max(events))
                self.event("production_lane_activity", count=len(events), latest=utc(max(events)))
            states = self.states()
            workers = external_workers(rows)
            if set(states) != set(LANES):
                raise RuntimeError("production ANE lane missing from management table")
            if any(state != "unloaded" for state in states.values()) or workers:
                last_activity = time.time()
                for lane, state in states.items():
                    if state != "unloaded":
                        self.touched = True
                        try:
                            result = self.rpc("model.unload", {"model_id": lane})
                            self.event("unload", lane=lane, state=result.get("state"))
                        except Exception as error:
                            self.event("unload_refused", lane=lane, error=str(error))
            elif time.time() - last_activity >= 120 and time.time() < QUIET_END:
                self.event("quiet_confirmed", seconds=time.time() - last_activity,
                           states=states, workers=workers)
                return True
            time.sleep(min(5, max(0, QUIET_END - time.time())))
        self.event("quiet_deadline_abort", states=self.states(), workers=external_workers(processes()))
        return False

    def remember(self, rows, root):
        for pid in descendants(rows, root):
            if pid in rows:
                self.owned[pid] = (rows[pid]["identity"], rows[pid]["exe"])
        # Private packed modules use aliases below the unique raw run directory.
        # This also catches one that was reparented between process snapshots.
        for pid, row in rows.items():
            if Path(row["exe"]).is_relative_to(self.out):
                self.owned[pid] = (row["identity"], row["exe"])

    def cleanup(self, child=None):
        signaled = []
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            rows = processes()
            for pid, identity in self.owned.items():
                row = rows.get(pid)
                if row and (row["identity"], row["exe"]) == identity:
                    if pid == self.prod:
                        raise RuntimeError("refusing to signal production")
                    try:
                        os.kill(pid, sig)
                        signaled.append(pid)
                    except ProcessLookupError:
                        pass
            if child is not None:
                try:
                    child.wait(timeout=2 if sig == signal.SIGTERM else 10)
                except subprocess.TimeoutExpired:
                    pass
            elif sig == signal.SIGTERM and signaled:
                time.sleep(2)
        rows = processes()
        remaining = [pid for pid, identity in self.owned.items() if pid in rows
                     and (rows[pid]["identity"], rows[pid]["exe"]) == identity]
        self.event("private_cleanup", signaled=sorted(set(signaled)), remaining=remaining)
        if remaining:
            raise RuntimeError("private processes did not exit; restoration may be blocked")

    def stage(self, name, argv, env):
        if time.time() >= END:
            raise RuntimeError("02:50Z hard window deadline before " + name)
        rows = processes()
        self.check_production(rows)
        workers = external_workers(rows)
        if workers or self.tail.read() or any(s != "unloaded" for s in self.states().values()):
            raise RuntimeError("production lane reloaded before " + name)
        self.record["stages"][name] = {"started": utc(), "load_before": command(["sysctl", "-n", "vm.loadavg"])}
        self.event("stage_start", stage=name)
        reason = None
        with (self.out / (name + ".log")).open("w") as log:
            child = subprocess.Popen([str(a) for a in argv], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    rows = processes()
                    self.remember(rows, child.pid)
                    self.check_production(rows)
                    live_owned = [pid for pid, identity in self.owned.items() if pid in rows
                                  and (rows[pid]["identity"], rows[pid]["exe"]) == identity]
                    workers = external_workers(rows, live_owned)
                    events = self.tail.read()
                    if workers or events:
                        reason = "production ANE activity: workers=" + str(workers) + ", log_events=" + str(len(events))
                        self.event("contamination", stage=name, workers=workers, log_events=len(events))
                        break
                    if time.time() >= END:
                        reason = "02:50Z hard window deadline"
                        break
                    if child.poll() is not None:
                        break
                    time.sleep(min(1, max(0, END - time.time())))
            finally:
                self.remember(processes(), child.pid)
                self.cleanup(child)
                child.wait(timeout=10)
        result = self.record["stages"][name]
        raw = self.out / ("packed-raw.json" if name == "packed" else "stress-raw.json")
        if raw.exists():
            payload = json.loads(raw.read_text())
            if name == "packed":
                result["arm_start_loads"] = payload.get("arm_start_loads", [])
        result.update(ended=utc(), exit_code=child.returncode, stopped_reason=reason,
                      valid=reason is None, load_after=command(["sysctl", "-n", "vm.loadavg"]))
        self.event("stage_end", stage=name, **result)
        return reason is None

    def restore(self):
        self.event("restoration_start")
        for lane in LANES:
            started = time.time()
            attempts = 0
            error = "restore not attempted"
            while time.time() - started < 180:
                attempts += 1
                try:
                    self.check_production(processes())
                    result = self.rpc("embed.query", {"model": lane, "id": "window-restore",
                                                       "text": "Neural Engine maintenance window restoration check"})
                    if not restored_vector(result, lane):
                        raise RuntimeError("restoration fingerprint/dimension/finite check failed")
                    self.record["restoration"][lane] = {"started": utc(started), "served_at": utc(),
                                                       "attempts": attempts, "fingerprint": result["fingerprint"],
                                                       "dims": len(result["vectors"][0]["vector"]), "passed": True}
                    self.event("restored", lane=lane, **self.record["restoration"][lane])
                    break
                except Exception as failure:
                    error = str(failure)
                    time.sleep(1)
            else:
                self.record["restoration"][lane] = {"started": utc(started), "ended": utc(),
                                                   "attempts": attempts, "passed": False, "error": error}
                self.event("restore_failed", lane=lane, **self.record["restoration"][lane])
        self.event("restoration_end", states=self.states())

    def run(self):
        self.preflight()
        if self.args.dry_run:
            return
        if time.time() >= QUIET_END:
            raise RuntimeError("too late to begin quiet check")
        if time.time() < START:
            self.event("waiting_for_window")
            time.sleep(START - time.time())
        self.event("window_started")
        try:
            if not self.quiet():
                self.record["outcome"] = "not run: no quiet interval by 02:15Z"
                return
            env = os.environ.copy()
            for key in ["TMPDIR", "SUBC_LAUNCH_NONCE", "SUBC_LAUNCH_NONCE_FD"]:
                env.pop(key, None)
            env.update(DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer",
                       SYNAPSE_PACKING_APPROVED="1", SYNAPSE_PACKING_UNLOADED_PRODUCTION_PID=str(self.prod),
                       SYNAPSE_COMPARE_ASSETS=str(self.args.assets.resolve()),
                       SYNAPSE_QWEN_WEIGHTS=str(self.args.weights.resolve()),
                       SYNAPSE_PACKING_INPUT=str(self.args.input.resolve()),
                       SYNAPSE_PACKING_OUT=str(self.out / "packed-raw.json"),
                       ANE_TEST_WORKER=str((self.args.assets / "ck-synapse-worker-ane-direct").resolve()),
                       ANE_TEST_PACKAGES=str(self.args.packages.resolve()))
            for key in ["SYNAPSE_PACKING_REPEATS", "SYNAPSE_PACKING_QUERY_SAMPLES"]:
                env.pop(key, None)
            if not self.stage("packed", [self.out / "ckdev-ane-packing-serving"], env):
                self.record["outcome"] = "packed stopped; stress not run"
                return
            if time.time() < END:
                self.stage("stress", [sys.executable, ROOT / "crates/synapse-worker-ane-direct/tests/run_stress.py",
                                      "--driver", self.args.driver.resolve(), "--wait-for-load",
                                      "--out", self.out / "stress-raw.json"], env)
            self.record["outcome"] = "see stage exit codes and unchanged gates"
        except Exception as error:
            self.record["outcome"] = str(error)
            self.event("window_error", error=str(error))
        finally:
            # Stop private work before restoration so its normal global ANE lock
            # cannot prevent production from reloading. Never signal production.
            try:
                self.cleanup()
            finally:
                self.restore()
                self.save()


class Tests(unittest.TestCase):
    def simulated_quiet(self, busy=False):
        from unittest.mock import Mock, patch
        window = object.__new__(Window)
        window.tail = Mock()
        window.tail.read.return_value = []
        window.check_production = Mock()
        window.event = Mock()
        window.touched = False
        ready = {lane: "ready" for lane in LANES}
        unloaded = {lane: "unloaded" for lane in LANES}
        window.states = Mock(return_value=ready if busy else unloaded)
        rpc = Mock(return_value={"state": "unloaded"})
        window.rpc = rpc
        clock = [START]
        def advance(seconds):
            clock[0] += seconds
        with patch.dict(globals(), processes=Mock(return_value={})):
            with patch("time.time", side_effect=lambda: clock[0]), patch("time.sleep", side_effect=advance):
                result = window.quiet()
        return result, clock[0], window, rpc

    def test_quiet_requires_two_full_minutes_without_lane_events(self):
        result, clock, window, rpc = self.simulated_quiet()
        self.assertTrue(result)
        self.assertGreaterEqual(clock - START, 120)
        rpc.assert_not_called()

    def test_quiet_aborts_at_0215_when_traffic_keeps_lanes_loaded(self):
        result, clock, window, rpc = self.simulated_quiet(busy=True)
        self.assertFalse(result)
        self.assertEqual(clock, QUIET_END)
        self.assertTrue(window.touched)
        self.assertEqual(rpc.call_args.args[0], "model.unload")

    def simulated_stage(self, worker=False, deadline=False, log_event=False):
        from unittest.mock import Mock, patch
        with tempfile.TemporaryDirectory() as root:
            window = object.__new__(Window)
            window.out = Path(root)
            window.prod = 1
            window.owned = {11: ("old", "/bin/ck-synapse-worker-ane-swift")}
            window.record = {"stages": {}}
            window.tail = Mock()
            window.tail.read.side_effect = [[], [START] if log_event else []]
            window.states = Mock(return_value={lane: "unloaded" for lane in LANES})
            window.check_production = Mock()
            window.event = Mock()
            window.cleanup = Mock()
            child = Mock(pid=10, returncode=0)
            child.poll.return_value = 0
            row = {"ppid": 0, "identity": "start", "exe": "/bin/private"}
            before = {1: {**row, "exe": "/bin/ck-synapse"}}
            after = {**before, 10: row}
            if worker:
                after[11] = {**row, "exe": "/bin/ck-synapse-worker-ane-swift"}
            clock = [START]
            def launch(*args, **kwargs):
                clock[0] = END if deadline else START
                return child
            with patch.dict(globals(), processes=Mock(side_effect=[before, after, after]),
                            command=Mock(return_value="{ 1 2 3 }")):
                with patch("subprocess.Popen", side_effect=launch), patch("time.time", side_effect=lambda: clock[0]):
                    valid = window.stage("packed", ["never executed"], {})
            return valid, window.record["stages"]["packed"]

    def test_stage_voids_arm_on_external_worker(self):
        valid, result = self.simulated_stage(worker=True)
        self.assertFalse(valid)
        self.assertIn("production ANE activity", result["stopped_reason"])

    def test_stage_voids_arm_on_production_log_activity(self):
        valid, result = self.simulated_stage(log_event=True)
        self.assertFalse(valid)
        self.assertIn("log_events=1", result["stopped_reason"])

    def test_stage_stops_at_hard_deadline(self):
        valid, result = self.simulated_stage(deadline=True)
        self.assertFalse(valid)
        self.assertEqual(result["stopped_reason"], "02:50Z hard window deadline")

    def test_stage_retains_successful_exit_without_contamination(self):
        valid, result = self.simulated_stage()
        self.assertTrue(valid)
        self.assertIsNone(result["stopped_reason"])

    def test_worker_scan_covers_both_lanes_and_excludes_only_owned_children(self):
        rows = {1: {"ppid": 0, "exe": "/bin/ck-synapse"},
                2: {"ppid": 1, "exe": "/bin/ck-synapse-worker-ane-direct"},
                3: {"ppid": 99, "exe": "/bin/ck-synapse-worker-ane-swift"},
                4: {"ppid": 9, "exe": "/bin/ck-synapse-worker-ane-direct"},
                9: {"ppid": 0, "exe": "/bin/test-driver"}}
        self.assertEqual(external_workers(rows), [2, 3, 4])
        self.assertEqual(external_workers(rows, descendants(rows, 9)), [2, 3])

    def test_log_guard_resets_for_admission_refusal_and_completion_not_management(self):
        for event in ["method=embed.query", "job admitted", "job refused", "job done", "branch=cold_load"]:
            self.assertTrue(lane_event("model_id=qwen3-embedding-0.6b-ane " + event))
        self.assertFalse(lane_event("model_id=qwen3-embedding-0.6b-ane method=model.unload"))
        self.assertFalse(lane_event("model_id=gte-modernbert-base-f16 job admitted"))

    def test_log_tail_waits_for_newline_and_follows_date_rotation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "synapse.2026-10-12.log"
            event = "2026-10-12T23:59:59.123Z INFO synapse: job admitted model_id=qwen3-embedding-0.6b-ane"
            path.write_text(event)
            tail = LogTail(Path(root))
            self.assertEqual(tail.read(), [])
            with path.open("a") as f:
                f.write("\n")
            self.assertEqual(len(tail.read()), 1)
            self.assertEqual(tail.read(), [])
            (Path(root) / "synapse.2026-10-13.log").write_text(event + "\n")
            self.assertEqual(len(tail.read()), 1)

    def test_historical_tail_skips_only_initial_partial_record(self):
        with tempfile.TemporaryDirectory() as root:
            event = "2026-10-12T23:59:59.123Z INFO synapse: job admitted model_id=qwen3-embedding-0.6b-ane\n"
            path = Path(root) / "synapse.log"
            path.write_text("x" * (256 * 1024) + event + event)
            tail = LogTail(Path(root))
            self.assertEqual(len(tail.read()), 1)
            self.assertEqual(tail.read(), [])

    def test_restoration_checks_actual_vector_and_fingerprint(self):
        result = {"fingerprint": "5f7ac7cdabc", "vectors": [{"vector": [0.0] * 1024}]}
        lane = "qwen3-embedding-0.6b-ane"
        self.assertTrue(restored_vector(result, lane))
        self.assertFalse(restored_vector(result, "gte-modernbert-base-ane-fp16"))
        result["vectors"][0]["vector"][0] = float("nan")
        self.assertFalse(restored_vector(result, lane))
        result["vectors"][0]["vector"] = [0.0] * 768
        self.assertFalse(restored_vector(result, lane))

    def test_cleanup_signals_only_recorded_identity_not_reused_pid_or_production(self):
        from unittest.mock import Mock, call, patch
        with tempfile.TemporaryDirectory() as root:
            window = object.__new__(Window)
            window.prod = 1
            window.out = Path(root)
            window.owned = {2: ("start", "/bin/private"), 3: ("old", "/bin/private")}
            window.event = lambda *args, **kwargs: None
            rows = {1: {"identity": "start", "exe": "/bin/ck-synapse"},
                    2: {"identity": "start", "exe": "/bin/private"},
                    3: {"identity": "new", "exe": "/bin/private"}}
            with patch.dict(globals(), processes=Mock(side_effect=[rows, rows, {}])):
                with patch("os.kill") as kill, patch("time.sleep"):
                    window.cleanup()
                    self.assertEqual(kill.call_args_list, [call(2, signal.SIGTERM),
                                                           call(2, signal.SIGKILL)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    for name in ["out", "assets", "driver", "weights", "input", "packages"]:
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
        return
    if any(getattr(args, name) is None for name in ["out", "assets", "driver", "weights", "input", "packages"]):
        parser.error("all asset/output paths are required")
    def interrupted(signum, frame):
        raise RuntimeError("supervisor interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Window(args).run()


if __name__ == "__main__":
    main()
