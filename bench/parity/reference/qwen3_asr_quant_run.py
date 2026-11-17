#!/usr/bin/env python3
"""Resumable public-only Q8 CPU campaign: prepare, run, progress, collect, pack."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

from qwen3_asr_reference import canonical, model_pin, sha256
from qwen3_asr_corpus import expand

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORPUS = HERE / "qwen3-asr-corpus.json"
FP32 = HERE.parent / "fixtures/qwen3-asr-1.7b/traces"
RAM_PER_PROCESS = 20_000_000_000
MAX_MEMBER = 16 * 1024 * 1024
MAX_BUNDLE = 3 * 1024**3


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, raw, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".publishing-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value, exclusive=False):
    return atomic_bytes(path, canonical(value) + b"\n", exclusive)


@contextmanager
def state_lock(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "supervisor.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another supervisor holds this state directory") from error
        yield


def corpus_cases():
    return [c for c in expand(read_json(CORPUS))["cases"] if not c.get("refusal")]


def source_files():
    files = [HERE.parent / name for name in ("Cargo.toml", "Cargo.lock", "models.json")]
    files += sorted((HERE.parent / "src").rglob("*.rs"))
    files += sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.json"))
    files += sorted(HERE.glob("requirements-*.txt"))
    files += [HERE / "setup_qwen3_asr_quant_run.sh", HERE / "QUANT-GATE-RUN.md"]
    files += sorted(FP32.glob("*.json"))
    files += [ROOT / "bench/lanes/stt-bias/score.py"]
    files += [ROOT / "crates/synapse-owned-decode/owned-decode-routing" / (name + ".rs")
              for name in ("error", "family", "q8artifact", "q8asr", "q8ingest", "tensor_source")]
    return sorted(set(files))


def identity():
    # Bind resumable results to both numerical code and every sealed input trace.
    files = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    value = {"files": files, "model": model_pin()}
    return hashlib.sha256(canonical(value)).hexdigest(), value


def verify_kit():
    manifest = read_json(ROOT / "quant-gate-kit.json")
    for name, digest in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError("upload digest mismatch: " + name)
    current, _ = identity()
    if current != manifest["identity"]:
        raise ValueError("upload identity mismatch")
    return manifest


def pack(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("upload archive must be new")
    key, inputs = identity()
    manifest = {"schema": "qwen-quant-kit-v1", "identity": key, **inputs}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in source_files():
            archive.add(path, arcname=str(Path("quant-gate-kit") / path.relative_to(ROOT)), recursive=False)
        raw = canonical(manifest)
        info = tarfile.TarInfo("quant-gate-kit/quant-gate-kit.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    return {"archive": str(destination), "sha256": sha256(destination),
            "bytes": destination.stat().st_size, "identity": key,
            "files": len(manifest["files"]) + 1}


def lpt(entries, processes):
    if processes < 1:
        raise ValueError("processes must be positive")
    queue = [(0, index) for index in range(processes)]
    heapq.heapify(queue)
    shards = [[] for _ in queue]
    for entry in sorted(entries, key=lambda row: (-row["weight"], row["id"])):
        cost, index = heapq.heappop(queue)
        shards[index].append(entry["id"])
        heapq.heappush(queue, (cost + entry["weight"], index))
    return shards


def available_ram():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("Linux MemAvailable is required for RAM admission")


def admit_processes(processes, available):
    maximum = available // RAM_PER_PROCESS
    if processes < 1 or processes > maximum:
        raise ValueError(f"{processes} processes require {processes * 20} GB; "
                         f"safe maximum is {maximum} at 20 GB/process; choose --processes explicitly")


def plan_entries(cases):
    entries = []
    for case in cases:
        baseline = FP32 / (case["id"] + ".json")
        row = read_json(baseline)
        prompt, generated = len(row["prompt_ids"]), len(row["generated_ids"])
        # Decoder work grows with generated length and attended context. The
        # squared prefill term also gives immediate-stop long audio nonzero cost.
        weight = max(1, generated * (prompt + generated) + prompt * prompt // 32)
        entries.append({"id": case["id"], "weight": weight,
                        "case_sha256": hashlib.sha256(canonical(case)).hexdigest(),
                        "fp32_sha256": sha256(baseline)})
    return entries


def prepare(cache, derive_binary, setup_started_at):
    from qwen3_asr_quant import verified_artifacts
    from qwen3_asr_reference import deterministic_runtime, fetch_model
    from run_qwen3_asr_quant import prepare_audio
    cache = Path(cache).resolve()
    deterministic_runtime()
    model = fetch_model(cache)
    derived = cache / "quant-derived"
    receipt = cache / "prepared.json"
    if derived.exists() and any(derived.iterdir()):
        if not receipt.is_file():
            raise ValueError("incomplete derivation cache: move it aside before setup retry")
        previous = read_json(receipt)
        artifacts = cache / previous["artifacts"]
    else:
        result = subprocess.run([str(derive_binary), "qwen3-asr-1.7b", str(model), str(derived)],
                                capture_output=True, text=True, check=True)
        print(result.stdout, end="", flush=True)
        candidates = list(derived.glob("owned-decode-q8/*/weights.q8_0"))
        if len(candidates) != 1:
            raise ValueError("expected exactly one Rust Q8 artifact")
        artifacts = candidates[0].parent
    verified_artifacts(artifacts)
    corpus = expand(read_json(CORPUS))
    for index, case in enumerate(corpus["cases"]):
        if not case.get("refusal"):
            prepare_audio(case, corpus["sources"], cache)
        if index % 200 == 0:
            print(json.dumps({"materialized_cases": index, "total": len(corpus["cases"])}), flush=True)
    key, _ = identity()
    value = {"schema": "qwen-quant-cache-v1", "identity": key,
             "model": str(model.relative_to(cache)), "artifacts": str(artifacts.relative_to(cache)),
             "setup_started_at": setup_started_at}
    atomic_json(receipt, value)
    for path in cache.rglob("*"):
        if path.is_symlink():
            raise ValueError("shared cache must not contain symlinks")
        path.chmod(path.stat().st_mode & ~0o222)
    cache.chmod(cache.stat().st_mode & ~0o222)
    return value


def archive_path(state, id):
    if not id or Path(id).name != id or id in (".", ".."):
        raise ValueError("unsafe case id")
    return Path(state) / "cases" / (id + ".tar.gz")


def publish_case(state, plan, entry, result):
    payloads = {name + ".json": canonical(result[name])
                for name in ("teacher", "profile") if result.get(name) is not None}
    meta = {"schema": "qwen-quant-case-v1", "identity": plan["identity"], **entry,
            "failure": result.get("failure"), "metrics": result.get("metrics", {}),
            "finished_at": time.time(),
            "payloads": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}}
    payloads["meta.json"] = canonical(meta)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, raw in payloads.items():
            if len(raw) > MAX_MEMBER:
                raise ValueError("case JSON exceeds archive size bound")
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    if not atomic_bytes(archive_path(state, entry["id"]), buffer.getvalue(), exclusive=True):
        raise ValueError("case archive already published: " + entry["id"])


def read_case(state, plan, entry):
    path = archive_path(state, entry["id"])
    payloads = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if (member.name not in ("meta.json", "teacher.json", "profile.json")
                    or member.name in payloads or not member.isfile() or member.size > MAX_MEMBER):
                raise ValueError("invalid case archive members: " + str(path))
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("missing archive payload")
            payloads[member.name] = stream.read()
    meta = json.loads(payloads.pop("meta.json"))
    if (meta["schema"] != "qwen-quant-case-v1" or meta["identity"] != plan["identity"]
            or any(meta[key] != value for key, value in entry.items())
            or meta["payloads"] != {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
            or "teacher.json" not in payloads
            or (not meta["failure"] and "profile.json" not in payloads)):
        raise ValueError("case archive identity or payload digest mismatch: " + str(path))
    result = {name.removesuffix(".json"): json.loads(raw) for name, raw in payloads.items()}
    for name in ("teacher", "profile"):
        if name in result and result[name]["id"] != entry["id"]:
            raise ValueError("case payload id mismatch")
    return {**result, "meta": meta}


def fail(state, kind, detail):
    atomic_json(Path(state) / "failure.json", {"kind": kind, "detail": detail,
                                              "time": time.time()}, exclusive=True)


def evaluator(cache):
    from qwen3_asr_quant import QuantReference
    from qwen3_asr_quant_gate import calibration, cap_failure, probe_failure
    from qwen3_asr_reference import pcm
    cache = Path(cache)
    receipt = read_json(cache / "prepared.json")
    reference = QuantReference(cache / receipt["model"], cache / receipt["artifacts"])
    cases = {c["id"]: c for c in corpus_cases()}

    def evaluate(id):
        case, baseline = cases[id], read_json(FP32 / (id + ".json"))
        audio = pcm(case, cache)
        begin = time.monotonic()
        teacher = reference.run(case, audio)
        teacher["audio_sha256"] = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
        teacher["fp32_trace_sha256"] = sha256(FP32 / (id + ".json"))
        if (teacher["prompt_ids"] != baseline["prompt_ids"] or teacher["span"] != baseline["span"]
                or teacher["audio_sha256"] != baseline["model_input_audio_sha256"]):
            raise ValueError("Q8 inputs differ from the seal: " + id)
        result = {"teacher": teacher, "profile": None,
                  "metrics": {"teacher_seconds": time.monotonic() - begin}}
        result["failure"] = probe_failure(case, teacher) or cap_failure(teacher, baseline)
        if result["failure"]:
            return result
        begin = time.monotonic()
        profile = reference.run(case, audio, forced_ids=teacher["generated_ids"][:-1], profile=True)
        result["profile"] = profile
        result["metrics"]["profile_seconds"] = time.monotonic() - begin
        calibrated = calibration([case], {id: teacher}, {id: profile})
        # Coverage is pooled over a whole fixture set. Tolerance tau is the
        # larger of 0.1 logit and twice the largest disagreement margin; its
        # >1-logit failure cannot be repaired by later fixtures.
        result["failure"] = next((f for f in calibrated["failures"] if f["gate"] == "tau-maximum"), None)
        return result
    return evaluate


def worker_loop(state, shard, make_evaluator):
    state = Path(state)
    plan = read_json(state / "plan.json")
    entries = {e["id"]: e for e in plan["entries"]}
    evaluate = None
    try:
        for id in plan["shards"][shard]:
            if (state / "failure.json").exists():
                return 2
            entry = entries[id]
            if archive_path(state, id).exists():
                result = read_case(state, plan, entry)
                if result["meta"]["failure"]:
                    fail(state, "gate-failed", result["meta"]["failure"])
                    return 2
                continue
            if evaluate is None:
                evaluate = make_evaluator()
            result = evaluate(id)
            result.setdefault("metrics", {})["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            publish_case(state, plan, entry, result)
            if result.get("failure"):
                fail(state, "gate-failed", result["failure"])
                return 2
        return 0
    except Exception as error:
        fail(state, "execution-error", {"shard": shard, "type": type(error).__name__, "message": str(error)})
        return 1


def public_checker(cases, wer_binary):
    from collections import Counter
    from qwen3_asr_quant_gate import calibration, quality, wer_counts
    baseline = {c["id"]: read_json(FP32 / (c["id"] + ".json")) for c in cases}
    totals = Counter(c["split"] for c in cases)
    counts = {}
    arms = {}
    for case in cases:
        arm = case.get("list_id") or ("none" if not case.get("terms") else
                                      "tier-" + str(case["tier"]) if case.get("tier") else "global")
        arms.setdefault((case["split"], arm, case.get("tier")), set()).add(case["id"])
    previous_ready, previous_quality = None, None

    def check(records):
        nonlocal previous_ready, previous_quality
        teachers = {id: r["teacher"] for id, r in records.items()}
        profiles = {id: r["profile"] for id, r in records.items()}
        seen = [c for c in cases if c["id"] in records]
        new = [c for c in seen if c["id"] not in counts]
        if new:
            counts.update(wer_counts(wer_binary, new, baseline, teachers))
        calibrated = calibration(seen, teachers, profiles)
        ready = Counter(c["split"] for c in seen)
        complete_sets = {split for split, total in totals.items() if ready[split] == total}
        failures = [f for f in calibrated["failures"]
                    if f["gate"] == "tau-maximum" or f.get("split") in complete_sets]
        ready_arms = {key for key, ids in arms.items() if ids <= records.keys()}
        if ready_arms != previous_ready:
            previous_quality = quality(cases, baseline, teachers, counts, partial=True)
            previous_ready = ready_arms
        graded = previous_quality
        if graded is None:
            raise ValueError("quality readiness was not initialized")
        failures.extend(graded["failures"])
        return {"failures": failures, "tau": calibrated["tau"],
                "complete_fixture_sets": sorted(complete_sets), "quality": graded,
                "engine_slices_allowed": False}
    return check


def stop_peers(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for child in children:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def parent_death_guard(expected_parent):
    import ctypes
    if (os.getppid() != expected_parent
            or ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0
            or os.getppid() != expected_parent):
        raise ValueError("cannot establish parent-death supervision")


def scan_cases(state, plan, records):
    changed = False
    known = {entry["id"] for entry in plan["entries"]}
    if any(path.name.removesuffix(".tar.gz") not in known for path in (state / "cases").glob("*.tar.gz")):
        raise ValueError("unexpected case archive")
    for entry in plan["entries"]:
        id = entry["id"]
        if id not in records and archive_path(state, id).exists():
            record = read_case(state, plan, entry)
            if record["meta"]["failure"]:
                fail(state, "gate-failed", record["meta"]["failure"])
            else:
                records[id] = record
            changed = True
    return changed


def supervise(state, plan, check, worker_command=None, poll_seconds=1.0):
    """One supervisor owns a state lock; a failed gate stops every process group."""
    state = Path(state)
    records, children, logs = {}, [], []
    command = worker_command or [sys.executable, str(Path(__file__).resolve()), "worker"]
    status = {"started_at": plan.get("first_started_at", time.time()),
              "status": "running", "completed": [], "engine_slices_allowed": False}

    def update_status():
        status["completed"] = [{"id": id, "finished_at": r["meta"]["finished_at"],
                                "weight": r["meta"]["weight"],
                                "peak_rss_bytes": r["meta"]["metrics"].get("peak_rss_bytes", 0)}
                               for id, r in sorted(records.items())]
        status["updated_at"] = time.time()
        atomic_json(state / "status.json", status)

    try:
        prior = state / "failure.json"
        if prior.exists():
            if read_json(prior)["kind"] == "gate-failed":
                raise ValueError("a definitive gate failure already stopped this run")
            history = state / "history"
            history.mkdir(exist_ok=True)
            os.replace(prior, history / f"failure-{time.time_ns()}.json")
        scan_cases(state, plan, records)
        if records and not (state / "failure.json").exists():
            result = check(records)
            atomic_json(state / "aggregate.json", result)
            if result["failures"]:
                fail(state, "gate-failed", result["failures"])
        if not (state / "failure.json").exists() and len(records) < len(plan["entries"]):
            (state / "logs").mkdir(exist_ok=True)
            for shard in range(plan["processes"]):
                stream = (state / "logs" / f"shard-{shard:02d}.log").open("ab")
                logs.append(stream)
                child = subprocess.Popen(command + ["--state", str(state), "--shard", str(shard),
                                                     "--supervisor-pid", str(os.getpid())],
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
        update_status()
        while children:
            if (state / "failure.json").exists():
                break
            changed = scan_cases(state, plan, records)
            if changed and not (state / "failure.json").exists():
                result = check(records)
                atomic_json(state / "aggregate.json", result)
                if result["failures"]:
                    fail(state, "gate-failed", result["failures"])
            exited_bad = [c for c in children if c.poll() not in (None, 0)]
            if exited_bad and not (state / "failure.json").exists():
                fail(state, "execution-error", {"exited_workers": [c.returncode for c in exited_bad]})
            if changed:
                update_status()
            if all(c.poll() is not None for c in children):
                scan_cases(state, plan, records)
                if len(records) != len(plan["entries"]) and not (state / "failure.json").exists():
                    fail(state, "execution-error", {"message": "workers exited with incomplete coverage"})
                if not (state / "failure.json").exists():
                    result = check(records)
                    atomic_json(state / "aggregate.json", result)
                    if result["failures"]:
                        fail(state, "gate-failed", result["failures"])
                break
            time.sleep(poll_seconds)
    except BaseException as error:
        fail(state, "execution-error", {"type": type(error).__name__, "message": str(error)})
    finally:
        stop_peers(children)
        for stream in logs:
            stream.close()
        failed = (state / "failure.json").exists()
        status["status"] = "stopped" if failed else "public-complete"
        if failed:
            status["failure"] = read_json(state / "failure.json")
        update_status()
    return 2 if (state / "failure.json").exists() else 0


def progress(state, now=None):
    state = Path(state)
    now = time.time() if now is None else now
    plan, status = read_json(state / "plan.json"), read_json(state / "status.json")
    complete = status["completed"]
    elapsed = max(1, now - status["started_at"])
    recent = [r for r in complete if now - 3600 <= r["finished_at"] <= now]
    window = min(3600, elapsed)
    rate = len(recent) * 3600 / window
    weight_done = sum(r["weight"] for r in complete)
    weight_left = sum(e["weight"] for e in plan["entries"]) - weight_done
    weight_rate = sum(r["weight"] for r in recent) / window
    basis = "last-hour"
    if not weight_rate:
        weight_rate = weight_done / elapsed
        basis = "since-supervisor-start"
    hours = weight_left / weight_rate / 3600 if weight_rate else None
    spent = plan["spent_before"] + max(0, now - plan["rental_start"]) / 3600 * plan["hourly_rate"]
    projected = spent + hours * plan["hourly_rate"] if hours is not None else None
    failure = read_json(state / "failure.json") if (state / "failure.json").exists() else None
    return {"status": "stopped" if failure else status["status"], "completed": len(complete), "total": len(plan["entries"]),
            "cases_per_hour_last_hour": rate, "projected_hours_remaining": hours,
            "projection_basis": basis, "estimated_cost_so_far": spent,
            "projected_total_cost": projected, "budget_dollars": plan["budget"],
            "over_budget": spent > plan["budget"] or (projected is not None and projected > plan["budget"]),
            "status_age_seconds": now - status["updated_at"],
            "observed_peak_process_rss_bytes": max((r.get("peak_rss_bytes", 0) for r in complete), default=0),
            "failure": failure,
            "engine_slices_allowed": False}


def run_campaign(args):
    from qwen3_asr_quant import verified_artifacts
    state, cache = args.state.resolve(), args.cache.resolve()
    admit_processes(args.processes, available_ram())
    key, _ = identity()
    receipt = read_json(cache / "prepared.json")
    if receipt["identity"] != key or cache.stat().st_mode & 0o222:
        raise ValueError("cache identity mismatch or cache is not read-only; rerun setup")
    verified_artifacts(cache / receipt["artifacts"])
    with state_lock(state):
        cases = corpus_cases()
        entries = plan_entries(cases)
        plan = {"schema": "qwen-quant-run-v1", "identity": key, "entries": entries,
                "processes": args.processes, "shards": lpt(entries, args.processes),
                "cache": str(cache), "hourly_rate": args.hourly_rate,
                "rental_start": args.rental_start or receipt["setup_started_at"],
                "spent_before": args.spent_before, "budget": 10.0,
                "first_started_at": time.time()}
        old_path = state / "plan.json"
        if old_path.exists():
            old = read_json(old_path)
            if old["identity"] != key or old["entries"] != entries:
                raise ValueError("state belongs to different numerical code or frozen inputs")
            plan["first_started_at"] = old["first_started_at"]
        # Sharding and mount paths may change on restart; case identity does not.
        atomic_json(old_path, plan)
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            return supervise(state, plan, public_checker(cases, args.wer_binary.resolve()))
        finally:
            signal.signal(signal.SIGTERM, previous)


def collect(state, destination, wer_binary):
    state, destination = Path(state).resolve(), Path(destination)
    if destination.exists():
        raise ValueError("collection archive must be new")
    with state_lock(state):
        plan = read_json(state / "plan.json")
        key, _ = identity()
        cases = corpus_cases()
        if plan["identity"] != key or plan["entries"] != plan_entries(cases):
            raise ValueError("collection identity mismatch")
        records = {}
        scan_cases(state, plan, records)
        if (state / "failure.json").exists() or len(records) != len(cases):
            raise ValueError("collection requires complete passing public coverage; inspect progress/failure.json")
        report = public_checker(cases, wer_binary)(records)
        if report["failures"]:
            fail(state, "gate-failed", report["failures"])
            raise ValueError("complete public grading failed")
        manifest = {"schema": "qwen-quant-bundle-v1", "identity": key, "files": {},
                    "engine_slices_allowed": False}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial")
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                def add(name, value):
                    raw = canonical(value)
                    manifest["files"][name] = hashlib.sha256(raw).hexdigest()
                    info = tarfile.TarInfo("generated/" + name)
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
                for id, record in sorted(records.items()):
                    add("teacher/" + id + ".json", record["teacher"])
                    add("forced-profile/" + id + ".json", record["profile"])
                add("public-report.json", report)
                add("plan.json", plan)
                raw = canonical(manifest)
                info = tarfile.TarInfo("generated/bundle-manifest.json")
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
            if temporary.stat().st_size > MAX_BUNDLE:
                raise ValueError("collection exceeds the 3 GiB transport bound")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"archive": str(destination), "sha256": sha256(destination),
            "bytes": destination.stat().st_size, "cases": len(records),
            "engine_slices_allowed": False}


def verify_generated(generated):
    generated = Path(generated).resolve()
    manifest = read_json(generated / "bundle-manifest.json")
    key, _ = identity()
    if manifest["identity"] != key:
        raise ValueError("bundle was produced by different code or frozen inputs")
    expected = {f"{folder}/{case['id']}.json" for case in corpus_cases()
                for folder in ("teacher", "forced-profile")}
    expected |= {"public-report.json", "plan.json"}
    if set(manifest["files"]) != expected:
        raise ValueError("bundle fixture coverage mismatch")
    for name, digest in manifest["files"].items():
        path = (generated / name).resolve()
        if not path.is_relative_to(generated) or sha256(path) != digest:
            raise ValueError("bundle digest mismatch: " + name)
    return {"verified_cases": len(expected - {"public-report.json", "plan.json"}) // 2,
            "engine_slices_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    package = sub.add_parser("pack")
    package.add_argument("--output", type=Path, required=True)
    sub.add_parser("verify-kit")
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--derive-binary", type=Path, required=True)
    prep.add_argument("--setup-started-at", type=float, default=time.time())
    run = sub.add_parser("run")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--wer-binary", type=Path, required=True)
    run.add_argument("--processes", type=int, default=28)
    run.add_argument("--hourly-rate", type=float, default=.45)
    run.add_argument("--rental-start", type=float)
    run.add_argument("--spent-before", type=float, default=0)
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--state", type=Path, required=True)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--supervisor-pid", type=int, required=True)
    status = sub.add_parser("progress")
    status.add_argument("--state", type=Path, required=True)
    collection = sub.add_parser("collect")
    collection.add_argument("--state", type=Path, required=True)
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--wer-binary", type=Path, required=True)
    verify = sub.add_parser("verify-bundle")
    verify.add_argument("--generated", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "pack":
        result = pack(args.output)
    elif args.action == "verify-kit":
        result = {"identity": verify_kit()["identity"], "verified": True}
    elif args.action == "prepare":
        result = prepare(args.cache, args.derive_binary.resolve(), args.setup_started_at)
    elif args.action == "run":
        import math
        if any(not math.isfinite(value) or value < 0 for value in (args.hourly_rate, args.spent_before)):
            parser.error("hourly rate and prior spend must be finite and nonnegative")
        if args.rental_start is not None and (not math.isfinite(args.rental_start) or args.rental_start <= 0):
            parser.error("rental start must be a finite positive epoch timestamp")
        raise SystemExit(run_campaign(args))
    elif args.action == "worker":
        # Linux kills an in-flight worker if its supervisor dies, including SIGKILL.
        parent_death_guard(args.supervisor_pid)
        state = args.state.resolve()
        plan = read_json(state / "plan.json")
        raise SystemExit(worker_loop(state, args.shard, lambda: evaluator(plan["cache"])))
    elif args.action == "progress":
        result = progress(args.state)
    elif args.action == "collect":
        result = collect(args.state, args.output, args.wer_binary.resolve())
    else:
        result = verify_generated(args.generated)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
