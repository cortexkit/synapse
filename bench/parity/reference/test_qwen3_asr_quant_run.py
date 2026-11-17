#!/usr/bin/env python3
"""Tiny public-run orchestration tests; no models, downloads or inference."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import qwen3_asr_quant_run as kit


def synthetic_plan(state, **options):
    entries = [{"id": id, "weight": weight, "case_sha256": "fixture", "fp32_sha256": "baseline"}
               for id, weight in (("a", 30), ("b", 20), ("c", 10))]
    plan = {"identity": "synthetic-not-sealable", "entries": entries, "processes": 2,
            "shards": [["a", "c"], ["b"]], "synthetic": options,
            "hourly_rate": .45, "spent_before": 0, "budget": 10,
            "rental_start": time.time(), "first_started_at": time.time()}
    kit.atomic_json(state / "plan.json", plan)
    return plan


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("synthetic worker did not reach its barrier")


def fake_factory(state):
    options = kit.read_json(state / "plan.json")["synthetic"]

    def evaluate(id):
        counter = state / (id + ".attempts")
        counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
        if options.get("blocked") == id:
            (state / "peer-pid").write_text(str(os.getpid()))
            while True:
                time.sleep(.1)
        if options.get("wait_for_peer") == id:
            wait_until(lambda: (state / "peer-pid").exists())
        if options.get("pause") == id and not (state / "resume-now").exists():
            while True:
                time.sleep(.1)
        return {"teacher": {"id": id}, "profile": {"id": id},
                "metrics": {"teacher_seconds": .01, "profile_seconds": .01},
                "failure": {"gate": "synthetic-definitive-failure", "id": id}
                if options.get("local_fail") == id else None}
    return evaluate


def fake_command():
    return [sys.executable, str(Path(__file__).resolve()), "--fake-worker"]


class Scheduling(unittest.TestCase):
    def test_all_6559_frozen_cases_have_one_deterministic_shard(self):
        entries = kit.plan_entries(kit.corpus_cases())
        shards = kit.lpt(entries, 28)
        flattened = [id for shard in shards for id in shard]
        self.assertEqual(len(flattened), 6559)
        self.assertEqual(len(set(flattened)), 6559)
        self.assertEqual(set(flattened), {row["id"] for row in entries})
        weights = {row["id"]: row["weight"] for row in entries}
        for shard in shards:
            self.assertEqual([weights[id] for id in shard], sorted((weights[id] for id in shard), reverse=True))

    def test_lpt_is_balanced_deterministic_and_longest_first(self):
        entries = [{"id": id, "weight": weight} for id, weight in (("b", 20), ("c", 10), ("a", 30))]
        self.assertEqual(kit.lpt(entries, 2), [["a"], ["b", "c"]])
        self.assertEqual(kit.lpt(list(reversed(entries)), 2), [["a"], ["b", "c"]])

    def test_ram_admission_names_safe_maximum_without_silent_reduction(self):
        kit.admit_processes(28, 560_000_000_000)
        with self.assertRaisesRegex(ValueError, "safe maximum is 27"):
            kit.admit_processes(28, 559_000_000_000)

    def test_state_lock_refuses_a_second_supervisor(self):
        with tempfile.TemporaryDirectory() as folder:
            with kit.state_lock(folder):
                with self.assertRaisesRegex(ValueError, "another supervisor"):
                    with kit.state_lock(folder):
                        self.fail("second lock was admitted")


class Resume(unittest.TestCase):
    def test_well_formed_archive_with_changed_payload_fails_hash_verification(self):
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            plan = synthetic_plan(state)
            entry = plan["entries"][0]
            kit.publish_case(state, plan, entry, {"teacher": {"id": "a"}, "profile": {"id": "a"}})
            path = kit.archive_path(state, "a")
            with tarfile.open(path) as archive:
                members = {}
                for member in archive.getmembers():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    members[member.name] = stream.read()
            members["teacher.json"] = b'{"id":"a","changed":true}'
            with tarfile.open(path, "w:gz") as archive:
                for name, raw in members.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
            with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
                kit.read_case(state, plan, entry)

    def test_killed_shard_reuses_verified_case_and_repeats_only_in_flight_case(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            plan = synthetic_plan(state, pause="c")
            child = subprocess.Popen(fake_command() + ["--state", str(state), "--shard", "0"],
                                     start_new_session=True)
            try:
                wait_until(lambda: (state / "c.attempts").exists())
                kit.read_case(state, plan, plan["entries"][0])
            finally:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            (state / "resume-now").touch()
            self.assertEqual(kit.worker_loop(state, 0, lambda: fake_factory(state)), 0)
            self.assertEqual((state / "a.attempts").read_text(), "1")
            self.assertEqual((state / "c.attempts").read_text(), "2")
            kit.read_case(state, plan, plan["entries"][2])
            self.assertFalse(list((state / "cases").glob(".publishing-*")))

    def test_corrupt_or_foreign_archive_is_never_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            plan = synthetic_plan(state)
            entry = plan["entries"][0]
            kit.publish_case(state, plan, entry, {"teacher": {"id": "a"}, "profile": {"id": "a"}})
            with self.assertRaises(ValueError):
                kit.read_case(state, {**plan, "identity": "other"}, entry)
            kit.archive_path(state, "a").write_bytes(b"incomplete gzip")
            with self.assertRaises(Exception):
                kit.read_case(state, plan, entry)
            with patch.object(kit, "publish_case", side_effect=AssertionError("must not overwrite")):
                self.assertEqual(kit.worker_loop(state, 0, lambda: self.fail("must not infer")), 1)
            self.assertEqual(kit.read_json(state / "failure.json")["kind"], "execution-error")


class StopRule(unittest.TestCase):
    def check_peer_gone(self, state):
        peer = int((state / "peer-pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(peer, 0)

    def test_worker_gate_failure_stops_blocked_peer_and_persists_case(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            plan = synthetic_plan(state, local_fail="a", blocked="b", wait_for_peer="a")
            result = kit.supervise(state, plan, lambda _r: {"failures": []}, fake_command(), .02)
            self.assertEqual(result, 2)
            self.assertEqual(kit.read_json(state / "failure.json")["kind"], "gate-failed")
            self.assertEqual(kit.read_case(state, plan, plan["entries"][0])["meta"]["failure"]["id"], "a")
            self.assertFalse(kit.archive_path(state, "c").exists())
            self.check_peer_gone(state)
            self.assertEqual(kit.supervise(state, plan, lambda _r: self.fail("must not resume failed gate"),
                                           fake_command(), .02), 2)

    def test_complete_aggregate_failure_stops_remaining_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            plan = synthetic_plan(state, blocked="b", wait_for_peer="a", pause="c")
            def check(records):
                return {"failures": [{"gate": "synthetic-complete-split"}] if "a" in records else []}
            self.assertEqual(kit.supervise(state, plan, check, fake_command(), .02), 2)
            self.assertEqual(kit.read_json(state / "failure.json")["detail"][0]["gate"],
                             "synthetic-complete-split")
            self.check_peer_gone(state)
            self.assertTrue(kit.archive_path(state, "a").exists())
            self.assertFalse(kit.archive_path(state, "c").exists())


class AggregateReadiness(unittest.TestCase):
    def test_partial_aggregation_refuses_missing_tier_dependencies(self):
        from qwen3_asr_quant_gate import quality
        cases = [{"id": f"e{i}-tier-30", "split": "earnings22-test-200", "ground_truth": "Alice",
                  "tier": 30, "terms": ["Alice"], "retained_terms": ["Alice"]} for i in range(200)]
        cases += [{"id": f"base{i}-control-30-{n}", "split": "speech", "ground_truth": "speech",
                   "tier": 30, "terms": ["Acme"], "list_id": str(n), "probe_set": "unrelated-list"}
                  for n in range(3) for i in range(400)]
        traces = {c["id"]: {"text": c["ground_truth"], "stop_reason": "stop", "context_tokens": 64}
                  for c in cases}
        counts = {c["id"]: {"words": 1000, "baseline_edits": 0, "actual_edits": 0} for c in cases}
        with self.assertRaisesRegex(ValueError, "dependencies"):
            quality(cases, traces, traces, counts, partial=True)

    def test_completed_control_insertion_set_does_not_wait_for_earnings(self):
        from qwen3_asr_quant_gate import quality
        for tier in (30, 100):
            cases, baseline, actual, counts = [], {}, {}, {}
            def add(case, completed=False):
                cases.append(case)
                id = case["id"]
                baseline[id] = {"text": case["ground_truth"], "stop_reason": "stop"}
                if completed:
                    actual[id] = {**baseline[id], "context_tokens": 64}
                    counts[id] = {"words": 1000, "baseline_edits": 0, "actual_edits": 0}
            for i in range(400):
                add({"id": f"base{i}", "split": "speech", "ground_truth": "speech", "terms": []})
            for i in range(200):
                add({"id": f"e{i}-global", "split": "earnings22-test-200", "ground_truth": "Alice", "terms": ["Alice"]})
                add({"id": f"e{i}-tier-{tier}", "split": "earnings22-test-200", "ground_truth": "Alice",
                     "tier": tier, "terms": ["Alice"], "retained_terms": ["Alice"]})
            for list_num in range(3):
                for i in range(400):
                    add({"id": f"base{i}-control-{tier}-{list_num}", "split": "speech", "ground_truth": "speech",
                         "terms": ["Acme"], "tier": tier, "list_id": str(list_num), "probe_set": "unrelated-list"}, True)
            actual[f"base0-control-{tier}-0"]["text"] = "speech Acme Acme"
            result = quality(cases, baseline, actual, counts, partial=True)
            self.assertEqual(result["accepted"], tier == 100)
            if tier == 30:
                self.assertEqual(result["failures"][0]["gate"], "control-insertions")
            self.assertEqual(result["tiers"], {})

    def test_partial_wer_waits_for_arm_completion(self):
        from qwen3_asr_quant_gate import quality
        cases = [{"id": id, "split": "speech", "ground_truth": "alpha", "terms": []} for id in ("a", "b")]
        baseline = {id: {"text": "alpha", "stop_reason": "stop"} for id in ("a", "b")}
        counts = {id: {"words": 100, "baseline_edits": 0, "actual_edits": 10} for id in ("a", "b")}
        partial = quality(cases, baseline, {"a": baseline["a"]}, {"a": counts["a"]}, partial=True)
        self.assertEqual(partial["failures"], [])
        self.assertEqual(partial["measurements"], {})
        self.assertEqual(quality(cases, baseline, baseline, counts, partial=True)["failures"][0]["gate"], "wer-delta")

    def test_coverage_waits_for_set_but_stops_before_other_set_completes(self):
        cases = [{"id": id, "split": "speech" if id != "c" else "other", "ground_truth": "alpha", "terms": []}
                 for id in ("a", "b", "c")]
        baseline = {"text": "alpha", "stop_reason": "stop"}
        def record(id):
            teacher = {**baseline, "prompt_ids": [1], "span": {"start": 0, "len": 1},
                       "generated_ids": [151645], "top_two_margins": [.01 if id == "a" else 1.]}
            profile = {"generated_ids": [151645], "forced_ids": [], "mode": "f16-boundaries",
                       "prompt_ids": [1], "span": teacher["span"]}
            return {"teacher": teacher, "profile": profile}
        with tempfile.TemporaryDirectory() as folder:
            for case in cases:
                (Path(folder) / (case["id"] + ".json")).write_text(json.dumps(baseline))
            with patch.object(kit, "FP32", Path(folder)), patch("qwen3_asr_quant_gate.wer_counts",
                    side_effect=lambda _b, rows, *_: {c["id"]: {"words": 100, "baseline_edits": 0, "actual_edits": 0}
                                                     for c in rows}):
                check = kit.public_checker(cases, Path("unused"))
                self.assertEqual(check({"a": record("a")})["failures"], [])
                result = check({"a": record("a"), "b": record("b")})
                self.assertEqual(result["failures"][0]["gate"], "nonvacuous-profile")
                self.assertEqual(result["complete_fixture_sets"], ["speech"])

    def test_progress_reports_recent_rate_cost_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            plan = synthetic_plan(state)
            plan.update(rental_start=100, first_started_at=100)
            kit.atomic_json(state / "plan.json", plan)
            kit.atomic_json(state / "status.json", {"started_at": 100, "updated_at": 3700,
                "status": "stopped", "completed": [{"id": "a", "weight": 30, "finished_at": 3690}]})
            kit.fail(state, "gate-failed", {"gate": "test"})
            result = kit.progress(state, now=3700)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["total"], 3)
            self.assertEqual(result["cases_per_hour_last_hour"], 1)
            self.assertEqual(result["projected_hours_remaining"], 1)
            self.assertEqual(result["projected_total_cost"], .9)
            self.assertEqual(result["failure"]["kind"], "gate-failed")


class Collection(unittest.TestCase):
    def collect_fixture(self, state, output):
        plan = synthetic_plan(state)
        for entry in plan["entries"]:
            kit.publish_case(state, plan, entry,
                             {"teacher": {"id": entry["id"]}, "profile": {"id": entry["id"]}})
        cases = [{"id": entry["id"]} for entry in plan["entries"]]
        with patch.object(kit, "identity", return_value=(plan["identity"], {})), \
                patch.object(kit, "corpus_cases", return_value=cases), \
                patch.object(kit, "plan_entries", return_value=plan["entries"]), \
                patch.object(kit, "public_checker", return_value=lambda _r: {"failures": []}):
            result = kit.collect(state, output, Path("unused"))
        return plan, cases, result

    def test_collect_json_only_bundle_and_verify_hashes(self):
        import tarfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, cases, result = self.collect_fixture(root / "run", root / "bundle.tar.gz")
            self.assertEqual(result["cases"], 3)
            self.assertFalse(result["engine_slices_allowed"])
            with tarfile.open(root / "bundle.tar.gz") as archive:
                self.assertTrue(all(m.name.endswith(".json") for m in archive.getmembers()))
                archive.extractall(root / "collected", filter="data")
            generated = root / "collected/generated"
            with patch.object(kit, "identity", return_value=(plan["identity"], {})), \
                    patch.object(kit, "corpus_cases", return_value=cases):
                self.assertEqual(kit.verify_generated(generated)["verified_cases"], 3)
                (generated / "teacher/a.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "bundle digest mismatch"):
                    kit.verify_generated(generated)

    def test_collect_refuses_transport_size_over_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(kit, "MAX_BUNDLE", 1):
                with self.assertRaisesRegex(ValueError, "transport bound"):
                    self.collect_fixture(root / "run", root / "bundle.tar.gz")
            self.assertFalse((root / "bundle.tar.gz").exists())
            self.assertFalse((root / "bundle.tar.partial").exists())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fake-worker":
        if "--supervisor-pid" in sys.argv:
            kit.parent_death_guard(int(sys.argv[sys.argv.index("--supervisor-pid") + 1]))
        state = Path(sys.argv[sys.argv.index("--state") + 1])
        shard = int(sys.argv[sys.argv.index("--shard") + 1])
        raise SystemExit(kit.worker_loop(state, shard, lambda: fake_factory(state)))
    unittest.main()
