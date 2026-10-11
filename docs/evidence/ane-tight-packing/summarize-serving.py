#!/usr/bin/env python3
"""Reduce a completed serving report to hashes, counters and independently computed gates."""
import argparse
import hashlib
import json
import math
import statistics
import unittest
from pathlib import Path


PIN = "5f7ac7cdbf63e71f04405b2130b2bde8eb7b307e5d929419bcd747b129806a6b"
HASH_KEYS = (
    "input_sha256", "meta_sha256", "checkpoint_sha256", "tokenizer_sha256",
    "module_sha256", "worker_sha256", "driver_sha256",
)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def latency(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("missing or invalid latency samples")
    return {"count": len(values), "p50_ms": percentile(values, .50), "p99_ms": percentile(values, .99)}


def digest(value, length=64):
    if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid digest")
    return value


def summarize(report, allow_incomplete=False):
    if (report["schema"], report["rows"], report["outer_calls"], report["calls_in_flight"], report["engine_quantum_rows"], report["fingerprint"]) != (1, 6341, 127, 2, 8, PIN):
        raise ValueError("unexpected workload, concurrency, quantum or identity")
    arms = report["arms"]
    complete = len(arms) >= 6 and len(arms) % 2 == 0
    if (not complete and not allow_incomplete) or len(arms) < 2:
        raise ValueError("incomplete alternating run")
    expected = [(r, packing) for r in range((len(arms) + 1) // 2) for packing in (False, True)][:len(arms)]
    if [(a["repetition"], a["packing"]) for a in arms] != expected:
        raise ValueError("arms are not matched and alternating")
    compact = []
    for arm in arms:
        if len(arm["calls_ms"]) != 127 or len(arm["query_ms"]) < 1000 or arm["background_calls"] < 2:
            raise ValueError("incomplete export or query/bulk stratum")
        if not math.isfinite(arm["export_seconds"]) or arm["export_seconds"] <= 0:
            raise ValueError("invalid export time")
        call_latency = latency(arm["calls_ms"])
        query_latency = latency(arm["query_ms"])
        for recorded, recomputed in ((arm["call_latency"], call_latency), (arm["query_latency"], query_latency)):
            if recorded["count"] != recomputed["count"] or any(not math.isclose(recorded[k], recomputed[k], abs_tol=1e-6) for k in ("p50_ms", "p99_ms")):
                raise ValueError("recorded latency disagrees with raw samples")
        before = arm["observations_before"]["wire"]
        after = arm["observations_after_export"]["wire"]
        packed_calls = after.get("EMBED_PACKED", 0) - before.get("EMBED_PACKED", 0)
        if packed_calls < 0:
            raise ValueError("worker counter regressed")
        compile_ms = arm["packed_compile_request_ms"]
        if any(not math.isfinite(v) or v <= 0 for v in compile_ms):
            raise ValueError("invalid preparation time")
        compact.append({
            "repetition": arm["repetition"], "packing": arm["packing"],
            "export_seconds": arm["export_seconds"], "call_latency": call_latency,
            "query_latency": query_latency, "identical_rows": arm["identical_rows"],
            "mismatched_rows": arm["mismatched_rows"], "vectors_sha256": digest(arm["vectors_sha256"]),
            "identical_queries": arm["identical_queries"], "background_calls": arm["background_calls"],
            "background_identical_rows": arm["background_identical_rows"],
            "background_mismatched_rows": arm["background_mismatched_rows"],
            "packed_export_worker_calls": packed_calls,
            "packed_total_worker_calls": after.get("EMBED_PACKED", 0),
            "single_export_worker_calls": after.get("EMBED_BATCH", 0) - before.get("EMBED_BATCH", 0),
            "max_engine_rows": after.get("max_embed_rows", 0),
            "packed_compile_request_ms": compile_ms,
        })
    serial = [a for a in arms if not a["packing"]]
    packed = [a for a in arms if a["packing"]]
    serial_seconds = statistics.median(a["export_seconds"] for a in serial)
    packed_seconds = statistics.median(a["export_seconds"] for a in packed)
    baseline = latency([v for a in serial for v in a["query_ms"]])
    concurrent = latency([v for a in packed for v in a["query_ms"]])
    p50_limit = baseline["p50_ms"] + max(baseline["p50_ms"] * .10, 5)
    p99_limit = baseline["p99_ms"] + max(baseline["p99_ms"] * .10, 5)
    identity = all(a["identical_rows"] == 6341 and not a["mismatched_rows"] and a["identical_queries"] == len(a["query_ms"]) and a["background_mismatched_rows"] == 0 for a in arms)
    identity = identity and len({a["vectors_sha256"] for a in compact}) == 1
    dispatch = all(0 < a["max_engine_rows"] <= 8 and (
        a["packed_export_worker_calls"] > 0 and bool(a["packed_compile_request_ms"]) if a["packing"]
        else a["packed_total_worker_calls"] == 0 and not a["packed_compile_request_ms"]
    ) for a in compact)
    export_pass = packed_seconds <= .80 * serial_seconds
    query_pass = concurrent["p50_ms"] <= p50_limit and concurrent["p99_ms"] <= p99_limit
    gates = {
        "serial_median_seconds": serial_seconds, "packed_median_seconds": packed_seconds,
        "packed_over_serial": packed_seconds / serial_seconds,
        "serial_queries": baseline, "packed_queries": concurrent,
        "query_p50_limit_ms": p50_limit, "query_p99_limit_ms": p99_limit,
        "all_vectors_bit_identical": identity, "packed_dispatch_and_off_switch": dispatch,
        "export_time_pass": export_pass, "query_latency_pass": query_pass,
        "passed": identity and dispatch and export_pass and query_pass and complete,
    }
    for key, value in gates.items():
        if isinstance(value, (bool, dict)):
            if report["gates"][key] != value:
                raise ValueError(f"driver and reducer disagree on {key}")
        elif not math.isclose(report["gates"][key], value, abs_tol=1e-6):
            raise ValueError(f"driver and reducer disagree on {key}")
    chip = report["chip"]
    if not isinstance(chip, str) or not chip.startswith("Apple M") or any(not (c.isascii() and (c.isalnum() or c == " ")) for c in chip):
        raise ValueError("unsupported chip identity")
    if report["query_row"] != 0:
        raise ValueError("query row changed")
    return {
        "schema": 1, "source_commit": digest(report["source_commit"], 40),
        "chip": chip, "fingerprint": PIN,
        **{key: digest(report[key]) for key in HASH_KEYS},
        "rows": 6341, "outer_calls": 127, "calls_in_flight": 2, "engine_quantum_rows": 8,
        "query_row": report["query_row"], "arms": compact, "gates": gates,
        "complete": complete, "completed_serial_arms": len(serial), "completed_packed_arms": len(packed),
        "packed_preparation_first_request_median_ms": statistics.median(a["packed_compile_request_ms"][0] for a in packed) if all(a["packed_compile_request_ms"] for a in packed) else None,
    }


def complete_fixture():
    arms = []
    for repetition in range(3):
        for packing in (False, True):
            before = {"EMBED_PACKED": 3, "max_embed_rows": 8} if packing else {"EMBED_BATCH": 1, "max_embed_rows": 8}
            after = {"EMBED_PACKED": 103, "max_embed_rows": 8} if packing else {"EMBED_BATCH": 101, "max_embed_rows": 8}
            arms.append({
                "repetition": repetition, "packing": packing,
                "export_seconds": 80 if packing else 100,
                "calls_ms": [100] * 127,
                "call_latency": {"count": 127, "p50_ms": 100, "p99_ms": 100},
                "query_ms": [15 if packing else 10] * 1000,
                "query_latency": {"count": 1000, "p50_ms": 15 if packing else 10, "p99_ms": 15 if packing else 10},
                "background_calls": 2, "identical_rows": 6341, "mismatched_rows": [],
                "identical_queries": 1000, "background_identical_rows": 128,
                "background_mismatched_rows": 0, "vectors_sha256": "d" * 64,
                "packed_compile_request_ms": [1000] if packing else [],
                "observations_before": {"wire": before}, "observations_after_export": {"wire": after},
            })
    return {
        "schema": 1, "rows": 6341, "outer_calls": 127, "calls_in_flight": 2,
        "engine_quantum_rows": 8, "fingerprint": PIN, "chip": "Apple M4", "query_row": 0,
        "source_commit": "b" * 40, **{k: "c" * 64 for k in HASH_KEYS}, "arms": arms,
        "gates": {"serial_median_seconds": 100, "packed_median_seconds": 80, "packed_over_serial": .8,
                  "serial_queries": {"count": 3000, "p50_ms": 10, "p99_ms": 10},
                  "packed_queries": {"count": 3000, "p50_ms": 15, "p99_ms": 15},
                  "query_p50_limit_ms": 15, "query_p99_limit_ms": 15,
                  "all_vectors_bit_identical": True, "packed_dispatch_and_off_switch": True,
                  "export_time_pass": True, "query_latency_pass": True, "passed": True},
    }


class Tests(unittest.TestCase):
    def test_complete_run_and_failed_dispatch_are_computed_from_raw_counters(self):
        report = complete_fixture()
        self.assertTrue(summarize(report)["gates"]["passed"])
        report["arms"][1]["observations_after_export"] = report["arms"][1]["observations_before"]
        report["gates"]["packed_dispatch_and_off_switch"] = False
        report["gates"]["passed"] = False
        self.assertFalse(summarize(report)["gates"]["packed_dispatch_and_off_switch"])

    def test_driver_gate_claim_cannot_override_recomputed_export_time(self):
        report = complete_fixture()
        report["arms"][3]["export_seconds"] = 81
        report["arms"][5]["export_seconds"] = 81
        with self.assertRaisesRegex(ValueError, "driver and reducer disagree"):
            summarize(report)

    def test_explicit_partial_report_never_clears_acceptance(self):
        report = complete_fixture()
        report["arms"] = report["arms"][:5]
        report["gates"]["packed_queries"]["count"] = 2000
        report["gates"]["passed"] = False
        summary = summarize(report, allow_incomplete=True)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["gates"]["passed"])
        self.assertEqual(summary["completed_packed_arms"], 2)

    def test_nearest_rank_p99_is_not_max(self):
        self.assertEqual(percentile(list(range(1, 101)), .99), 99)
        self.assertEqual(latency([4, 1, 2, 3])["p50_ms"], 2)

    def test_invalid_samples_and_hashes_refuse(self):
        for values in ([], [0], [float("nan")], [-1]):
            with self.assertRaises(ValueError):
                latency(values)
        with self.assertRaises(ValueError):
            digest("/private/machine/path")

    def test_incomplete_run_cannot_be_summarized(self):
        report = {"schema": 1, "rows": 6341, "outer_calls": 127, "calls_in_flight": 2, "engine_quantum_rows": 8, "fingerprint": PIN, "arms": []}
        with self.assertRaisesRegex(ValueError, "incomplete alternating run"):
            summarize(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="retain partial evidence, explicitly marked not passed")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], exit=True)
    if not args.raw:
        parser.error("a completed raw report is required")
    data = args.raw.read_bytes()
    summary = summarize(json.loads(data), args.allow_incomplete)
    summary["raw_report_sha256"] = hashlib.sha256(data).hexdigest()
    summary["raw_report_bytes"] = len(data)
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")
