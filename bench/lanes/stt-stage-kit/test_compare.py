import unittest
from typing import Any

import compare


def record(engine, instrumented, clip, segment=0, count=1, total=10) -> dict[str, Any]:
    return dict(schema="matched-stages-v1", run_id="same-run", engine=engine, instrumented=instrumented,
                clip_id=clip, invocation_id="primary", population="corpus", warm=True, observation="on",
                segment_id=segment, segment_count=count, rung=128, model_total_ms=total,
                encoder_ms=3, predictor_ms=2, joint_ms=1, readback_pack_ms=1,
                greedy_bookkeeping_ms=1, remaining_model_ms=2, valid_frames=100,
                padded_frames=128, encoder_calls=1, decode_steps=8, predictor_calls=5, accepted_tokens=4)


def population():
    return [record(e, on, c) for e in ("synapse", "fluidaudio") for on in (False, True) for c in ("a", "b")]


class CompareTests(unittest.TestCase):
    def test_join_uses_ids_and_aggregates_segments_before_statistics(self):
        rows = population()
        rows.remove(next(v for v in rows if v["engine"] == "synapse" and v["instrumented"] and v["clip_id"] == "b"))
        rows += [record("synapse", True, "b", i, 2) for i in range(2)]
        rows.reverse()
        joined = compare.aggregate(rows, ["a", "b"])
        self.assertEqual(joined["synapse", True]["b"]["model_total_ms"], 20)
        self.assertEqual(joined["synapse", True]["b"]["predictor_calls"], 10)
        self.assertEqual(joined["synapse", True]["b"]["joint_steps"], 16)
        summary = compare.summarize(joined)
        self.assertEqual(summary["synapse"]["stages"]["model_total_ms"], {"median": 15, "p95": 20, "mean": 15})
        self.assertEqual(summary["synapse"]["instrumentation_overhead_percent"], 50)

    def test_repeats_noncorpus_and_cold_calls_do_not_enter_population(self):
        rows = population()
        for change in ({"invocation_id": "repeat-1"}, {"population": "first-load"}, {"warm": False}):
            extra = record("synapse", True, "not-in-corpus", total=900)
            extra.update(change)
            rows.append(extra)
        summary = compare.summarize(compare.aggregate(rows, ["b", "a"]))
        self.assertEqual(summary["synapse"]["stages"]["model_total_ms"]["mean"], 10)

    def test_duplicate_or_incomplete_segments_are_rejected(self):
        rows = population()
        with self.assertRaisesRegex(ValueError, "duplicate primary"):
            compare.aggregate(rows + [rows[0]], ["a", "b"])
        rows[0]["segment_count"] = 2
        with self.assertRaisesRegex(ValueError, "incomplete segments"):
            compare.aggregate(rows, ["a", "b"])

    def test_missing_clip_or_mixed_run_is_not_a_partial_comparison(self):
        with self.assertRaisesRegex(ValueError, "unmatched clips"):
            compare.aggregate(population()[:-1], ["a", "b"])
        rows = population()
        rows[0]["run_id"] = "other"
        with self.assertRaisesRegex(ValueError, "common run ID"):
            compare.aggregate(rows, ["a", "b"])

    def test_residual_is_independent_and_tolerance_is_checked_per_segment(self):
        rows = population()
        target = next(v for v in rows if v["engine"] == "synapse" and v["instrumented"])
        target["model_total_ms"] = 10.25
        joined = compare.aggregate(rows, ["a", "b"], .3)
        self.assertEqual(joined["synapse", True]["a"]["residual_ms"], .25)
        with self.assertRaisesRegex(ValueError, "residual outside tolerance"):
            compare.aggregate(rows, ["a", "b"], .2)
        rows.remove(target)
        rows += [record("synapse", True, "a", 0, 2, 12), record("synapse", True, "a", 1, 2, 8)]
        with self.assertRaisesRegex(ValueError, "residual outside tolerance"):
            compare.aggregate(rows, ["a", "b"])

    def test_invalid_times_and_missing_stages_are_rejected(self):
        for value in (float("nan"), -1, True):
            rows = population()
            rows[0]["model_total_ms"] = value
            with self.assertRaisesRegex(ValueError, "invalid model_total_ms"):
                compare.aggregate(rows, ["a", "b"])
        rows = population()
        del rows[4]["encoder_ms"]
        # Index 4 is FluidAudio baseline, whose stage fields are not required.
        del rows[6]["encoder_ms"]
        with self.assertRaisesRegex(ValueError, "missing FluidAudio"):
            compare.aggregate(rows, ["a", "b"])

    def test_preaggregated_fluid_windows_and_derived_other_are_labelled(self):
        rows = population()
        for row in rows:
            if row["engine"] == "fluidaudio":
                row.update(aggregated_segments=True, segment_count=3)
                for stage in ("readback_pack_ms", "greedy_bookkeeping_ms", "remaining_model_ms"):
                    del row[stage]
        summary = compare.summarize(compare.aggregate(rows, ["a", "b"]))
        self.assertEqual(summary["fluidaudio"]["stages"]["other_unsplit_ms"]["mean"], 4)
        self.assertNotIn("residual_ms", summary["fluidaudio"]["stages"])
        self.assertIn("unmeasured", compare.table(summary))


if __name__ == "__main__":
    unittest.main()
