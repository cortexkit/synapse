"""Small synthetic text fixtures only: safe for portable test runners."""

import unittest
from pathlib import Path

from safety import TARGET, limits_exceeded, private_path
from score import aggregate, alignment, norm, score_clip, spans


def fixture(reference="we met luma today", terms=None):
    terms = ["Luma"] if terms is None else terms
    return {"id": "example", "dataset": "synthetic-unit-test", "reference": reference,
            "expected": terms, "vocabulary": terms, "lists": {"none": [], "global": terms}}


class ScoringTests(unittest.TestCase):
    def test_colli_normalization(self):
        self.assertEqual(norm("OVH-1, AFT's 4/28/20!"), "ovh one aft s four twenty eight twenty")
        self.assertEqual(norm("Café & pi_runner"), "caf pi runner")

    def test_alignment_edits_and_empty_sequences(self):
        self.assertEqual(alignment([], ["x"]), [(None, 0)])
        self.assertEqual(alignment(["x"], []), [(0, None)])
        self.assertEqual(alignment(["a", "b"], ["a", "c"]), [(0, 0), (1, 1)])

    def test_aliases_and_longest_nonoverlapping_spans(self):
        self.assertEqual(spans(["ovh", "one", "luma", "labs"],
                               ["OVH 1|OVH one", "Luma", "Luma Labs"]),
                         [(0, 2, "OVH 1|OVH one"), (2, 4, "Luma Labs")])

    def test_inserted_name_is_not_aligned_recall(self):
        result = score_clip(fixture(), "Luma we met today")
        self.assertEqual(result["colli_hits"], 1)
        self.assertEqual(result["term_hits"], 0)
        self.assertEqual(result["false_insertions"], 1)

    def test_aligned_multiword_term_and_duplicate_false_insertion(self):
        result = score_clip(fixture("we met luma labs today", ["Luma Labs"]),
                            "we met luma labs today luma labs")
        self.assertEqual(result["term_occurrences"], 1)
        self.assertEqual(result["term_hits"], 1)
        self.assertEqual(result["term_predictions"], 2)
        self.assertEqual(result["false_insertions"], 1)

    def test_nonlist_error_mask_separates_name_insertions(self):
        result = score_clip(fixture(), "we met luna today luma")
        self.assertEqual(result["word_edits"], 2)
        self.assertEqual(result["nonlist_edits"], 0)
        self.assertEqual(result["nonlist_words"], 3)
        result = score_clip(fixture(), "we just met luma today")
        self.assertEqual(result["nonlist_edits"], 1)

    def test_control_counts_occurrences_not_unique_names(self):
        result = score_clip(fixture("we met today"), "luma we met luma today")
        self.assertEqual(result["term_occurrences"], 0)
        self.assertEqual(result["false_insertions"], 2)

    def test_aggregate_is_micro_averaged_and_contains_no_text(self):
        a, b = fixture("one two"), fixture("one two three four five six")
        a["id"], b["id"] = "a", "b"
        predictions = [{"id": key, "system": "test", "list": "none", "seconds": 1.0,
                        "text": text} for key, text in [("a", "one"), ("b", b["reference"])]]
        result = aggregate([a, b], predictions)[0]
        self.assertEqual(result["wer"], 1 / 8)
        self.assertIsNone(result["precision"])
        self.assertNotIn("reference", result)
        self.assertNotIn("text", result)
        self.assertNotIn("id", result)

    def test_incomplete_duplicate_and_bad_latency_fail_closed(self):
        prediction = {"id": "example", "system": "test", "list": "none", "seconds": 1., "text": "hello"}
        with self.assertRaises(ValueError):
            aggregate([fixture()], [])
        with self.assertRaises(ValueError):
            aggregate([fixture(), {**fixture(), "id": "missing"}], [prediction])
        with self.assertRaises(ValueError):
            aggregate([fixture()], [prediction, prediction])
        with self.assertRaises(ValueError):
            aggregate([fixture()], [{**prediction, "seconds": float("nan")}])
        with self.assertRaises(ValueError):
            aggregate([fixture()], [{**prediction, "text": None}])

    def test_private_outputs_cannot_escape_target(self):
        self.assertEqual(private_path(TARGET / "tests" / "safe.jsonl"), TARGET / "tests" / "safe.jsonl")
        with self.assertRaises(ValueError):
            private_path(Path(__file__).parent / "transcripts.jsonl")

    def test_watchdog_gates_growth_not_standing_swap(self):
        first = {"rss_bytes": 2**30, "swap_bytes": 12 * 2**30, "pageouts": 100000, "free_percent": 60}
        self.assertFalse(limits_exceeded(first, [first, {**first, "pageouts": 100066}]))
        self.assertFalse(limits_exceeded(first, [first, {**first, "pageouts": 125000}]))
        for field, value in [("rss_bytes", 20 * 2**30 + 1),
                             ("swap_bytes", 13 * 2**30 + 1),
                             ("free_percent", 24), ("pageouts", 125001)]:
            with self.subTest(field=field):
                self.assertTrue(limits_exceeded(first, [first, {**first, field: value}]))

    def test_pageout_window_does_not_accumulate_entire_run(self):
        first = {"rss_bytes": 2**30, "swap_bytes": 12 * 2**30, "pageouts": 100000, "free_percent": 60}
        self.assertFalse(limits_exceeded(first, [{**first, "pageouts": 200000},
                                                 {**first, "pageouts": 201000}]))


if __name__ == "__main__":
    unittest.main()
