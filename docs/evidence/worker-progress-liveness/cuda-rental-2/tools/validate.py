"""Check NVIDIA certification records, timings, progress summaries and cleanup.

Expected model fingerprints come from a separate catalog snapshot, rather
than from the certification records being checked.
"""
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "be6cd118a40cc8cc5e999fad9e1e880e43dcd644"
BASELINE = "5f6d7419fc42a5ebf6ed1b8eab2115b6cc84fe5a"
MODELS = {"gte-modernbert-base", "gte-reranker-modernbert-base", "qwen3-embedding-0.6b", "qwen3-reranker-0.6b"}


class RentalEvidence(unittest.TestCase):
    def setUp(self):
        self.records = [json.loads(p.read_text()) for p in (ROOT / "records").rglob("*.json")]
        self.pins = json.loads((ROOT / "pins.json").read_text())["pins"]
        self.summary = json.loads((ROOT / "summary.json").read_text())

    def test_cell_coverage(self):
        expected = {(commit, row, model) for commit, rows in [
            (CANDIDATE, ["cuda-linux-nvidia", "vulkan-linux-nvidia"]),
            (BASELINE, ["cuda-linux-nvidia"])] for row in rows for model in MODELS}
        actual = {(r["source_commit"], r["row_id"], r["model"]) for r in self.records}
        self.assertEqual(len(self.records), 12)
        self.assertEqual(actual, expected)

    def test_fingerprint_pins(self):
        for record in self.records:
            backend = record["row_id"].split("-", 1)[0]
            expected = self.pins[record["model"]][backend]
            self.assertEqual(record["fingerprint"], expected, (record["row_id"], record["model"]))
            self.assertEqual(record["parity"]["fingerprint"], expected)

    def test_parity_and_admission(self):
        for record in self.records:
            self.assertEqual(record["status"], "passed")
            parity = record["parity"]
            self.assertEqual(parity["failures"], [])
            self.assertTrue(all(parity["gates"].values()))
            self.assertEqual(parity["completed_fixtures"], 17 if record["operation"] == "embed" else 125)
            self.assertEqual(parity["completed_fixtures"], parity["expected_fixtures"])
            self.assertEqual(record["admission"], {
                "tokens_8192": {"outcome": "processed", "worker_requests": 1, "truncated": False, "diverted": False},
                "tokens_8193": {"outcome": "sequence_too_long", "worker_requests": 0, "truncated": False, "diverted": False}})

    def test_hardware_identity(self):
        for record in self.records:
            self.assertEqual(record["machine"]["instance_id"], "55712356")
            backend = record["row_id"].split("-", 1)[0]
            self.assertEqual(record["machine"]["driver"], "595.84" if backend == "cuda" else "NVIDIA 595.84")
            if backend == "cuda":
                self.assertEqual(record["machine"]["driver_api"], 13020)
            self.assertEqual(record["machine"]["gpu"]["name"], "NVIDIA GeForce RTX 5090")
        for backend in ["cuda", "vulkan"]:
            uuids = {r["machine"]["gpu"]["uuid_sha256"] for r in self.records if r["row_id"] == f"{backend}-linux-nvidia"}
            self.assertEqual(len(uuids), 1)

    def test_timing_samples(self):
        import statistics
        self.assertEqual(len(self.summary["timings"]), 24)
        for shape in self.summary["timings"]:
            self.assertEqual(shape["successful_samples"], 6)
            self.assertEqual(shape["samples_attempted"], 6)
            self.assertEqual(len(shape["warm_samples_ms"]), 5)
            self.assertEqual(statistics.median(shape["warm_samples_ms"]), shape["median_warm_ms"])
            self.assertGreater(shape["first_shape_ms"], 0)
            self.assertIn(shape["tokens"], [512, 8192])

    def test_progress_summary(self):
        with (ROOT / "progress.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 316)
        self.assertEqual(len(self.summary["progress"]), 16)
        for group in self.summary["progress"]:
            selected = [r for r in rows if all(r[k] == group[k] for k in ["source", "backend", "model"])]
            gaps = [float(r["max_between_events_ms"]) for r in selected if r["max_between_events_ms"]]
            self.assertEqual(len(selected), group["exchanges"])
            self.assertEqual(sum(int(r["progress_units"]) for r in selected), group["completed_work_events"])
            self.assertEqual(max(gaps), group["max_between_events_ms"])

    def test_spend_and_cleanup(self):
        spend = json.loads((ROOT / "spend.json").read_text())
        self.assertEqual(spend["instance_ids"], [55712356])
        self.assertEqual(spend["final_instances"], [])
        self.assertEqual(spend["final_volumes"], [])
        self.assertAlmostEqual(spend["account_credit_before"] - spend["account_credit_final"], spend["account_debit"], places=12)
        self.assertGreater(spend["account_debit"], 0)
        self.assertLess(spend["account_debit"], 3.00)


if __name__ == "__main__":
    unittest.main(verbosity=2)
