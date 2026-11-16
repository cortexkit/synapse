"""Validate committed reference outputs and receipts without the external tensor cache."""

import hashlib
import json
import math
from pathlib import Path
import unittest

from qwen3_asr_corpus import calibration_trace, expand, token_rate_report
from qwen3_asr_reference import canonical, model_pin, sha256
from run_qwen3_asr_full import check_case, check_termination, validate_policy
from seal_qwen3_asr import GENERATION_PIN, verify

HERE = Path(__file__).resolve().parent
PARITY = HERE.parent
ROOT = PARITY / "fixtures/qwen3-asr-1.7b"


class SealedTeacher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = expand(json.loads((HERE / "qwen3-asr-corpus.json").read_text()))
        cls.reference = json.loads((ROOT / "reference.json").read_text())
        cls.regrade = json.loads(verify(ROOT, cls.reference["offline_regrade"]).read_text())

    def test_registration_and_separate_generation_grading_identities(self):
        key = "qwen3-asr-1.7b.ref-v1.transformers-4.57.6.seed-0"
        index = json.loads((PARITY / "fixtures/index.json").read_text())
        self.assertEqual(index[key]["path"], "fixtures/qwen3-asr-1.7b/reference.json")
        self.assertEqual(index[key]["sha256"], sha256(ROOT / "reference.json"))
        self.assertEqual(index[key]["seed"], 0)
        self.assertEqual(index[key]["transformers_version"], "4.57.6")
        self.assertEqual(self.reference["provenance"]["model"], model_pin())
        pin = json.loads(GENERATION_PIN.read_text())
        for field in ("source_sha256", "corpus_sha256", "reference_rules_sha256"):
            self.assertEqual(self.reference["provenance"][field], pin[field])
        self.assertEqual(self.regrade["generation_revision"], pin["revision"])
        self.assertEqual(self.regrade["grading"]["revision"], "21ffed30e9c7c43e51da727b5a7e738446a6a65c")
        self.assertIsNone(self.reference["source_run_checksums_sha256"])
        self.assertEqual(self.regrade["validated_processes"], 46)
        self.assertEqual(self.regrade["validated_cases"], 6560)
        self.assertEqual(self.regrade["original_failure"]["type"], "ValueError")
        self.assertIn("earnings22-0024-tier-100", self.regrade["original_failure"]["error"])
        self.assertIn("FAILED.json", self.regrade["source_bundle"])
        for absent in ("SHA256SUMS", "run-complete.json", "calibration.json"):
            self.assertNotIn(absent, self.regrade["source_bundle"])
        validate_policy(self.reference["generation_policy"])

    def test_all_6560_trace_receipts_preserve_the_original_venue_payloads(self):
        cases = {case["id"]: case for case in self.corpus["cases"]}
        self.assertEqual(len(cases), 6560)
        self.assertEqual(set(self.reference["cases"]), set(cases))
        self.assertEqual({p.stem for p in (ROOT / "traces").glob("*.json")}, set(cases))
        lineage = hashlib.sha256(canonical(self.reference["provenance"])).hexdigest()
        policy = hashlib.sha256(canonical(self.reference["generation_policy"])).hexdigest()
        tensors = {}
        for id, case in cases.items():
            receipt = self.reference["cases"][id]
            path = verify(ROOT, receipt)
            row = json.loads(path.read_text())
            check_case(row, case)
            self.assertEqual(row["run_provenance_sha256"], lineage, id)
            self.assertEqual(row["corpus_sha256"], self.reference["corpus_sha256"], id)
            self.assertEqual(row["reference_rules_sha256"], self.reference["provenance"]["reference_rules_sha256"], id)
            if not case.get("refusal"):
                check_termination(row)
                self.assertEqual(row["generation_policy_sha256"], policy, id)
                self.assertEqual(row["cap"], 64 + math.ceil(25 * case["sample_count"] / 16000), id)
                if row["stop_reason"] == "stop":
                    self.assertEqual(row.pop("sealed_max_new_tokens"), row["cap"], id)
                    self.assertEqual(row.pop("sealed_tokens_per_second"), 25, id)
                else:
                    self.assertNotIn("sealed_max_new_tokens", row, id)
                for tensor in (row["mel"], row["encoder"]):
                    tensors[tensor["cache_path"]] = tensor
            self.assertEqual(hashlib.sha256(canonical(row)).hexdigest(), receipt["source_trace_sha256"], id)
            self.assertEqual(receipt["source_trace_sha256"], self.regrade["source_bundle"]["traces/" + id + ".json"]["sha256"], id)
        sealed_tensors = {tensor["cache_path"]: tensor for tensor in self.regrade["external_tensors"]}
        self.assertEqual(tensors, sealed_tensors)
        self.assertEqual(len(tensors), 3022)
        self.assertEqual(sum(t["bytes"] for t in tensors.values()), 2685671936)

    def test_fixed_serving_cap_no_list_calibration_and_exact_cap_census(self):
        rows = []
        for case in self.corpus["cases"]:
            if not case.get("refusal"):
                row = json.loads((ROOT / "traces" / (case["id"] + ".json")).read_text())
                rows.append(calibration_trace(row, case))
        report = token_rate_report(rows)
        self.assertEqual(report, self.reference["token_rate"])
        self.assertEqual(report, self.regrade["calibration"])
        self.assertEqual(report["measured_tokens_per_second"], 24)
        self.assertEqual(report["tokens_per_second"], 25)
        self.assertEqual(report["speech_fixtures"], 1995)
        self.assertEqual(report["complete_cap_checked_fixtures"], 2006)
        self.assertEqual(report["maximum_rate_fixture"], "fleurs-el-test-1102888255749710863")
        self.assertAlmostEqual(report["highest_speech_tokens_per_second"], 15.898617511520737)
        self.assertEqual(report["maximum_name_list_rate_fixture"], "earnings22-0024-tier-100")
        self.assertAlmostEqual(report["highest_name_list_tokens_per_second"], 22.466666666666665)
        self.assertEqual(report["ten_minute_cap"], 15064)
        self.assertEqual(report["kv_positions_without_bias"], 22879)
        self.assertEqual(report["kv_positions_at_2048_bias_tokens"], 24927)
        self.assertEqual(report["capped_fixtures"]["count"], 2)
        self.assertEqual(report["capped_fixtures"]["by_tier"]["300"]["ids"], [
            "earnings22-0106-tier-300", "earnings22-0290-tier-300"])
        self.assertEqual(sum(c["count"] for c in report["capped_fixtures"]["by_split"].values()), 2)


if __name__ == "__main__":
    unittest.main()
