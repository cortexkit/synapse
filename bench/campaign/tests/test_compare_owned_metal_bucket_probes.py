import copy
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "compare-owned-metal-bucket-probes.py"
SPEC = importlib.util.spec_from_file_location("bucket_compare", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def sample_run():
    vector = [0.25, -0.5]
    digest = COMPARE.vector_sha256(vector)
    metadata = {
        "probe_version": "representative-text-v1",
        "family": "gte-modernbert",
        "dtype": "f16",
        "model_config_sha256": "config",
        "model_weights_sha256": "weights",
        "tokenizer_sha256": "tokenizer",
        "max_tokens": 8192,
        "attention_units": 67_108_864,
        "execution": "explicit",
        "bucket_policy_version": 1,
        "engine_identity": {},
        "first_use_runs_per_case": 1,
        "measured_repeats_per_case": 2,
        "case_order": ["case"],
    }
    data = {
        "metadata": metadata,
        "cold_load_s": 1.0,
        "cases": [
            {
                "name": "case",
                "rows": [
                    {
                        "id": "row",
                        "seed_text": "text",
                        "target_tokens": 2,
                        "input_ids": [10, 11],
                    }
                ],
                "real_tokens": 2,
                "first_use_engine_wall_s": 0.5,
                "warm_engine_wall_s": [1.0, 3.0],
                "repeat_vector_sha256": [[digest], [digest]],
                "vectors": [vector],
            }
        ],
    }
    calls = [
        {"shapes": [{"items": 1, "max_tokens": 2, "batch": 1, "seq": 64}], "engine_wall_s": wall}
        for wall in (0.5, 1.0, 3.0)
    ]
    return data, calls, 3


class ComparisonValidationTests(unittest.TestCase):
    def test_rejects_mismatched_metadata(self):
        baseline = sample_run()[0]
        candidate = copy.deepcopy(baseline)
        candidate["metadata"]["tokenizer_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "metadata mismatch: tokenizer_sha256"):
            COMPARE.validate_compatible(baseline, candidate)

    def test_rejects_input_length_mismatch(self):
        data, calls, _ = sample_run()
        data["cases"][0]["rows"][0]["target_tokens"] = 3
        with self.assertRaisesRegex(ValueError, "input length mismatch"):
            COMPARE.validate_run(data, calls, "run")

    def test_rejects_same_length_but_different_input_ids(self):
        baseline = sample_run()[0]
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["rows"][0]["input_ids"][1] = 99
        with self.assertRaisesRegex(ValueError, "input rows or token IDs mismatch"):
            COMPARE.validate_compatible(baseline, candidate)

    def test_rejects_vector_row_count_mismatch(self):
        data, calls, _ = sample_run()
        data["cases"][0]["vectors"] = []
        with self.assertRaisesRegex(ValueError, "vector row count mismatch"):
            COMPARE.validate_run(data, calls, "run")

    def test_rejects_vector_dimension_mismatch_without_zip_truncation(self):
        baseline = sample_run()[0]
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["vectors"][0].append(0.75)
        with self.assertRaisesRegex(ValueError, "vector dimension mismatch"):
            COMPARE.validate_compatible(baseline, candidate)
        with self.assertRaisesRegex(ValueError, "vector row count mismatch"):
            COMPARE.compare_vectors([[1.0]], [[1.0], [2.0]])

    def test_rejects_repeat_digest_mismatch(self):
        data, calls, _ = sample_run()
        data["cases"][0]["repeat_vector_sha256"][1][0] = "wrong"
        with self.assertRaisesRegex(ValueError, "repeat 1 vector digest mismatch"):
            COMPARE.validate_run(data, calls, "run")

    def test_aggregate_throughput_uses_total_tokens_over_total_time(self):
        baseline = sample_run()
        candidate = copy.deepcopy(baseline)
        candidate[0]["metadata"]["bucket_policy_version"] = 2
        output = COMPARE.build_comparison(baseline, candidate)
        self.assertEqual(output["cases"][0]["baseline"]["aggregate_real_tokens_per_s"], 1.0)
        self.assertEqual(output["aggregate"]["baseline_real_tokens_per_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
