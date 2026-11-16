"""Offline checks for the public corpus, immutable pins and fail-closed seals."""

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from generate_qwen3_asr import pilot_cases
from qwen3_asr_corpus import (SEED, audio_tokens, compact, control_lists, distractors,
                             expand, max_new_tokens, occurs, score_normalize,
                             tier_list, validate_annotations)
from qwen3_asr_reference import artifact, canonical, merged_tensor_index, model_pin, sha256
from seal_qwen3_asr import register_reference, seal, verify

HERE = Path(__file__).resolve().parent


class CorpusRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = expand(json.loads((HERE / "qwen3-asr-corpus.json").read_text()))

    def test_audio_length_helper_chunk_boundaries(self):
        for frames, expected in [(0, 0), (1, 1), (8, 1), (9, 2), (99, 13),
                                 (100, 13), (101, 14), (137, 18), (800, 104), (57000, 7410)]:
            self.assertEqual(audio_tokens(frames), expected)
        with self.assertRaises(ValueError):
            audio_tokens(-1)

    def test_collli_vocabulary_normalization_and_aliases(self):
        self.assertEqual(score_normalize("ACME-1, 28 & 20 4"), "acme one twenty eight twenty four")
        self.assertTrue(occurs("Other|ACME-1", "An ACME one call"))
        self.assertFalse(occurs("Ann", "annual revenue"))

    def test_cut_pad_excludes_reference_terms_and_extends_both_vocabularies(self):
        row = {"id": "row", "global_terms": ["A", "B"], "ground_truth": "Ann works at Acme one",
               "vocabulary": ["A", "B", "local"]}
        pool = ["A", "B", "Ann", "Alias|ACME-1", "Distractor", "Other"]
        arm = tier_list(row, pool, 4)
        self.assertEqual(arm["terms"][:2], ["A", "B"])
        self.assertEqual(set(arm["padding_terms"]), {"Distractor", "Other"})
        self.assertEqual(set(arm["vocabulary"]), {"A", "B", "local", "Distractor", "Other"})
        self.assertEqual(tier_list(row, [], 1)["terms"], ["A"])
        self.assertEqual(tier_list(row, [], 2)["padding_terms"], [])
        with self.assertRaises(ValueError):
            distractors(["Ann"], ["Ann"], 1, "insufficient")

    def test_reuses_all_parakeet_speech_and_keeps_their_audio_hashes(self):
        original = json.loads((HERE / "parakeet-corpus.json").read_text())
        by_id = {c["id"]: c for c in self.corpus["cases"]}
        self.assertEqual(len(original["cases"]), 1200)
        for case in original["cases"]:
            for field in ("audio_sha256", "ground_truth", "sample_count"):
                self.assertEqual(by_id[case["id"]][field], case[field])

    def test_each_c4_tier_has_1200_pairs_and_no_reference_distractors(self):
        for tier in (30, 100, 300):
            controls = [c for c in self.corpus["cases"] if c.get("tier") == tier
                        and c.get("probe_set") == "unrelated-list"]
            self.assertEqual(len(controls), 1200)
            self.assertEqual(len({c["list_id"] for c in controls}), 3)
            for row in controls:
                self.assertEqual(len(row["terms"]), tier)
                self.assertEqual(row["terms"], row["vocabulary"])
                self.assertFalse(any(occurs(t, row["ground_truth"]) for t in row["terms"]))
            arms = [c for c in self.corpus["cases"] if c.get("tier") == tier
                    and c["split"] == "earnings22-test-200"]
            biased = {c["source_row"]: c for c in arms if not c["baseline"]}
            baseline = {c["source_row"]: c for c in arms if c["baseline"]}
            self.assertEqual(len(biased), 200)
            for source_row, row in biased.items():
                self.assertEqual(len(row["terms"]), tier)
                self.assertEqual(row["vocabulary"], baseline[source_row]["vocabulary"])
                self.assertTrue(all(t in row["vocabulary"] for t in row["padding_terms"]))
                self.assertFalse(any(occurs(t, row["ground_truth"]) for t in row["padding_terms"]))

    def test_sealed_probe_counts_and_seeded_manual_review_prefix(self):
        cases = self.corpus["cases"]
        for split in ("near-silence", "noise", "single-name"):
            self.assertEqual(sum(c["split"] == split for c in cases), 50)
        annotations = self.corpus["curation"]
        text = {r["utterance_id"]: r["ground_truth"] for r in annotations["reviews"]}
        candidates = [{"utterance_id": id, "ground_truth": text.get(id, "")}
                      for id in annotations["population_ids"]]
        self.assertEqual(len(candidates), 2620)
        self.assertEqual(len(validate_annotations(annotations, candidates)), 50)
        changed = copy.deepcopy(annotations)
        changed["reviews"][0], changed["reviews"][1] = changed["reviews"][1], changed["reviews"][0]
        with self.assertRaisesRegex(ValueError, "seeded review prefix"):
            validate_annotations(changed, candidates)
        for row in (c for c in cases if c["split"] == "single-name"):
            self.assertEqual(row["terms"][0], row["noun"])
            self.assertFalse(any(occurs(t, row["ground_truth"]) for t in row["terms"][1:]))

    def test_long_form_references_are_whole_utterances_not_repeated_audio(self):
        for row in (c for c in self.corpus["cases"] if c["split"] == "long-form"):
            self.assertIn(row["sample_count"], (120 * 16000, 300 * 16000, 570 * 16000))
            self.assertEqual(row["ground_truth"], " ".join(p["ground_truth"] for p in row["parts"]))
            self.assertEqual(sum(p["sample_count"] for p in row["parts"]) + row["trailing_silence_samples"], row["sample_count"])
            ids = [p["utterance_id"] for p in row["parts"]]
            self.assertEqual(ids, sorted(set(ids)))

    def test_compact_manifest_is_lossless_and_pilot_covers_required_cases(self):
        self.assertEqual(expand(compact(self.corpus)), self.corpus)
        chosen = pilot_cases(self.corpus["cases"])
        self.assertEqual(len(chosen), 20)
        self.assertIn("long-form-570-list", chosen)
        self.assertIn("near-silence-00", chosen)
        self.assertTrue(any("tier-300" in name for name in chosen))

    def test_generation_cap_helper_keeps_the_additive_short_clip_allowance(self):
        self.assertEqual(max_new_tokens(9600000), 3664)
        self.assertEqual(max_new_tokens(16001), 71)

    def test_pinned_reference_identity_and_all_artifact_hashes_present(self):
        pin = model_pin()
        self.assertEqual(pin["hf_revision"], "7278e1e70fe206f11671096ffdd38061171dd6e5")
        self.assertEqual(pin["operation"], "transcribe")
        self.assertEqual(len(pin["files"]), 10)
        for key in ("checkpoint_digest", "tokenizer_digest"):
            self.assertRegex(pin[key], r"^sha256:[a-f0-9]{64}$")
        self.assertEqual(len(pin["files"]["model-00001-of-00002.safetensors"]), 64)


class Seals(unittest.TestCase):
    def test_external_tensor_seals_detect_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory)
            receipt = artifact(root, b"\0" * 16, "ignored", True)
            path = verify(root, receipt)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify(root, receipt)
            with self.assertRaisesRegex(ValueError, "unsafe artifact path"):
                verify(root, {"path": "../outside", "bytes": 0, "sha256": ""})

    def test_sealer_refuses_incomplete_reference(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory)
            (root / "traces").mkdir()
            with self.assertRaisesRegex(ValueError, "coverage differs"):
                seal(HERE / "qwen3-asr-corpus.json", root, root / "sealed")

    def test_sharded_header_merge_is_canonical_and_refuses_duplicate_tensors(self):
        a, b = {"a": {"shape": [2]}}, {"b": {"shape": [3]}}
        self.assertEqual(canonical(merged_tensor_index([b, a])), b'{"a":{"shape":[2]},"b":{"shape":[3]}}')
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            merged_tensor_index([a, a])

    def test_registration_preserves_existing_reference_entries(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory)
            destination = root / "fixtures" / "qwen3-asr-1.7b"
            destination.mkdir(parents=True)
            (destination / "reference.json").write_text('{"cases":{}}')
            legacy = {"legacy": {"model": "unchanged", "sha256": "a" * 64}}
            (root / "fixtures/index.json").write_text(json.dumps(legacy))
            register_reference(destination, root)
            actual = json.loads((root / "fixtures/index.json").read_text())
            self.assertEqual(actual["legacy"], legacy["legacy"])
            entry = actual["qwen3-asr-1.7b.ref-v1.transformers-4.57.6.seed-0"]
            self.assertEqual(entry["sha256"], sha256(destination / "reference.json"))


if __name__ == "__main__":
    unittest.main()
