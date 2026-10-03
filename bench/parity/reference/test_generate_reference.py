"""Tests for the reference generator's refusal rules and committed fixtures.

Run with: python3 -m unittest discover -s bench/parity/reference
These tests need neither torch nor transformers: the generator imports them
only after its preflight passes, and the scoring backend is replaced by a
recording fake.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_reference as gr  # noqa: E402

PARITY_DIR = Path(__file__).resolve().parent.parent


class FakeTokenizer:
    """One id per space-separated word; special tokens per the grammar."""

    def __init__(self, grammar: dict):
        self.grammar = grammar

    def __call__(self, text, pair=None, add_special_tokens=True):
        ids = [1000 + (sum(map(ord, word)) % 997) for word in text.split()]
        if pair is not None:
            ids = ids + [5] + [1000 + (sum(map(ord, w)) % 997) for w in pair.split()]
        if add_special_tokens:
            ids = [1] + ids + [2]
        return {"input_ids": ids}


class RecordingBackend:
    calls: list = []

    def __init__(self, snapshot, entry, seed):
        RecordingBackend.calls.append(("load", entry["hf_revision"], seed))
        self.tokenizer = FakeTokenizer(entry["grammar"])
        self.entry = entry

    def score(self, batch):
        RecordingBackend.calls.append(("score", len(batch)))
        if self.entry["operation"] == "embed":
            return [[1.0, 0.0] for _ in batch]
        return [0.5 for _ in batch]


class PreflightTests(unittest.TestCase):
    def setUp(self):
        RecordingBackend.calls = []
        self.workdir = Path(tempfile.mkdtemp())
        shutil.copy(PARITY_DIR / "models.json", self.workdir / "models.json")

    def tearDown(self):
        shutil.rmtree(self.workdir)

    def run_generator(self, version, seed):
        return gr.run(
            ["--hf-cache", str(self.workdir), "--model", "gte-modernbert-base", "--seed", str(seed)],
            installed_version=version,
            backend_factory=RecordingBackend,
            parity_dir=self.workdir,
        )

    def test_transformers_5_16_0_is_refused_before_scoring(self):
        with self.assertRaisesRegex(gr.Refused, "5.16.0"):
            self.run_generator("5.16.0", 0)
        self.assertEqual(RecordingBackend.calls, [])
        self.assertFalse((self.workdir / "fixtures").exists())

    def test_seed_1_is_refused_before_scoring(self):
        with self.assertRaisesRegex(gr.Refused, "seed 1"):
            self.run_generator("5.16.1", 1)
        self.assertEqual(RecordingBackend.calls, [])
        self.assertFalse((self.workdir / "fixtures").exists())

    def test_5_16_1_with_seed_0_runs_and_scores(self):
        path = self.run_generator("5.16.1", 0)
        self.assertEqual(RecordingBackend.calls[0], ("load", gr.load_manifest(self.workdir)["models"]["gte-modernbert-base"]["hf_revision"], 0))
        self.assertTrue(any(call[0] == "score" for call in RecordingBackend.calls))
        document = json.loads(path.read_text())
        self.assertEqual(document["fixture_set_id"], "gte-modernbert-base.ref-v1.transformers-5.16.1.seed-0")
        lengths = {case["id"]: len(case["input_ids"]) for case in document["cases"]}
        for length in gr.BOUNDARY_LENGTHS:
            self.assertEqual(lengths[f"boundary-{length}"], length)
        self.assertEqual(lengths["long-8192"], 8192)


class CommittedFixtureTests(unittest.TestCase):
    """The committed reference fixtures match their index and contain every
    case category listed in generate_reference.py's module docstring."""

    def setUp(self):
        self.manifest = gr.load_manifest()
        self.index = json.loads((PARITY_DIR / "fixtures" / "index.json").read_text())

    def test_every_model_has_a_fixture_set_named_by_version_and_seed(self):
        reference = self.manifest["reference"]
        for slug in self.manifest["models"]:
            set_id = gr.fixture_set_id(slug, reference["reference_transformers_version"], reference["reference_seed"])
            self.assertIn(set_id, self.index, slug)
            self.assertIn("transformers-5.16.1", set_id)
            self.assertIn("seed-0", set_id)

    def test_digests_and_coverage(self):
        for set_id, entry in self.index.items():
            path = PARITY_DIR / entry["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"], set_id)
            document = json.loads(path.read_text())
            self.assertEqual(document["fixture_set_id"], set_id)
            categories = {case["category"] for case in document["cases"]}
            required = {"short", "shape_boundary", "batched", "long"}
            if document["operation"] == "rerank":
                required |= {"pool_10", "pool_100"}
                pools = {}
                for case in document["cases"]:
                    if "pool" in case:
                        pools[case["pool"]] = pools.get(case["pool"], 0) + 1
                self.assertEqual(pools, {"pool-10": 10, "pool-100": 100}, set_id)
            self.assertTrue(required <= categories, (set_id, categories))
            long_cases = [case for case in document["cases"] if case["category"] == "long"]
            self.assertTrue(long_cases and all(len(case["input_ids"]) == 8192 for case in long_cases), set_id)


if __name__ == "__main__":
    unittest.main()
