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
from unittest.mock import Mock, call, patch

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

    def classifier_logit(self, ids):
        RecordingBackend.calls.append(("logit", len(ids)))
        return 0.0

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
        for key, value in gr.RUNTIME_SETTINGS.items():
            self.assertEqual(document["reference"][key], value)
        lengths = {case["id"]: len(case["input_ids"]) for case in document["cases"]}
        for length in gr.BOUNDARY_LENGTHS:
            self.assertEqual(lengths[f"boundary-{length}"], length)
        self.assertEqual(lengths["long-8192"], 8192)


class RuntimeSettingsTests(unittest.TestCase):
    def test_environment_limits_are_set_before_library_imports(self):
        import subprocess

        command = (
            "import os; os.environ['OMP_NUM_THREADS']='8'; os.environ['MKL_NUM_THREADS']='8'; "
            "import generate_reference; "
            "assert os.environ['OMP_NUM_THREADS']=='1'; "
            "assert os.environ['MKL_NUM_THREADS']=='1'; "
            "assert 'torch' not in __import__('sys').modules"
        )
        subprocess.run([sys.executable, "-c", command], cwd=Path(gr.__file__).parent, check=True)

    def test_torch_configuration_precedes_loading_and_all_models_use_eager(self):
        torch = Mock(__version__="2.14.0", float32="fp32")
        transformers = Mock()
        events = Mock()
        events.attach_mock(torch, "torch")
        events.attach_mock(transformers, "transformers")
        manifest = gr.load_manifest()
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.object(gr, "_TORCH_CONFIGURED", False):
            for entry in manifest["models"].values():
                gr.TorchBackend(Path("snapshot"), entry, 0)
        torch.set_num_threads.assert_called_once_with(1)
        torch.set_num_interop_threads.assert_called_once_with(1)
        torch.use_deterministic_algorithms.assert_called_once_with(True)
        self.assertEqual(events.mock_calls[:3], [
            call.torch.set_num_threads(1),
            call.torch.set_num_interop_threads(1),
            call.torch.use_deterministic_algorithms(True),
        ])
        torch.manual_seed.assert_called_with(0)
        for loader in (transformers.AutoModel, transformers.AutoModelForSequenceClassification, transformers.AutoModelForCausalLM):
            for invocation in loader.from_pretrained.call_args_list:
                self.assertEqual(invocation.kwargs, {"dtype": "fp32", "attn_implementation": "eager"})
            self.assertTrue(loader.from_pretrained.called)

    def test_unpinned_torch_is_refused_before_configuration(self):
        torch = Mock(__version__="2.13.0")
        with self.assertRaisesRegex(gr.Refused, "torch 2.14.0"):
            gr.configure_torch(torch)
        torch.set_num_threads.assert_not_called()


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for relative in (gr.CATALOG_PATH, gr.EVIDENCE_PATH):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gr.REPO_ROOT / relative, path)
        RecordingBackend.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def generate(self, version="5.16.1"):
        return gr.run(
            ["--hf-cache", str(self.root), "--catalog", "--output-dir", str(self.root / "output")],
            installed_version=version, backend_factory=RecordingBackend, repo_root=self.root,
        )

    def test_catalog_mode_preserves_inputs_and_metadata_and_stages_both_schemas(self):
        original = (self.root / gr.CATALOG_PATH).read_bytes()
        path = self.generate()
        self.assertEqual((self.root / gr.CATALOG_PATH).read_bytes(), original)
        before = json.loads(original)
        after = json.loads(path.read_text())
        for old, new in zip(before["models"], after["models"]):
            if "self_check" not in old:
                self.assertEqual(old, new)
                continue
            self.assertEqual(new["self_check"]["inputs"], old["self_check"]["inputs"])
            self.assertEqual(new["self_check"]["fixture_revision"], 2)
            self.assertIn("transformers 5.16.1, torch 2.14.0", new["self_check"]["reference_tool"])
            self.assertIn("generate_reference.py --catalog", new["self_check"]["reference_tool"])
            self.assertEqual({k: v for k, v in old.items() if k != "self_check"}, {k: v for k, v in new.items() if k != "self_check"})
            if new["task"] == "embed":
                self.assertEqual(new["self_check"]["reference"]["vectors"], [[1.0, 0.0]] * 8)
            else:
                self.assertEqual(new["self_check"]["reference"]["scores"], [[0.5] * len(group["candidates"]) for group in old["self_check"]["inputs"]])
        evidence = json.loads((self.root / "output" / gr.EVIDENCE_PATH).read_text())
        old_evidence = json.loads((self.root / gr.EVIDENCE_PATH).read_text())
        for field in ("upstream", "files", "corpus_id", "pairs"):
            self.assertEqual(evidence[field], old_evidence[field])
        for old, new in zip(old_evidence["items"], evidence["items"]):
            for field in ("id", "query", "candidates"):
                self.assertEqual(old[field], new[field])
            self.assertEqual(new["raw_logits"], [0.0] * len(new["candidates"]))
            self.assertEqual(new["scores"], [0.5] * len(new["candidates"]))
        self.assertEqual(len([event for event in RecordingBackend.calls if event[0] == "load"]), 3)
        first = path.read_bytes()
        self.assertEqual(self.generate().read_bytes(), first)

    def test_catalog_version_refusal_precedes_loading_or_writing(self):
        with self.assertRaisesRegex(gr.Refused, "5.17.0"):
            self.generate("5.17.0")
        self.assertEqual(RecordingBackend.calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_catalog_mismatching_upstream_is_refused_before_loading(self):
        path = self.root / gr.CATALOG_PATH
        document = json.loads(path.read_text())
        document["models"][0]["upstream"]["revision"] = "wrong"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(gr.Refused, "upstream pins disagree"):
            self.generate()
        self.assertEqual(RecordingBackend.calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_catalog_qwen_terminal_eos_is_added_only_once(self):
        entry = gr.load_manifest()["models"]["qwen3-embedding-0.6b"]
        self.assertEqual(gr.catalog_embed_ids(entry, lambda text: [42], "text"), [42, 151643])
        self.assertEqual(gr.catalog_embed_ids(entry, lambda text: [42, 151643], "text"), [42, 151643])
        gte = gr.load_manifest()["models"]["gte-modernbert-base"]
        self.assertEqual(gr.catalog_embed_ids(gte, lambda text: [50281, 42, 50282], "text"), [50281, 42, 50282])

    def test_manifest_file_digest_mismatch_refuses_before_backend_loading(self):
        entry = gr.load_manifest()["models"]["gte-modernbert-base"]
        with patch.object(gr, "sha256_file", return_value="0" * 64), patch.object(gr, "TorchBackend") as backend:
            with self.assertRaisesRegex(gr.Refused, "manifest pins"):
                gr.load_backend(self.root, entry, 0)
        backend.assert_not_called()


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
            self.assertEqual({k: document["reference"][k] for k in gr.RUNTIME_SETTINGS}, {
                "num_threads": 1, "num_interop_threads": 1, "deterministic_algorithms": True,
                "attn_implementation": "eager", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            })
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
