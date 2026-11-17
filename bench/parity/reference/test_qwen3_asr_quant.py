"""Ordinary quant-gate checks; synthetic fixtures never certify the model."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from qwen3_asr_quant_gate import calibration, cap_failure, probe_failure, quality, seal_profile


def traces(id="one", split="speech", margins=(.2, .3)):
    case = {"id": id, "split": split, "ground_truth": "alpha", "terms": []}
    teacher = {"generated_ids": [1, 151645], "top_two_margins": list(margins),
               "prompt_ids": [42], "span": {"start": 0, "len": 1}}
    profile = {**teacher, "generated_ids": [1, 151645], "forced_ids": [1], "mode": "f16-boundaries"}
    return [case], {id: teacher}, {id: profile}


class Calibration(unittest.TestCase):
    def test_floor_and_strictly_above_tau_counts_stop(self):
        cases, teacher, profile = traces(margins=(.1, .8))
        result = calibration(cases, teacher, profile)
        self.assertEqual(result["tau"], .1)
        self.assertEqual(result["fixtures"], {"one": {"graded": 1, "exempt": 1}})
        self.assertFalse(result["accepted"])

    def test_disagreement_uses_teacher_margin_not_profile_margin(self):
        cases, teacher, profile = traces(margins=(.3, .9))
        profile["one"]["generated_ids"][0] = 2
        profile["one"]["top_two_margins"] = [100, 100]
        result = calibration(cases, teacher, profile)
        self.assertEqual(result["tau"], .6)

    def test_tau_over_one_rejected(self):
        cases, teacher, profile = traces(margins=(.51, 3))
        profile["one"]["generated_ids"][0] = 2
        self.assertEqual(calibration(cases, teacher, profile)["failures"][0]["gate"], "tau-maximum")

    def test_vacuous_profile_rejected_per_set_not_globally(self):
        cases, teacher, profile = traces(margins=(.4, .4))
        c, t, p = traces("probe", "noise", (.01, .01))
        result = calibration(cases + c, teacher | t, profile | p)
        self.assertEqual(result["failures"], [{"gate": "nonvacuous-profile", "split": "noise", "graded": 0, "exempt": 2}])

    def test_exactly_ninety_percent_accepted(self):
        cases, teacher, profile = traces()
        teacher["one"].update(generated_ids=list(range(10)), top_two_margins=[.01] + [1.] * 9)
        profile["one"].update(generated_ids=list(range(10)), forced_ids=list(range(9)))
        self.assertTrue(calibration(cases, teacher, profile)["accepted"])

    def test_forcing_must_use_teacher_prefix_even_after_divergence(self):
        cases, teacher, profile = traces()
        profile["one"]["forced_ids"] = [99]
        with self.assertRaisesRegex(ValueError, "teacher-forced"):
            calibration(cases, teacher, profile)

    def test_missing_or_malformed_trace_refused(self):
        cases, teacher, profile = traces()
        with self.assertRaisesRegex(ValueError, "coverage"):
            calibration(cases, teacher, {})
        teacher["one"]["top_two_margins"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "teacher-forced"):
            calibration(cases, teacher, profile)

    def test_partial_corpus_cannot_seal_or_admit_engines(self):
        cases, teacher, profile = traces()
        with self.assertRaisesRegex(ValueError, "entire frozen corpus"):
            seal_profile(cases, teacher, profile, {}, Path("unused"), {"correct": 9, "total": 10})


class Caps(unittest.TestCase):
    def test_new_q8_cap_is_a_regression_even_if_text_is_good(self):
        trace = {"id": "case", "stop_reason": "cap", "cap": 2, "generated_ids": [1, 2]}
        failure = cap_failure(trace, {"stop_reason": "stop"})
        assert failure is not None
        self.assertEqual(failure["gate"], "decode_limit_reached")

    def test_sealed_cap_allows_exact_prefix_but_never_fabricated_eos(self):
        trace = {"id": "case", "stop_reason": "cap", "cap": 2, "generated_ids": [1, 2]}
        baseline = {"stop_reason": "cap", "cap": 2}
        self.assertIsNone(cap_failure(trace, baseline))
        trace["generated_ids"] = [1, 151645]
        self.assertIsNotNone(cap_failure(trace, baseline))
        trace["generated_ids"] = [1]
        self.assertIsNotNone(cap_failure(trace, baseline))
        trace.update(cap=3, generated_ids=[1, 2, 3])
        self.assertIsNotNone(cap_failure(trace, baseline))

    def test_quality_scores_a_bounded_teacher_instead_of_automatically_rejecting_it(self):
        case = {"id": "case", "split": "speech", "ground_truth": "alpha", "terms": []}
        baseline = {"text": "alpha", "stop_reason": "cap", "cap": 2}
        trace = {**baseline, "id": "case", "generated_ids": [1, 2]}
        result = quality([case], {"case": baseline}, {"case": trace},
                         {"case": {"words": 1, "baseline_edits": 0, "actual_edits": 0}})
        self.assertTrue(result["accepted"])


class Quality(unittest.TestCase):
    def test_probes_score_parsed_text_without_companion(self):
        case = {"id": "probe", "split": "near-silence", "terms": ["Acme"]}
        self.assertIsNone(probe_failure(case, {"text": "acmeology", "no_speech": True}))
        failure = probe_failure(case, {"text": "ACME!", "no_speech": True})
        assert failure is not None
        self.assertEqual(failure["forbidden_occurrences"], 1)
        case.update(split="single-name", terms=["Alice", "Acme"], noun="Alice")
        self.assertIsNone(probe_failure(case, {"text": "Alice"}))
        self.assertIsNotNone(probe_failure(case, {"text": "Alice and Acme"}))

    def test_wer_no_per_split_exception_and_exact_half_point_allowed(self):
        cases = [{"id": "a", "split": "el", "ground_truth": "words", "terms": []},
                 {"id": "b", "split": "de", "ground_truth": "words", "terms": []}]
        rows = {id: {"text": "words", "stop_reason": "stop"} for id in ("a", "b")}
        counts = {"a": {"words": 200, "baseline_edits": 0, "actual_edits": 1},
                  "b": {"words": 10000, "baseline_edits": 100, "actual_edits": 0}}
        self.assertTrue(quality(cases, rows, rows, counts)["accepted"])
        counts["a"]["actual_edits"] = 2
        self.assertEqual(quality(cases, rows, rows, counts)["failures"][0]["split"], "el")

    def test_quality_refuses_partial_arms(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            quality([{"id": "missing"}], {}, {}, {})

    def test_earnings_false_occurrences_not_wer_edits(self):
        cases = [{"id": "a", "split": "earnings22-test-200", "ground_truth": "Alice",
                  "terms": ["Alice", "Acme"], "vocabulary": ["Alice", "Acme"]}]
        before = {"a": {"text": "Alice", "stop_reason": "stop"}}
        after = {"a": {"text": "Alice Acme Acme Acme", "stop_reason": "stop"}}
        counts = {"a": {"words": 10000, "baseline_edits": 0, "actual_edits": 3}}
        result = quality(cases, before, after, counts)
        self.assertEqual(result["failures"][0]["gate"], "earnings-names")
        self.assertEqual(result["failures"][0]["q8"]["false"], 3)

    def test_c4_selects_largest_passing_tier_and_refuses_thirty_failure(self):
        cases, rows, counts = [], {}, {}

        def add(c, text, cost=65):
            cases.append(c)
            rows[c["id"]] = {"text": text, "stop_reason": "stop", "context_tokens": cost}
            counts[c["id"]] = {"words": 1000, "baseline_edits": 0, "actual_edits": 0}

        for i in range(400):
            add({"id": f"base{i}", "split": "clean" if i < 200 else "other", "ground_truth": "speech", "terms": []}, "speech")
        for i in range(200):
            add({"id": f"e{i}-global", "split": "earnings22-test-200", "ground_truth": "Alice", "terms": ["Alice"]}, "Alice")
        for tier in (30, 100, 300):
            for i in range(200):
                add({"id": f"e{i}-tier-{tier}", "split": "earnings22-test-200", "tier": tier,
                     "ground_truth": "Alice", "terms": ["Alice"], "retained_terms": ["Alice"]}, "Alice", tier + 1)
            for list_num in range(3):
                for i in range(400):
                    add({"id": f"base{i}-control-{tier}-{list_num}", "split": "clean" if i < 200 else "other",
                         "ground_truth": "speech", "terms": ["Acme"], "tier": tier,
                         "list_id": f"control-{tier}-{list_num}", "probe_set": "unrelated-list"}, "speech", tier + 1)
        baseline = copy.deepcopy(rows)
        for i in range(2):
            rows[f"base{i}-control-300-0"]["text"] = "speech Acme"
        result = quality(cases, baseline, rows, counts)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["tiers"]["300"]["passed"])
        self.assertEqual(result["provisional_bias_limit"], 128)
        rows["base0-control-30-0"]["text"] = "speech Acme Acme"
        result = quality(cases, baseline, rows, counts)
        self.assertFalse(result["accepted"])
        self.assertEqual(set(result["tiers"]), {"30"})


class Runtime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_rust_block_decode_signed_scale_not_requantization(self):
        from qwen3_asr_quant import dequantize
        raw = b"\x00\x38" + bytes([0, 1, 127, 128, 255] + [2] * 27)
        self.assertEqual(dequantize(raw, [1, 32]).tolist()[0][:5], [0., .5, 63.5, -64., -.5])
        with self.assertRaisesRegex(ValueError, "layout"):
            dequantize(raw, [1, 31])

    def test_half_boundary_not_global_half_or_logits_cast(self):
        from qwen_asr.core.transformers_backend import modeling_qwen3_asr as up
        from qwen3_asr_quant import half, numerical_profile
        torch = self.torch
        root = torch.nn.Module()
        root.thinker = torch.nn.Module()
        root.thinker.model = torch.nn.Module()
        root.thinker.model.layers = torch.nn.ModuleList([])
        root.thinker.audio_tower = torch.nn.Module()
        class ZeroAttention(torch.nn.Module):
            def forward(self, hidden_states, **_kwargs):
                return torch.zeros_like(hidden_states)
        class EncoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn_layer_norm = torch.nn.Identity()
                self.self_attn = ZeroAttention()
                self.final_layer_norm = torch.nn.Identity()
                self.fc1 = torch.nn.Identity()
                self.activation_fn = torch.nn.Identity()
                self.fc2 = ZeroAttention()
            def forward(self, hidden_states, *_args):
                return (hidden_states,)
        root.thinker.audio_tower.layers = torch.nn.ModuleList([EncoderLayer()])
        root.thinker.linear = torch.nn.Linear(1, 1, bias=False)
        root.thinker.lm_head = torch.nn.Linear(1, 1, bias=False)
        root.thinker.conv = torch.nn.Conv2d(1, 1, 1, bias=False)
        for parameter in root.parameters():
            parameter.data.fill_(1.0003)
        x = torch.ones(1, 1)
        original_attention = up.eager_attention_forward
        with numerical_profile(root):
            self.assertEqual(root.thinker.linear(x).item(), 1.)
            self.assertNotEqual(root.thinker.lm_head(x).item(), 1.)
            self.assertNotEqual(root.thinker.conv(torch.ones(1, 1, 1, 1)).item(), 1.)
            self.assertEqual(root.thinker.audio_tower.layers[0](torch.tensor([1.0003]), None)[0].item(), 1.)
            _output, probabilities = up.eager_attention_forward(
                types.SimpleNamespace(num_key_value_groups=1, training=False),
                torch.tensor([[[[.1234]]]]), torch.tensor([[[[.2345], [.3456]]]]),
                torch.ones(1, 1, 2, 1), None, 1.)
            self.assertTrue(torch.equal(probabilities, half(probabilities)))
            self.assertTrue(all(p.dtype == torch.float32 for p in root.parameters()))
        self.assertIs(up.eager_attention_forward, original_attention)
        self.assertNotEqual(root.thinker.linear(x).item(), 1.)

    def test_hash_mismatch_precedes_any_load(self):
        from qwen3_asr_quant import verified_artifacts
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "weights.q8_0").write_bytes(b"poison")
            with self.assertRaisesRegex(ValueError, "artifact_poisoned"):
                verified_artifacts(path)

    def test_private_output_only_aggregates(self):
        import numpy as np
        import soundfile as sf
        from run_qwen3_asr_private_quant import aggregate
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            sf.write(path / "private.wav", np.zeros(1600), 16000)
            (path / "manifest.json").write_text(json.dumps([{"audio": "private.wav", "terms": ["SECRETNAME"],
                       "reference": "Hello SECRETNAME", "expected": ["SECRETNAME"]}]))
            reference = types.SimpleNamespace(run=lambda *_a, **_k: {"text": "Hello SECRETNAME", "stop_reason": "stop"})
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = aggregate(path / "manifest.json", reference)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual((result["correct"], result["total"]), (1, 1))
            self.assertNotIn("SECRETNAME", json.dumps(result))

    def test_native_and_python_private_diagnostics_are_suppressed(self):
        import subprocess
        import sys
        code = ("from run_qwen3_asr_private_quant import private_output\n"
                "import os, sys\n"
                "with private_output():\n"
                " print('SECRETNAME')\n"
                " print('SECRETNAME', file=sys.stderr)\n"
                " os.write(1, b'SECRETNAME')\n"
                " os.write(2, b'SECRETNAME')\n"
                "print('aggregate-only')\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "aggregate-only\n")
        self.assertEqual(result.stderr, "")


class PublicAudio(unittest.TestCase):
    def test_completed_stage_archive_survives_until_next_publication(self):
        import tarfile
        from run_qwen3_asr_quant import archive_output
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "pilot"
            (output / "teacher").mkdir(parents=True)
            (output / "teacher/case.json").write_text('"sealed-case"')
            archived = archive_output(output)
            with tarfile.open(archived) as archive:
                stream = archive.extractfile("pilot/teacher/case.json")
                assert stream is not None
                self.assertEqual(stream.read(), b'"sealed-case"')
            (output / "forced-profile").mkdir()
            (output / "forced-profile/case.json").write_text('"forced-case"')
            archive_output(output)
            with tarfile.open(archived) as archive:
                self.assertIn("pilot/forced-profile/case.json", archive.getnames())
                self.assertIn("pilot/teacher/case.json", archive.getnames())

    def test_every_frozen_member_resolves_to_a_pinned_source(self):
        from qwen3_asr_corpus import expand
        from run_qwen3_asr_quant import CORPUS, source_key
        corpus = expand(json.loads(CORPUS.read_text()))
        sources = corpus["sources"]
        members = []
        for case in corpus["cases"]:
            members.extend(case.get("parts", [case]))
        for member in members:
            if "audio_sha256" in member:
                self.assertIn(source_key(member), sources)
        single = next(c for c in corpus["cases"] if c["split"] == "single-name")
        self.assertNotIn("source", single)
        self.assertEqual(source_key(single), "librispeech-test-clean")
        earnings = next(c for c in corpus["cases"] if c["split"] == "earnings22-test-200")
        self.assertEqual(source_key(earnings), "earnings22")

    def test_generated_probe_does_not_fetch(self):
        from run_qwen3_asr_quant import prepare_audio
        with patch("prepare_parakeet.fetch", side_effect=AssertionError("unexpected fetch")):
            prepare_audio({"recipe": "noise"}, {}, Path("unused"))

    def test_missing_member_materialized_from_pinned_parquet(self):
        import hashlib
        import pyarrow as pa
        import pyarrow.parquet as pq
        from run_qwen3_asr_quant import prepare_audio
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            raw = b"public audio member"
            digest = hashlib.sha256(raw).hexdigest()
            source = path / "public.parquet"
            pq.write_table(pa.Table.from_pylist([{"audio": {"bytes": raw}}]), source)
            case = {"audio_sha256": digest, "source": "public"}
            with patch("prepare_parakeet.fetch", return_value=source) as fetch:
                prepare_audio(case, {"public": {"url": "https://public/file.parquet", "sha256": "pin"}}, path)
                fetch.assert_called_once_with("https://public/file.parquet", "pin", path / "sources")
            self.assertEqual((path / "audio" / digest).read_bytes(), raw)
            with patch("prepare_parakeet.fetch", side_effect=AssertionError("cached member")):
                prepare_audio(case, {}, path)


if __name__ == "__main__":
    unittest.main()
