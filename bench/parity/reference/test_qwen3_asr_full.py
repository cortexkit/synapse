"""Offline runner, process-reuse controls and two independent cap obligations."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from qwen3_asr_corpus import (PROVISIONAL_SLOPE, SLUG, audio_tokens, calibration_trace, dump, expand,
                              max_new_tokens, token_rate_report)
from qwen3_asr_reference import (artifact, canonical, generation_config_record, generation_stop_reason,
                                model_pin, sha256)
from run_qwen3_asr_full import (SOURCE_FILES, check_case, check_termination, checksums, compare_determinism,
                                determinism_plan, finish, full_run, ordered_shard,
                                run_sequence, validate_policy)
from seal_qwen3_asr import (GENERATION_PIN, seal, source_bundle_receipts, verify_full_run,
                            verify_generation_revision)

HERE = Path(__file__).resolve().parent


def row(id="speech", split="fleurs-el-test", samples=64000, speech_tokens=50):
    count = audio_tokens(samples // 160)
    return {"id": id, "split": split, "tier": None, "sample_count": samples, "ground_truth": "public",
            "generated_ids": [11528, 17860, 151704] + [10] * (speech_tokens - 1) + [151645],
            "top_two_margins": [0.25] * (speech_tokens + 3), "language_prefix_tokens": 3,
            "prompt_ids": [1] * 8 + [151676] * count + [1] * 7, "span": {"start": 8, "len": count},
            "context_tokens": 0, "has_name_list": False, "max_new_tokens": max_new_tokens(samples, PROVISIONAL_SLOPE),
            "stop_reason": "stop", "cap": max_new_tokens(samples, PROVISIONAL_SLOPE),
            "tokens_per_second": PROVISIONAL_SLOPE, "encoder_width": 2048,
            "generation_max_length": count + 15 + max_new_tokens(samples, PROVISIONAL_SLOPE),
            "valid_mel_frames": samples // 160, "mel_frames": samples // 160}


def policy():
    pin = model_pin()
    return {"pinned_generation_config": json.loads((HERE.parent / "checkpoints" / SLUG / "generation_config.json").read_text()),
            "pinned_generation_config_sha256": pin["files"]["generation_config.json"],
            "actual_logits_processors": [], "do_sample": False, "generation_mode": "greedy_search",
            "effective_generation_config_without_length_limits": {
                "do_sample": False, "num_beams": 1, "num_return_sequences": 1}}


class CapRules(unittest.TestCase):
    def test_speech_slope_excludes_language_prefix_but_keeps_stop(self):
        report = token_rate_report([row(), row("probe", "frontend", 1600, 1)])
        self.assertEqual(report["highest_speech_tokens_per_second"], 12.5)
        self.assertEqual(report["highest_raw_tokens_per_second"], 40)
        self.assertEqual(report["maximum_rate_fixture"], "speech")
        self.assertEqual(report["measured_tokens_per_second"], 19)
        self.assertEqual(report["tokens_per_second"], 25)
        self.assertEqual(report["ten_minute_cap"], 15064)
        self.assertEqual(report["kv_positions_without_bias"], 22879)
        self.assertEqual(report["kv_positions_at_2048_bias_tokens"], 24927)
        self.assertEqual(report["complete_cap_checked_fixtures"], 2)

    def test_sub_two_second_speech_and_probes_do_not_set_the_slope(self):
        report = token_rate_report([row("base", samples=32000, speech_tokens=6),
                                    row("short", "librispeech-test-clean", 30400, 30),
                                    row("probe", "noise", 1600, 1)])
        self.assertEqual(report["measured_tokens_per_second"], 6)
        self.assertEqual(report["speech_fixtures"], 1)
        with self.assertRaisesRegex(ValueError, ">=2-second speech"):
            token_rate_report([row("only-probe", "noise", 1600, 1)])

    def test_complete_cap_still_grades_each_probe(self):
        with self.assertRaisesRegex(ValueError, "complete-cap headroom failed: probe"):
            token_rate_report([row(samples=32000, speech_tokens=6),
                               row("probe", "noise", 16000, 57)])

    def test_stop_token_is_part_of_the_speech_rate(self):
        self.assertEqual(token_rate_report([row(samples=32000, speech_tokens=9)])["measured_tokens_per_second"], 7)

    def test_all_public_speech_families_can_set_the_maximum(self):
        for split in ("librispeech-test-clean", "librispeech-test-other", "earnings22-test-200", "long-form",
                      *(f"fleurs-{lang}-test" for lang in ("de", "fr", "es", "it", "pl", "nl", "ru", "el"))):
            with self.subTest(split=split):
                report = token_rate_report([row("base", samples=32000, speech_tokens=6), row("fast", split)])
                self.assertEqual(report["maximum_rate_fixture"], "fast")

    def test_slope_above_twenty_five_stops_before_sealing(self):
        self.assertEqual(token_rate_report([row(samples=48000)])["measured_tokens_per_second"], 25)
        fast = row(speech_tokens=100)
        listed = calibration_trace({**fast, "context_tokens": 6}, {"terms": ["public name"]})
        report = token_rate_report([row("no-list", samples=32000, speech_tokens=6), listed])
        self.assertEqual(report["measured_tokens_per_second"], 6)
        self.assertEqual(report["tokens_per_second"], 25)
        self.assertEqual(report["highest_name_list_tokens_per_second"], 25)
        self.assertEqual(report["maximum_name_list_rate_fixture"], "speech")
        self.assertEqual(report["complete_cap_checked_fixtures"], 1)
        with self.assertRaisesRegex(ValueError, "exceeds 25 at speech"):
            token_rate_report([fast])
        with self.assertRaisesRegex(ValueError, "terms/context disagree"):
            calibration_trace(fast, {"terms": ["public name"]})

    def test_cap_census_has_exact_ids_for_all_tiers_and_split_intersections(self):
        capped = []
        splits = ("fleurs-el-test", "earnings22-test-200")
        for split in splits:
            for label, tier in (("none", None), ("30", 30), ("100", 100), ("300", 300)):
                trace = row(split + "/" + label, split)
                trace.update(tier=tier, stop_reason="cap")
                trace["generated_ids"] = [11528, 17860, 151704] + [10] * (trace["cap"] - 3)
                trace["top_two_margins"] = [0.25] * trace["cap"]
                capped.append(trace)
        report = token_rate_report([row("stopped", samples=32000, speech_tokens=6), *capped])
        census = report["capped_fixtures"]
        self.assertEqual(census["count"], 8)
        self.assertNotIn("stopped", census["ids"])
        for label in ("none", "30", "100", "300"):
            self.assertEqual(census["by_tier"][label], {"count": 2,
                             "ids": ["fleurs-el-test/" + label, "earnings22-test-200/" + label]})
            for split in splits:
                self.assertEqual(census["by_split"][split]["count"], 4)
                self.assertEqual(census["by_split"][split]["by_tier"][label],
                                 {"count": 1, "ids": [split + "/" + label]})

    def test_eos_at_the_cap_is_stop_and_a_full_non_eos_prefix_is_cap(self):
        trace = row(speech_tokens=1)
        check_termination(trace)
        trace["cap"] = trace["max_new_tokens"] = len(trace["generated_ids"])
        trace["generation_max_length"] = len(trace["prompt_ids"]) + trace["cap"]
        check_termination(trace)
        self.assertEqual(generation_stop_reason(trace["generated_ids"], trace["cap"], [151645]), "stop")
        trace["generated_ids"][-1] = 10
        trace["stop_reason"] = "cap"
        check_termination(trace)
        trace["cap"] += 1
        with self.assertRaisesRegex(ValueError, "ended early without a stop token"):
            check_termination(trace)

    def test_generation_policy_refuses_length_dependent_processors(self):
        evidence = policy()
        validate_policy(evidence)
        evidence["actual_logits_processors"] = ["ForcedEOSTokenLogitsProcessor"]
        with self.assertRaisesRegex(ValueError, "processor evidence changed"):
            validate_policy(evidence)

    def test_effective_policy_serializes_runtime_token_tensors_not_length_limits(self):
        class TokenTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [151643, 151645]

        class Config:
            def to_dict(self):
                return {"max_length": 100, "max_new_tokens": 25, "do_sample": False,
                        "_eos_token_tensor": TokenTensor(), "forced_eos_token_id": None}

        recorded = generation_config_record(Config())
        self.assertEqual(json.loads(canonical(recorded)), {
            "do_sample": False, "_eos_token_tensor": [151643, 151645], "forced_eos_token_id": None})


class FakeAudio:
    def __init__(self, value):
        self.value = value

    def astype(self, *_args, **_kwargs):
        return self

    def tobytes(self):
        return self.value.encode()


class FakeReference:
    loads = 0

    def __init__(self, _directory):
        FakeReference.loads += 1
        self.generation_policy = policy()

    def run(self, case, _audio, output, slope):
        assert slope == PROVISIONAL_SLOPE
        trace = row(case["id"], case["split"], case["sample_count"], 1)
        trace["ground_truth"] = case["ground_truth"]
        trace["tier"] = case.get("tier")
        if case.get("fake_emits_past_cap"):
            trace["generated_ids"] = [11528, 17860, 151704] + [10] * (trace["cap"] - 3)
            trace["top_two_margins"] = [0.25] * trace["cap"]
            trace["stop_reason"] = "cap"
        trace["generation_policy_sha256"] = hashlib.sha256(canonical(self.generation_policy)).hexdigest()
        trace["mel"] = artifact(output, b"\0" * (trace["mel_frames"] * 128 * 4), "", True)
        trace["encoder"] = artifact(output, b"\0" * (trace["span"]["len"] * 2048 * 4), "", True)
        return trace


class FullRunner(unittest.TestCase):
    def test_shell_entry_sets_single_threads_and_keeps_runtime_caches_under_output(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory)
            interpreter = root / "python-stub"
            interpreter.write_text('#!/usr/bin/env python3\nimport json, os, sys\nprint(json.dumps({"env": dict(os.environ), "args": sys.argv[1:]}))\n')
            interpreter.chmod(0o755)
            output = root / "output"
            env = {**os.environ, "PYTHON": str(interpreter)}
            result = json.loads(subprocess.check_output(["bash", str(HERE / "full_run.sh"), "2", str(output)], env=env, text=True))
            self.assertEqual(result["args"], [str(HERE / "run_qwen3_asr_full.py"), "--workers", "2", "--output", str(output)])
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
                self.assertEqual(result["env"][name], "1")
            self.assertEqual(result["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            for name in ("XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "JOBLIB_TEMP_FOLDER"):
                self.assertTrue(Path(result["env"][name]).is_relative_to(output / "cache"))
            self.assertFalse(output.exists())

    def test_shards_preserve_order_and_cover_every_case_once(self):
        cases = [{"id": str(i)} for i in range(17)]
        shards = [ordered_shard(cases, index, 4) for index in range(4)]
        self.assertEqual([c["id"] for c in shards[2]], ["2", "6", "10", "14"])
        self.assertEqual(sorted(c["id"] for shard in shards for c in shard), sorted(c["id"] for c in cases))
        with self.assertRaises(ValueError):
            ordered_shard(cases, 4, 4)

    def test_reused_models_run_after_different_predecessors_and_drift_stops(self):
        cases = expand(json.loads((HERE / "qwen3-asr-corpus.json").read_text()))["cases"]
        plan = determinism_plan(cases)
        by_id = {case["id"]: case for case in cases}
        provenance = {"corpus_sha256": "a" * 64, "reference_rules_sha256": "b" * 64}
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            FakeReference.loads = 0
            reports = [run_sequence(root, [by_id[id] for id in plan[label]], label,
                                    provenance, FakeReference, lambda c, _cache: FakeAudio(c["id"]))
                       for label in ("check-a", "check-b")]
            self.assertEqual(FakeReference.loads, 2)
            result = compare_determinism(plan, *reports)
            self.assertEqual(len(result), 3)
            for evidence in result.values():
                self.assertNotEqual(*evidence["preceding_fixtures"].values())
            changed = copy.deepcopy(reports[1])
            changed["cases"][plan["targets"][0]]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "model-reuse determinism check failed"):
                compare_determinism(plan, reports[0], changed)

    def test_existing_outputs_cannot_be_mixed_or_resumed(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory)
            (root / "old-trace.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "cannot be mixed or resumed"):
                full_run(root, 1)

    def test_checksum_manifest_supports_external_cache_relocation_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory) / "output"
            receipt = artifact(root, b"\0" * 16, "", True)
            dump(root / "trace.json", {"encoder": receipt})
            dump(root / "cache" / "not-retained.json", {})
            self.assertEqual(checksums(root), 2)
            cache = Path(directory) / "cache"
            shutil.move(root / "external", cache)
            self.assertEqual(len(verify_full_run(root, cache)), 2)
            (cache / receipt["cache_path"]).write_bytes(b"x" * 16)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_full_run(root, cache)

    def test_complete_bundle_seals_both_actual_and_derived_caps_and_rejects_mixed_venues(self):
        cases = [{"id": "speech-" + str(i), "split": "fleurs-el-test", "sample_count": 32000,
                  "sample_rate": 16000, "ground_truth": "public"} for i in range(3)]
        cases[1].update(fake_emits_past_cap=True, tier=300)
        plan = {"targets": [c["id"] for c in cases],
                "check-a": ["warm-a", *[c["id"] for c in cases]],
                "check-b": ["warm-b", *[c["id"] for c in reversed(cases)]]}
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            corpus_path = root / "corpus.json"
            dump(corpus_path, {"cases": cases})
            provenance = {"model": model_pin(), "corpus_sha256": sha256(corpus_path),
                          "reference_rules_sha256": sha256(HERE / "qwen3-asr-reference.json"),
                          "workers": 1, "source_sha256": {name: sha256(HERE / name) for name in SOURCE_FILES}}
            dump(root / "provenance.json", provenance)
            rules = json.loads((HERE / "qwen3-asr-reference.json").read_text())
            dump(root / "loaded-weight-facts.json", rules["checkpoint_weight_facts"])
            reports = [run_sequence(root, order, label, provenance, FakeReference,
                                    lambda c, _cache: FakeAudio(c["id"]))
                       for label, order in (("check-a", cases), ("check-b", list(reversed(cases))), ("shard-0", cases))]
            determinism = compare_determinism(plan, reports[0], reports[1])
            finish(root, cases, provenance, determinism)
            with patch("seal_qwen3_asr.determinism_plan", return_value=plan):
                report = seal(corpus_path, root, root / "sealed")
                self.assertEqual(report["measured_tokens_per_second"], 6)
                self.assertEqual(report["tokens_per_second"], 25)
                trace = json.loads((root / "sealed/traces/speech-0.json").read_text())
                self.assertEqual(trace["max_new_tokens"], 114)
                self.assertEqual(trace["sealed_max_new_tokens"], 114)
                self.assertEqual(trace["generated_ids"], [11528, 17860, 151704, 151645])
                capped = root / "sealed/traces/speech-1.json"
                self.assertEqual(capped.read_bytes(), (root / "traces/speech-1.json").read_bytes())
                self.assertNotIn("sealed_max_new_tokens", json.loads(capped.read_text()))
                self.assertEqual(report["capped_fixtures"]["by_tier"]["300"], {"count": 1, "ids": ["speech-1"]})
                changed = root / "traces/speech-0.json"
                original = json.loads(changed.read_text())
                original["run_provenance_sha256"] = "0" * 64
                changed.write_bytes(canonical(original))
                checksums(root)
                with self.assertRaisesRegex(ValueError, "reference lineage mismatch"):
                    seal(corpus_path, root, root / "bad-seal")

    def test_offline_regrade_refuses_generation_inputs_from_a_different_revision(self):
        pin = json.loads(GENERATION_PIN.read_text())
        self.assertEqual(verify_generation_revision(pin), "76cc6a4b10f03c40c73ae82baeb55ab83bf793bc")
        changed = copy.deepcopy(pin)
        changed["source_sha256"]["qwen3_asr_reference.py"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory)
            dump(root / "corpus.json", {"cases": [{"id": "one"}]})
            dump(root / "traces/one.json", {})
            dump(root / "provenance.json", changed)
            with self.assertRaisesRegex(ValueError, "generation inputs do not match revision 76cc6a4b"):
                seal(root / "corpus.json", root, root / "sealed", offline_regrade=True)
            self.assertFalse((root / "sealed").exists())

    def test_offline_regrade_preserves_failed_bundle_and_records_separate_grading_identity(self):
        cases = [{"id": id, "split": "fleurs-el-test", "sample_count": 32000,
                  "ground_truth": "public"} for id in ("warm-a", "warm-b", "one", "two", "three")]
        by_id = {c["id"]: c for c in cases}
        plan = {"targets": ["one", "two", "three"], "check-a": ["warm-a", "one", "two", "three"],
                "check-b": ["warm-b", "three", "two", "one"]}
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            generated = root / "returned"
            corpus_path = root / "corpus.json"
            dump(corpus_path, {"cases": cases})
            provenance = {"model": model_pin(), "corpus_sha256": sha256(corpus_path),
                          "reference_rules_sha256": sha256(HERE / "qwen3-asr-reference.json"),
                          "workers": 1, "source_sha256": {name: sha256(HERE / name) for name in SOURCE_FILES}}
            dump(generated / "provenance.json", provenance)
            rules = json.loads((HERE / "qwen3-asr-reference.json").read_text())
            dump(generated / "loaded-weight-facts.json", rules["checkpoint_weight_facts"])
            reports = [run_sequence(generated, [by_id[id] for id in plan[label]], label, provenance,
                                    FakeReference, lambda c, _cache: FakeAudio(c["id"]))
                       for label in ("check-a", "check-b")]
            run_sequence(generated, cases, "shard-0", provenance, FakeReference, lambda c, _cache: FakeAudio(c["id"]))
            finish(generated, cases, provenance, compare_determinism(plan, *reports))
            for name in ("SHA256SUMS", "run-complete.json", "calibration.json"):
                (generated / name).unlink()
            dump(generated / "FAILED.json", {"error": "speech slope 34 exceeds 25 at earnings22-0024-tier-100; stop before sealing", "type": "ValueError"})
            pin_path = root / "generation-pin.json"
            pin = json.loads(GENERATION_PIN.read_text())
            dump(pin_path, {**pin, **{key: provenance[key] for key in ("source_sha256", "reference_rules_sha256", "corpus_sha256")}})
            before = source_bundle_receipts(generated)
            with patch("seal_qwen3_asr.GENERATION_PIN", pin_path), \
                    patch("seal_qwen3_asr.grading_identity", return_value={"revision": "b" * 40, "source_sha256": {}}), \
                    patch("seal_qwen3_asr.determinism_plan", return_value=plan):
                seal(corpus_path, generated, root / "sealed", offline_regrade=True)
            self.assertEqual(source_bundle_receipts(generated), before)
            receipt = json.loads((root / "sealed/offline-regrade.json").read_text())
            self.assertEqual(receipt["generation_revision"], pin["revision"])
            self.assertEqual(receipt["grading"]["revision"], "b" * 40)
            self.assertEqual(receipt["source_bundle"], before)
            self.assertEqual(receipt["validated_cases"], 5)
            self.assertEqual(receipt["validated_processes"], 3)
            self.assertEqual(receipt["calibration"]["tokens_per_second"], 25)

    def test_case_guards_reject_wrong_transcripts_and_placeholder_spans(self):
        case = {"id": "one", "split": "frontend", "sample_count": 16000, "ground_truth": "public"}
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory:
            root = Path(directory)
            trace = FakeReference(root).run(case, None, root, PROVISIONAL_SLOPE)
            check_case(trace, case)
            trace["ground_truth"] = "different"
            with self.assertRaisesRegex(ValueError, "case identity differs"):
                check_case(trace, case)
            trace["ground_truth"] = "public"
            trace["prompt_ids"][8] = 0
            with self.assertRaisesRegex(ValueError, "tensor/prompt shape differs"):
                check_case(trace, case)


if __name__ == "__main__":
    unittest.main()
