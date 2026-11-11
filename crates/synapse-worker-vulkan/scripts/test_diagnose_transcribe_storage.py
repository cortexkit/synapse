"""Portable checks for the scratch-reference diagnostic's seal and trace logic."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name("diagnose-transcribe-storage.py")
SPEC = importlib.util.spec_from_file_location("storage_diagnostic", MODULE)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def step(duration=4):
    return dict(frame_index=417, token_top1=7990, token_top2=497,
                duration_top1=duration, duration_top2=3 if duration == 4 else 4,
                token_logit1=-3.620, token_logit2=-7.889,
                duration_logit1=11.35513, duration_logit2=11.35067)


class StorageDiagnosticTests(unittest.TestCase):
    def test_selective_fp32_sets_do_not_widen_other_tensors(self):
        self.assertTrue(diag.keep_fp32("encoder.layers.21.conv.norm.running_var", "f32-last-three"))
        self.assertFalse(diag.keep_fp32("encoder.layers.20.conv.norm.running_var", "f32-last-three"))
        self.assertTrue(diag.keep_fp32("encoder.layers.0.self_attn.relative_k_proj.weight", "f32-attention"))
        self.assertFalse(diag.keep_fp32("encoder.layers.0.self_attn.bias_u", "f32-attention"))
        self.assertFalse(diag.keep_fp32("encoder_projector.weight", "f32-attention"))
        self.assertTrue(diag.keep_fp32("encoder.layers.5.feed_forward2.linear1.weight", "f32-feed-forward"))
        self.assertFalse(diag.keep_fp32("decoder.embedding.weight", "f32-feed-forward"))

    def test_duration_pair_and_nonexempt_margin(self):
        result = diag.first_finding([step()], [step(3)], 0.001)
        self.assertEqual(result["step"], 0)
        self.assertFalse(result["margin_exempt"])
        self.assertEqual(result["heads"][0]["head"], "duration")
        self.assertEqual(result["heads"][0]["reference_pair"], [4, 3])
        self.assertEqual(result["heads"][0]["engine_pair"], [3, 4])
        self.assertAlmostEqual(result["heads"][0]["reference_margin"], 0.00446)
        self.assertTrue(diag.first_finding([step()], [step(3)], 0.005)["margin_exempt"])

    def test_frame_or_length_difference_is_not_exempt(self):
        shifted = step()
        shifted["frame_index"] = 418
        self.assertFalse(diag.first_finding([step()], [shifted], 10)["margin_exempt"])
        self.assertFalse(diag.first_finding([step()], [], 10)["margin_exempt"])
        self.assertFalse(diag.first_finding([], [step()], 10)["margin_exempt"])
        self.assertIsNone(diag.first_finding([step()], [step()], 0.001))

    def test_hash_size_and_path_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b"diagnostic"
            (root / "tensor").write_bytes(raw)
            artifact = dict(path="tensor", bytes=len(raw), sha256=diag.hashlib.sha256(raw).hexdigest())
            self.assertEqual(diag.read_artifact(artifact, root, root), raw)
            for changed in (dict(bytes=1), dict(sha256="0" * 64), dict(path="../tensor"),
                            dict(cache_path="tensor"), dict(path="/tensor")):
                with self.assertRaises(ValueError):
                    diag.read_artifact({**artifact, **changed}, root, root)

    def test_trace_record_layout_is_shared_wire_size(self):
        self.assertEqual(diag.TRACE.size, 40)
        record = step()
        raw = diag.TRACE.pack(*(record[k] for k in diag.FIELDS), 1)
        decoded = dict(zip(diag.FIELDS, diag.TRACE.unpack(raw)[:9]))
        self.assertIsNone(diag.first_finding([record], [decoded], 0.001))
        self.assertIn("duration_logit1", json.dumps(decoded))


if __name__ == "__main__":
    unittest.main()
