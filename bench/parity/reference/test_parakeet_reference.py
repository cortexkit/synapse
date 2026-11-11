import unittest

import parakeet_reference as ref


class GreedyTests(unittest.TestCase):
    def test_always_nonblank_zero_duration_has_exact_frame_cap(self):
        trace, triples = ref.greedy(lambda t, previous: ([3., 0., -1.], [4., 0.]),
                                    3, blank=2, cap=10, durations=(0, 1))
        self.assertEqual(triples, [[0, t, 0] for t in range(3) for _ in range(10)])
        self.assertEqual(len(trace), 30)
        self.assertEqual([i for i, step in enumerate(trace) if step["cap_hit"]], [9, 19, 29])

    def test_positive_duration_at_cap_does_not_add_an_extra_frame(self):
        calls = 0

        def joint(t, previous):
            nonlocal calls
            calls += 1
            return [3., 0., -1.], ([0., 4.] if calls % 10 == 0 else [4., 0.])

        trace, triples = ref.greedy(joint, 3, blank=2, cap=10, durations=(0, 1))
        self.assertEqual(len(trace), 30)
        self.assertEqual([x[1] for x in triples], [t for t in range(3) for _ in range(10)])
        self.assertEqual([x[2] for x in triples], ([0] * 9 + [1]) * 3)
        self.assertFalse(any(x["cap_hit"] for x in trace))

    def test_blank_zero_duration_advances_and_preserves_blank_input(self):
        calls = []

        def joint(frame, previous):
            calls.append((frame, previous))
            return [-2., -1., 3.], [4., 0.]

        trace, triples = ref.greedy(joint, 3, blank=2, durations=(0, 1))
        self.assertEqual(calls, [(0, 2), (1, 2), (2, 2)])
        self.assertEqual(triples, [])
        self.assertEqual(len(trace), 3)

    def test_lowest_id_wins_both_ties(self):
        self.assertEqual(ref.top_two([1., 1., 1.]), (0, 1, 1., 1.))

    def test_nonfinite_logits_refused(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            ref.top_two([0., float("nan")])

    def test_ladder_has_every_rung_boundary_and_ten_minutes(self):
        rows = ref.ladder()
        self.assertEqual(len(rows), 22)
        for row in rows:
            self.assertEqual(row["sample_count"] % 16, 0)
            self.assertEqual(ref.frame_counts(row["sample_count"])[2], row["encoder_frames"])
        self.assertEqual(rows[-1]["sample_count"], 9600000)

    def test_hop_boundaries(self):
        self.assertEqual(ref.frame_counts(16000 - 1), (100, 99, 13))
        self.assertEqual(ref.frame_counts(16000), (101, 100, 13))
        self.assertEqual(ref.frame_counts(16000 + 1), (101, 100, 13))

    def test_word_boundaries_zero_duration_and_final_clamp(self):
        class Tokenizer:
            pieces = {1: "▁hello", 2: "!", 3: "▁world"}

            def convert_ids_to_tokens(self, id):
                return self.pieces[id]

            def decode(self, ids):
                return "".join(self.pieces[id].replace("▁", " ") for id in ids).lstrip()

        result = ref.words_from_triples(Tokenizer(), [[1, 1, 0], [2, 1, 0], [3, 2, 4]], 3200)
        self.assertEqual(result, {"text": "hello! world", "no_speech": False,
                         "words": [{"word": "hello!", "start_ms": 80, "end_ms": 160},
                                   {"word": "world", "start_ms": 160, "end_ms": 200.0}]})
        self.assertEqual(ref.words_from_triples(Tokenizer(), [], 16000),
                         {"text": "", "words": [], "no_speech": True})


class TransportTests(unittest.TestCase):
    def test_transport_hash_ignores_source_mtime(self):
        import os
        from pathlib import Path
        import tempfile
        import generate_parakeet as generator

        scratch = Path(__file__).resolve().parents[1] / "target/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            source = root / "output"
            source.mkdir()
            (source / "tensor.f32le").write_bytes(bytes(16))
            generator.pack_output(source, root / "first.tar")
            before = generator.ref.sha256_file(root / "first.tar")
            os.utime(source / "tensor.f32le", (123456789, 123456789))
            os.utime(source, (987654321, 987654321))
            generator.pack_output(source, root / "second.tar")
            self.assertEqual(before, generator.ref.sha256_file(root / "second.tar"))
            self.assertEqual((root / "first.seals.json").read_bytes(),
                             (root / "second.seals.json").read_bytes())


class RecipeTests(unittest.TestCase):
    def test_float_wav_peak_timestamp_is_zero_and_audio_is_unchanged(self):
        import io
        import struct
        import numpy as np
        import soundfile as sf
        import generate_parakeet as generator

        audio = np.array([[0.125, -0.25], [0.5, -0.75]], dtype=np.float32)
        raw = generator.wav_bytes(audio, 44100, "FLOAT")
        position, timestamps = 12, []
        while position + 8 <= len(raw):
            chunk, size = struct.unpack_from("<4sI", raw, position)
            if chunk == b"PEAK":
                timestamps.append(struct.unpack_from("<I", raw, position + 12)[0])
            position += 8 + size + (size & 1)
        self.assertEqual(timestamps, [0])
        decoded, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        self.assertEqual(rate, 44100)
        self.assertTrue(np.array_equal(decoded, audio))

    def test_ladder_concatenates_and_trims_without_a_base_field(self):
        import numpy as np
        from pathlib import Path
        from unittest.mock import patch
        import generate_parakeet as generator

        parts = [{"audio_sha256": "audio-a"}, {"audio_sha256": "audio-b"}]
        audio = np.array([[0.125], [-0.25]], dtype=np.float32)
        with patch.object(generator, "CACHE", Path(".")), \
                patch.object(generator.ref, "sha256_file", side_effect=lambda p: p.name), \
                patch("soundfile.read", return_value=(audio, 16000)), \
                patch.object(generator, "resample", side_effect=lambda a, source, target: a.copy()):
            actual = generator.pcm({"recipe": "ladder", "parts": parts, "sample_count": 5})
        self.assertEqual(actual.tolist(), [0.125, -0.25, 0.125, -0.25, 0.125])


class MaskFixTests(unittest.TestCase):
    def test_fully_masked_query_is_finite_and_zero(self):
        import torch
        import parakeet_mask_fix
        from types import SimpleNamespace
        module = SimpleNamespace(num_key_value_groups=1, training=False)
        query = torch.zeros(1, 1, 2, 4)
        values = torch.ones_like(query)
        bias = torch.tensor([[[[0., float("-inf")], [float("-inf"), float("-inf")]]]])
        output, weights = parakeet_mask_fix.eager_attention_forward(
            module, query, query, values, bias, 1.)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output[0, 1].tolist(), [[0., 0., 0., 0.]])
        self.assertEqual(weights.tolist(), [[[[1., 0.], [0., 0.]]]])

    def test_unpadded_attention_is_bit_identical(self):
        import torch
        import parakeet_mask_fix
        from types import SimpleNamespace
        from transformers.models.parakeet.modeling_parakeet import eager_attention_forward
        module = SimpleNamespace(num_key_value_groups=1, training=False)
        query = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 24
        for bias in (None, torch.zeros(1, 2, 3, 3)):
            expected = eager_attention_forward(module, query, query, query, bias, .5)
            actual = parakeet_mask_fix.eager_attention_forward(module, query, query, query, bias, .5)
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected[1], actual[1]))


if __name__ == "__main__":
    unittest.main()
