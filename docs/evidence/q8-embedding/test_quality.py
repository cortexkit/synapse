"""Small public controls: the expected results do not use the measured formulas."""
import unittest

import numpy as np
import torch

from quality import W8A8Linear, channel_int8, gguf_name, int8_dot, overlap, q8_0, rankings, tau


class QuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_q8_blocks_and_rounding(self):
        # Scale is 1 and 2 respectively; half steps must round away from zero.
        w = torch.zeros((1, 64))
        w[0, :4] = torch.tensor([127, -127, 0.5, -0.5])
        w[0, 32:36] = torch.tensor([254, -254, 1, -1])
        expected = w.clone()
        expected[0, 2:4] = torch.tensor([1, -1])
        expected[0, 34:36] = torch.tensor([2, -2])
        torch.testing.assert_close(q8_0(w), expected, rtol=0, atol=0)
        torch.testing.assert_close(q8_0(torch.zeros(2, 32)), torch.zeros(2, 32))

    def test_q8_matches_independent_package(self):
        from gguf import GGMLQuantizationType, dequantize, quantize
        rng = np.random.default_rng(3)
        w = rng.normal(size=(7, 96)).astype(np.float32)
        expected = dequantize(quantize(w, GGMLQuantizationType.Q8_0), GGMLQuantizationType.Q8_0)
        np.testing.assert_array_equal(q8_0(torch.from_numpy(w)).numpy(), expected)

    def test_channel_has_separate_output_scales(self):
        w = torch.tensor([[127., 0.5, -0.5], [254., 1., -1.], [0., 0., 0.]])
        q, s = channel_int8(w)
        torch.testing.assert_close(s, torch.tensor([[1.], [2.], [0.]]))
        torch.testing.assert_close(q, torch.tensor([[127, 1, -1], [127, 1, -1], [0, 0, 0]], dtype=torch.int8))

    def test_int32_dot_cpu(self):
        torch.manual_seed(2)
        a = torch.randint(-127, 128, (16, 1536), dtype=torch.int8)
        b = torch.randint(-127, 128, (1536, 32), dtype=torch.int8)
        expected = a.long() @ b.long()
        actual = int8_dot(a, b)
        self.assertEqual(actual.dtype, torch.int32)
        torch.testing.assert_close(actual.long(), expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not present")
    def test_int32_dot_mps_extremes_and_cancellation(self):
        torch.manual_seed(3)
        a = torch.randint(-127, 128, (16, 3072), dtype=torch.int8)
        b = torch.randint(-127, 128, (3072, 32), dtype=torch.int8)
        a[0] = 127
        b[:, 0] = 127
        b[:, 1] = -127
        b[::2, 2] = 127
        b[1::2, 2] = -127
        expected = a.long() @ b.long()
        actual = int8_dot(a.to("mps"), b.to("mps")).cpu()
        self.assertEqual(actual.dtype, torch.int32)
        torch.testing.assert_close(actual.long(), expected, rtol=0, atol=0)

    def test_w8a8_dynamic_token_and_bias(self):
        layer = torch.nn.Linear(32, 16)
        with torch.no_grad():
            layer.weight.fill_(1.)
            layer.bias.fill_(2.)
        quantized = W8A8Linear(layer)
        x = torch.stack([torch.ones(32), torch.ones(32) * 3, torch.zeros(32)]).reshape(1, 3, 32)
        expected = torch.tensor([34., 98., 2.]).view(1, 3, 1).expand(1, 3, 16)
        torch.testing.assert_close(quantized(x), expected)

    def test_qwen_tensor_mapping(self):
        self.assertEqual(gguf_name("layers.3.self_attn.q_norm.weight"), "blk.3.attn_q_norm.weight")
        self.assertEqual(gguf_name("embed_tokens.weight"), "token_embd.weight")
        with self.assertRaises((KeyError, ValueError)):
            gguf_name("unexpected.weight")


class MetricTests(unittest.TestCase):
    def test_overlap_is_sol_top10_not_all_relevant(self):
        self.assertEqual(overlap(list(range(10, 20)), list(range(20)), 50), 0)
        self.assertEqual(overlap([2, 0, 1], [0, 1], 3), 1)
        self.assertEqual(overlap([], [], 0), 0)
        self.assertEqual(overlap([0, 2], [0, 1], 2), 0.5)

    def test_tau_complete_orders_and_small_pools(self):
        self.assertEqual(tau([0, 1, 2], [2, 1, 0]), -1)
        self.assertAlmostEqual(tau([0, 1, 2], [0, 2, 1]), 1 / 3)
        self.assertIsNone(tau([0], [0]))

    def test_tied_scores_preserve_baseline_identity_order(self):
        pools = [{"candidates": [{"id": "z"}, {"id": "a"}]}]
        vectors = np.array([[1., 0.], [0.5, 0.5], [0.5, -0.5]])
        self.assertEqual(rankings(vectors, pools, [(0, [1, 2])]), [["z", "a"]])


if __name__ == "__main__":
    print("torch", torch.__version__, "numpy", np.__version__, flush=True)
    unittest.main(verbosity=2)
