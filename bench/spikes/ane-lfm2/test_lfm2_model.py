#!/usr/bin/env python3
"""Focused tests for the static LFM2 conversion graph."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from lfm2_decode import StatefulConvMixer  # pyright: ignore[reportMissingImports]
from lfm2_model import LFM2ConvMixer, apply_rope, read_config


class LFM2ModelTests(unittest.TestCase):
    def test_short_conv_matches_causal_unflipped_cross_correlation(self) -> None:
        hidden_size = 2
        identity = torch.eye(hidden_size)
        mixer = LFM2ConvMixer(
            hidden_size=hidden_size,
            in_proj=torch.cat((identity, identity, identity), dim=0),
            conv_weight=torch.tensor([[[1.0, 2.0, 3.0]], [[-1.0, 0.5, 2.0]]]),
            out_proj=identity,
        )
        values = torch.tensor([[[[9.0], [1.0], [2.0], [3.0]], [[7.0], [2.0], [1.0], [4.0]]]])
        mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.int32)

        actual = mixer(values, mask)
        expected = torch.zeros_like(actual)
        kernels = ((1.0, 2.0, 3.0), (-1.0, 0.5, 2.0))
        active_values = ((0.0, 1.0, 2.0, 3.0), (0.0, 2.0, 1.0, 4.0))
        for channel in range(hidden_size):
            products = [value * value for value in active_values[channel]]
            for position, gate in enumerate(active_values[channel]):
                convolution = 0.0
                for tap, weight in enumerate(kernels[channel]):
                    source = position - 2 + tap
                    if source >= 0:
                        convolution += weight * products[source]
                expected[0, channel, position, 0] = gate * convolution
        torch.testing.assert_close(actual, expected)

    def test_stateful_conv_keeps_three_products_and_uses_newest_sample(self) -> None:
        source = LFM2ConvMixer(
            hidden_size=1,
            in_proj=torch.tensor([[1.0], [1.0], [1.0]]),
            conv_weight=torch.tensor([[[1.0, 10.0, 100.0]]]),
            out_proj=torch.tensor([[1.0]]),
        )
        mixer = StatefulConvMixer(source, hidden_size=1)
        outputs = []
        for value in (1.0, 2.0, 3.0):
            hidden = torch.tensor([[[[value]]]])
            outputs.append(float(mixer(hidden).item()))
        self.assertEqual(outputs, [100.0, 820.0, 2823.0])
        torch.testing.assert_close(
            mixer.conv_state.flatten(),
            torch.tensor([1.0, 4.0, 9.0]),
        )

    def test_rope_uses_half_split_pairs(self) -> None:
        hidden = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        cos = torch.zeros_like(hidden)
        sin = torch.ones_like(hidden)
        actual = apply_rope(hidden, cos, sin)
        torch.testing.assert_close(actual, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))

    def test_config_uses_checkpoint_mlp_width_instead_of_block_ff_dim(self) -> None:
        tensors = {
            "embed_tokens.weight": torch.zeros((16, 4)),
            "layers.0.feed_forward.w1.weight": torch.zeros((6, 4)),
            "layers.0.feed_forward.w2.weight": torch.zeros((4, 6)),
            "layers.0.feed_forward.w3.weight": torch.zeros((6, 4)),
            "layers.0.conv.conv.weight": torch.zeros((4, 1, 3)),
        }
        raw = {
            "block_dim": 4,
            "block_ff_dim": 99,
            "num_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "full_attn_idxs": [],
            "conv_L_cache": 3,
            "norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "vocab_size": 16,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
            config = read_config(path, tensors)
        self.assertEqual(config.intermediate_size, 6)
        self.assertEqual(config.layer_types, ("conv",))


if __name__ == "__main__":
    unittest.main()
