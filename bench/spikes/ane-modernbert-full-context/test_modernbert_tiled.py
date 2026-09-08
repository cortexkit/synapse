#!/usr/bin/env python3
"""Unit tests for bounded full-context ModernBERT attention semantics."""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock

import torch

import modernbert_tiled as tiled
from norm_reproducer import NormVariants
from run_stages import stage_can_continue


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    local_window: int | None,
) -> torch.Tensor:
    query_bhsd = query.permute(0, 1, 3, 2)
    key_bhsd = key.permute(0, 1, 3, 2)
    value_bhsd = value.permute(0, 1, 3, 2)
    scores = torch.matmul(query_bhsd, key_bhsd.transpose(-1, -2)) * query.shape[-2] ** -0.5
    key_padding = (1.0 - attention_mask.float()).reshape(attention_mask.shape[0], 1, 1, -1)
    additive = key_padding * tiled.MASK_MIN_VALUE
    if local_window is not None:
        positions = torch.arange(query.shape[-1])
        permitted = (positions[:, None] - positions[None, :]).abs() <= local_window // 2
        additive = torch.where(
            permitted.reshape(1, 1, query.shape[-1], query.shape[-1]),
            additive,
            torch.full_like(scores, tiled.MASK_MIN_VALUE),
        )
    probabilities = torch.softmax(scores + additive, dim=-1, dtype=torch.float32)
    output = torch.matmul(probabilities, value_bhsd)
    return output.permute(0, 1, 3, 2).reshape(
        query.shape[0], query.shape[1] * query.shape[2], 1, query.shape[-1]
    )


class TileRangeTests(unittest.TestCase):
    def test_boundaries_and_final_partial_tile_are_complete_and_ordered(self) -> None:
        self.assertEqual(tiled.tile_ranges(9, 4), ((0, 4), (4, 8), (8, 9)))
        flattened = [index for start, end in tiled.tile_ranges(9, 4) for index in range(start, end)]
        self.assertEqual(flattened, list(range(9)))

    def test_diagnostic_positions_cover_local_and_query_tile_boundaries(self) -> None:
        self.assertEqual(tiled.diagnostic_positions(1024), (0, 63, 64, 255, 256, 1023))
        labels = tiled.diagnostic_checkpoint_labels(2)
        self.assertEqual(
            labels,
            (
                "token_embeddings",
                "embeddings_norm",
                "layer_0.attention_residual",
                "layer_0.output",
                "layer_1.attention_residual",
                "layer_1.output",
                "final_norm",
            ),
        )

    def test_rejects_non_positive_shapes(self) -> None:
        for length, tile_size in ((0, 1), (1, 0), (-1, 2)):
            with self.subTest(length=length, tile_size=tile_size):
                with self.assertRaises(ValueError):
                    tiled.tile_ranges(length, tile_size)


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.query = torch.randn(2, 3, 4, 9)
        self.key = torch.randn(2, 3, 4, 9)
        self.value = torch.randn(2, 3, 4, 9)
        self.mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 0]],
            dtype=torch.int32,
        )

    def test_global_attention_matches_dense_with_padding_and_partial_tile(self) -> None:
        expected = dense_attention(self.query, self.key, self.value, self.mask, None)
        actual = tiled.query_tiled_attention(
            self.query, self.key, self.value, self.mask, query_tile_size=4, local_window=None
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_local_attention_matches_dense_across_query_tile_boundaries(self) -> None:
        expected = dense_attention(self.query, self.key, self.value, self.mask, 4)
        actual = tiled.query_tiled_attention(
            self.query, self.key, self.value, self.mask, query_tile_size=4, local_window=4
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_streaming_reference_matches_dense_for_global_and_local_masks(self) -> None:
        for local_window in (None, 4):
            with self.subTest(local_window=local_window):
                expected = dense_attention(
                    self.query, self.key, self.value, self.mask, local_window
                )
                actual = tiled.streaming_reference_attention(
                    self.query,
                    self.key,
                    self.value,
                    self.mask,
                    query_tile_size=4,
                    key_tile_size=5,
                    local_window=local_window,
                )
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_global_query_tile_retains_distant_key_value_visibility(self) -> None:
        query = torch.ones(1, 1, 2, 9)
        key = torch.zeros(1, 1, 2, 9)
        value = torch.zeros(1, 1, 2, 9)
        mask = torch.ones(1, 9, dtype=torch.int32)
        baseline = tiled.query_tiled_attention(
            query, key, value, mask, query_tile_size=4, local_window=None
        )
        value[:, :, :, 8] = 90.0
        changed = tiled.query_tiled_attention(
            query, key, value, mask, query_tile_size=4, local_window=None
        )
        self.assertGreater(float((changed[:, :, :, 0] - baseline[:, :, :, 0]).abs().max()), 1.0)

    def test_local_query_does_not_see_keys_outside_its_permitted_window(self) -> None:
        query = torch.ones(1, 1, 2, 9)
        key = torch.zeros(1, 1, 2, 9)
        value = torch.zeros(1, 1, 2, 9)
        mask = torch.ones(1, 9, dtype=torch.int32)
        baseline = tiled.query_tiled_attention(
            query, key, value, mask, query_tile_size=4, local_window=4
        )
        value[:, :, :, 8] = 90.0
        changed = tiled.query_tiled_attention(
            query, key, value, mask, query_tile_size=4, local_window=4
        )
        torch.testing.assert_close(changed[:, :, :, 0], baseline[:, :, :, 0])

    def test_padded_key_is_not_visible(self) -> None:
        query = torch.ones(1, 1, 2, 5)
        key = torch.zeros(1, 1, 2, 5)
        value = torch.zeros(1, 1, 2, 5)
        mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.int32)
        baseline = tiled.query_tiled_attention(
            query, key, value, mask, query_tile_size=3, local_window=None
        )
        value[:, :, :, 4] = 10_000.0
        changed = tiled.query_tiled_attention(
            query, key, value, mask, query_tile_size=3, local_window=None
        )
        torch.testing.assert_close(changed, baseline)

    def test_score_and_local_mask_intermediates_are_bounded(self) -> None:
        score_shapes: list[tuple[int, ...]] = []
        mask_shapes: list[tuple[int, ...]] = []
        real_einsum = torch.einsum
        real_mask = tiled._tile_additive_mask

        def recording_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
            result = real_einsum(equation, *operands)
            if equation == "bchq,bkhc->bkhq":
                score_shapes.append(tuple(result.shape))
            return result

        def recording_mask(*args: Any, **kwargs: Any) -> torch.Tensor:
            result = real_mask(*args, **kwargs)
            mask_shapes.append(tuple(result.shape))
            return result

        with mock.patch("torch.einsum", side_effect=recording_einsum), mock.patch.object(
            tiled, "_tile_additive_mask", side_effect=recording_mask
        ):
            tiled.query_tiled_attention(
                self.query, self.key, self.value, self.mask, query_tile_size=4, local_window=4
            )
        self.assertTrue(score_shapes)
        self.assertLessEqual(max(math.prod(shape) for shape in score_shapes), 2 * 8 * 1 * 4)
        self.assertLessEqual(max(shape[1] for shape in mask_shapes), 8)
        self.assertNotIn((2, 9, 1, 9), mask_shapes)


class NormReproducerTests(unittest.TestCase):
    def test_actual_value_paths_preserve_axis_and_layout_semantics(self) -> None:
        torch.manual_seed(29)
        sequence_last = torch.randn(1, 9, 8)
        channel_input = sequence_last.transpose(1, 2).unsqueeze(2).contiguous()
        position_ids = torch.arange(9, dtype=torch.int32).unsqueeze(0)
        model = NormVariants(torch.rand(8), sequence_last.squeeze(0))
        outputs = model(sequence_last, channel_input, position_ids)
        for candidate in outputs[1:]:
            self.assertTrue(torch.allclose(outputs[0], candidate, atol=1e-6, rtol=1e-6))


class StageGateTests(unittest.TestCase):
    def test_only_a_passed_parity_gate_can_advance_to_a_longer_stage(self) -> None:
        self.assertTrue(stage_can_continue("passed"))
        self.assertFalse(stage_can_continue("parity_failed"))
        self.assertFalse(stage_can_continue("failed"))
        self.assertFalse(stage_can_continue(None))


class RopeTests(unittest.TestCase):
    def test_rope_matches_split_half_reference(self) -> None:
        query = torch.randn(1, 2, 4, 7)
        key = torch.randn(1, 2, 4, 7)
        cos, sin = tiled.build_rope_tables(7, 4, 10_000.0)
        actual_query, actual_key = tiled.apply_rope(query, key, cos, sin)

        def reference(value: torch.Tensor) -> torch.Tensor:
            conventional = value.permute(0, 1, 3, 2)
            half = conventional.shape[-1] // 2
            rotated = torch.cat((-conventional[..., half:], conventional[..., :half]), dim=-1)
            conventional_cos = cos.permute(0, 1, 3, 2)
            conventional_sin = sin.permute(0, 1, 3, 2)
            return (conventional * conventional_cos + rotated * conventional_sin).permute(0, 1, 3, 2)

        torch.testing.assert_close(actual_query, reference(query))
        torch.testing.assert_close(actual_key, reference(key))


if __name__ == "__main__":
    unittest.main()
