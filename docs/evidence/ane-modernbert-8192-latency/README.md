# GTE ModernBERT ANE full-context latency investigation

This investigation tested whether the 8192-token Core ML export was accidentally computing full sequence-square attention in ModernBERT's local layers. It was measured on 2026-09-08 on the same M5 Max (`Mac17,6`), macOS 27.0 build `26A5425a`, model snapshot, rotation seed, and toolchain as the rotation-conditioning evidence. `EVIDENCE.json` contains the machine-readable measurements.

## Source result: the proposed missing band is already present

The checkpoint says `global_attn_every_n_layers=3`, `local_attention=128`, `num_hidden_layers=22`, and `num_attention_heads=12`. Transformers 5.16.1 converts the legacy global interval into eight `full_attention` layers at indices 0, 3, 6, 9, 12, 15, 18, and 21, with 14 `sliding_attention` layers. Its `sliding_window` property is `local_attention // 2`, or 64, and the model selects separate full and bidirectional sliding-window masks by layer type.

The spike follows that schedule in `ModernBertLayer`: indices divisible by three are global and all other layers receive the 128-token local window. More importantly, `query_tiled_attention` does not score the complete sequence in a local layer. For each query tile `[q0, q1)`, it slices keys and values to `[max(0, q0 - 64), min(S, q1 + 64))` before either einsum. An interior 256-query tile therefore scores a `384 x 256` rectangle per head, then masks that rectangle to the exact inclusive-radius band. A tile boundary never removes an allowed key.

The tested hypothesis—local layers first compute `S x S` scores and only then mask—was therefore false. Local attention in the existing export is `O(S * (256 + 128))`, not `O(S²)`. Eight genuinely global layers still have an `O(S²)` term and retain every key for every query as the model requires. The reference source checked for this conclusion was Transformers 5.16.1 [`configuration_modernbert.py`](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/modernbert/configuration_modernbert.py#L113-L122) and [`modeling_modernbert.py`](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/modernbert/modeling_modernbert.py#L250-L255).

## Existing scaling curve (before)

These are the accepted rotation-conditioned measurements, not extrapolations. Every row passed the unchanged 0.999 minimum-cosine gate and kept all expensive operations ANE-preferred.

| Stage | Warm predict | Load | Cosines by row | Preferred CPU / ANE | ANE einsum / softmax |
|---:|---:|---:|---|---:|---:|
| 1024 | 32.638 ms | 36.756 s | 0.9999590 / 0.9999255 / 0.9999814 | 13 / 11,660 | 2,112 / 1,056 |
| 2048 | 97.750 ms | 152.322 s | 0.9999825 / 0.9999255 / 0.9999814 | 21 / 20,712 | 4,224 / 2,112 |
| 4096 | 417.119 ms | 190.534 s | 0.9999752 / 0.9999255 / 0.9999814 | 37 / 38,816 | 8,448 / 4,224 |
| 8192 | 1553.157 ms | 735.348 s | 0.9999038 / 0.9999255 / 0.9999814 | 69 / 75,024 | 16,896 / 8,448 |

Warm latency changes by 3.00x, 4.27x, and 3.72x at successive doublings. Attention dispatch count itself is linear: with 256-query tiles, softmax count is exactly `22 layers * 12 heads * ceil(S / 256)`. The shape of each global score grows with `S`, however, so the eight global layers supply the quadratic work even while operation count grows linearly.

## Same-session 1024 control

A fresh unchanged 256-query-tile package was built after the experimental arms to control for machine state. Its five warm predictions were 32.765, 32.697, 32.647, 32.677, and 32.645 ms (median 32.677 ms); load was 34.664 s. The three row cosines were 0.9999590, 0.9999255, and 0.9999814. MLComputePlan preferred 11,660 operations on ANE and 13 on CPU, including all 90 convolutions, 2,112 einsums, 1,056 softmaxes, and 45 norm reductions on ANE. This reproduces the original 32.638 ms result closely.

## Negative 1024 arms

Only 1024 results appear in this table. None is an 8192 measurement. Each convertible arm passed all three parity rows and kept every expensive operation ANE-preferred, but each lost the 1024 latency screen.

| 1024 arm | Warm median | Change vs control | Load | Cosines by row | Preferred CPU / ANE | Expensive attention placement | Result |
|---|---:|---:|---:|---|---:|---|---|
| Control, tile 256 | 32.677 ms | — | 34.664 s | 0.9999590 / 0.9999255 / 0.9999814 | 13 / 11,660 | 2,112 einsum + 1,056 softmax on ANE | keep |
| Tile 128 | 52.950 ms | +62.0% | 96.773 s | 0.9999590 / 0.9999255 / 0.9999778 | 21 / 20,712 | 4,224 einsum + 2,112 softmax on ANE | reject |
| Tile 512 | 40.894 ms | +25.1% | 25.650 s | 0.9999590 / 0.9999255 / 0.9999764 | 9 / 7,134 | 1,056 einsum + 528 softmax on ANE | reject |
| Group all 12 heads in local layers | 46.510 ms | +42.3% | 24.564 s | 0.9999590 / 0.9999255 / 0.9999814 | 13 / 4,968 | 768 einsum + 112 matmul + 440 softmax on ANE | reject |
| Group pairs of heads in local layers | 38.338 ms | +17.3% | 30.139 s | 0.9999590 / 0.9999255 / 0.9999814 | 13 / 10,484 | 768 einsum + 672 matmul + 720 softmax on ANE | reject |
| Reuse masks across heads | 37.101 ms | +13.5% | 49.403 s | 0.9999590 / 0.9999255 / 0.9999814 | 13 / 11,660 | 2,112 einsum + 1,056 softmax on ANE | reject |

The dispatch experiment is especially informative. Grouping all local heads reduced the MIL graph from 30,397 to 12,365 operations and materially improved load time, but Core ML lowered the grouped local contractions to larger batched matmuls. Warm prediction became 42.3% slower instead of faster. Grouping pairs gave the same direction with a smaller penalty. Fewer plan operations are therefore not sufficient on this shape; the per-operation tensor shape and lowering matter.

The tile-size arms establish a local optimum among the tested values. Tile 512 improved load time by 26.0% but hurt warm latency by 25.1%; tile 128 hurt both. Reusing masks produced the same lowered operation and placement counts as the control, so the compiler already removed that source-level repetition and the measured package did not improve.

A fourth source prototype used `Tensor.unfold` to form each query's exact 129-key offset window, eliminating even the masked-away portions of the `384 x 256` local rectangle. Eager parity passed, but coremltools 9.0 rejected conversion with `Unsupported fx node unfold, kind unfold`. It never loaded or predicted, so it has no latency or placement result and is not presented as one.

## Stage decision and conclusion

No candidate advanced to 2048. Consequently there is no accepted “after” value at 1024, 2048, 4096, or 8192; the unchanged before curve remains the result. This follows the staged method rather than spending 4096/8192 conversion and load time on changes already slower at 1024. The prototype source changes were reverted, while the unchanged rotation conditioning and 0.999 gate remain intact.

The evidence refutes the initial mechanism and two obvious pivots. Local layers are already banded before score computation. Reducing dispatches or changing query-tile size can improve package load, but the tested forms increase warm latency without moving expensive work to CPU. Further work should target a Core ML-supported sliding-window primitive or the eight global layers' kernel/layout cost; it should not claim a missing local band or retain one of these regressions.
