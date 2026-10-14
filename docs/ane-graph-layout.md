# Neural Engine graph layout

This page explains the tensor construction in
`bench/spikes/ane-direct-probe/src/bin/modernbert_full.rs`. It is a graph-layout
reference, not a hardware performance report or a release-certification claim.

## Tensor axes

The graph uses NCHW tensors. For a single sequence, hidden features occupy the
channel axis and sequence positions occupy the width axis; height is one.
The flat offset for channel `c` and position `w` is `c * width + w`.
Changing an axis requires changing both the graph shape and host-side packing.
A reshape preserves element order; it does not perform a transpose.

Projection weights and activation layouts must agree on which dimension is
contracted. A graph that compiles can still pair the wrong channels or positions.
Inputs with distinct values across both axes are needed to detect that mistake.
The probe keeps the independent CPU reference separate from graph construction.

## Attention layout

Queries, keys and values are split into heads. Attention scores retain separate
head, query and key dimensions: each query is normalized across its allowed keys,
not across heads or other queries. Grouping heads into one tensor operation must
preserve that separation when reshaping and restoring the context tensor.

Local attention slices keys and values to the query tile's allowed neighbourhood
before scoring. The distance mask then excludes keys outside the exact local
window. Global layers retain the full key dimension. Tiling must never remove an
allowed key at a boundary or join unrelated sequences.

## Softmax reduction axis

The score layout `[1, heads, queries, keys]` is transposed to
`[1, keys, heads, queries]`, normalized on axis 1, then transposed back before the
value multiplication. The reduced dimension is still keys; only its tensor axis
changes. Normalizing a head or query dimension would compute different attention.

Transpose and reduction changes can alter floating-point rounding even when their
real-arithmetic equations agree. Keep the independent reference and accuracy gate;
compilation success or a matching output shape is not an accuracy proof.

## Layer grouping

An executable can contain consecutive layers when it preserves every attention,
normalization, projection and residual dependency. Intermediate tensors stay in
the graph; the last output must have the same logical layout as the next input.
The probe's sequence-512 path groups two layers, while longer sequences keep the
requested grouping. Requested and compiled grouping are recorded separately so
reports cannot silently relabel the executed graph.

The final pooling and normalization use the complete encoder output. Combining
layers is not permission to omit a layer, change a mask, or alter the readout.
Different grouping must still be checked against the unchanged CPU comparator.

## Verification boundary

Run the portable harness refusal tests from the repository root:

```sh
bash bench/campaign/ane-direct-embed-harness.sh --self-test .
```

This self-test protects comparator, input and scoring integrity. It does not
execute Neural Engine graphs; numerical qualification still needs supported
hardware, the pinned checkpoint and every unchanged reference row.
