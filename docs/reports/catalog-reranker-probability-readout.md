# Native catalog Metal reranker probability readout

## Reproduction

The task-base revision `f1149e55ea631e70496d4310e1e32a50023ee408` has identical
serving-module source, owned-engine source, and catalog data to the provided
signed release at `d08bf55c4d5bc479baa7729bf6891fed5e24fbe2` used for reproduction
(verified with `git diff --exit-code` on those paths). That release served the
catalog lane `gte-reranker-modernbert-base-metal` as raw logits under pin
`6da1fa7f1dfa0c7c883dd34ca9c3e5d06e50ceaaabbb13d49fd7b7b291b0b9b9`.
The ignored `catalog_metal_rerank_fixture_readout_probe` downloads from a local
pinned mirror, loads through the catalog, and calls `rerank.score` with the first
64-candidate pool request used by certification. It uses a development binary
alias and an isolated store and lease root. Its comment documents the command.

The reference is the sealed CPU-fp32 parity fixture, not a transformed expected
value computed from the observation. All four sampled composed `input_ids`
matched the fixture exactly.

| Case | Observed logit | Sealed probability | Absolute error after sigmoid |
|---|---:|---:|---:|
| pool100-000 | 3.3118042945861816 | 0.964831531047821 | 2.38e-8 |
| pool100-007 | 3.7117819786071777 | 0.976148784160614 | 5.02e-8 |
| pool100-042 | -0.07958893477916718 | 0.4801078736782074 | 5.39e-6 |
| pool100-063 | 0.104514479637146 | 0.5261048078536987 | 5.38e-8 |

For `pool100-007`, the untransformed error is `2.7356331944465637`, reproducing
the certification refusal. The sampled token identities and sigmoid errors
rule out a query/document template mismatch as the explanation for that error.

## Decision points

The owned engine looks up runtime `profile` in
`crates/synapse-engine-owned/src/lib.rs:503-508`; at lines 572-575 it selects
`load_catalog_family` when present, otherwise `load_model_family`.
Native Metal catalog lane construction does not install a `profile` flag,
so the engine takes the ordinary loader and retains raw-logit output.

`crates/synapse-engine-owned/src/modernbert.rs:167-170` calls
`load_with_config(path, precision, config, false)` for the ordinary loader;
`load_catalog` at lines 1421-1427 passes `true`. The classifier score readout
at lines 416-424 uses that `catalog_readout` boolean to select sigmoid or raw logits:

```rust
if self.catalog_readout {
    1.0 / (1.0 + (-logit).exp())
} else {
    logit
}
```

The native catalog self-check in `numerical_catalog_check` separately applies
`1.0 / (1.0 + (-f64::from(*x)).exp())` to the raw head result. Before this fix,
ordinary serving returned the engine scores unchanged. Consequently self-check
could pass while serving/certification compared logits with probabilities.

ANE-direct, CUDA, Vulkan, and profile-loaded Metal already transform their
catalog reranker heads into probabilities:

- ANE-direct: `backend.rs:492-514` applies sigmoid to the classifier logit.
- CUDA: `model.rs:992-1032` applies sigmoid to the classifier output.
- Vulkan: `runtime.rs:1546-1558` selects shader operation 13;
  `shaders/common.glsl:224-225` computes `1/(1+exp(-x[i]-float(w[0])))`.
- Profile-loaded Metal: `modernbert.rs:1421-1427` enables `catalog_readout`;
  the classifier readout at lines 416-424 applies sigmoid to the raw logit.
- Reference: `bench/parity/reference/generate_reference.py:295-301` seals
  `sigmoid(logit)`.

## Fix and identity

The module's common rerank execution boundary now selects catalog probability
readout by reserved-lane ownership, independently of the optional profile flag.
For native Metal engine logits it applies sigmoid once; already-probabilistic
profile backends pass through. Non-catalog lanes pass through byte-for-byte.
The raw engine head and native raw-engine self-check are unchanged.

Only the native catalog reranker identity acquires
`rerank_readout=sigmoid_classifier_logit`. Its new pin, computed by the production
`catalog_native_lane_spec`/`build_stored_model_config` construction, is
`f6c27197647b9d6302971dcda763d23391d5c7e7d45e78606e826bdf0bac4c4d`.
This is an intentional numeric-profile change from the old pin above. All other
catalog pins remain unchanged. Legacy `gte-reranker-modernbert-base-f32` retains
`2fa5f24c0208f30c6db4bf18eb66bc0b2c46f765882cb744fcc58cd080b2b92d`.

Portable module tests replay the independently captured native head bytes for
fixed fixture token inputs through the real execution boundary. They verify the
legacy bytes/fingerprint, all six catalog rerank lanes across Metal/ANE/CUDA/Vulkan,
and numeric fingerprints computed through the production lane constructor. The
ignored Metal engine test replays
that same catalog checkpoint through the ordinary legacy loader and checks its
exact raw-head bytes; the production legacy fingerprint is pinned independently
by the frozen preload fixture. No production store or production checkpoint is
read. The unchanged portable Qwen3 ANE golden bytes and golden test guard against
accidentally changing an unrelated catalog backend's numeric identity.

## Release verification

A clean-stamped, signed release candidate built from fix commit
`6b9674c78863eee9a74896a96096d98830df7c40` passed the
`metal-m5/gte-reranker-modernbert-base` certification cell, covering all sealed
score/token fixtures, ranking pools, and sequence-length admission checks:

- `status=passed`, 125 of 125 fixtures complete; score, ranking, token semantics,
  finiteness, dimensions and 8192-token coverage pass (all record gates true);
- worst absolute score error `6.9141387939453125e-6`, below the fp32 `0.005` bound;
- the 10- and 100-document ranking pools have Kendall tau `1.0` and top ten
  matching the sealed fp32 reference rankings;
- 8192 tokens processed without truncation/diversion; 8193 refused before execution.

The short real-path probe also passed with `pool100-007` (candidate 7 in the
100-document pool for the Vulkan GPU-selection query) returning exactly
`0.976148784160614`, equal to its sealed reference (absolute error zero).
The normalized certification record is retained at
`docs/evidence/certification/6b9674c78863eee9a74896a96096d98830df7c40/metal-m5/gte-reranker-modernbert-base.json`.
Only the structured record is committed, not raw runtime logs or machine paths.
