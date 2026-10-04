# Direct ANE worker (experimental)

`ck-synapse-worker-ane-direct` implements the v2 supervised Unix-socket worker
boundary using the vendored private Apple Neural Engine binding. Other platforms
build a typed-refusal stub without the private framework dependencies. Private
framework resolution occurs at `--probe-floor` or LOAD, not HELLO/PING.

LOAD checks the manifest profile, operation, package digest, required heads and
tensor dtypes and stores weights without compiling. Shape admission compiles
one fp16 executable per transformer layer. Eviction drops all of that shape's
executables before acknowledging. Requests are serialized; inference cannot race
an eviction. The module owns cross-worker admission/lease coordination. The
worker retains inherited stdin (the module's lane-lock descriptor) and watches
supervisor EOF during compilation and execution.

ModernBERT uses the converted Hadamard weights and fp32 rotation projections;
centering is performed in the original basis at rotation_in, not in the rotated
basis. Masks use the composed token count, not equality with the pad id: Qwen's
terminal EOS has the same id as its padding token and must remain attended.
Transformer computation stays in ANE graphs; pooling, final normalization and
the two reranker heads are fp32 CPU stages.

## Current hardware limitation — not release ready

On an Apple M5 Max, isolated one-model admission succeeds at 128, 256, 512, 1024
and 2048 for gte-modernbert-base. At 4096, compiling/loading layer 14 fails with
`Program load failed — no ANE resources (transient; retry) (underlying=0x5)`.
This is a measured failure of the required ladder, not an authorized model drop.
The worker therefore does **not** yet meet the required seven-rung admission
and simultaneous-request stress criteria with up to four resident shapes per
model and eight overall.
It needs a graph/executable resource reduction before certification. No stress
success artifact is fabricated.

Four-model short-input real-weight parity at 128 was measured against the
committed fp32 fixtures:

| Model | Result | First inference milliseconds, excluding compile (debug build) |
|---|---|---:|
| gte-modernbert-base | cosine 0.999562587 | 1068.531 |
| gte-reranker-modernbert-base | score 0.96428573; expected 0.9641748070716858; absolute error 0.000110924 | 1394.890 |
| qwen3-embedding-0.6b | cosine 0.999903201 | 29.424 |
| qwen3-reranker-0.6b | score 0.9999784; expected 0.999980926513672; absolute error 0.000002503 | 32.431 |

These short-input measurements are not all-row certification or a latency
comparison against Metal. The four upstream checkpoints and converted packages
were hash-checked against the manifest before the test; packages are not stored
in git. The model-backed tests are ignored in ordinary CI and explicitly require
`ANE_TEST_PACKAGES` to point to the four converted packages:

```
ANE_TEST_PACKAGES=../../target/ane-direct-packages cargo test -p synapse-worker-ane-direct real_weight_short_parity_all_four_profiles -- --ignored --nocapture
ANE_TEST_PACKAGES=../../target/ane-direct-packages cargo test -p synapse-worker-ane-direct real_weight_admission_all_ladder_shapes -- --ignored --nocapture
```

The second command currently fails at the measured resource limit. It must go
green before implementing and running the full simultaneous supervisor stress
experiment. The separate killed-holder/process-replacement experiment and
full padded golden fixture comparison are also still required.
