# Direct ANE worker (experimental)

`ck-synapse-worker-ane-direct` implements the v2 supervised Unix-socket worker
boundary using the vendored private Apple Neural Engine binding. Other platforms
build a typed-refusal stub without the private framework dependencies. Private
framework resolution occurs at `--probe-floor` or LOAD, not HELLO/PING.

Do not redirect `TMPDIR`: Apple's private compiler requires it to be unset or
exactly the user's `DARWIN_USER_TEMP_DIR` (`getconf DARWIN_USER_TEMP_DIR`). A
worktree directory or even a subdirectory of the per-user temp directory fails
with `verifyBundleAtPath: invalid model`. The worker leaves `TMPDIR` unchanged.

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

An initial debug ladder run on Apple M5 Max admitted 128 through 2048, then
failed loading layer 14 at 4096 with
`Program load failed — no ANE resources (transient; retry) (underlying=0x5)`.
The ladder experiment dropped all earlier shapes before each admission; the
failure occurred while accumulating the current shape's layer executables, not
while retaining six shapes. This is a measured failure of the required ladder,
not an authorized model drop.
Subsequent fresh release processes retained all 22 layer executables at each
of 2048, 4096 and 8192 successfully. Layer 14 also loaded alone at 4096 and 8192.
The old failure is therefore not an established per-program or executable-count
ceiling. See [the resource measurements](resource-experiments.md) for each
executable's payload, I/O allocations and machine load.

The worker still has no passing simultaneous-request stress result. That test
must complete 28 shape/model admissions, eight two-rung embedding requests and
one three-rung reranker pool through FIFO admission and unleased eviction. All
outputs must match their reference fixtures, every residency sample must stay
within four shapes per model and eight overall, and no leased eviction,
`shape_not_admitted` or `no ANE resources` error may occur. No stress success
artifact is fabricated and no resource-management redesign has been selected.

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

The recorded debug run of the second command failed at 4096; the later
fresh-process experiments do not establish a persistent resource ceiling.
The simultaneous-request supervisor test described above remains required.
The separate lock-lifetime test must kill the module while it holds the lane
lock and has resident shapes/in-flight work, then show that a replacement gets
`ane_lane_busy` until every old worker exits. That test and full padded golden
fixture comparison are also still required.
