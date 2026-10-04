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
after earlier shapes had been cleared. This was a real transient load failure,
not an authorized model drop.
Subsequent fresh release processes retained all 22 layer executables at each
of 2048, 4096 and 8192 successfully. Layer 14 also loaded alone at 4096 and 8192.
The old failure is therefore not an established per-program or executable-count
ceiling. See [the resource measurements](resource-experiments.md) for each
executable's payload, I/O allocations and machine load.

Warm 512-token gte embedding latency is now 26.800ms after batching the exact
fp32 rotation matrices on CPU Accelerate, instead of scalar per-token dense
loops. See [the before/after stage table](latency-stages.md); transformer graphs
and cached ANE requests are unchanged.

A load-specific `no ANE resources` failure drops all previously loaded layer
executables and returns `ane_resources_exhausted` without admitting the shape.
The module supervisor then evicts its lane-wide LRU unleased shape and retries
once; another exhaustion (or no unleased victim) becomes a transient refusal
with a 250ms retry delay. For other admission errors the module still restarts
the worker and forgets every shape previously counted for that worker.

The worker still has no passing simultaneous-request stress result. That test
must complete 28 shape/model admissions, eight two-rung embedding requests and
one three-rung reranker pool through FIFO admission and unleased eviction. All
outputs must be finite, reranker repeats must be byte-identical, every residency
sample must stay within four shapes per model and eight overall, and no leased
eviction, `shape_not_admitted` or `no ANE resources` error may occur. Numerical
parity remains a separate check; no stress success artifact is fabricated.

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

## Development integration tests

`tests/fixtures/direct-ane-padding.json` pins post-padding ids and additive masks
for all four models, including a real token whose id equals the pad id. Its
independent generator uses only the shared token fixtures, not model inference.
The worker consumes the same padding helper checked by this golden test.

The module's ignored hardware tests run real release workers through the
production handshake, inherited-lock spawning and residency supervisor. Build
the worker with `cargo build --release --locked -p synapse-worker-ane-direct`.
Set `ANE_TEST_WORKER` to its absolute path and `ANE_TEST_PACKAGES` to the four
converted-package directory. Run the killed-holder test with:

```
cargo test --locked -p synapse-module killed_holder_with_resident_work_keeps_lane_busy_until_old_worker_exits -- --ignored --nocapture
```

Run the 37-request stress harness with:

```
python3 crates/synapse-worker-ane-direct/tests/run_stress.py --wait-for-load --timeout 7200 --out crates/synapse-worker-ane-direct/evidence/stress-dev.json
```

Omit `--out` to use a temporary JSON file. The Rust test validates the committed
development schema and refuses to start if the one-minute load is at least16.
The report records machine identifiers, OS build, source commit, harness profile
and all three load averages. The test driver may use debug mode (the unrelated
owned-decode release build requires the Metal developer toolchain), but the
spawned direct-ANE workers use the release binary. It is development evidence, never release certification.
Reranker pools at128/512/2048 must be finite with byte-identical repeats. The
Metal ordering comparison is explicitly skipped when that lane/package is not
available; a2048-rung parity fixture remains a parity-crate follow-up, not a new
worker-authored model reference generator.
