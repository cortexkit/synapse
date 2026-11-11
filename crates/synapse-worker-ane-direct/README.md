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

The module's ignored hardware tests run real workers through the
production handshake, inherited-lock spawning and residency supervisor. Build
the worker with `cargo build --locked -p synapse-worker-ane-direct`.
Set `ANE_TEST_WORKER` to its absolute path and `ANE_TEST_PACKAGES` to the four
converted-package directory. Run the killed-holder test with:

```
cargo test --locked -p synapse-module killed_holder_with_resident_work_keeps_lane_busy_until_old_worker_exits -- --ignored --nocapture
```

Run the 37-request stress harness with:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer python3 crates/synapse-worker-ane-direct/tests/run_stress.py --wait-for-load --timeout 28800 --out crates/synapse-worker-ane-direct/evidence/stress-dev.json
```

Omit `--out` to use a temporary JSON file. The Rust test validates the committed
development schema and refuses to start unless load1<16 and load5<20. The
launcher polls `sysctl -n vm.loadavg` every60 seconds; `--driver <path>` runs an
already-built supervisor test executable without invoking Cargo after the wait.
The report records machine identifiers, OS build, source commit, harness profile
and all three load averages. By default the driver and workers use debug builds;
set `ANE_TEST_WORKER` explicitly to measure a different worker build. Full Xcode
must be selected with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
so dependencies compile their real Metal kernels. After changing that setting,
run `cargo clean -p synapse-engine-owned` to invalidate an earlier stub build. The generated stress report provides development evidence, not
release certification.
Reranker pools at128/512/2048 must be finite with byte-identical repeats. The
Metal ordering comparison is explicitly skipped when that lane/package is not
available; a2048-rung parity fixture remains a parity-crate follow-up, not a new
worker-authored model reference generator.

## Parakeet transcription hardware gate

The opt-in `parakeet::hardware::certify_parakeet` test runs the real 24-program
FastConformer encoder, accelerated encoder projection, and shared-core token-and-
duration transducer (TDT) loop.
The hardware owner must arrange exclusive Neural Engine access. The test takes
the production `~/Library/Caches/ck-synapse/ane-direct.lock` nonblocking and
refuses while any cooperating ANE client holds it; it never uses a private lock.
Do not stop another client or bypass the lock to run the gate.

Build before opening the exclusive hardware window (full Xcode is required):

```sh
bash crates/synapse-worker-ane-direct/tests/run_parakeet.sh build
```

The `probe` mode compiles and releases independent subsampling stages, complete
subsampling, conformer components and one complete block, the encoder projection,
and the complete first encoder layer. It never submits an inference request. Each candidate prints its
operation count (excluding constants), NCHW input/output shapes, and pass/fail;
failed candidates do not stop the remaining probes. Probe mode takes the same
machine-wide lock as inference and needs only `ANE_PARAKEET_SNAPSHOT` and
`ANE_PARAKEET_REPORT` below. After building, run it during the owner's window:

```sh
unset TMPDIR
bash crates/synapse-worker-ane-direct/tests/run_parakeet.sh probe
```

Independent stages whose frequency width is below the binding's 64-column
input minimum also get explicitly labeled padded-width probes. The padded
variants test operator support, not the exact production shape. The complete
subsampling and first-layer probes retain their original shapes.

`probe-attention` narrows the compile-only run to the complete attention graph
and seven independent attention pieces: QKV projections; content scores with
softmax and values; dynamic and constant position projections; position-score
matmul; relative shift; and the query/key mask with the -10000 sentinel. The
constant position-projection probe adds a dynamic input only at its output to
test that constant branch inside a valid, input-bearing ANE program.

`probe-corrected` compiles the static relative-score gather, corrected attention,
one complete corrected conformer block, and corrected layer 0. The gather uses
per-query compile-time slice bounds and small, bounded-fan-in concatenations.
It stores no dense selection matrix and retains the shared relative-position
table, so selection metadata grows linearly with encoder frames rather than
quadratically. The encoder still uses exactly 24 executables per shape. This
changes internal graph operations, not the number of worker dispatches; latency
at the maximum duration must still be measured on hardware. The original
pad/reshape/slice shift remains available for diagnostic comparison in `probe-attention`,
not in serving or in the corrected probe group.

Set the asset roots and a retained report path, then run smoke first:

```sh
export ANE_PARAKEET_SNAPSHOT="$HOME/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
export ANE_PARAKEET_MIRROR="$PWD/bench/parity/.cache-out"
export ANE_PARAKEET_WAVS="$PWD/target/scratch/parakeet-wavs"
export ANE_PARAKEET_REPORT="$PWD/.cortexkit/internal-docs/investigations/ane-parakeet-smoke.json"
unset TMPDIR
bash crates/synapse-worker-ane-direct/tests/run_parakeet.sh smoke
export ANE_PARAKEET_REPORT="$PWD/.cortexkit/internal-docs/investigations/ane-parakeet-full.json"
bash crates/synapse-worker-ane-direct/tests/run_parakeet.sh full
```

The kit converts the pinned checkpoint to a checksum-verified ANE package on
first use, retaining it under `target/scratch/ane-parakeet-kit/`. Alternatively
set `ANE_PARAKEET_PACKAGE` to an existing converted package; loading still checks
its manifest digest. Tensor artifacts and prepared WAVs are checksum-checked.

Full mode grades the encoder's minimum per-frame cosine, all ten word error rate
(WER) splits, and the reference trace rule: frame indices and token/duration
argmax decisions must match, except a changed argmax at the first differing step
is exempted only when the frame index agrees and each differing head's reference
top-1 minus top-2 logit margin is below the profile's `tau_margin`. Frame-only
or trace-length differences fail. Emitted triples
(token id, encoder frame index, duration) must also match the decisions implied
by the engine's entire trace. Every stored-tensor fixture
also checks bit-identical triples and valid frame counts with observation on
and off, in the order off/off/on. Each pair records frame and triple counts,
the first differing triple index with both triples, and encoder/projection
readback bit equality and maximum absolute differences. These are read-only
copies after inference. Encoder and trace gate outcomes are recorded even if
the repeat comparison fails. Reports include 24-executable inventories, peak live executables
(at most four shapes / 96 programs), completed-work progress gaps including the
terminal interval, and the largest passing full-attention ladder duration.
Smoke runs only three short fixtures and is not certification. To try a smaller
envelope, set `ANE_PARAKEET_MAX_DURATION_MS` to a ladder duration present in the
fixture set; full mode
still requires coverage of every non-ladder fixture. No maximum is certified
unless the entire full run succeeds. These are engine development measurements,
not a substitute for the release runner's end-to-end module-path certification
(see [backend certification](../../docs/backend-certification.md)).

For numerical diagnosis, every stored-encoder case reports all per-frame cosines,
their minimum/mean/quantiles/maximum, and counts below 0.999 and 0.9999. Optional
`ANE_PARAKEET_LAYER_DUMP_DIR` captures completed encoder layers as raw row-major
f32 tensors without changing the graph. `ANE_PARAKEET_LAYER_REFERENCE_INDEX` can
name a checksum-indexed, pinned upstream CPU layer reference for one fixture;
the kit compares every layer and reports the first stage below 0.9999. It also
runs an isolated subsampling diagnostic, releasing that extra executable before
the 24-program encoder runs. The peak count of simultaneously live diagnostic
and encoder executables is reported separately and must remain within the
100-executable budget. Unset these variables for duration/performance measurements.
The small padded-attention unit fixture comes from the actual transformers
5.16.1 positional-encoding and attention classes with deterministic diagonal
projections, both position biases, and the pinned mask fix (a finite sentinel
before softmax and zeroed masked probabilities afterwards). Its scalar row-major replay
checks the lowered position table and gather against an independent upstream
output rather than only comparing two versions of the worker's shift formula.

`component-probe` diagnoses late-layer arithmetic and masking using a
checksum-indexed bundle of CPU reference input/output tensors at component
boundaries, named by
`ANE_PARAKEET_COMPONENT_REFERENCE_INDEX`. It first runs the normal encoder and
captures full layer-21/22 outputs, including padded positions, under
`ANE_PARAKEET_LAYER_DUMP_DIR`. Each isolated layer-22/23 normalization,
feed-forward, attention, convolution, and full-block program then runs on the
same CPU seam input. Masked components additionally replace only the padded
rows with the actual ANE predecessor's padded rows, or with zeros. The valid
rows are identical across these controls. Reports include pairwise output
differences and each frame's index and zero-based distance from the valid end.
A separate readback checks the three-stage downsampled mask against the exact
valid encoder prefix. Only one temporary program is live beside the normal
24-program shape, so the peak is 25 executables, within the budget of 100.
This is an owner-run diagnostic, not a certification result; the machine-wide
Neural Engine lock and explicit opt-in are still mandatory.

The opt-in runner is collect-all: a numeric mismatch is recorded and the next
fixture still runs. Every available transcript contributes to its split's WER,
including cases whose encoder or trace gate fails. `wer_details` includes raw
word/edit counts, both WERs, their percentage-point delta, expected/observed case
counts, and explicit incomplete status. Trace findings retain the first differing
step and margins, total differing steps, and up to 16 differing-step examples.
They also check that emitted (token id, frame index, duration) triples exactly
match the non-blank decisions in the engine's entire trace. Encoder distributions are available wherever a
sealed reference encoder tensor exists; other cases still require finite output
and the exact valid-frame count. Encoder, trace, WER, progress, inventory and
observation-parity gates are unchanged, and any failure keeps the final test red.
No maximum duration is certified by a run containing failures.

Each completed case is appended immediately to a `.cases.jsonl` receipt beside
`ANE_PARAKEET_REPORT`. The complete JSON is checkpointed every 25 cases and at
shape admissions, so repeated serialization does not dominate inference. Case
records include admission, preparation, encoder and host-decoder timing, as well
as every measured progress gap. Per-rung summaries retain cache-hit/cold-compile
counts and estimate the full run using measured encoder time and sealed joint/
predictor work. Missing rung measurements are reported as unknown, not zero.

Run `timing` before sizing an owner window. It selects the largest in-envelope
ladder fixture per encoder rung plus one non-tensor corpus fixture for each
populated corpus rung. The latter measures actual frontend work rather than only
reading sealed mel tensors. `ANE_PARAKEET_TIMING_MAX_RUNG=512` limits an initial
pilot to smaller rungs; 1024 covers every corpus shape class, including two
utterances above 40 seconds. Omitted large-rung costs remain unknown. Full and
timing modes unset layer-dump diagnostic variables to avoid contaminating latency
measurements. With the configured 600000 ms envelope (not a certification claim), full mode processes 1244
in-envelope cases and checks three above-limit ladder refusals. The 76 in-envelope
stored-tensor cases run twice with observation disabled, then once with it
enabled, giving 1396 engine calls in total. A timing
pilot is not certification, and it still returns failure for unchanged gates
that do not pass.
