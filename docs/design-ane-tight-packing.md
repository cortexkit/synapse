# Serving direct Neural Engine embeddings with tight packing

## Status and scope

**Design for approval; not an implemented serving policy.** This change is an
execution optimization for the pinned Qwen3-Embedding-0.6B direct-ANE lane, not a
new embedding space. It does not enable other models, the batch-axis layout,
384/512-column packing, cross-request batching, or a larger machine-wide ANE
budget. This report was prepared from source and committed evidence at
`02d9efba98474ef58ef7abc0012d79a68995257b`; no hardware run was performed.

Approval has two parts: approve the packing/residency/error contract below, then
enable it on a device only after the correctness and serving-performance gates.
The M4 prototype results justify integration, not an unconditional production
speed claim on the M5 Max.

## Approval decisions

Approved with these rulings, which take precedence over any text below that
reads otherwise:

1. **Recommendations 1-4 are approved:** worker-local packing within one engine
   call, distinct packed shape keys inside the single 100-executable budget, bit
   equality against standalone as the identity gate, and an eight-row first
   rollout. Recommendation 5 (64-row bulk look-ahead) is a separate later step;
   it starts only after the eight-row rollout passes its gates on both Macs.
2. **Packed 256 is prepared at lane load, not in maintenance windows.** After the
   lane's existing self-check succeeds, the worker admits the packed 256 shape
   through the supervisor and runs the exact packed-vs-standalone qualification.
   This happens off the request path and does not hold the lane lock. If the
   budget has no free capacity and no unleased victim exists, skip it and retry
   the next time the lane has no requests in flight. A request never waits on
   packed preparation; until it is qualified, rows run standalone.
3. **No public identity change.** The lane id, engine identity, numeric profile
   and fingerprint stay byte-identical. If implementation finds any hashed field
   must change, stop and escalate; do not hard-code the old hash.
4. **One off switch, no other knobs.** A module config switch disables packed
   dispatch and serves standalone with the same fingerprint. Fill thresholds,
   pass counts and widths are code constants backed by measurements, not config.
5. **Old workers refuse the packed mode.** The packed execution mode is added to
   worker protocol v2 so that a worker without it refuses the request by name;
   it is never interpreted as single-row or batch-axis execution.

## Evidence and its limits

The source prototype is
[`multirow.rs`](../crates/synapse-worker-ane-direct/src/multirow.rs), with tight
variable-length packing in
[`tight_packing.rs`](../crates/synapse-worker-ane-direct/src/tight_packing.rs),
replay in
[`multirow_replay.rs`](../crates/synapse-worker-ane-direct/src/multirow_replay.rs),
and the encoder audit in
[`tight_audit.rs`](../crates/synapse-worker-ane-direct/src/tight_audit.rs).
Nothing in serving calls the prototype today. Equal-width slots and tight
segments are different graphs; the recommendation is the latter.

Source anchors for the serving baseline:

- `crates/synapse-module/src/routes.rs` is absent at this revision. The intended
  routing policy is **`src/lib.rs::recommended_batch_for_engine`** (lines
  2222–2243), not a missing shim. Its direct-lane advice uses the shared eight-row,
  3,072-token constants (lines 215–216). `plan_embedding_engine_batches`
  (11908–11932) length-sorts and enforces the row cap; bulk execution also clamps
  the token quantum (11803–11805) and yields between calls (11889–11892).
- `worker.rs::sequences` (351–387) runs each row serially. The worker request loop
  (245–271) admits/evicts synchronously and ACKs eviction after release.
- `worker_host/mod.rs::ane_residency`: budget/admission/error policy
  (3431–3441), integer shape key/slot accounting/soft-pin preference
  (3747–3813), and `AneServing::infer_permitted` (5925–6005) are the integration
  points. Serving already groups by published rung, leases one rung at a time,
  and detaches I/O with lifetime guards. Packed grouping must cross those
  single-rung groups *within the same call*, or 128- and 256-rung rows cannot
  fill the same tight pass.

Read the [worker capacity/reclamation evidence](../crates/synapse-worker-ane-direct/evidence/)
and the [lane profile](evidence/ane-lane-profile/README.md) alongside the
[tight-packing report](evidence/ane-tight-packing/README.md) and its
[raw audit](evidence/ane-tight-packing/raw/tight-audit.json). The lane profile's
cold-compile incident describes the older locking behavior; it is not a reason
to reintroduce compilation under the serving lane lock.

| Observed on the M4 mini | Result | Qualification |
| --- | --- | --- |
| Tight 256, 64-row planning windows | 101.12 s; 2,877 passes; 92.2% fill | All 6,341 export rows, 679,090 composed tokens, original 127 metadata batches |
| Tight 256, 8-row windows | 116.24 s; 3,317 passes; 80.0% fill | Windows split within each metadata batch; tails never merged across calls |
| Paired single-row baseline for tight 256 | 155.23 s | About 155.7 s in the earlier planner experiment; compare paired arms, not rounded numbers across runs |
| Corrected tight output vs natural-rung single-row execution | 6,341/6,341 identical; max absolute difference 0.0 | Every accepted replay; the export contains no row over 256 tokens |
| Mixed full 256-column runtime pass | 35.41 ms | Includes mask/RoPE construction, staging, dispatch, readback and normalization; not an RPC p99 |
| Tight 384 / 512, 64-row windows | 121.43 / 133.27 s | Slower than 256; 512 at eight rows takes 159.58 s, slower than its baseline |
| Nearest-rounded runtime RoPE | Failed standalone parity on row 39 | Must use the graph-constant truncating encoder, not ordinary runtime conversion |

Eight-row slicing leaves a substantial gain: **1.34×** throughput over its paired
serial baseline. Moving from eight to 64 improves throughput another **1.15×**
(15.12 s less export wall time), not another 1.54×. The measured 1.54× is the
64-window gain over serial. All these inference timings exclude installation,
tokenization, compilation and parity comparison; they do include planning and
CPU output work. They have no in-wall evictions, other direct models, module IPC,
client deadlines or concurrent serving. They do not establish call p50/p99.

The M5 Max [capacity evidence](../crates/synapse-worker-ane-direct/evidence/capacity-development.md)
supports a shared executable pool and conservative **100 live reservations**,
not larger packed throughput. Compiler payload bytes are not executable RSS.
The [pooled reclamation evidence](../crates/synapse-worker-ane-direct/evidence/live-budget-pools-development.md)
reports a failed 37-way debug cold flood: 29 admission-deadline failures, although
no hardware resource refusals remained. It is not a successful release stress
test. Keep bounded admission and process-exit recovery; packing does not fix
cold-start queue overload.

## Recommendations, ranked by confidence

1. **High — pack within one engine call in the worker.** Never collect rows from
   independent requests. Reuse tight 256, runtime block-causal masks, row-local
   rotary positions and constant-compatible fp16 coefficients. Keep the existing
   standalone numerical path and exact input composition.
2. **High — account packed programs as distinct resident shapes.** Packed 256
   and single 256 each cost 28 executables; neither substitutes for the other in
   inventory. Keep the global 100 budget, leases, acknowledged eviction and
   detached cancellation drain. Soft-pin single 128, not the packed shape.
3. **High — exact equality is the identity requirement.** Retain the lane and
   engine numerical identity only if every fingerprint input stays unchanged
   and exact packed-vs-single gates pass. A cosine threshold alone cannot justify
   saying that AFT needs no re-embedding.
4. **Medium — ship first with the existing eight-row scheduling quantum.** It
   isolates the correctness/residency integration from a fairness change and
   already has a measured whole-export benefit. A single short row still uses
   single 128 rather than paying for a 256-column pass.
5. **Medium, conditional — raise the direct lane's bulk planning cap to 64 rows,
   not its interactive scheduling quantum.** Preserve deadline checks and yields
   between bounded execution quanta. Roll out this cap only after the memory,
   identity-envelope and mixed-traffic p99 gates below; there is no evidence for
   making every direct call a non-preemptible 64-row RPC.
6. **Low — extrapolating an M4 latency or a wider layout to M5.** Do neither.
   Measure M5 separately. Unmeasured performance and admission limits remain
   findings in the final unresolved section, not assumed implementation facts.

## Placement, batching and latency

### Choose worker-local packing, not module-wide coalescing

| Placement | Benefits | Costs and risks |
| --- | --- | --- |
| Worker packs rows belonging to one engine call | Owns token IDs, model weights, IOSurfaces, masks, last-token gather and CPU tail; no extra IPC of activations; one owner/deadline and existing all-or-error response | Fill is limited by the call's planning window; cannot accelerate lone unrelated queries by combining them |
| Module gathers across concurrent calls | Could fill sparse calls and share layer dispatch cost even for unrelated one-row requests | Requires a batching-delay policy, per-row owners/deadlines, cancellation isolation, fairness, shared-pass lifetime and response scatter across requests; waiting to fill increases query tails; does not create parallel ANE streams |

**Choose the first.** The measured export gain is available without the second
scheduler. Tokenization/admission stay in the module. Numerical packing, runtime
operands, execution and unpacking stay in the worker. The module may use the same
pure lengths-only planner to identify required shapes and execution quanta; it
must not collect rows from another request, materialize activations or implement
a second independent numerical packer. Promote/share the deterministic planner
rather than maintain two algorithms that can disagree about capacity or order.

### Separate planning size from the execution quantum

An API batch, a planning window and a worker exchange are not the same thing.
Initial integration keeps **at most eight rows per engine call** and the current
token budget. It uses packing only when the packed program is resident and
qualified. Eight-row planning is the measured 116.24 s kernel replay opportunity,
not the 101.12 s result from planning up to 64 rows at a time.

After the eight-row rollout passes its gates, the proposed bulk policy for the
pinned Qwen3 direct-ANE lane is **64 rows and at most 8,192 composed
tokens of look-ahead within one logical call**, further clamped to the stored
profile's existing `max_sequences`/`max_batch_tokens` and ordinary admission
limits. The source-default certified envelope is 8,192 context tokens, 8,192
batch tokens, 3,072 micro-batch tokens and 64 sequences; a two-pass packed exchange
uses at most 512 tokens and 64 rows, inside those defaults. Deployments with
smaller stored limits use smaller windows/quanta, not a new numeric profile.

The current 3,072-token scheduling cap would often cut an average export window
well before 64 rows (the export averages about 107 composed tokens/row). Keep
that cap for initial integration. The follow-up decouples bounded look-ahead
(up to 8,192 tokens) from per-exchange accounting (at most 512 packed tokens),
without changing `jobs.bulk_quantum_tokens` or `inline` configuration. A retained
3,072-token look-ahead cannot claim the measured full 64-window fill/time; nor can
an 8,192-token window claim an unclamped replay when a metadata batch exceeds it.

Do not raise shared `MAX_ENGINE_BATCH_ITEMS` or
`DEFAULT_ENGINE_BATCH_TOKEN_BUDGET`: their planner/advice affects Metal, CUDA,
Core ML and reranking as well as direct embedding.
Advertising 64 without updating the direct-specific scheduler clamps is not an
implementation. Do not change clients' admission or the certified numeric
profile merely to obtain a larger look-ahead window. Expose the 64-row limit as
direct-Qwen **bulk** advice/selection, leaving interactive execution bounded. If
the existing advice schema cannot distinguish bulk planning from execution size,
retain eight-row advice until a capability-gated bulk hint exists; do not simply
return 64 to every client. Clients must submit the larger logical bulk call to
use its look-ahead; eight separate API calls are never merged by this design.

For 64-row look-ahead, compute **first-fit decreasing (FFD)**: sort by decreasing
composed length with stable original-index ties, then place each row in the first
256-column bin with enough space. Schedule the resulting **whole packed passes**,
rather than reslicing original input order into eight-row windows. Send at most
**two 256-column passes per worker exchange**
(512 real tokens maximum), retaining original-index mappings on the host. A
single pass may contain more than eight tiny rows, but never rows from another
call. The worker validates the declared pass boundaries against token lengths,
creates the actual packed buffers and runs them; do not let it recombine bins
across exchanges. At most 64 input rows can contribute to an exchange. Singles
and over-width rows use their own natural-rung exchanges. Release the stream,
execution permit and shape lease between exchanges and re-enter the existing
fair bulk scheduling path. Preserve each logical call's deadline and result
scatter state across these quanta. Scheduler completion accounting must count the
new execution quanta so it cannot finish before all planned passes have run.
This requires an explicit direct-only execution-plan/mode protocol extension,
not a scalar shape-hint trick.

Two passes are a proposed fairness bound, not a measured p99 guarantee. At the
M4's observed 35.41 ms/pass they suggest about 71 ms of packed compute plus RPC
work; at most one dispatched exchange still drains after cancellation. By
contrast, 101.12 s / 127 metadata batches is about 796 ms of average inference
work per metadata batch, **not** a 64-row p50/p99 or maximum. Holding the lane for
all of that work in one 64-row RPC would worsen head-of-line blocking. The
baseline export has up to 64 rows per metadata batch, not 127 identical full
64-row calls. Measure quanta and user calls separately.

Use standalone 128 for a <=128-token singleton instead of a sparsely occupied
256 pass. A singleton of 129–256 tokens may use qualified packed 256 (its isolated
case is part of the exact gate); if packing is unavailable, use single 256.
Never wait for unrelated rows to fill a pass. Do not introduce a guessed fill
threshold: evaluate sparse tails in the measurement matrix before adding one.

### Memory limits

Increasing the planning window does not multiply the fixed packed tensor width.
For the pinned hidden size 1,024 and head dimension 128, the prototype's persistent
fp16 inputs are two activation surfaces, one dense mask and two rotary tables:
`2 * 1024 * 256 * 2 + 256 * 256 * 2 + 2 * 128 * 256 * 2` = **1.25 MiB nominal**,
before any surface alignment. This is visible buffer payload, not the ANE's
opaque intermediate/compiler allocation. Fp32 staging/readback and the model's
embedding weights are additional.

Sixty-four 1,024-D fp32 output vectors are **256 KiB**, versus 32 KiB for eight.
Token storage is `4 * admitted_tokens` bytes before codec copies; retain the
certified token limit and bound in-flight windows through the existing request
admission. Do not buffer all 6,341 export rows in a worker or grow one window while
waiting for lane access. Keep one detached state per admitted call, release row
buffers/results promptly and count IPC/codec duplication in peak RSS tests.
Compilation/resident weights and concurrent pending calls can dominate these
small arithmetic bounds. No measured packed RSS or 64-call memory safety on the
M5 is claimed; the 64-row rollout needs that evidence.

## Residency and cold-shape behavior

### Distinct program identities, one budget

Extend the residency and worker protocol shape key to distinguish
`Single { width }` from `TightQwen { width: 256, program_revision }`, scoped to the
same model/package owner. The internal program revision prevents reuse of stale
packed programs; it is not a new public embedding engine. Update admission,
inventory, inference selection, leases, recency, eviction, recovery and profiling
together. An integer `256` must never name both kinds. A capability-negotiated
packed execution mode must be refused by an old worker, not silently interpreted
as single-row or batch-axis execution. Keep original inference available.

Reserve **28 executables for each Qwen key**, single or packed, from the existing
module-wide **100** before compilation. Counts come from the pinned manifest's
layer count, not the row count or packed width. Count pending admissions and
acknowledgment-pending evictions. Keep per-model/overall shape caps as secondary
limits, not reasons to exceed 100. Never add an independently owned prototype
`TightProgram` beside the supervisor's inventory.

| Resident example | Charge | Consequence |
| --- | ---: | --- |
| Qwen single 128 + tight 256 | 56 | Preferred warm set; 44 left for other direct models |
| Those plus Qwen single 256 or one longer single rung | 84 | At most 16 left; a fourth Qwen shape would require 112 and cannot be admitted without eviction |
| Qwen single 128 + tight 256 + two GTE 22-layer shapes | 100 | Valid mixed-model set; adding another Qwen rung must evict or wait |
| Three Qwen shapes plus one GTE shape | 106 | Invalid, even though each model's individual shape cap might permit it |

Soft-pin **single 128** through `State.smallest_rungs`: choose it as an eviction
victim only when no other unleased resident shape is available. This is an
unleased-victim preference, not reserved capacity or immunity from eviction.
Tight 256 is ordinary leased/LRU residency. Do not pin both and starve longer
requests or other models. Single 256 remains a distinct natural-rung fallback,
not a mandatory permanent third shape. Other single rungs remain demand-admitted.
A long row may displace tight 256; the next short batch must still be serviceable
through single 128/256 rather than assuming packed residency survives.

Use current cross-model admission fairness and unleased-victim selection for
required shapes. Optional packing warm-up must not evict another active model's
working set just to optimize Qwen. If no free budget or suitable unleased Qwen
victim exists, skip optional warming. The 100 is a measured conservative policy,
not a promise that an unrelated ANE client cannot cause resource exhaustion.

### Request flow and eviction

1. Module composition/admission validates the entire engine call, records original
   indices and the absolute deadline, and selects packed eligibility by the exact
   pinned profile and current qualification state.
2. Plan from lengths; partition into packed passes and natural-rung singles.
   A resident/qualification snapshot is only an eligibility hint, not a lease.
3. Acquire the serving-lifetime guard using the existing lock order. For each
   shape/quantum, obtain only its lease, then the worker stream, then execution
   permit; validate remaining deadline before writing. Packed leasing is
   nonblocking and must honor supervisor fairness: if unavailable, replan those
   rows as natural-rung singles and use normal required-shape admission. Never
   dispatch a nonresident key or lease all required rungs at once: a call mixing
   128/256/512/1024 would otherwise exceed 100 or deadlock while retaining its own
   eviction candidates.
4. Worker validates the execution mode/plan, stages inputs and returns the
   existing ordered vector result. Host scatters checked results, releases that
   quantum's resources and yields before another exchange.
5. Detached inference owns the lifetime guard and the current exchange's lease,
   stream and permit through response drain. Cancellation stops undispatched
   work, not accounting for dispatched work.

An eviction may select only an unleased key. The worker drops the complete
program, cached requests and surfaces inside an autorelease pool and acknowledges
only after the pool drains. Refund the reservation only after that ACK. Failed or
partial compilation rolls back its programs inside a pool; do not publish a
partial shape. Keep the current one-victim retry/resource-exhaustion recovery
policy: persistent exhaustion returns a transient refusal; restart must confirm
owned-process exit before clearing reservations or reconnecting/reloading models.
Do not recursively evict/compile until a request succeeds.

### No hidden cold packing compile

Packing is optional. A request with a cold/evicted/unqualified packed key uses
standalone; it does **not** synchronously compile tight 256 to earn a prospective
35 ms pass. Prepare/qualify tight 256 as explicit lane warm-up in an approved idle
or maintenance window, or leave it disabled. Shape admission still uses the
supervisor and lifecycle rules. Do not start speculative compiles on every sparse
call or every eviction. Compile latency must remain visible in status/telemetry.

Required standalone rungs retain the existing deadline-bounded, single-flight
admission path. Compile waits remain decoupled from the catalog lane mutex and
execution permits, as in [the compile-lane design](design-ane-compile-lane.md).
**That does not make compilation parallel with inference:** the worker loop and
stream still serialize the compile exchange and warm inference. Admission of a
long rung can block unrelated warm work on that worker until compile completes;
packing must preserve `shape_compiling`/deadline classification, not label this
as loading a model or claim a latency win. Async compilation or a second ANE
worker would be a separate design and hardware qualification.

## Fingerprint, engine identity and AFT compatibility

[`crates/synapse-core/src/fingerprint.rs`](../crates/synapse-core/src/fingerprint.rs)
(lines 65–113) hashes the stable serialized **entire `NumericProfile`** to obtain
`numeric_profile_id`, then hashes `[model_digest, quant, numeric_profile_id]` for
the fingerprint. This is a metadata hash, not a hash of output vectors or worker
source. `every_fingerprint_input_changes_identity_independently` (375–427) proves
that engine name/version/build flags, tokenizer, pooling, normalization, dtype,
flash-attention policy, all four certified shape-envelope fields, templates,
thread policy, operation, grammar, kernel revision, rotation, converted package
and manifest-profile digest each affect identity.

**Decision:** keep `qwen3-embedding-0.6b-ane`, its `ane-direct-worker` engine
identity and its existing numerical profile. Do not rename the lane to a packing
engine, add a packing flag to `EngineIdentity.build_flags`, change a serialized
kernel revision, or modify the pinned manifest/profile to advertise the
optimization. Use internal program/capability/telemetry metadata for dispatch
kind and qualification. This policy is valid only after exactness qualification;
if changing any hashed field is actually necessary, it is an identity change
and must be declared, not hidden by retaining a hard-coded old hash.

The module constructs the envelope from model context, `inline.max_tokens`,
`jobs.bulk_quantum_tokens` and `inline.max_items` (`lib.rs:3439–3444`). Source
defaults are **8,192 / 8,192 / 3,072 / 64**, respectively (`lib.rs:199–214` and
catalog context validation at 3144–3165). `CatalogProfile::apply_numeric_profile`
(3246–3300) adds the manifest/package/grammar digests and the existing
`ANE_DIRECT_KERNEL_REVISION = "ane-direct-graph-v1"`; it does not rewrite that
envelope. Profile engine identity is `ane-direct-worker`, version `protocol-v2`,
with `risk_class=abort_capable`, the pinned `profile` ID, `compute_dtype=f16` and
`storage_dtype=f16` (3189–3217, 3708–3748). These are source defaults/rules, not
an inspection of the live store or a reconstruction of the supplied hash prefix.

A change to the scheduling advice/row constant **alone** is not among those
hashed fields. A change to `certified_shape.max_sequences`, `max_batch_tokens`
or `max_micro_batch_tokens` **is**, even when sampled vectors remain identical.
Before enabling larger quanta, snapshot and compare the complete production
numeric profile, fingerprint and response engine identity in a regression test;
validate the proposed calls against the existing envelope rather than expanding
that envelope. The larger lengths-only planning window can encompass multiple
already-admitted bounded worker exchanges; it must not be misrepresented as a
newly certified numerical batch size.

**AFT: no re-embed for the unchanged-identity, exact-output optimization.** AFT is
the code-search index client supplying the engram export. The supplied index
identity is this lane (`5f7ac7cd…`, a fingerprint prefix, not a Git revision).
Its old document vectors and new query/document vectors then remain in the same
space and the fingerprint comparison remains unchanged.
No alias, mixed provenance or migration is introduced. This confirms the
intended compatibility outcome, **not** the general claim that bit-identical
output automatically preserves a fingerprint: an engine/version/envelope/profile
edit changes the hash independently. If such an edit is required, stop this
rollout and decide migration/explicit evidence-backed aliasing separately. The
core alias rules (116–225) even permit a retracted alias to require migration;
packing must not manufacture an alias or bypass an existing provenance check.
AFT's external client implementation/live store was not inspected for this
report; the compatibility conclusion uses the supplied index identity and
Synapse's source rules, not a claimed live-index verification.

## Numerical contract and proof

The worker must retain these invariants from the prototype:

- Compose/tokenize documents exactly as today, including the terminal token;
  plan from **composed token counts**, not characters or unprefixed text.
- First-fit decreasing is deterministic: decreasing length, original index as
  the tie-breaker; each nonempty row occurs exactly once. Concatenate real tokens
  without per-row slot padding. Only the unused suffix is padded.
- Use `[1, hidden, 1, 256]`, not `[rows, hidden, 1, width]`. The batch-axis graph
  can compile and return finite but wrong vectors; rejecting it is a correctness
  constraint, not a performance preference.
- For each real query, allow only earlier/equal keys in its own segment. Padding
  queries attend only themselves to keep softmax finite; real queries cannot see
  padding. Reset RoPE positions to zero at every segment boundary.
- Encode both rotary tables with **`ane::f32_to_fp16_bytes`**, decode those fp16
  bits to exactly representable fp32, then use the existing runtime staging path.
  Ordinary `TensorData::copy_from_f32` on the original fp32 tables uses a different
  rounding policy and is not equivalent. Do not change the standalone encoder.
- Rebuild/copy the entire activation, mask and rotary inputs for every pass.
  Serialize access to a program's reusable IOSurfaces and cached requests. Gather
  each segment's last real token, run the existing CPU RMS/L2 tail without changing
  arithmetic order, and scatter by original row index.

### CI gates to add during implementation

1. **Portable structural tests:** exhaustive small length lists plus seeded larger
   lists cover deterministic ties, mixed lengths, suffix padding, zero/overflow
   rejection, one occurrence per input, exact order restoration, boundary lengths
   1/127/128/129/255/256/257 and long-row fallback. Test masks against an independent
   segment-membership/causality oracle, not the packer's own offsets alone. Include
   cancellation, stale-plan rejection, and partial-pass tails.
2. **Encoder regression:** preserve
   `runtime_rope_coefficients_match_actual_graph_constant_bits` in
   `tight_packing.rs`. Compare actual serialized `Graph.constant` payload bits to
   staged mask/rotary values; duplicating the truncation algorithm in the expected
   result is not an independent oracle. Where staging is architecture-specific,
   this gate belongs on macOS CI as well as the portable encoder tests.
3. **Protocol/residency tests with fake workers:** distinguish packed/single 256,
   charge all pending layers, retain the charge until eviction ACK, retain leases
   through cancelled RPC drain, exercise deadline/FIFO behavior and resource
   exhaustion recovery. Prove a mixed call does not need all its fallback rungs
   leased simultaneously. Warm short traffic must not become `model_loading`
   because another request wants a cold shape.
4. **Apple hardware qualification CI:** independently run the unchanged natural-
   rung single-row path, save its final vectors, then pack the same inputs. Require
   equality of **every `f32::to_bits()`**, correct count/order, finite 1,024-D unit
   vectors and max absolute difference 0.0. Run pinned fixtures alone and mixed;
   change neighbours, reorder/repeat rows and alternate reused-buffer geometry;
   place short rows before/after the 128-column tile boundary and at the end of a
   pass. Include 129–256-token rows and >256 fallback. Replay all 6,341 export rows
   with both eight- and 64-row planning, verifying input/package/tokenizer hashes.
   A private export is a controlled test artifact, never committed text.
5. **Independent existing reference gate:** retain the original fp32 fixture,
   tokenizer composition and serving self-check tolerances unchanged. Exactness
   against standalone is an *additional* gate, not a replacement for correctness
   against the model reference. Linux-only CI cannot certify the private Apple
   compiler/runtime; release qualification requires the Apple gate.

The existing replay `parity` function asserts cosine >= 0.9999 and reports
`max_abs`/`identical`; those diagnostics are not an exactness assertion. Its
`a == b` comparison also does not distinguish signed zero. The evidence reports
zero difference and identical outputs, but the new serving gate must assert
bit equality rather than merely continuing to print those fields.

Prove non-vacuity during implementation: restoring nearest-rounded RoPE must fail
an exactness test that exercises nonzero positions (including the recorded row-39
case); allowing cross-segment attention must fail a neighbour-isolation test;
premature budget refund must fail the ACK-ordering test. Record each exact failing
test and unaffected controls, stage before mutating, restore every mutant, and
keep production controls unchanged. A hardware-dependent control skipped by CI
is not a successful mutation proof.

### On-device self-check

Before a packed program becomes eligible for serving, run a supplemental exact
check on the same device, package, engine identity and compiled program. Obtain
independent standalone outputs at their natural 128/256 rungs, then compare the
packed outputs byte-for-byte. Use a fixed, versioned suite of fitting fixture
rows covering unequal lengths, multiple neighbours, tile crossings, first/last
positions, suffix padding and alternating geometry. Retain the existing model
self-check. A mismatch disables **packing only**, records a diagnostic and uses
the original lane; it must not silently publish changed embeddings.

Requalify after packed recompilation/worker recovery and whenever the OS/runtime,
package, tokenizer, numeric profile or packing implementation changes. Store the
qualification key and outcome outside the numeric profile. Do not treat a prior
M4 audit as this M5 device's self-check. Standalone reference vectors can be
captured before releasing a temporary single-256 shape, then compared after its
acknowledged eviction and packed admission; this avoids requiring three Qwen
shapes simultaneously just to check two independently computed paths. All of
these admissions still go through the supervisor. If budget or deadline prevents
qualification, leave packing unavailable and serve standalone; never bypass the
budget to perform a self-check.

## Failure modes and externally visible behavior

| Condition | Required behavior |
| --- | --- |
| Row >256 composed tokens | Do not truncate, split into independent embeddings, or send to a wider packed graph. Use the existing smallest single-row ladder rung that fits, up to the lane's admitted limit (ladder maximum 8,192). Above that, preserve current overflow/truncation disclosure or rejection policy before worker I/O. |
| Mixed lengths, including 129–256 | FFD packs only fitting rows; masks/RoPE use exact segment lengths. Scatter every vector to its original index. Long rows run singly, one required shape lease at a time. A partially filled pass is valid; it does not wait for future callers. |
| Empty sequence, invalid token ID, malformed sizes/plan | Validate the entire worker call before its first dispatch: nonempty rows, exact raw-frame length, checked length arithmetic, vocabulary bounds, one occurrence per row, plan capacity and mode/profile match. Return the existing invalid-request/worker error; never insert a zero vector or silently omit a row. Ordinary empty text may compose to a terminal-token row and is not the same as an empty token sequence. |
| One row or pass fails at runtime | The worker exchange is all-or-error, as today. Do not expose partial vectors as success or rerun selected rows after publishing others. A worker `ERR` and a broken channel retain their existing distinct recovery policies. Preserve the logical job's existing durable checkpoint/page semantics across already completed quanta, but do not mark a failed quantum complete. |
| Nonfinite vector, wrong count/dimensions/order or exact self-check mismatch | Fail that quantum/qualification, retain diagnostics, disable packing for the affected qualification key. Serve unchanged standalone only once its normal readiness is established; do not hide a channel fault behind an immediate retry on the same stream. |
| Packed shape evicted between planning and lease | Replan to natural-rung singles. A held lease forbids eviction; stale generation/plan responses must not release or populate a new slot. |
| Budget full/all victims leased | Required shapes wait under existing FIFO/starvation protection and the smaller of admission timeout/request deadline; optional packed warming is skipped. Never borrow executable slots or retain several leases while waiting for another. |
| Hardware resource exhaustion | Retain `ane_resources_exhausted` (250 ms retry guidance), bounded eviction/retry and confirmed-exit recovery. This is possible even below 100 due to other machine users or opaque runtime retention. |
| Caller cancels before dispatch | Remove its queued future/remaining quanta and buffers; do not create an admission or exchange after the request is abandoned. An already-dispatched admission/eviction finishes its accounting task. Add explicit caller-abandonment signalling for new multi-quantum plans; dropping a join handle alone is not such a signal. |
| Caller cancels or deadline expires after dispatch | Return the existing cancellation/deadline result to that caller; detached drain retains the lifetime guard, stream, permit, buffers and lease until reply or channel recovery finishes. Discard abandoned results and skip further quanta. Never abort an ANE pass or release its live surfaces early. |
| One cancelled request supposedly shares a pass with another | This cannot occur in this design: all pass rows have one call owner. Validate that invariant. If cross-request coalescing is proposed later, it needs a separate ownership/deadline design; its executing shared pass would have to drain without invalidating any other owner's results. |
| Deadline during compile or stream wait overlapping compile | Preserve **`shape_compiling`**, lane ID and shape, safe-to-retry-same-request and **1,000 ms** retry guidance. Retain the recorded wait cause even if compile has finished by error mapping time. Add an additive program-kind detail to distinguish single 256 from tight 256; do not change the existing width field into an incompatible object. |
| Deadline in ordinary budget/eviction/permit/inference work | Preserve `deadline_exceeded`, not `shape_compiling` merely because some later operation happens to compile. Check the absolute deadline at each queue/lease/stream/permit boundary and before writing a new quantum. No automatic retry past the deadline. |
| Stream queue timeout before write | No bytes were dispatched: return the proper deadline cause without faulting/restarting the worker. The inference I/O watchdog starts after stream acquisition; dispatched I/O failure still drains or performs the current channel recovery. |

`shape_compiling` means the model is already loaded and the retry may join the
same admission or find it resident. It is not a new packing error and must not
be rewritten to `model_loading`. Preserve the machine-global direct-ANE owner
lock; none of these fallback or warm-up paths bypasses it or starts a competing
compiler.

## Measurement plan and acceptance bar

### Matched before/after runs on both Macs

Run the same release worker/module and pinned model/tokenizer/fixture/export
hashes on the **M4 mini** and **this M5 Max**, in separately approved windows with
no competing direct-ANE owner. Do not stop production or bypass its lock to obtain
a window. Prebuild before measuring, record hardware/RAM, OS build, compiler,
worker/host revisions, active other lanes, load and UTC/monotonic boundaries. Save
raw results outside build/cache directories; include counts/refusals and input
hashes, not private texts. A refused run is a refusal record, not a timing sample.

Measure three serving arms against an interleaved serial baseline:

1. unchanged eight-row serial serving;
2. eight-row tight-256 serving with the same admission/scheduler limits;
3. direct-only 64-row look-ahead, bounded two-pass exchanges, including actual
   token-budget clamps. Unlike the existing in-process 64-row replay, this arm
   includes module admission, IPC and scheduling across several worker exchanges.

Use the existing replay/audit harness as a separate kernel diagnostic, and the
private catalog/module harness from the lane profile for serving. Extend it to
record every user call and worker quantum with stable owner/pass IDs; overlapping
requests must not be attributed by overlapping time windows alone. Keep whole-
export kernel and end-to-end serving clocks separate. Preserve original metadata
batch boundaries/order in the export comparison, then measure the production
length-sorted scheduler separately: the eight-row prototype uses consecutive
slices, whereas production sorts by length before slicing. Do not substitute one
workload while calling the result an exact reproduction.

| Workload | Required measurements |
| --- | --- |
| Warm singles at 32/128/256/512/1024 tokens | End-to-end latency, layer/worker wall, queue/permit/stream waits; singleton fallback latency |
| Calls of 1/2/4/8/16/32/64 rows, real mixed lengths and adversarial sparse tails | Latency **per logical call and per worker exchange**, fill, passes, dispatches, rows/tokens per second; separate fixed row count/token distribution strata |
| All 6,341 original export rows | Whole-export inference and end-to-end time, every output compared exactly, token/order hashes, pass counts, IPC/CPU/compile/setup totals |
| Foreground one-row queries plus sustained bulk, two concurrent calls, 8 concurrent calls | Foreground and bulk **p50/p99**, queue time, fairness, deadline/refusal rates and completion throughput; report successful latencies and failures separately, not success-only tails |
| Single 128 + packed 256 with other direct embedding/rerank models | Live reservations, leased/pending/evicting occupancy, other-model p50/p99, packed eviction/refill, avoided optional compiles |
| Cold packed preparation; cold single 512/1024 during bulk; all victims leased | Compile duration, queue wait, first-after-compile call, `shape_compiling` details/retry, eviction ACK/refund, recovery and resumed short-row latency |
| Cancel queued and dispatched requests; short deadlines at each wait boundary | Returned error, undispatched-work suppression, drain time and time until next caller can run; no cross-owner vector/result loss |
| Repeated warm/evict/recompile/recover cycles with concurrent pending windows | Peak worker/module RSS and memory-pressure/swap observations, post-drain plateau, opaque allocation retention, maximum reserved executables and correctness after reuse |

For warm per-call distributions collect **at least 1,000 successful calls per key
size/traffic stratum** over multiple interleaved windows, preserving all refusals
and load observations. Publish sample count, p50/p99 definition, range and a
bootstrap confidence interval; a five-repeat probe does not establish p99. Run
at least three alternating full-export repetitions per arm after a separately
reported full warm-up, and report medians/ranges. Cold compile samples and
first-after-reload samples are separate, not dropped as warm outliers. Do not
interpret zero private hardware counters as zero device work; use wall timers
where the runtime does not supply usable counters.

### Gates for enabling packing

- **Correctness/identity are hard gates:** all covered packed-vs-single output
  bits match, all existing model-reference gates pass, original output order and
  fingerprint/engine fields remain unchanged. One changed vector disables the
  optimization; no accuracy-tolerance relaxation or automatic alias.
- **Safety is a hard gate:** reservations never exceed 100; no leased eviction,
  early refund, nonresident dispatch or unaccounted partial compile; cancellation
  drain and confirmed-exit recovery tests pass. Concurrent mixed-model release
  tests must complete within the unchanged deadlines or return the specified
  bounded refusal, not resource leaks or silent hangs. The historical failed
  cold flood must remain explicitly classified as failed, not used as a pass.
- **Eight-row rollout:** on each device, median matched end-to-end whole-export
  time must improve by **at least 20%** over serial (time <=0.80× baseline), with
  a demonstrated improvement across repeat windows. The M4 kernel observed about
  25% less time, leaving limited room for added serving overhead. Reproduce the
  kernel result within **10%** of a contemporaneous same-machine replay; do not
  gate M5 on 101.12/116.24 s absolute M4 numbers.
- **Query/other-lane protection:** warm foreground p50/p99 must be no worse than
  `baseline + max(10% of baseline, 5 ms)` in each matched mixed-traffic stratum;
  deadline/refusal rates must not increase. This is a proposed product acceptance
  threshold, not a measured property. Check distributions including queueing.
- **64-row bulk rollout:** improve matched end-to-end export time at least **5%**
  beyond the eight-row packed arm while passing the same foreground/other-lane
  p99 bar. Record the actual planning sizes after token/envelope clamps. If it
  fails, retain eight-row packing; do not enlarge the exchange quantum to force
  a throughput pass. The 64-row user-call latency and bounded-quantum latency
  both need distributions, not an average inferred from export time.
- **Memory:** no OOM, added swap/pressure episode or monotonic post-drain RSS
  growth during repeated turnover. Establish the peak RSS of each arm including
  compile and concurrent pending calls on each device before selecting an
  operational headroom limit; 1.25 MiB of explicit surfaces is not that limit.

These performance thresholds are approval proposals. Run them on both devices;
a correctness-qualified but slower device leaves packing disabled there. A
materially different OS/compiler build repeats qualification and matched runs.

## Implementer handoff and rollout order

1. Freeze current numeric-profile/engine/fingerprint bytes in compatibility tests.
   Add the distinct packed shape key and capability-negotiated mode to core worker
   protocol, module residency/inventory and worker storage/cleanup. Do not enable
   serving until all budget/lease/error-path tests pass.
2. Integrate the existing constant-compatible tight-256 kernel. Add complete-call
   prevalidation and exact result-count/order checks; share only the pure planner.
   Host planning must happen before current per-rung partitioning. Preserve the
   unchanged single-row path, lease-per-quantum discipline and wait-cause history.
3. Add supplemental device qualification with independently captured standalone
   results and proper lifecycle guarding; optional cold preparation remains
   explicit. Initially opt in only a qualified pinned Qwen lane, at eight rows.
   Compare profile bytes and responses before declaring AFT compatibility.
4. Run matched M4/M5 serving qualification, including mixed-model turnover,
   deadlines/cancellation and foreground tails. Enable eight-row packing only on
   devices passing the gates, with a nonnumeric off switch reverting dispatch to
   standalone without changing the fingerprint.
5. Separately implement/test direct-only 64-row look-ahead with two-pass exchanges
   and explicit envelope/token bounds. Do not raise unrelated engine advice or
   apply the cap to reranking. Promote only if incremental throughput, memory
   and mixed-traffic p99 gates pass; rollback retains eight-row packing.

## Unresolved

These are approval/measurement findings, not permission to choose an unmeasured
answer during implementation:

- **Effective deployed envelope and protocol compatibility.** The source defaults
  permit the proposed two-pass/64-row bounds, but this report did not read the
  live stored profile or overridden configuration. Verify its complete bytes
  before rollout; smaller limits must clamp dispatch, not change identity.
  Confirm that capability-gated packed keys/modes can be added without changing
  the public `EngineIdentity.version` or numeric kernel revision. If a required
  metadata change alters the hash, it cannot share the no-re-embed claim. Do not
  infer compatibility solely from the 64-row prototype hardware replay: that
  Rust test is ignored by default, invoked explicitly, and bypasses normal module
  scheduling/admission rather than demonstrating its public-call limits.
- **M5 performance and serving tails.** There is no M5 tight-packing inference
  result or production p99 in the supplied evidence. Is two passes per exchange
  the right fairness bound on each device? Does the 64-row policy improve the
  token-clamped production workload after extra IPC? Keep it gated until measured.
- **Operational memory headroom.** Peak packed executable/intermediate RSS,
  compiler retention and concurrent pending-call cost are unmeasured. Establish
  a device-specific headroom limit; neither source payload bytes nor explicit
  surface arithmetic supplies one.
- **Preparation windows and churn.** Explicit idle/maintenance preparation avoids
  hidden cold packing work, but its availability and frequency of packed eviction
  under mixed models are unknown. The serial worker cannot protect warm-request
  latency during a required compile merely by releasing the lane mutex. Async
  compile, permanent pinning or a second worker is not silently approved here.
- **Release qualification infrastructure.** Reserve an Apple CI/qualification
  runner and controlled export artifact with verifiable hashes. Portable tests
  alone cannot certify the private compiler. The prior M5 debug cold flood failed
  and the release stress run was deferred; obtain a separately labelled release
  result without relaxing deadlines to convert that finding into a pass.
