# Synapse wire contract v1 (consumer snapshot)

Status: SNAPSHOT of v1.1 with explicit catalog installation and owned-only local embedding/reranking.
Authoritative examples: `crates/synapse-module/tests/skeleton_e2e.rs`. This document
restates the consumer contract;
on disagreement, fix the document against the executable API tests.

## 1. Envelope (inference payloads)

Checks: `embed_query_preloaded_minilm_returns_vectors_and_envelope` and
`embed_batch_preloaded_minilm_preserves_order_and_envelope` (skeleton e2e).

Management responses wrap operation payloads as `{"result": <payload>}`.
The fields below describe inference payloads, not every management result.
Download statuses are flat inside `result`.

```json
{
  "fingerprint": "…",            // strict identity string — persist VERBATIM, compare only, never parse
  "table_epoch": 2,               // frozen at the last published alias-table epoch
  "dims": 384,
  "provenance": { "engine": { "engine": "owned-metal", "version": "…", "build_flags": {} } },
  "module_generation": 3,         // bumps every module boot — cheap restart detection mid-conversation
  "equivalent_to": [],            // aliases removed; retained as an empty array
  …op payload flattened here…
}
```

- fingerprint = hash(model digest, quant, numeric_profile_id); the numeric
  profile folds engine identity, sanitized-tokenizer digest, pooling,
  normalization, dtype, shape envelope, thread class. Anything that moves
  vector space mints a new string. Alias flips do NOT change fingerprints.
- Per-item fields on embed responses: `real_token_counts`,
  `truncation_disclosures` (`submitted_tokens`, `effective_tokens`, `truncated`),
  and `vectors` with `id`, `vector`, `content_sha256`, and `submitted_sha256`
  (see Divergence detector below).

## 2. Errors (stable codes, typed recovery)

Checks: `stable_errors_are_exhaustively_documented_and_keep_wire_names`
(`wire_contract` integration target) and
`embed_query_deadline_one_returns_typed_rejection` (skeleton e2e).

Every error: {code, class: transient|permanent, retry_after_ms?,
safe_to_retry_same_request, message, details?}. `details` is an optional object.
Transient errors may carry retry_after_ms. Never poll-hammer
a permanent code.

Error-code vocabularies are surface-specific. Core `StableErrorCode` supplies
`queue_full`, `deadline_exceeded`, `model_loading`, `not_certified`,
`substitution_rejected`, `artifact_invalid`, `engine_crashed`,
`probe_required`, `migration_required`, and `module_restarted` here.
Owned-decode refusals use a separate vocabulary:
`grammar_disabled` is not a `StableErrorCode`; it is emitted for constrained
`microllm.oneshot` requests when grammar is disabled by module or approval
configuration, or when no certified enabled owned-decode grammar lane is
available.

The [local transcription surface](#local-batch-transcription-media-providerv1)
uses the `media-provider/v1` envelope and refusal vocabulary, not the catalog
result envelope below. In particular, undeclared engines use `unsupported`
(not `unknown_model` or `backend_unavailable`), changed replay bindings use
`client_job_id_reuse` (not `idempotency_conflict`), and audio limits use
`input_too_large`. Media error details live in singular `detail`.

- `sequence_too_long` — class `permanent`, no `retry_after_ms`, not safe to
  retry the same request. A catalog-installed lane refuses a composed input
  exceeding its 8192-token limit before job diversion or execution. `details`:
  `{tokens, max_tokens, item_id?}`; `item_id` identifies the submitted item when
  available. No truncation disclosure or job id is emitted. **Consumer
  disposition:** split or trim the input before resubmitting.
- `owned_cuda_unsupported` — class `permanent`. Emitted before an owned-CUDA
  worker is created when capability gating finds no usable CUDA capability
  evidence, a driver API below 12040, or compute capability below 7.5.
  **Consumer disposition:** do not retry on the unchanged host; select a
  non-CUDA lane or provide a compatible driver and device before retrying.
- `model_not_installed` — class `permanent`, no `retry_after_ms`, not safe to
  retry the same request. Emitted when a catalog model is requested on a backend
  that has no install record for the entry's current manifest. `details`:
  `{catalog_id, lane_id | null, download_op: "models.download",
  download_job_id | null}`; `download_job_id` names the non-terminal download
  job for the entry's current manifest, if one is running. **Consumer
  disposition:** call `models.download` (or wait for the named job) before
  retrying.
- `unknown_model` — class `permanent`, no `retry_after_ms`, not safe to retry
  the same request. Emitted when the requested id is neither a catalog id, a
  declared catalog lane id, nor a registered model. `details`: `{model_id}`
  (the requested id). **Consumer disposition:** do not retry; correct the id.
- `self_check_failed` — class `permanent`, no `retry_after_ms`, not safe to
  retry the same request. Emitted when a catalog lane's persisted self-check
  has failed for the current self-check key. `details`: `{model_id, backend,
  check_id, reason}`; `check_id` is the lowercase hex sha256 of the JCS
  serialization of the self-check key. **Consumer disposition:** do not retry
  on the unchanged install; the check reruns only after the key changes or the
  model is removed and downloaded again.
- `backend_unavailable` — class `permanent`, no `retry_after_ms`, not safe to
  retry the same request. Emitted when a catalog entry has no backend runnable
  on this machine, or the selected backend is not runnable here. `details`:
  `{catalog_id, lane_id | null, backends[] {backend, reason}}`. **Consumer
  disposition:** do not retry on this host; select another model or host.
- `model_in_use` — class `permanent`, no `retry_after_ms`, not safe to retry
  the same request. Emitted when a removal or unload would disturb a model that
  is still held. `details`: `{catalog_id, holders[]}`, each holder
  `{kind: "loaded", model_id}` | `{kind: "job", job_id}` |
  `{kind: "self_check", check_id}` | `{kind: "request", lease_id}`.
  **Consumer disposition:** release or wait out the named holders, then issue
  a new request.
- `download_failed` — class `transient`, `retry_after_ms: 1000`, safe to retry
  the same request. Emitted when a model download fails on the network, on an
  HTTP error status, or on full storage. `details`: `{file, reason: network |
  http_status | storage_full, http_status | null}`. **Consumer disposition:**
  retry the download after `retry_after_ms`.

### Catalog result-envelope rule and reused errors

Semantic refusals specified with details use
`{"result":{"module_generation": <u64>, "error": <wire error>}}`, including
semantic `invalid_request` from download, remove, unload and wrong-task serving.
Malformed or unknown-field params remain channel-level `invalid_request` with a
message. A failed download carries `error` beside `state` in its flat status.

- `artifact_invalid` — permanent, not safe to retry unchanged. Digest/size
  mismatch or missing/wrong-size cached blob: `{file, expected_sha256,
  actual_sha256, expected_size, actual_size, recovery_op: "models.download"}`.
  Loaded fingerprint mismatch: `{model_id, expected_fingerprint, actual_fingerprint}`.
  Corrupt blobs are quarantined and affected installs/pins invalidated; other
  blobs are untouched. Explicit download repairs the file; serving never fetches.
- `module_restarted` — transient, safe to retry the same request. The module
  restarted before this job or download finished, so that job id stays terminal;
  resubmitting the request starts a new attempt. Non-terminal downloads fail and
  clean up on restart; committed installs survive. No catalog-specific details
  object is added.
- `deadline_exceeded` — transient, not safe to retry the same request; request budget expired before
  readiness. The deadline is part of the request, and an inline batch records
  nothing under its `request_key`, so resending it unchanged recomputes from the
  start under the same budget. Resend it smaller, or with a fresh deadline and
  key. A same-key resend of a durable job returns that job while it is active.
  No catalog-specific details object is added.
- Wrong-task `invalid_request` — permanent, not safe to retry unchanged;
  `{catalog_id, requested_task, catalog_task}`, without load or network I/O.
- Derived-id management `invalid_request` — permanent, not safe to retry unchanged;
  download/remove carry `{model_id, catalog_id}`, including undeclared reserved
  backends. Unload with a catalog id carries `{catalog_id, lane_ids[]}`.
- `substitution_rejected` — permanent, not safe to retry unchanged. A lane id
  combined with another fingerprint carries `{lane_id, required_fingerprint}`.
  Unknown fingerprints also refuse without load. Aliases and `allow_equivalent`
  never substitute catalog fingerprints.
- `engine_crashed` — transient on catalog paths, `retry_after_ms: 250`, safe to
  retry; includes load/runtime failure or the 2000 ms self-check ceiling.
  A quarantined catalog worker also uses this code, never `probe_required`:
  `retry_after_ms` is the remaining time in its fixed ten-minute cool-down and
  `details` is `{lane_id, quarantined: true}`. The next serving request after
  expiry starts a fresh worker with zero strikes. Unload/reload or a module
  restart clears catalog quarantine immediately. Non-catalog worker quarantine
  still reports permanent `probe_required`.
  Worker crash, protocol fault and no-progress hang each add one consecutive
  strike; two strikes quarantine. Only fully drained successful serving work
  (embed, rerank/score or generation) clears strikes. LOAD/reload, PING,
  admission/eviction and setup never clear them. Caller-deadline expiry on live
  work neither strikes nor clears; its eventual detached serving success does.
- `model_loading` — transient, `retry_after_ms: 250`, safe to retry. Inline
  serving waits up to `min(remaining deadline, 5000 ms)` for load/self-check;
  this code is used only when that cap expires with deadline remaining. A
  request that reaches a lane while its load-time numerical self-check runs
  waits for it until its own deadline and is then answered `model_loading`.
  Never returned because another request is executing or a Neural Engine
  shape is compiling.
  Admitted batch jobs wait within their execution budget and never fail with it.
- `shape_compiling` — transient, `retry_after_ms` at least 1000, safe to retry
  the same request. Direct Neural Engine lanes only. The worker compiles each
  padded sequence shape the first time it is used (seconds to minutes) and
  cannot run any inference while it compiles. The module compiles one shape at
  a time across all direct-ANE lanes, cheapest first. A request waits through a
  compile as long as its own deadline allows.
  This code is returned when that deadline expires while the request is
  blocked on a compile: the shape it needs is being compiled (several requests
  share one compile) or is queued for its compile, or it was waiting for the
  lane's worker while another shape's compile occupied it at some point of
  that wait. It is also returned before the deadline, as soon as the module
  predicts the shape cannot be ready by then (or within its 600 s admission
  bound). `retry_after_ms` is the module's predicted time until the shape is
  ready, and never less than 1000. When the module can no longer tell (more
  than 16 compiles since the wait began), it answers `shape_compiling`.
  `details`: `{lane_id, shape}`,
  where `shape` is the shape being compiled. Waits that expire on anything else
  keep `deadline_exceeded`.
  **Consumer disposition:** the model is loaded; retry after `retry_after_ms`.
  The compile continues without the request, so a retry joins it or finds the
  shape ready. A queued compile that nobody retries is dropped after twice the
  promised delay. A deadline that covers a first-time compile (about 30 s for
  short inputs, minutes for 4096 tokens and more) avoids this code.
- `idempotency_conflict` — permanent, not safe to retry unchanged; replace a
  download key bound to a different digest.

## 3. Common request fields (acceptance constraints)

Checks: `embed_query_deadline_one_returns_typed_rejection` (skeleton e2e) and
`catalog_multi_backend_resolver_refuses_substitution_before_load` (catalog e2e).

deadline_ms (absolute budget), max_queue_ms (fast-fail bound — admission is
atomic accept-or-reject against it), required_fingerprint,
 allow_equivalent (accepted and ignored),
required_epoch, target_fingerprint (embed.query: route only to a matching or
certified-equivalent profile, else substitution_rejected/not_certified).

## 4. Flow-scoped routes

Checks: `tests::flow_routes_serve_diagnostics_and_event_bodies_only` and
`tests::the_flow_refusal_names_the_method_and_the_flow` (module library).

Synapse declares `flow-scopes/v1` and lists `diagnostics.findings` and `events_get`
in `flow_operations`. Routes whose scope carries a `flow_id` serve only these
read-only queries. Every other method, including other queries, mutations and
undeclared names, gets the channel-level error `flow_scope_refused`, naming the
method and the flow. Query classification is not an unattended-access policy:
inference queries can load models and consume hardware. The other operations on
routes without a `flow_id` are unaffected.

Both queries accept head scopes and daemon-authorized flow scopes;
bare worker/ephemeral scopes and unauthorized flows receive `permission_refused`.
Unscoped management routes remain available to operator clients. Route scope,
not request params, establishes flow identity and authorization.

## 5. Operations

Checks: `wire_contract_documents_every_management_operation` (module library)
compares the registry with the inventory below, including on CRLF checkouts;
`stable_errors_are_exhaustively_documented_and_keep_wire_names` (`wire_contract`
integration target) checks the core error vocabulary. The operation-specific
skeleton and catalog e2e tests prove runtime behavior; inventory checks alone do
not prove numerical execution.

The management registry in this snapshot is `embed.query`, `embed.batch`,
`embed.result`, `rerank.score`, `microllm.oneshot`,
`owned_decode.admit_session`, `owned_decode.decode`, `owned_decode.snapshot`,
`owned_decode.continue`, `owned_decode.abort`, `owned_decode.close`,
`owned_decode.session_status`, `owned_decode.disable`, `owned_decode.revoke`,
`model.status`, `model.unload`, `models.list`, `models.catalog`,
`models.download`, `models.download.cancel`, `models.remove`,
`cache.pin`, `cache.gc`, `admission.status`,
`approvals.migrate_owned_decode`, `approvals.enable`, `approvals.disable`,
`approvals.emergency_rollback`, `certify.observations`, `lanes.pause`,
`lanes.resume`, `lanes.status`, `diagnostics.findings`, `events_get`, `role.describe`,
`media.submit`, `media.status`, `media.result`, `media.cancel`,
`media.upload_begin`, `media.upload_append`, `media.upload_status`,
`media.upload_commit`, `media.read`, `certify.transcribe_observation`,
`certify.transcribe_measurements`, and `certify.transcribe_observe_rerun`.

### Local batch transcription (`media-provider/v1`)

This surface follows `cortexkit-role-media` 0.1.11. Successful RESPONSE bodies
are exactly `{result: answer}`; ERROR bodies are `{code, message, detail}` with
`detail.class` (`permanent` or `transient`). Local transcription needs neither
a vault credential nor a host binding and writes no usage or price record.
Identity comes from the route principal (the caller identity supplied by the
daemon), never request params. Foreign job/object ids are indistinguishable from
unknown ids. Flow-scoped routes refuse
all of these operations with `flow_scope_refused`.

- **role.describe** {} — query; declares group `jobs`,
  `jobs_survive_restart: false`, and engines `{kind: "transcribe", provider:
  "local", model, local: true, formats: ["audio/wav"], latency_class: "batch",
  max_input_bytes: 268435456, max_duration_ms, languages}`. The same engine facts appear in
  `role_declarations["media-provider/v1"]`. No `bias_limit` is declared.
  Every Parakeet engine declares the catalog's 25 languages: bg, cs, da, de, el,
  en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk.
- **media.submit** {client_job_id, kind: "transcribe", model, input: {audio,
  language?, bias?}, decision_id?, deadline_ms?} — mutate; returns `{job_id}`.
  `client_job_id` is 1–128 Unicode scalar values and is scoped to the principal.
  Replaying equal decoded JSON returns the original id, including after failure
  or restart, without new work. A changed `kind`, `input`, `model`, `decision_id`
  or `deadline_ms` refuses with `client_job_id_reuse {field}`, in that order.
  Refused submissions remember nothing. An undeclared audio MIME refuses permanent
  `unsupported {field: "input.audio.mime"}`. Language matching ignores case and
  accepts a hint extending a declared tag at a subtag boundary (`en-GB` matches
  `en`); other hints refuse `unsupported {field: "input.language"}`. A matching
  hint is accepted but does not steer decoding: Parakeet detects language itself
  and the result omits `language`. Any supplied bias, including empty terms,
  refuses `unsupported {field: "input.bias"}` because no bias limit is declared.
  Malformed input instead refuses `invalid_params`: for example a wrong field
  type, invalid base64, or bytes declared as WAV which cannot be decoded as WAV.
- **media.status** {job_id} — query; returns `{status, failure?}` where status is
  `queued`, `running`, `done`, `failed` or `cancelled`. Exactly failed jobs carry
  `{code, class, message, retry_after_ms?, detail?}`. Deadlines include queue time.
- **media.result** {job_id} — query; done jobs return `{text, words: [{text,
  start_ms, end_ms}], no_speech}` with no language. Answers exceeding 3 MiB are
  stored as JSON and returned through `{output_object: {mime, bytes, sha256,
  object_id}}`, readable with `media.read`. Certification mode adds the
  observation descriptor below. Other statuses refuse with `job_not_done
  {status}`. Unknown/foreign jobs refuse with `unknown_job`.
  Words begin with the first emitted token and at every tokenizer piece starting
  with `▁`; other pieces join the current word. Each word is the trimmed tokenizer
  decode of its token ids. Start is 80 ms times the first token's encoder-frame
  index; end is 80 ms times `(last token's frame index + max(duration, 1))`,
  including duration-zero emissions. Both times clamp to
  the original audio duration, then truncate to the contract's integer
  milliseconds. Text decodes all emitted tokens. `no_speech` is true exactly
  when no tokens were emitted, in which case text is empty and words is `[]`.
- **media.cancel** {job_id} — mutate; returns `{status}`. Cancellation is terminal
  immediately; the first terminal status wins. In-flight work drains on its lane
  owner and its output is discarded. Cancellation does not count as a worker
  crash or consume the crash budget that can quarantine a lane.
- **media.upload_begin** {mime, bytes, sha256} — mutate; returns `{upload_id}`.
- **media.upload_append** {upload_id, offset, data} — mutate; appends contiguous
  base64 data, at most 2 MiB decoded per request, and returns `{received}`.
  Refusals include `upload_chunk_too_large`, `upload_offset_mismatch {received}`,
  and `upload_size_mismatch`, without changing the committed offset.
- **media.upload_status** {upload_id} — query; returns `{state, received}`.
- **media.upload_commit** {upload_id} — mutate; verifies size and lowercase hex
  SHA-256, then returns `{state: "committed", sha256, mime, bytes}`. The upload id
  becomes the object id. States are `open`, `committed`, `gone`. Unknown/foreign
  uploads give `unknown_upload`; lost open uploads give `upload_gone`.
- **media.read** {object_id, offset, length} — query; returns `{data}` base64,
  with length 1–2,097,152 bytes. Unknown/foreign/gone objects give `unknown_object`;
  uncommitted uploads give `upload_not_committed`. Object facts are authoritative;
  mismatches give `object_mismatch {field: "input.audio"}`.
- **certify.transcribe_observation** {job_id, part, page} — query; available only
  in certification observation mode, otherwise `certify_observation_disabled`.
  Like the two certification operations below, it is principal-scoped and refused
  on [flow-scoped routes](#4-flow-scoped-routes), identified by the daemon-stamped
  flow scope; only `diagnostics.findings` and `events_get` are allowed there.
  Part is `encoder_output` or `decode_trace`. Each reply is `{data}` containing
  base64 of at most 512 KiB raw bytes. The result's `observation` contains
  `{lane_id, fingerprint, valid_encoder_frames, triples, encoder_output,
  decode_trace}`, where triples is base64 and each large part is
  `{bytes, sha256, pages}`. Descriptors count raw bytes, use their lowercase
  SHA-256, and count zero-based pages of at most 524,288 raw bytes (an empty
  part has zero pages). Pages are principal-scoped, reject an invalid part or
  page with `invalid_params {field}`, and expire with their job result.
- **certify.transcribe_measurements** {job_id} — query; certification observation
  mode only (`certify_observation_disabled` otherwise), principal-scoped to a
  completed, unexpired job. Returns exactly `{self_check_wall_ms,
  longest_progress_interval_ms, ane_shapes, peak_device_memory_bytes,
  peak_storage_buffer_bytes}`. These values are frozen atomically at the original
  job's completion; later inference and observe-off reruns cannot change them.
  `self_check_wall_ms` is unsigned integer milliseconds for the lane's first-load
  self-check: its three sealed short clips, timed from after model load and shape
  admission to the last clip's text compare, covering each clip's WAV decode,
  frontend, engine call and text check. It excludes model load and compile. `longest_progress_interval_ms` is unsigned
  integer milliseconds for the largest gap between consecutive completed-work
  progress or complete observation frames (an `ENCODER_OUTPUT` or `DECODE_TRACE`
  header and its entire validated raw payload, never a partial frame), including inference start and end as
  boundaries. Setup/load/compile and queue time are excluded; a no-progress
  request reports its whole inference interval, not zero. Timer heartbeats do not
  reduce the interval. `ane_shapes` is empty off ANE; on ANE it is the cumulative
  admitted encoder-rung inventory, including evicted shapes, through completion:
  `[{encoder_frames, executables: [{id, layers}], cpu_stages}]`. Encoder frames
  and layer indices are integers. Each rung has 24 uniquely named executables
  partitioning layers 0–23 exactly once; `cpu_stages` holds diagnostic stage names.
  `peak_device_memory_bytes` is unsigned integer bytes, positive on CUDA/Vulkan
  and null elsewhere: the request's engine-owned device-allocation high-water
  mark, including persistent model, input, rung and workspace allocations.
  CUDA counts explicit engine allocations, not other processes or opaque
  CUDA/cuBLAS internal allocations. `peak_storage_buffer_bytes` is unsigned
  integer bytes, positive on Vulkan and null elsewhere, for the largest allocated
  storage-buffer range used by the request. Both nullable fields are always
  present. Unmeasured required telemetry refuses with `measurement_unavailable`
  naming `certify.transcribe_measurements.<field>`, never a fabricated zero.
  Worker protocol v4 is unchanged: `TRANSCRIPT` replies optionally carry
  `peak_device_memory_bytes` and `peak_storage_buffer_bytes`, omitted when not measured.
- **certify.transcribe_observe_rerun** {job_id, audio} — mutate; certification
  observation mode only (`certify_observation_disabled` otherwise), principal-scoped
  to the original completed, unexpired job. `audio` is a fresh committed upload
  `{object_id, mime: "audio/wav", bytes, sha256}` of the identical WAV. Ownership,
  all object facts, actual bytes and equality to the original audio SHA-256 are
  checked before work or writes. A SHA mismatch refuses with `object_mismatch
  {field: "audio.sha256"}`. The original pinned lane/fingerprint is required;
  normal lane locking, pause gates and residency budgets still apply. The rerun
  forces observe=false and returns exactly `{observe: false,
  valid_encoder_frames, triples}`, with triples as base64 little-endian i32
  `(token_id, frame_index, duration)` records. The fresh upload is deleted after
  use, including inference failure; original measurements and results are unchanged.
  Both operations return `unknown_job` for foreign/unknown jobs and `job_not_done`
  for non-completed jobs. Neither operation is allowed on a flow route.

#### Engine ids and declaration states

The request's `model` names one of these engines, as advertised by `role.describe`:

| Engine id | Backend selection | Platform |
| --- | --- | --- |
| `parakeet-tdt-0.6b-v3` | First runnable backend in catalog order: Metal, ANE, CUDA, Vulkan | macOS, Linux, Windows |
| `parakeet-tdt-0.6b-v3-metal` | In-process owned Metal | macOS |
| `parakeet-tdt-0.6b-v3-ane` | Direct Neural Engine worker | macOS |
| `parakeet-tdt-0.6b-v3-cuda` | Owned CUDA worker | Linux, Windows |
| `parakeet-tdt-0.6b-v3-vulkan` | Owned Vulkan worker | Linux, Windows |

The catalog id therefore prefers Metal over ANE on macOS and CUDA over Vulkan
on Linux and Windows, subject to backend availability and device floors. To
request the Neural Engine explicitly, use `parakeet-tdt-0.6b-v3-ane`. Selection
happens before checking install, self-check, pause or quarantine state. Neither
the catalog id nor a lane id falls back when its selected backend cannot serve.
There is no CPU lane, hosted fallback or live streaming transcription.

| State of the selected backend | Engine declared? | Serving outcome |
| --- | --- | --- |
| Not runnable on this machine | No | Permanent `unsupported {field: "model"}`; no work or storage |
| Runnable, not installed | Yes | Permanent `model_not_installed {download_op: "models.download"}` |
| Installed, persisted self-check failure | Yes | Permanent `self_check_failed {model_id, backend, check_id, reason}`; no `retry_after_ms` |
| Installed, transient load/self-check fault or worker quarantine | Yes | `engine_crashed` with `retry_after_ms`; quarantine also names `lane_id` and `quarantined: true` |
| Installed, no serving refusal | Yes | Job accepted for execution |

An unknown model also refuses `unsupported {field: "model"}`. Permanent
self-check failures rerun only after a change to the self-check key (catalog id,
manifest digest, backend, fingerprint, engine identity, OS build or fixture
revision), or after the model is removed and downloaded again.
A load/runtime failure or the 2000 ms self-check ceiling
is transient instead. Neural Engine pause (`lanes.pause` scope `neural_engine`)
keeps the engine declared but refuses `lane_paused`, without selecting another
backend. Installs use `models.download`; no probe or approval is required.

#### Admission limits and retention

Each lane runs one transcription at a time and admits at most 16 non-terminal
jobs; catalog jobs count against their selected lane. Upload reservations and
stored audio total at most 1 GiB. Refusals beyond either bound are transient
`queue_full` with `retry_after_ms`, before storing any job/audio. Accepted input
is RIFF/WAVE PCM16 or float32 (including extensible variants), 8–192 kHz and
1–8 channels, up to 256 MiB. Malformed, unsupported or non-finite WAV input
refuses `invalid_params {field: "audio"}` before engine work. A stated duration
over the limit refuses `input_too_large {field: "input.audio", limit:
"max_duration_ms"}` at submit. A decoded duration/frame overflow instead makes
the accepted job fail with the same permanent code and details before loading
a host, reserving compiled programs for the request's frame count, or running
inference. Duration failures also carry
`{duration_ms, max_duration_ms, valid_encoder_frames, max_encoder_frames}`.
The byte limit uses `input_too_large {field: "input.audio", limit:
"max_input_bytes"}`.

`max_duration_ms` is the selected lane's certified limit, not a global promise
of ten-minute support. Input duration is `ceil(sample_frames * 1000 / sample_rate)`
before resampling. The resampled 16 kHz sample count also determines
`valid_encoder_frames`, which must not exceed that lane's certified frame limit.
Both checks precede engine work; there is no silent truncation or hidden
chunking. Zero-sample WAVs, truncated chunks, other codecs/bit depths and
out-of-range channel counts or rates are invalid audio.

The direct Neural Engine Parakeet lane (`parakeet-tdt-0.6b-v3-ane`) currently
serves audio up to 81,850 ms, the longest duration certified on hardware; Metal,
CUDA and Vulkan Parakeet lanes keep their 600,000 ms limit. A catalog backend may
declare a positive `certified_max_duration_ms` no greater than its profile's
`max_duration_ms`; the serving limit is the smaller of the two, or the profile
limit when no cap is declared. `role.describe`, `models.catalog` and
`models.list` report this serving limit as `max_duration_ms`, and both duration
checks above use it. The cap is serving policy only: it changes no lane
fingerprint, profile digest or installed-byte manifest digest.

Audio bytes live only in owner-only `media-audio` beneath the data folder
(0700 directory and 0600 files on Unix; the user data ACL on Windows), explicitly
excluded from Engram backups. SQLite holds metadata, replay bindings and results,
never inline recordings. Open uploads expire after one hour without append and
become gone on restart. Committed uploads survive restart and remain at least
one hour after commit, or the later of job end and last result/read. Terminal
jobs/results/replays survive restart; queued/running jobs become failed with
transient `module_restarted`. Retention sweeps remove expired audio and results;
startup also removes orphaned audio. These retention, replay and word-shape
rules apply to this media surface.

#### Reading certification observation frames

Certification observations copy results without changing inference. Outside
`certify_observation` mode the result omits `observation`, and
`certify.transcribe_observation` refuses `certify_observation_disabled`.
The descriptor's `lane_id` and `fingerprint` identify the lane that actually ran,
including when the job was submitted under the catalog id. All raw values are
little-endian:

| Field or paged part | Raw encoding |
| --- | --- |
| `triples` (inline base64) | Consecutive i32 `(token_id, frame_index, duration)` triples, 12 bytes each |
| `encoder_output` (paged) | Row-major f32, `valid_encoder_frames * 1024` values |
| `decode_trace` (paged) | 40-byte records: i32 `frame_index`, `token_top1`, `token_top2`, `duration_top1`, `duration_top2`; f32 `token_logit1`, `token_logit2`, `duration_logit1`, `duration_logit2`; u32 `flags` |

Trace duration ids 0–4 select durations of 0, 1, 2, 3 or 4 encoder frames,
respectively (80 ms per frame). Flag bit 0 means emitted and bit 1 means cap hit,
with all other bits zero. `encoder_output` and `decode_trace`
are descriptors only, never inline raw arrays. For each part, read page indices
`0` through `pages - 1` with `{job_id, part, page}`, base64-decode each reply's
`data`, and concatenate in index order. Verify the concatenation against the
descriptor's `bytes` and SHA-256 before decoding records. Page boundaries are
byte offsets, not record boundaries; an empty part requires no reads. Pages are
deleted with the job and remain private to its route principal.

### Other management operations

- **diagnostics.findings** {window_ms?} — read-only query. Omitted/null params
  mean `{}`; `window_ms` omitted/null defaults to 21,600,000 ms (six hours).
  Otherwise it must be an integer in 1–604,800,000 inclusive (seven days).
  Non-object params, extra params and invalid windows receive `invalid_request`;
  an unreadable health store receives `store_failure`, not an empty diagnosis.
  Returns `{result: {generated_at_ms, window_ms, findings}}`, with an empty
  `findings: []` for healthy evidence. The module's Unix-epoch millisecond clock
  supplies `generated_at_ms`. Each finding contains `check`, `key`, `severity`,
  positive `count`, `first_seen_ms`, `last_seen_ms`, `summary`, `samples`, and
  `hint`. Guidance appears only in `hint`; clients must not execute it automatically.
  Findings are severity ordered (`error`, then `warn`); ties use the order below.
  Checks read existing runtime catalog, certification, admission and self-check
  state: they never load a model, execute inference, run a probe or repair state.
  All current checks describe standing conditions, even if they predate the
  requested window; there is no recent-history scan.
  - `catalog_self_check_failed` (`warn`): installed runtime catalog lanes with a
    failed current on-device self-check. Samples contain `lane_id` and
    `checked_at_ms`; the earliest check supplies `first_seen_ms` (report time if
    unknown). Raw failure reasons are not returned.
  - `worker_quarantined` (`warn`): runtime lanes with a current in-memory worker
    crash fence or a cached quarantine load refusal. Samples contain `lane_id`
    and `cached`: true means the refusal has no recorded expiry and may outlive
    the fence. No quarantine start time is held, so both evidence timestamps use
    report time. Host-wide quarantine counts are not attributed to unrelated lanes.
  `last_seen_ms` is report time for these standing checks. Counts precede sample
  trimming; samples are sorted by lane id, capped at five per finding. The entire
  compact response body stays strictly below 65,536 bytes: samples are removed
  from the first finding with evidence until it fits, then tail findings if
  necessary, preserving retained counts. There is no pagination or truncation
  flag. Only lane identifiers, check times and bounded status evidence are
  returned—never prompts, inputs, embeddings, raw errors, artifact paths or
  machine-profile details. Performance-stale alone is omitted because it does
  not refuse serving and must not wake an owner as a certification failure.

- **lanes.pause** {scope: "neural_engine", duration_ms, reason} — mutate. Starts
  a volatile pause of every direct-ANE lane, including cold lanes.
  `duration_ms` must be positive and at most 7,200,000 (120 minutes); a blank
  reason, unknown scope, or a second active pause is `invalid_request`.
  Returns `{paused_until_ms, reason, lane_ids, phase, hardware_free}` immediately.
  `phase: "draining"` means admitted requests are finishing normally. The module
  then unloads affected workers and releases the global direct-ANE lock.
  The duration starts at acceptance and includes draining; it is never extended
  to compensate for a slow or failed drain. Expiry stops waiting for drain and
  immediately re-acquires the lock instead of starting another hardware-free window.
  Wait for `lanes.status` to report `phase: "paused", hardware_free: true` before
  starting a private measurement daemon. Metal, rerank on non-ANE engines, and
  other unaffected lanes continue serving.
- **lanes.resume** {} — mutate. Requests early expiry; idempotent when running.
  Returns the same status shape. It stops waiting for drain immediately, without
  cutting off admitted requests or abandoning an unload's lifecycle guards.
- **lanes.status** {} — query. Returns the pause status above, or
  `{phase: "running", hardware_free: false, lane_ids: []}` with the pause fields
  absent. Expiry is automatic using a monotonic timer, even without another call.
  After expiry or resume the module re-acquires the global lock **before** opening
  admission. If a private daemon still holds it, `phase: "waiting_for_lock"`
  remains paused and a warning is logged on each one-second retry. No worker is
  loaded without the lock. The pause is not persisted: a module restart ends it.
  `models.list` adds `paused_until_ms` and `pause_reason` on affected lane rows
  only while paused, including drain and lock-reacquisition time.
  `phase: "drain_failed"` reports an unsuccessful drain with
  `drain_error: {lane_id, stage, code, message}` (`lane_id: null` for the global
  lock). It always has `hardware_free: false`; measurement scripts must not start
  on this status. Retained lock descriptors are retried until expiry or resume.
  Unload failures cannot be treated as confirmed worker exit and do not release
  the global lock. The deadline/resume ends either failure state, logs the blocked
  lane or global stage and error, then proceeds through lock re-acquisition and
  clears the pause. An unload already running keeps its lane non-ready and its
  lifecycle/serving guards until it finishes. Resumed callers use ordinary model
  loading/deadline errors for that lane rather than an indefinitely extended
  `lane_paused`. `hardware_free` is never reported for an incomplete drain.

While a lane is paused, inference is refused before admission, loading, numerical
self-checks, or shape compilation with stable code `lane_paused`, class
`transient`, `safe_to_retry_same_request: true`. `retry_after_ms` is the remaining
pause duration, with a minimum of 1000 (also used while waiting for the global
lock after expiry). Details are `{lane_id, fingerprint, paused_until_ms, reason}`.
`lane_id` is the exact serving model/lane id, not a backend label or worker id.
For catalog lanes `fingerprint` is the pinned catalog fingerprint, the same value
as a successful reply, readable without loading. For other lanes it is omitted
if not known while paused; absence means **not known while paused**, not null or
a different identity. Callers may use `lane_id` plus `fingerprint` to confirm the
pause is for their lane. Semantic refusals use the normal result error envelope.
New durable `embed.batch` submissions are refused before job diversion. Already
queued jobs that encounter the fence fail transiently with the same `lane_paused`
error; admitted execution drains normally. Replay the identical `embed.batch`
request (same `request_key` and content) after the pause clears to start a fresh
attempt and reuse retained checkpoints. Checkpoint reuse is bounded by job result retention.
Pause/resume are mutate operations and therefore refused on flow-scoped routes.

Operator examples: `ck-synapse-opctl lanes pause --minutes 45 --reason measurement`,
`ck-synapse-opctl --json lanes status`, `ck-synapse-opctl lanes resume`.

- **certify.observations** {} — query, certification-only surface. Consumers
  must not use this operation or depend on its output. It refuses with the typed
  channel error `certify_observation_disabled` unless the generated certification
  config enables `certify_observation` (false by default). When enabled, it returns
  `inventories`, per-worker `worker_requests` counts, an independent supervisor
  `admitted_count`, and `available`. In-process Metal uses no worker IPC, so
  its complete snapshot has empty inventories and request counts and zero admissions.
  CUDA and Vulkan report request counts without ANE placement inventories.
  Direct ANE reports placement from its residency supervisor. Supervisor
  observation capture retains every successful
  ADMITTED inventory across eviction and confirmed-exit restart, separately from
  its cumulative admission count. Observation never changes computation.
- **Certification observation responses** — consumers must not use this surface.
  With `certify_observation` enabled, inline embed and rerank responses include a
  top-level `observation`: `input_ids` are the engine-bound token sequences;
  catalog Qwen rerank also includes `readout_ids: [yes, no]`. Legacy unframed
  worker rerank instead discloses `query_ids` and `candidate_ids` as sent. Disabled
  mode omits the block entirely and records nothing.
- **embed.query** {model, text, …constraints} — interactive class, the
  latency path. Full envelope incl. fingerprint/equivalent_to (one-comparison
  hot path).
- **embed.batch** {model, items[{id, text}], …} — inline under the byte/item
  budget; over budget → job: {job_id} back immediately.
  - **embed.result** {job_id, cursor?, max_page_bytes?} — committed pages are
    readable while the job is still running; `pages_available` on job status
    is the visible committed-page count. The default page is 512 KiB; page
    order is job-internal (length-sorted execution) — items carry ids, KEY
    WRITES BY ID, never by position. Pages survive restart until the result-retention TTL.
    `request_key` makes resubmission idempotent: the same key and digest attach
    to or resume the same work; a different digest is `idempotency_conflict`.
    After a terminal restart failure, a fresh attempt keeps prior committed
    pages readable and resumes by item id.
- **rerank.score** {model, query, candidates[], …} — RAW per-candidate
  scores (no server-side ordering opinions); interactive ≤20 candidates,
  bulk beyond. Catalog `qwen3-reranker-0.6b` declares ANE, CUDA and Vulkan lanes,
  but no Metal lane. Free-form local reranking accepts
  owned engines only, never llama.cpp.
- **microllm.oneshot** {model, prompt, max_tokens, grammar?, …} — greedy.
  `max_tokens` is capped by module config `microllm_max_tokens` (default 512);
  requests above the ceiling are rejected with both numbers in the error.
  `grammar` (GBNF): unset or empty = free-text. When set, constrained
  decoding is permitted only if module config `grammar_enabled` is true (default
  false); otherwise grammar requests fail closed with typed `grammar_disabled`.
  When the constrained runtime is unavailable or not certified, grammar requests
  fail closed with the same reason; the retired `grammar_unavailable_in_build` code
  is not emitted.
- **`models.catalog`** (query) `{query?, task?, runnable_here?, installed?}` →
  `{catalog_revision, models[]}`. Unknown params are rejected. Filters intersect;
  query is trimmed, case-insensitive substring search over id, name and description;
  task matches exactly. Rows are ordered by id. Each row carries `{id, task, name,
  description, default_for_task, upstream {hf_repo, revision}, manifest_digest,
  download_bytes, install_state, backends[]}`. Install state is `not_installed`,
  `installed` or `stale`; `installed: false` includes both not-installed and stale.
  A backend row is `{backend, lane_id, fingerprint, runnable, reason, installed,
  self_check}`; self-check is null without a current install. Reason is null when
  runnable, otherwise `not_supported_on_platform`, `device_missing`,
  `device_below_floor` or `worker_missing`. Download bytes sum distinct manifest
  files needed by runnable backends, independent of installation (zero with none).
  Catalog values and pinned upstream revisions come from the compiled document.
  Transcribe backend rows additionally carry their certified `max_duration_ms`;
  token-shaped limits are not part of those rows.
- **`models.download`** (mutate) `{catalog_id, request_key?}` → a flat download
  status, initially `queued` or `downloading`, before off-request-path fetching
  completes. No automatic downloads occur. Checks, in order: a derived lane id
  is `invalid_request {model_id, catalog_id}`; an absent catalog id is
  `unknown_model {model_id}`; no runnable backend is `backend_unavailable
  {catalog_id, lane_id | null, backends[] {backend, reason}}`. Blank request keys
  are omitted; default is `models.download:<catalog_id>:<manifest_digest>`.
  Keys bind to the JCS SHA-256 of `{catalog_id, manifest_digest}`: different
  content is `idempotency_conflict`; non-terminal requests share one job per
  entry manifest. A committed binding is reusable while the entry is complete.
  Size-correct digest blobs are reused without network; only missing or
  wrong-size blobs are fetched. Every fetched body is size- and SHA-256-verified.
  The shared flat status shapes are `{job_id, state: queued}`, `{job_id,
  state: downloading, bytes_done, bytes_total}`, `{job_id, state: verifying}`,
  `{job_id, state: committed}`, `{job_id, state: failed, error}` and `{job_id,
  state: cancelled}`. Byte counts appear only while downloading and error only
  on failure. A zero-network-byte job moves directly from queued to committed.
  `model.status {job_id}` accepts these jobs and adds `kind: "models.download"`.
  Failure codes include `download_failed {file, reason: network | http_status |
  storage_full, http_status | null}`, `artifact_invalid {file, expected_sha256,
  actual_sha256, expected_size, actual_size, recovery_op: "models.download"}`
  and `module_restarted`. Downloads have a 60000 ms no-body-byte timeout, not
  an absolute execution TTL. Terminal jobs and key bindings expire together
  after the configured result-retention TTL; non-terminal jobs are not purged.
- **`models.download.cancel`** (mutate) `{job_id}` → the same flat download
  status. Cancellation is idempotent and affects every caller attached to the
  shared job. Terminal jobs remain unchanged. An unknown or expired job id is
  `invalid_request`. Publication, commit, cancel and failure share a publish
  fence: exactly one terminal transition wins, and a committed job is never
  cleaned up by cancellation. Cancel/failure removes staging and only newly
  published blobs with no remaining install, pin, registration or job root.
- **`models.remove`** (mutate) `{catalog_id}` → `{catalog_id, removed_manifests[],
  freed_bytes}`. Removes every current and stale install and its self-check
  rows, then unlinks only unreferenced member blobs and owned compiled packages.
  Freed bytes are the on-disk bytes actually unlinked. Checks, in order: a
  derived lane id is `invalid_request {model_id, catalog_id}`; an absent catalog
  id is `unknown_model {model_id}`; any active holder is `model_in_use
  {catalog_id, holders[]}`. Holders are `{kind: loaded, model_id}`, `{kind: job,
  job_id}`, `{kind: self_check, check_id}` or `{kind: request, lease_id}`.
  An active first download is a holder even before any install commits. Only
  after the in-use check may an uninstalled entry return empty manifests and
  zero freed bytes. Malformed or unknown-field params remain channel errors;
  the listed semantic refusals are result-envelope wire errors with details.
- **models.list** — catalog and per-model state and fingerprints. The top-level
  compatibility fields are `table_epoch: 2` and `alias_rows: []`; per-model
  `table_epoch` remains null where exposed.
  recommended_batch, per-row token ceiling (`max_tokens`), ceiling source
  (`max_tokens_source`), bucket/shape ladder (`bucket_ladder`), output dimensions
  (`dims`), dtype/precision (`dtype`), device class (`device_class`), serving
  certification flag (`certified`), and warm-load cost hint (`warm_load_cost_hint_ms`).
  - `recommended_batch` (ADVISORY batch sizing per model; admission remains the
    enforcement — consumers should not carry their own batch knobs). When present,
    `recommended_batch` is `{ "rows": positive_integer, "token_budget": positive_integer }`;
    `rows` is the advised request-row cap and `token_budget` is the corresponding
    aggregate token cap across an entire batch. The field is omitted when an engine
    has no measured policy.
  - `max_tokens` (positive integer) — the per-row token ceiling the lane will accept.
    Sourced from what the module actually enforces at request time (the largest loaded
    bucket on bucketed lanes such as ANE and Metal, or catalog config when unloaded).
    Consumers must use this row limit rather than inferring a per-row cap from
    `recommended_batch.token_budget`.
  - `max_tokens_source` (`"worker_bucket"` | `"runtime_bucket"` | `"catalog_unloaded"` | `"catalog"`) —
    discloses the origin of `max_tokens`. Distinguishes live enforced bucket limits from
    unloaded catalog values so consumers know whether live worker state was consulted.
  - `bucket_ladder` (array of positive integers, e.g. `[128, 256, 512]`) — the ladder
    of sequence lengths the lane accepts. Present when the lane is loaded and enforces
    discrete bucket envelopes; omitted when unloaded or when the lane accepts continuous
    lengths up to `max_tokens`.
  - `dims` (positive integer) — output embedding vector dimensions. Present when
    reported by the loaded model; omitted when unknown.
  - `dtype` (string, e.g. `"f16"`, `"f32"`, `"bf16"`) — numeric precision of the lane.
  - `device_class` (`"metal"` | `"ane"` | `"cpu"` | `"cuda"`) — execution device class.
  - `certified` (boolean, optional) — for parked owned-decode lanes, whether matching
    certification evidence exists under the current machine profile. Profile preloads
    use self-checks instead of certification evidence. Llama generation omits the field.
    **`certified: true` does not mean the lane will serve.** Certification and serving approval
    are separate facts recorded by separate writes: a lane that is certified but not approved
    refuses every request (or, for a substitutable unconstrained request, is answered by a
    different lane). A consumer deciding where to send traffic wants `serving_admission` below.
  - `serving_admission` (`"enabled"` | `"disabled"`, optional) — whether this lane is approved
     to take traffic.
    `"enabled"` requires both an enabled approval row and current certification evidence.
    Omitted for lane classes with no approval concept — absence means "not applicable", never
    "enabled".
  - `serving_admission_reason` (string, optional) — why `serving_admission` is `"disabled"`,
    one of `approval_absent`, `approval_disabled`, `not_certified`, `approval_unavailable`,
    `self_check_failed`, or
    a recorded operator reason. Absent when the lane is enabled or has no approval concept.
  - `warm_load_cost_hint_ms` (float) — observed load duration hint in milliseconds,
     sourced from observed load duration. This is an advisory hint, not a
     performance guarantee or upper bound; actual
    load latency depends on residency (whether the model is already loaded in memory) and
    system load. Omitted if no measurement has been recorded for the model.

  Local embed/rerank rows never report `device_class: cpu`; the current catalog
  reports `metal` only. Other device classes may appear for profile preloads.
  Catalog rows are an exception to approval-based fields above. They add
  `self_check` (`state: pending | passed | failed`, with identity/evidence when
  available).
  For one release catalog `certified` mirrors `self_check.state == passed` and
  `serving_admission` is enabled unless self-check failed, then disabled with
  reason `self_check_failed`. No approval row is consulted. Next release removes
  these two compatibility fields from catalog rows in favor of `self_check`.
  Catalog rows exist for runnable, current-manifest installed backends only,
  have lane id `<catalog_id>-<backend>`, one fingerprint, and initial state
  `unloaded` with `max_tokens_source: catalog_unloaded`. Unload keeps the row;
  remove deletes it. Owned-decode approval gates are unchanged.
  Transcribe rows instead carry `max_duration_ms` and omit `max_tokens`,
  `max_tokens_source`, `recommended_batch`, `bucket_ladder`, `dims`, `pooling`,
  `normalize` and `attention_units`; their envelope counts audio duration and
  encoder frames, not text tokens.
- **model.status** — query by exactly one of `job_id` (a `models.download` job)
  or `model_id` (a registered lane). Job status reports download progress and
  its terminal outcome; lane status reports its runtime load state.

### Catalog serving and migration

The compiled catalog (`catalog_revision: 2`) has five frozen ids:
`gte-modernbert-base` (embed,
`Alibaba-NLP/gte-modernbert-base`, revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`),
`gte-reranker-modernbert-base` (rerank, `Alibaba-NLP/gte-reranker-modernbert-base`,
revision `f7481e6055501a30fb19d090657df9ec1f79ab2c`), `qwen3-embedding-0.6b`
(embed, `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`),
`qwen3-reranker-0.6b` (rerank, `Qwen/Qwen3-Reranker-0.6B`, revision
`e61197ed45024b0ed8a2d74b80b4d909f1255473`), and `parakeet-tdt-0.6b-v3`
(transcribe, `nvidia/parakeet-tdt-0.6b-v3`, revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`). Transcribe backend platforms are
listed in the engine table above. `models.catalog` retains non-runnable backend
rows with a reason such as `not_supported_on_platform`; `runnable_here: true`
filters to entries with a backend runnable on the current machine.
There is no CPU fallback.

A catalog id selects the first runnable backend in catalog order; a declared
lane id or fingerprint pins a backend. The selected lane never falls back to
another backend when loading, failed or uninstalled. Serving uses cached local
files only, lazily loads and runs self-check, never network I/O. Catalog lanes
never read knob assignments or require probes/approvals. The self-check key is
catalog id, manifest digest, backend, fingerprint, engine identity, OS build and
fixture revision. Key changes reset it to pending; persisted numerical failure
refuses without rerunning. Remove and redownload also permits a new check.

Worker quarantine is also a catalog serving error, not a probe/approval gate:
it returns transient `engine_crashed` with `{lane_id, quarantined: true}` and a
positive `retry_after_ms` equal to the remaining fixed ten-minute cool-down.
The lane stays selected; no other backend is substituted. On expiry the next
serving request starts a fresh worker with zero strikes. Explicit unload/reload
or module restart clears the cool-down immediately.

The same consecutive-fault/cool-down rule applies to direct ANE channels.
Quarantine closes and confirms the old worker's exit before refunding resident
shapes, and does not spawn/recompile replacements until recovery is allowed.
Live work has no worker total-time cap: only lack of completed-work progress
for 120 seconds during inference (600 during load/compile/setup) is a hang.
Caller deadlines return `deadline_exceeded` without a strike and submitted work
drains on its owner so the next request cannot receive a stale reply. Existing
direct-ANE compile-blocked deadline waits retain `shape_compiling`.

#### Removed or rejected paths and their replacements

- ORT and MiniLM product/catalog lanes are removed: use a catalog id and explicitly
  download it. ORT remains only as a bench reference, not a portable CPU floor.
- Omitted-model local embed/rerank knob routing is removed: specify a model or
  use the task's `default_for_task` entry (the two GTE entries).
- Unregistered embed/rerank ids no longer return `model_loading`: use the
  `unknown_model` correction flow. Uninstalled catalog ids/lanes return
  `model_not_installed`; call `models.download` rather than polling a load.
- Catalog `certified`/`serving_admission` are one-release compatibility projections:
  consumers must move to `self_check` before those fields are removed next release.
- Unpinned `source=hf`, revision `main`, URLs containing `resolve/main`, and
  resolved HTTP(S) assets without SHA-256 are rejected before network I/O:
  supply a pinned 40-hex revision and a digest for each resolved network asset,
  regardless of `source` (including `source=file`).
 - Preloads using a catalog id or `<catalog_id>-<backend>` fail startup naming
   the entry. Use `models.download` for those ids. Undeclared derived ids are
   `unknown_model` for serving/status/unload, but `invalid_request` for
   download/remove; use a declared lane or catalog id.
- `model.unload` with a catalog id is rejected: unload a declared lane id from
  `details.lane_ids`, then use `models.remove` to delete installation files.

- **admission.status** — advisory snapshot: queue depths, current knob,
   registered lane counts, and job outcome counters.
  Job counters are per-process since start:
  - `jobs_minted`: jobs admitted in this process;
  - `jobs_completed`: minted jobs that reached a successful terminal state;
  - `jobs_failed`: minted jobs that reached a failed terminal state;
  - `jobs_inherited`: jobs reaching a terminal state in this process that were
    admitted by a previous process (startup reconciliation of orphaned jobs or
    resumed jobs);
  - `jobs_open`: derived in-flight count satisfying the identity
    `jobs_open = jobs_minted + jobs_inherited - jobs_completed - jobs_failed`
    (saturating to zero, never underflowing).
  Download jobs count as minted; `committed` counts completed, and `failed`
  or `cancelled` counts failed, preserving the identity above.
   `catalog_lanes` counts registered lanes, including unloaded ones.
   `certified_lanes` counts lanes with current owned-decode evidence, not
   catalog self-check passes. Catalog lanes report `certification_required: false`,
   `certification_status: not_required`, and their `self_check`.
  The `lanes` array remains resident-only because its
  `meeting_deadlines`, rolling start-delay, waiter, and in-flight values describe
  live execution state. The contract lives in per-request budgets, not this
  snapshot.
- **cache.pin / cache.gc** — model cache management (content-addressed,
  shared-lease readers, two-phase GC; GC never deletes under a live reader
  or a foreign pin).

### Owned-decode lane certification

Parked owned-decode lanes serve only with matching persisted certification and
an enabled approval. `models.list` projects evidence as `certified` and projects
serving permission separately as `serving_admission` and
`serving_admission_reason`. No on-device operation writes new certification
rows. The llama.cpp worker path does not require this certification or approval.
### Owned-decode session operations (envelope v2)

Every successful operation below has the management response envelope
`{"result": <payload>}`. Malformed parameters are channel errors with
`invalid_request`; the owned-decode domain refusals below instead use
`{"result":{"module_generation": <u64>, "error": {"code", "class":
"permanent", "safe_to_retry_same_request": false, "message"}}}`. The domain
refusal list is exhaustive for the checks in these handlers; a worker dispatch
can additionally return the normal owned-route errors described by its route.

#### `owned_decode.admit_session`

- Params: `{catalog_fingerprint, caller_id, context_ceiling_tokens,
  generation: {mode: "greedy_top1"}, kv_configuration:
  {block_size_tokens, recurrent_state_grain_tokens}}`. All fields are required
  and unknown fields are rejected. The only accepted generation mode is
  `greedy_top1`; KV block size is one of 256, 512, or 1024 tokens and the
  recurrent-state grain must be nonzero.
- Result payload: `{session_id, catalog_fingerprint, approval_generation,
  reservation: {reserved_embed_rerank_bytes, reserved_artifact_weight_bytes,
  reserved_session_kv_bytes, context_ceiling_tokens}}`.
- Errors: `invalid_request`, `sampling_unsupported`, `artifact_unapproved`,
  `artifact_disabled`, `artifact_revoked`, `artifact_mismatch`,
  `unsupported_machine`, `invalid_context_ceiling`, `insufficient_memory`,
  `incompatible_resident_artifact`, `invalid_kv_configuration`,
  `owned_decode_unavailable`, and `store_failure`.
- Semantics: admission reserves the certified artifact and session KV budget,
  records the serving-session approval generation, and writes no session on a
  refusal. A session can only be admitted against the requested catalog
  fingerprint's current serving approval and matching certification.

#### `owned_decode.decode`

- Params: `{session_id, req_id, prompt, max_tokens, grammar?, deadline_ms?,
  max_queue_ms?}`. All fields before `?` are required; `req_id` is nonempty and
  `max_tokens` is positive. `grammar` is free text when absent or empty.
- Result payload: `{session_id, req_id, frames: [FrameEnvelope...]}`. Each
  frame is envelope-v2: `{protocol: "owned-decode-envelope-v2",
  protocol_version: 2, req_id, session_id, stream_seq, kind, ...}`. `stream_seq`
  starts at 1 and is monotonic. A progress frame has
  `{kind: "progress", progress: {committed_token_ids: [u32...],
  committed_token_count}}`; the IDs are newly committed, while the count is
  cumulative. The terminal is `{kind: "final"|"error", terminal: {req_id,
  session_id, committed_token_count, tokens_emitted, decode_fingerprint,
  processing_fingerprint, runtime_config_digest, worker_generation,
  derived_digest?, terminal_state, decode_mode, speculative_telemetry?}}`.
  The session handler emits serial terminals; `speculative_telemetry` is absent.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `decode_scheduler_busy`, `request_cancelled`, `worker_protocol_error`,
  `scheduler_boundary_error`, `store_failure`, `owned_decode_unavailable`, and
  normal owned-route errors such as `grammar_disabled`, `queue_full`,
  `deadline_exceeded`, `model_loading`, `artifact_invalid`, and
  `engine_crashed`.
- Semantics: the response carries every frame produced by this invocation. A
  token becomes visible only after its scheduler and durable serving-session
  boundary both commit. The terminal accounts for exactly that committed prefix;
  an emergency revoke returns an `artifact_revoked` error terminal after the
  last committed progress frame rather than retracting tokens.

#### `owned_decode.snapshot`

- Params: `{session_id, position_tokens}`; both fields are required and unknown
  fields are rejected.
- Result payload: `{session_id, retained_kv_session_id, retained_position,
  reused_blocks}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `invalid_kv_configuration`, `invalid_kv_alignment`, `artifact_unapproved`,
  `artifact_disabled`, `artifact_revoked`, `artifact_mismatch`,
  `incompatible_resident_artifact`, `store_failure`, and
  `owned_decode_unavailable`.
- Semantics: snapshots are accepted only while the session is idle, at the LCM
  boundary of its KV block size and recurrent-state grain. The retained state
  is durable and belongs to that session and catalog fingerprint.

#### `owned_decode.continue`

- Params: `{session_id, retained_kv_session_id, req_id?}`; unknown fields are
  rejected.
- Result payload: `{session_id, retained_kv_session_id, retained_position,
  reused_blocks}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `retained_kv_unavailable`, `artifact_unapproved`, `artifact_disabled`,
  `artifact_revoked`, `artifact_mismatch`, `incompatible_resident_artifact`,
  `store_failure`, and `owned_decode_unavailable`.
- Semantics: continuation is only for the matching session's retained state at
  an idle boundary. When `req_id` is supplied, its retained stream prefix must
  match the snapshot position, so continuation neither replays nor skips an
  already committed token.

#### `owned_decode.abort`

- Params: `{session_id, req_id, retain_kv}`; all fields are required and
  unknown fields are rejected.
- Result payload: `{session_id, req_id, abort: "requested", disposition:
  "removed_queued"|"deferred_to_boundary"}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`, and
  `owned_decode_unavailable`.
- Semantics: abort is an acknowledgement, not a terminal. The active decode
  observes it at its next committed boundary. If retained, its decode result
  carries `cancelled: {req_id, generation_id, committed_token_count}` beside
  the final error frame; status recovery uses that same committed count.

#### `owned_decode.close`

- Params: `{session_id}`; unknown fields are rejected.
- Result payload: `{session_id, closed: true, unload_artifact}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `routing_error`, `store_failure`, and `owned_decode_unavailable`.
- Semantics: close is idempotent after a successful close, but cannot close an
  active decode. It completes durable serving state and may release the
  artifact only when no active sessions still require it.
- Retention: a closed session (closed by `owned_decode.close` or by an
  `artifact_revoked` terminal) stays known for 15 minutes after it closed, so a
  retried close and a status query after a lost frame both still see it. After
  that window the module forgets the session: `owned_decode.close`,
  `owned_decode.session_status`, `owned_decode.snapshot`,
  `owned_decode.continue`, and `owned_decode.decode` all return
  `unknown_session` for it. A repeated close does not restart the window. Open
  sessions are never forgotten, however long ago they were admitted.

#### `owned_decode.session_status`

- Params: `{session_id, req_id}`; both fields are required and unknown fields
  are rejected.
- Result payload: `{session_id, req_id, committed_token_count, state,
  retained_kv_session_id?}`. `state` is either `"in_flight"` or
  `{state: "terminal", terminal_state: "completed"|"aborted"|
  "artifact_disabled"|"artifact_revoked"|"failed"}`.
- Errors: `invalid_request`, `unknown_session`, `unknown_request`, and
  `owned_decode_unavailable`.
- Semantics: use this result after a lost or gapped envelope-v2 frame. It is
  authoritative for the monotonic committed-token count and never revises
  history already committed by a progress frame. A closed session's terminal
  state stays available for 15 minutes after it closed; after that window the
  session is no longer known and status returns `unknown_session` (see the
  retention rule under `owned_decode.close`).

#### `owned_decode.disable`

- Params: `{catalog_fingerprint, reason}`; both fields are required and
  unknown fields are rejected.
- Result payload: `{approval: {schema_revision, catalog_fingerprint,
  certification_record_id, artifact_id, state: "disabled"|"revoked",
  reason, approved_by, approved_at_ms, updated_at_ms, generation,
  semantic_digest}, invalidated_retained_states, active_sessions,
  termination_requested_sessions, unload_artifact}`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: disable fences new session admissions and invalidates retained KV
  states for that catalog fingerprint, but lets an already active decode finish
  normally. A previously revoked serving approval remains revoked. When
  `unload_artifact` is true, the module also unloads the resident decode worker
  for the model named by the catalog's certification record, whether or not any
  session for that catalog is still known.

#### `owned_decode.revoke`

- Params: `{catalog_fingerprint, reason}`; both fields are required and
  unknown fields are rejected.
- Result payload: the same `ServingControlOutcome` shape as
  `owned_decode.disable`, with `approval.state: "revoked"`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: revoke fences new admissions, invalidates retained state, and
  marks active sessions for `artifact_revoked` terminal accounting at their
  next committed boundary. It also requests the in-memory scheduler revoke and
  can unload the artifact when no active session remains.

### Model unload

#### `model.unload`

- Params: `{model_id}`. `model_id` is required; this handler does not reject
  additional fields.
- Result payload: `{module_generation, model_id, fingerprint, state:
  "unloaded", engine, task}`.
- Errors: `invalid_request`, `unknown_model`, `model_in_use`, `model_loading`,
  and `engine_crashed`. A catalog id carries `{catalog_id, lane_ids[]}`; use a
  declared lane id. An undeclared derived lane is `unknown_model`.
- Semantics: unloading a ready model removes its loaded engine and owned-decode
  dispatch cache, but keeps the catalog model registered and its cached artifact
  available for later lazy reload.

### Owned-decode approval administration

Approval rows are identified by the pair `(model_id, decode_fingerprint)`, not
by model ID alone. The approval mutation path is a fenced store transaction:
each changed identity receives a monotonic `generation`, and migration compares
its single pinned seed marker before writing (the store epoch/CAS boundary).
These wire operations do not accept a caller-supplied expected epoch. The
server recomputes the row's semantic digest in the same transaction.

#### `approvals.migrate_owned_decode`

- Params: `{seed_revision, schema_revision}`; both fields are required and
  unknown fields are rejected.
- Result payload: `{outcome, seed_revision, rows, marker, rendering}`. Outcomes
  include `applied`, `already_applied`, `invalid_seed`, `unmappable_identity`,
  `duplicate_identity`, and `transaction_failed`; non-applied outcomes leave
  approval rows and the migration marker unchanged.
- Errors: `invalid_request`, `approval_migration_state_corrupt`, and
  `store_failure`.
- Semantics: the one-shot migration validates the pinned seed, resolves every
  seed identity to exactly one eligible catalog model, compares its marker under
  the fence, then writes all approval rows and the marker atomically.

#### `approvals.enable`

- Params: `{model_id, decode_fingerprint, grammar_enabled}`; all fields are
  required and unknown fields are rejected.
- Result payload: `{model_id, decode_fingerprint, enabled, grammar_enabled,
  approved_by, approved_at_ms, semantic_digest, generation}`.
- Errors: `invalid_request`, `operator_identity_unavailable`, and
  `store_failure`.
- Semantics: an authenticated operator identity supplies `approved_by`. Enable
  creates or re-enables exactly the requested identity; a new row starts at
  generation 0 and a re-enable increments its generation. The model ID must
  resolve to exactly one owned-metal-decode generate catalog entry.

#### `approvals.disable`

- Params: `{model_id, decode_fingerprint, reason}`; all fields are required and
  unknown fields are rejected.
- Result payload: `{model_id, decode_fingerprint, enabled: false,
  disabled_reason}`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: disable changes exactly one approval identity. The reason must be
  nonempty; disabling a missing identity or a storage validation failure is
  reported as `store_failure` by this handler.

#### `approvals.emergency_rollback`

- Params: `{reason}`; `reason` is required and unknown fields are rejected.
- Result payload: `{disabled, reason}`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: the reason must be nonempty. One fenced transaction disables every
  owned-decode approval identity across the approval lanes, writes that reason,
  recomputes each changed row digest, and increments each changed row generation.
  This approvals-table rollback does not itself invoke `owned_decode.revoke` or
  terminate already active serving sessions.

## 6. Divergence detector (content_sha256 and submitted_sha256)

Check: `embed_query_row_over_ceiling_reports_submitted_and_content_hashes`
(skeleton e2e) proves submitted versus effective content disclosure.

Each embed item response echoes both `content_sha256` and `submitted_sha256`.
`submitted_sha256` is the SHA-256 hex digest of the exact received text bytes
before tokenization and before any truncation. `content_sha256` is the SHA-256
hex digest of the text that was ACTUALLY EMBEDDED (the decoded kept tokens when
the input was truncated).

A difference between `content_sha256` and `submitted_sha256` means the item was
truncated and is not a transport error; `truncation_disclosures` details the
submitted versus effective token counts. Consumers using hashes as transport-
divergence detectors should verify against `submitted_sha256`. Equal hashes
mean nothing was dropped; different hashes mean the row was truncated.
This field is additive and backwards-compatible; existing consumers reading only
`content_sha256` are unaffected.

## 7. Consumer integration

Checks: `embed_query_row_over_ceiling_reports_submitted_and_content_hashes`
(skeleton e2e).

- Persist the envelope fingerprint verbatim. Treat a different fingerprint
  as a different vector space; Synapse does not substitute another fingerprint.
- Resume durable work with the same request key and bound digest, and collect
  every committed page by item ID rather than assuming page order.
- Verify transport input against `submitted_sha256`; use `content_sha256` and
  truncation disclosures to identify the text actually represented by a vector.

## 8. Operational notes

Checks: `serving_admission_channel_fault_restarts_exactly_once` (worker host).

- Crash domain: llama.cpp runs in supervised single-lane workers; engine
  crashes return typed errors and the supervisor enforces a crash budget.
  Every request ends in result, typed error,
  or transport closure.
- Health and diagnostics read existing state. Failed catalog self-checks and
  worker quarantine are visible through the catalog and diagnostics surfaces.
- module_generation on every envelope: if it changes mid-conversation,
  re-poll jobs (prior-generation jobs are terminal module_restarted).


## 9. Durable pages

Every embed job status carries `pages_available` (download jobs do not).
`embed.result` serves committed pages while the job is queued, running, done, or
failed. A page commit atomically publishes item results, page metadata and the
incremented visible count. Previously committed pages remain readable after a
later failure.

The request digest binds the operation, model, constraints, ordered item IDs and
per-item content hashes. Reusing a request key with different content returns
permanent `idempotency_conflict`. Same-key/same-digest resubmission resumes from
committed item IDs.

`embed.query`, `embed.batch` and `rerank.score` ignore retired request fields
`accept_declared` and `allow_equivalent`. `microllm.oneshot` remains strict:
it accepts but ignores `allow_equivalent`, and does not accept `accept_declared`.

Owned-decode responses may add optional `chain_k` provenance. Legacy readers must
ignore additive provenance fields; this introduces no request field.

## 10. Compatibility and load-time verification

Checks: `tests::wire_contract_documents_every_management_operation` (module
library), `all_sixteen_profiles_have_sealed_model_specific_subsets` and
`tampered_subset_digest_refuses_and_names_profile` (certification self-checks).

### Grammar refusal retirement

`grammar_unavailable_in_build` is retired in favor of `grammar_disabled`. For a
constrained `microllm.oneshot` request on a platform or machine without a
certified owned-decode lane, the stable `grammar_disabled` refusal is returned;
when applicable, the original owned-decode refusal remains available as additive
`underlying_owned_decode_refusal_id` provenance. Constrained requests remain
owned-only: they never fall back to llama GBNF or retry unconstrained.

**Consumer notice:** callers matching the legacy `grammar_unavailable_in_build`
string must switch to `grammar_disabled`; the legacy code is no longer emitted.

### Profile preload load-time checks

A preload with a manifest `profile` (`<slug>.<lane>`) uses catalog-style numeric
admission, independently of `certify_observation`. Before serving it, the module
runs a quiet, persisted self-check using a compact sealed subset derived from
`bench/parity/fixtures`. All sixteen profiles use this single reference source;
the existing catalog-install references are unchanged. An absent reference,
invalid seal, or numerical mismatch refuses with `self_check_failed`, naming
the profile. Failed checks stay failed for the same model fingerprint, engine,
OS build, manifest and reference subset digest. Free-form embed and rerank
preloads are rejected at startup; llama generation preloads are unaffected.
Parked `owned-metal-decode` preloads retain their certification and approval gates.

The fixed subset is `short-0`, `boundary-127`, `boundary-128`, `boundary-129`,
plus `pool10-000`, `pool10-001`, `pool10-002` for rerankers. It covers short input,
both sides of the 128-token shape boundary, and a small ranking pool without
running the costly 8192-token case at every load. The original fixture seals are
verified during deterministic subset generation; at load the subset digest is
checked against its committed subset index, the source seal is matched to the
original committed fixture index, and fixture/model/revision identity is checked
by the parity evaluator. Numerical and semantic gates and manifest tolerances
are unchanged. Full-corpus and 8192-token coverage remain certification gates,
not claims made by the small load-time check.

The `ckdev-synapse-certify run --row <row> --model <catalog_id> --assets
<aggregate-dir> --checkout <root> --mirror <dir>` producer certifies each catalog
backend on its declared hardware rows: `metal-m5`, `ane-m5`, `cuda-*`, and
`vulkan-*`. The mirror serves pinned files from
`<hf_repo>/resolve/<revision>/<file>`. The runner checks `models.catalog`, downloads
through `models.download` and polls `model.status` to `committed`. Its first
fixture request loads and checks `<catalog_id>-<backend>`; `model_loading` replies
are retried within a bounded deadline and never count as completed fixtures.
`models.list` must then report a passed self-check and exactly the catalog pin,
and every successful fixture response must carry that same fingerprint.
Direct ANE records include every successful `ADMITTED` inventory and the
independent admission count. An unavailable observation stream is never
interpreted as an empty, complete inventory. Current schema-3 records never hold
a hardware UUID, only its salted SHA-256 digest
(`machine.platform_uuid_sha256` on Apple rows, `machine.gpu.uuid_sha256` on
CUDA and Vulkan rows); `crates/synapse-certify/README.md` gives the exact
construction.

## 11. Fleet bus event hints

Synapse publishes the following versioned subjects. `{acct}` is the bus account
resolved by `cortexkit-bus-client`; the `event.synapse.*` suffixes are logical
subject names, not unqualified NATS subjects.

| Subject | JSON body (inline or fetched) |
| --- | --- |
| `ck.{acct}.event.synapse.job_page.v1` | `{event_id, job_id, page_seq, job_state}` |
| `ck.{acct}.event.synapse.findings_changed.v1` | `{event_id, generation, worst_severity}` |
| `ck.{acct}.event.synapse.download_state.v1` | `{event_id, download_id, state}` |

`event_id` is a fresh UUID v4 and is also sent as `Nats-Msg-Id = event_id`, so
retries use the same message id. Identifiers are opaque strings; `page_seq` and
`generation` are integers. Job-page events follow committed embed pages (zero-based
`page_seq`) or media terminal states (`page_seq: 0`). Download events follow each
actual `models.download` state change, including initial `queued`. Findings events
carry only the generation and worst severity (`ok`, `warn`, or `error`), not finding
bodies; readers re-read `diagnostics.findings`.

Events are hints to **look now**, not replacements for polling. Existing replies
and polling remain unchanged and authoritative. Fact and outbox writes
commit together; publication is asynchronous, may be delayed during bus outages,
and can be retried. All published outbox rows are retained for seven days. Synapse
declares no durable consumer. Flow-scoped routes allow only `diagnostics.findings`
and `events_get`.

### Findings claim-check and `events_get`

`job_page` and `download_state` carry their bodies inline. Only `findings_changed`
is declared in the HELLO manifest's `events`, at version 1 with `headers: ["event_key"]`.
Its notice has an **empty payload**, `Nats-Msg-Id` and `event_key` equal to the
event's UUID, and `Ck-Content-Digest` equal to the bare lowercase SHA-256 hex of
the exact stored body bytes. The shared bus stream codec writes the digest and
message-id headers; no findings text or job content is published.

The read-only query `events_get` accepts
`{event_key, event_name: "findings_changed", version: 1}` and replies with
`{body, digest, event_key, event_name, published_at}`. `body` is a JSON **string**
containing the original serialized `{event_id, generation, worst_severity}`,
not a newly computed finding. `digest` is the same bare lowercase SHA-256 hex as
the notice. An unknown key, another event name, or another version is refused
`event_not_visible`. `published_at` is RFC 3339 UTC and always denotes the event's
store commit time, not the outbox publisher's internal sent marker. Fetches work
before the publish acknowledgment, so a fast notice reader never races that marker.

The exact stored bytes remain fetchable even after newer findings generations
exist, for at least seven days after publication. Samples pinned by a stale-file
test live in `crates/synapse-module/tests/samples/findings-changed-v1/`.
