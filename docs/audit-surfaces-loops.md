# Audit: dead public surfaces + unbounded self-heal loops (synapse-module, synapse-core, synapse-engine-owned)

Status: AUDIT (findings over fixes). Evidence at file:line. Zero-risk one-liners fixed
inline are named in the commit message; everything else is a written finding.

Scope: production crates `synapse-module`, `synapse-core`, `synapse-engine-owned`.
`bench/` and `tools/` are out of scope. The `synapse-worker-*` crates are out of
scope as audit targets but count as binary roots for reachability (the llama worker
binary `ck-synapse-worker-llama` consumes `synapse-core`).

Detector: transitive reachability from binary entry points, not one-hop import.
Binary roots seeded from every file of each binary crate:
- `crates/synapse-module/src/main.rs` (`ck-synapse`) → `synapse_module::run_from_env`
  → `SynapseHandler` → `dispatch_request` (the only root that pulls in the module lib).
- `crates/synapse-module/src/bin/subc_call.rs`, `inline_embed_throughput.rs` are
  wire-client CLIs that import `subc_client_rs` only — they do NOT pull in
  `synapse_module`/`synapse_core` and are not roots for the module lib.
- `crates/synapse-module/src/bin/timeout_worker.rs` is a mock worker (test fixture).
- `crates/synapse-worker-llama/src/main.rs` (`ck-synapse-worker-llama`) is a binary
  root that consumes `synapse-core` (`worker_framing_sync`, `worker_protocol`).
Re-exports (`pub use` in `synapse-core/src/lib.rs`) were resolved before classifying:
consumers name the re-exported symbol (`synapse_core::Fingerprint`), not the module
(`synapse_core::fingerprint::Fingerprint`), so a module with zero direct `mod`-path
imports can still be fully live.

Validation against known-live subsystems (required before trusting the detector):
- Owned-metal embed path (served production traffic this week): `main.rs` →
  `run_from_env` → `initialize` → `from_catalog` → `load_catalog_model_blocking`
  → `OwnedMetalEmbedEngine::new` + `EmbedEngine::load` → `embed_batch`. Detector
  classifies `synapse-engine-owned` `{lib, runtime, modernbert, qwen3}` and the
  `EmbedEngine` trait path as reachable-and-production-called. QUIET on the live
  path. ✓
- Worker_host llama spawn path: `main.rs` → `load_worker_backend_blocking` →
  `WorkerEngine::new` → `WorkerHost::start_worker` → `prepare_listener`/
  `accept_worker_handshake` (re-exported from `synapse-core::worker_transport`/
  `worker_framing`). Detector classifies `worker_host`, `worker_transport`,
  `worker_framing` as reachable-and-production-called. QUIET. ✓
- Certification probe chain: `dispatch_request` → `probe.start` →
  `execute_probe_job` → `execute_embed_probe_for_model`/`execute_rerank_probe_for_model`/
  `execute_generate_probe_for_model` → `store_probe_cert_row`/`store_probe_perf_row`
  → store tables. Detector classifies the full chain as
  reachable-and-production-called. QUIET. ✓

Verdict classes used below:
- **reachable-and-production-called**: transitively reachable from a binary root
  and called from a production dispatch path (not only tests).
- **reachable-but-e2e-only**: reachable from the binary root, but the only caller
  outside the crate's own tests is the e2e suite (`crates/synapse-module/tests/`).
  Contract-live-but-unconsumed fleet-side.
- **unreachable-from-roots**: no transitive path from any binary entry point.
- **test-only**: reachable only from `#[cfg(test)]` modules.
- **never-wired** vs **orphaned-by-refactor**: git history of the wiring site.

---

## Part 1: dead-surface audit

### Table 1a: wire operations → internal chain → verdict

Every op registered in `management_operations()` (lib.rs:8201) dispatches from
`dispatch_request` (lib.rs:1906), which is reachable from `main.rs` → `run_from_env`
→ `SynapseHandler::handle`. All 19 ops are reachable-and-production-called at the
first hop. The audit question is whether each op's full chain to the engine/store
layer it claims to reach is actually composed.

| Wire op | Dispatch site | Internal chain | Chain verdict | Wired history |
|---|---|---|---|---|
| `models.list` | lib.rs:1908 | `catalog_snapshot` → `models_list_payload` | reachable-and-production-called | always wired |
| `embed.query` | lib.rs:1912 | `embed_query` → `resolve_model_for_request` → `execute_embedding` → `EmbedEngine::embed_batch` (local) OR `remote_embed_query` → `RemoteGateway::embed` (remote) | reachable-and-production-called | always wired |
| `embed.batch` | lib.rs:1913 | `embed_batch` → inline `execute_embedding` OR `submit_embed_batch_job` → `execute_embed_batch_job` (local) OR `submit_remote_embed_batch_job` → `execute_remote_embed_batch_job` (remote) | reachable-and-production-called | always wired |
| `embed.result` | lib.rs:1914 | `embed_result` → `get_job` + `get_job_page` (reads `result_pages`) | reachable-and-production-called | always wired |
| **`job.resume`** | lib.rs:1915 | `job_resume` → `store.resume_paused_job` (sets state=`queued`) → **NOTHING re-spawns execution** | **broken chain — see Finding S1** | **never-wired** (born this way in 69e9eb1) |
| `rerank.score` | lib.rs:1916 | `rerank_score` → `resolve_model_for_request` → `execute_rerank` → `RerankEngine::rerank` (worker only; Ort/Owned rejected) | reachable-and-production-called | always wired |
| **`microllm.oneshot`** | lib.rs:1917 | `microllm_oneshot` → `execute_generate` → `GenerateEngine::generate` (worker). **Grammar field hardcoded `None` at lib.rs:4672; non-empty grammar always rejected with `grammar_unavailable_in_build` even when `grammar_enabled=true`** | **broken chain for grammar — see Finding S2** | **never-wired** (born this way in 6a34edc) |
| `model.load` | lib.rs:1918 | `model_load` → `execute_model_load_job` → `load_catalog_model_blocking` → engine `load` | reachable-and-production-called | always wired |
| `model.status` | lib.rs:1929 | `model_status` → `model_slot_snapshot` | reachable-and-production-called | always wired |
| `model.unload` | lib.rs:1919 | `model_unload` → `unload_embedding_model_blocking` | reachable-and-production-called | always wired |
| `cache.pin` | lib.rs:1920 | `cache_pin` → `ModelCache::pin` | reachable-and-production-called | always wired |
| `cache.gc` | lib.rs:1921 | `cache_gc` → `ModelCache::gc_digest` / `gc_to_watermark` → `gc_all` | reachable-and-production-called | always wired |
| `probe.start` | lib.rs:1922 | `probe_start` → `execute_probe_job` (spawn) | reachable-and-production-called | always wired |
| `probe.status` | lib.rs:1923 | `probe_status` → `job_status_payload` | reachable-and-production-called | always wired |
| `probe.report` | lib.rs:1924 | `probe_report` → `lane_measurement_rows` + `knob_assignments` | reachable-and-production-called | always wired |
| `aliases.check_index` | lib.rs:1925 | `aliases_check_index` → `AliasTable::check_index` | reachable-and-production-called | always wired |
| `alias.retract` | lib.rs:1926 | `alias_retract` → `mutate_alias_pair` → `store.retract_alias_pair` | reachable-and-production-called | always wired |
| `alias.declare` | lib.rs:1927 | `alias_declare` → `mutate_alias_pair` → `store.declare_alias_pair` | reachable-and-production-called | always wired |
| `admission.status` | lib.rs:1928 | `admission_status` → scheduler + execution stats + lane measurements | reachable-and-production-called | always wired |

### Table 1b: internal subsystem entry points → caller classification → verdict

| Surface | File:line | Callers outside own file | Verdict | Wired history |
|---|---|---|---|---|
| `LaneScheduler::admit` | scheduler.rs:238 | lib.rs:5086 (`execute_embedding_quanta`) | reachable-and-production-called | always wired |
| `LaneScheduler::next_dispatch` | scheduler.rs:286 | lib.rs:5106 | reachable-and-production-called | always wired |
| `LaneScheduler::complete_dispatch` | scheduler.rs:301 | lib.rs:5146 | reachable-and-production-called | always wired |
| **`LaneScheduler::snapshot`** | scheduler.rs:226 | **none** (zero callers anywhere) | **unreachable-from-roots** | **never-wired** (born in a5bd7f1, never called) |
| **`SchedulerStateSnapshot`** | scheduler.rs:172 | **none** (only constructed by `snapshot`) | **unreachable-from-roots** | **never-wired** (born in a5bd7f1) |
| `decide_admission` (free fn) | scheduler.rs:59 | lib.rs:1367 (`admit_inline`) | reachable-and-production-called | always wired |
| `ModelCache::acquire_read` | cache.rs:159 | lib.rs:2932 | reachable-and-production-called | always wired |
| `ModelCache::ingest` | cache.rs:169 | lib.rs:3055,3065,3080,3097,8082 | reachable-and-production-called | always wired |
| `ModelCache::pin` | cache.rs:255 | lib.rs:8090 | reachable-and-production-called | always wired |
| `ModelCache::read_meta` | cache.rs:264 | internal only (pin, gc_digest) | reachable-and-production-called (internal) | always wired |
| `ModelCache::gc_digest` | cache.rs:283 | lib.rs:8123 | reachable-and-production-called | always wired |
| `ModelCache::gc_all` | cache.rs:357 | internal only (`gc_to_watermark` cache.rs:444,455) | reachable-and-production-called (internal) | always wired |
| `ModelCache::gc_to_watermark` | cache.rs:429 | lib.rs:8126 | reachable-and-production-called | always wired |
| `ModelCache::total_blob_bytes` | cache.rs:388 | internal only (`gc_to_watermark`) | reachable-and-production-called (internal) | always wired |
| `ModelCache::default_root` | cache.rs:125 | lib.rs:1203 | reachable-and-production-called | always wired |
| `WorkerHost::load_model` | worker_host/mod.rs:198 | lib.rs (via `EmbedEngine::load` trait) | reachable-and-production-called | always wired |
| `WorkerHost::embed_batch` | worker_host/mod.rs:228 | lib.rs (via trait) | reachable-and-production-called | always wired |
| `WorkerHost::rerank` | worker_host/mod.rs:272 | lib.rs (via trait) | reachable-and-production-called | always wired |
| `WorkerHost::generate` | worker_host/mod.rs:320 | lib.rs:5863 (`execute_generate`) | reachable-and-production-called | always wired |
| `WorkerHost::unload` | worker_host/mod.rs:372 | lib.rs (via trait) | reachable-and-production-called | always wired |
| `WorkerHost::ping` | worker_host/mod.rs:403 | lib.rs:6662 (`ane_placement_share_for_model`) | reachable-and-production-called | always wired |
| `WorkerHost::health_snapshot` | worker_host/mod.rs:460 | lib.rs:8032 (`worker_health_for_model`) | reachable-and-production-called | always wired |
| `WorkerEngine::new` | worker_host/mod.rs:718 | lib.rs:2778 | reachable-and-production-called | always wired |
| `RemoteGateway::embed` | gateway.rs:210 | lib.rs:3723,4100,4221 | reachable-and-production-called | always wired |
| `RemoteGateway::ensure_certified` | gateway.rs:225 | lib.rs:3685,4017,4173 | reachable-and-production-called | always wired |
| `RemoteGateway::calibrate` | gateway.rs:235 | runtime.rs:827 (`check_profile`), lib.rs:6238 (probe) | reachable-and-production-called | always wired |
| `RemoteGateway::predicted_finish_ms` | gateway.rs:189 | lib.rs:3692,4056 | reachable-and-production-called | always wired |
| `RemoteGateway::profiles`/`catalog_entries`/`logical_handle`/`provenance`/`is_remote`/`profile` | gateway.rs:134-208 | lib.rs (multiple) | reachable-and-production-called | always wired |
| `CircuitBreaker::admit` | runtime.rs:256 | gateway.rs:199,483 | reachable-and-production-called | always wired |
| `CircuitBreaker::record_success`/`record_failure` | runtime.rs:308,322 | gateway.rs:601,650,658,699 | reachable-and-production-called | always wired |
| **`CircuitBreaker::state`** | runtime.rs:359 | **test only** (runtime.rs:1267,1286) | **test-only** | **never-wired** |
| **`BreakerStateSnapshot`** | runtime.rs:204 | **test only** (constructed by `state`, used in tests) | **test-only** | **never-wired** |
| **`BreakerLease::is_half_open_probe`** | runtime.rs:232 | **test only** (runtime.rs:1265) | **test-only** | **never-wired** |
| `ProviderPool::acquire` | runtime.rs:69 | gateway.rs:494 | reachable-and-production-called | always wired |
| **`ProviderPool::subbatch_tokens`** | runtime.rs:107 | **test only** (runtime.rs:1339) | **test-only** | **never-wired** (born in 0613d74) |
| `CredentialManager::acquire`/`acquire_for_job` | runtime.rs:623,651 | gateway.rs:501, runtime.rs (tests) | reachable-and-production-called | always wired |
| `SentinelContinuityCheck::check` | runtime.rs:941 | lib.rs:5010 (`apply_checkpoint_continuity`) | reachable-and-production-called | always wired |
| `OwnedMetalEmbedEngine::new` | lib.rs:279 | lib.rs:2694 | reachable-and-production-called | always wired |
| `OwnedMetalEmbedEngine::load` (trait) | lib.rs:473 | lib.rs:2696 | reachable-and-production-called | always wired |
| `OwnedMetalEmbedEngine::tokenizer_policy` (trait) | lib.rs:442 | lib.rs:2699 | reachable-and-production-called | always wired |
| **`OwnedMetalEmbedEngine::load_from_dir`** | lib.rs:299 | **none** (zero callers anywhere) | **unreachable-from-roots** | **never-wired** (born in 3284a2d, never called) |
| **`OwnedMetalEmbedEngine::embed_tokens`** | lib.rs:336 | **none** (zero callers anywhere) | **unreachable-from-roots** | **never-wired** (born in 3284a2d) |
| **`synapse_engine_owned::capabilities()`** | lib.rs:139 | **test only** (lib.rs:740) | **test-only** | **never-wired** (born in 3284a2d) |
| **`synapse_engine_owned::tokenizer_policy_for_package`** | lib.rs:189 | **none** (zero callers anywhere) | **unreachable-from-roots** | **never-wired** (born in 3284a2d) |
| **`OwnedExecutionConfig`** | lib.rs:107 | only in `load_from_dir` signature (itself dead) | **unreachable-from-roots** | **never-wired** |
| **`ExecutionMode`** | lib.rs:100 | only in `OwnedExecutionConfig` (itself dead) | **unreachable-from-roots** | **never-wired** |
| **`Capability`** | lib.rs:121 | only in `capabilities()` (itself test-only) | **test-only** | **never-wired** |
| `synapse_engine_owned::detect_family` | lib.rs:172 | lib.rs:1705 | reachable-and-production-called | always wired |
| `synapse_engine_owned::engine_identity` | lib.rs:236 | lib.rs (via re-export) | reachable-and-production-called | always wired |
| `synapse_core::worker_framing` (async) | core lib.rs:11 | `worker_transport` (unix.rs, windows.rs) → `worker_host` | reachable-and-production-called | always wired |
| `synapse_core::worker_framing_sync` | core lib.rs:12 | `synapse-worker-llama` binary (runner.rs) | reachable-and-production-called (from llama worker root) | always wired |
| `synapse_core::worker_transport` | core lib.rs:14 | `worker_host` (via re-exported `prepare_listener`/`accept_worker_handshake`), `synapse-worker-llama` | reachable-and-production-called | always wired |
| `synapse_core::cache`/`engine`/`envelope`/`error_contract`/`fingerprint`/`machine_profile`/`scheduler`/`tokenizer`/`worker_protocol` | core lib.rs:3-13 | re-exported and used by module + workers | reachable-and-production-called | always wired |

### Table 1c: producer → consumer pairs (rows written that nothing reads)

Question answered: for each store table / queue / obligation, is the writer
production-reachable AND is the reader production-reachable?

| Table / queue | Writer (reachable?) | Reader (reachable?) | Verdict |
|---|---|---|---|
| `module_meta` | `next_module_generation` (lib.rs:1181 ✓), `bump_table_epoch_tx` (store.rs:2213 ✓) | `next_module_generation` (✓), `alias_table` (✓) | live pair |
| `jobs` | `admit_job` (lib.rs:4786,4056 ✓), `resume_paused_job` (lib.rs:5409 ✓), `fail_prior_generation_incomplete_jobs` (lib.rs:1186 ✓) | `get_job` (lib.rs:5437,4903,5415 ✓), `claim_job_attempt` (lib.rs:4168,4834 ✓) | live pair — BUT `resume_paused_job` writes state=`queued` and no reader ever selects `queued` jobs for re-dispatch (Finding S1) |
| `result_pages` | `commit_job_page` (lib.rs:4318,4966 ✓) | `get_job_page` (lib.rs:5445 ✓), `admit_job` page-count (store.rs:1427 ✓) | live pair |
| `remote_checkpoints` | `commit_job_page` (store.rs:1609 ✓) | `checkpoint_count` (lib.rs:5007 ✓), `committed_item_ids` (lib.rs:4182,4852 ✓) | live pair |
| `alias_rows` | `declare_alias_pair`/`retract_alias_pair` (lib.rs via `mutate_alias_pair` ✓) | `alias_table` (lib.rs many ✓) | live pair |
| `cert_rows` | `store_cert_row` (probe execution, sentinel calibration ✓) | `get_cert_row`/`latest_cert_row`/`has_stale_cert_row`/`get_declared_cert_row`/`declared_cert_row_for_fingerprint` (✓) | live pair |
| `perf_rows` | `store_perf_row` (probe execution ✓) | `get_perf_row`/`latest_perf_row`/`current_perf_rows` (✓) | live pair |
| `knob_assignments` | `replace_knob_assignments` (lib.rs:6401 ✓) | `knob_assignment` (lib.rs:2332 ✓), `knob_assignments` (lib.rs:7869 ✓) | live pair |
| `models` | `upsert_model` (lib.rs:1419,3165 ✓) | `catalog_models`/`catalog_snapshot` (✓) | live pair |
| `remote_url_bindings` | `bind_remote_profile_url` (lib.rs:1401 ✓), `sweep_remote_url_bindings` (lib.rs:1408 ✓) | same (SELECT in both ✓) | live pair |
| worker crash-budget records (`crashes`/`quarantined`) | `record_crash` (worker_host/mod.rs:684 ✓) | `health_snapshot` (worker_host/mod.rs:462 ✓) | live pair |
| gateway continuity checkpoints | `SentinelContinuityCheck::check` writes cert row (runtime.rs:925 ✓) | `check_profile` reads cert row (runtime.rs:859 ✓) | live pair |
| **`jobs` state=`queued` after `job.resume`** | `resume_paused_job` (lib.rs:5409 ✓ writer) | **no reader — no queue drainer selects `queued` jobs for re-dispatch** | **producer with no consumer — Finding S1** |
| `purge_expired_jobs` (retention sweep) | n/a (deletes) | **only called lazily from `embed.result` (lib.rs:5434); no background sweep** | **consumer is lazy-only — unpolled jobs (e.g. paused-then-resumed, or abandoned) never purged** |

### Findings (surfaces)

#### Finding S1 — `job.resume` → `paused_needs_reauth` chain is broken (never-wired, HIGH)

The wire contract (docs/wire-contract-v1.md:240) says: "Consumers resume with
`job.resume {job_id}` after repairing credentials." The implementation:

- `job_resume` (lib.rs:5398) calls `store.resume_paused_job` (store.rs:1729),
  which sets the job state back to `queued` and returns.
- `job_resume` then returns `job_status_payload` — it does NOT re-spawn the
  execution task.
- All job execution is spawned at submit time only (`submit_embed_batch_job`
  lib.rs:4819, `submit_remote_embed_batch_job` lib.rs:4154, `probe_start`
  lib.rs:6118). There is NO background queue drainer — no `SELECT ... FROM jobs
  WHERE state = 'queued'` exists anywhere in the codebase.
- Result: a resumed job sits in `queued` until the execution TTL expires, then
  becomes a terminal failure. The consumer sees `state: "queued"` indefinitely.

The store layer works (test `vault_locked_pauses_and_resumes_a_durable_job` at
runtime.rs:1621 proves `resume_paused_job` sets state=`queued`), but the
composition — re-spawning the execution task after resume — was never implemented.
Git history: born this way in commit 69e9eb1 "add gateway durable store and wire
v1.1"; `job_resume` was never wired to re-spawn. This is (a) never-wired, not a
regression.

Proposed remedy: `job_resume` should, after `resume_paused_job` returns true,
re-spawn the appropriate execution task (`execute_embed_batch_job` or
`execute_remote_embed_batch_job`) based on the job's op type, mirroring the
submit-time spawn. Alternatively, add a background queue drainer that selects
`queued` jobs and dispatches them.

#### Finding S2 — `microllm.oneshot` grammar path is never forwarded (never-wired, MEDIUM)

The wire contract (docs/wire-contract-v1.md:70-73) says: "When set, constrained
decoding runs only if module config `grammar_enabled` is true (default false);
otherwise typed `invalid_request` naming the gate. If the llama worker build
lacks GBNF support, grammar requests fail with reason `grammar_unavailable_in_build`."

The implementation (lib.rs:4570-4582):
```rust
match params.grammar.as_deref() {
    None | Some("") | Some(raw) if raw.trim().is_empty() => {}
    Some(_) if !state.runtime.grammar_enabled => {
        return channel_error("invalid_request", "microllm.oneshot grammar requires grammar_enabled=true ...");
    }
    Some(_) => {
        return channel_error("invalid_request", "grammar_unavailable_in_build");
    }
}
```
When `grammar_enabled=true` and a non-empty grammar is supplied, the code
unconditionally returns `grammar_unavailable_in_build` — it never checks the
worker build and never forwards the grammar. Then at lib.rs:4672:
```rust
grammar: None,  // hardcoded
```
The `GenerateRequest.grammar` field is plumbed end-to-end through
`execute_generate` (lib.rs:5863) → `WorkerEngine::generate` → `WorkerHost::generate`
(worker_host/mod.rs:332 `request.grammar.clone()`) → worker protocol
(worker_protocol.rs:94) → llama runner (runner.rs:667 `grammar_rule`, 1044
`LlamaSampler::grammar`). The llama worker DOES implement GBNF
(`LlamaSampler::grammar` at runner.rs:1045). The e2e test
`microllm_grammar_unavailable_in_build_when_gate_enabled` (skeleton_e2e.rs:2772)
asserts the broken behavior.

This is exactly the fleet bug class: built (worker GBNF sampler), tested (worker
protocol round-trips grammar), plumbed (worker_host forwards it), but the module
dispatch never sets it. Git history: born this way in 6a34edc; never wired.

Proposed remedy: when `grammar_enabled=true`, forward `params.grammar` to
`GenerateRequest.grammar` instead of hardcoding `None` and rejecting. Remove
the unconditional `grammar_unavailable_in_build` arm; keep it only for the case
where the worker build actually lacks GBNF (detect at worker handshake or return
worker error).

#### Finding S3 — Dead public surfaces in synapse-engine-owned (never-wired, LOW)

- `OwnedMetalEmbedEngine::load_from_dir` (lib.rs:299): zero callers anywhere. The
  module uses the `EmbedEngine::load` trait method instead.
- `OwnedMetalEmbedEngine::embed_tokens` (lib.rs:336): zero callers anywhere. The
  module uses `EmbedEngine::embed_batch`/`embed_one` instead.
- `synapse_engine_owned::capabilities()` (lib.rs:139): only called from a test
  (lib.rs:740). The module never queries capabilities.
- `synapse_engine_owned::tokenizer_policy_for_package` (lib.rs:189): zero
  callers anywhere. The module uses `EmbedEngine::tokenizer_policy` instead.
- `OwnedExecutionConfig` (lib.rs:107), `ExecutionMode` (lib.rs:100),
  `Capability` (lib.rs:121): only used by the dead surfaces above.

All born in commit 3284a2d "integrate owned Metal embedding engine" and never
called. (a) never-wired dormant code. Cost: confusion (a reader sees a public
`load_from_dir` and assumes it's the load path; it isn't).

Proposed remedy: delete or gate behind `#[cfg(test)]`/`#[allow(dead_code)]` with
a comment explaining the trait method is the production path.

#### Finding S4 — Dead public surfaces in synapse-core scheduler (never-wired, LOW)

- `LaneScheduler::snapshot` (scheduler.rs:226): zero callers anywhere.
- `SchedulerStateSnapshot` (scheduler.rs:172): only constructed by `snapshot`.

Born in a5bd7f1 "wire inline embedding scheduler"; never called. (a) never-wired.

Proposed remedy: delete or `#[allow(dead_code)]`.

#### Finding S5 — Test-only public surfaces in remote runtime (never-wired, LOW)

- `CircuitBreaker::state` (runtime.rs:359): test only.
- `BreakerStateSnapshot` (runtime.rs:204): test only.
- `BreakerLease::is_half_open_probe` (runtime.rs:232): test only.
- `ProviderPool::subbatch_tokens` (runtime.rs:107): test only.

All never-wired. These are observability/test helpers exposed as `pub(super)`
that no production path calls. `subbatch_tokens` was intended for bulk sub-batch
sizing (design-remote-gateway.md names "target_subbatch_ms") but the gateway
never calls it — bulk requests send the full batch to the provider.

Proposed remedy: gate behind `#[cfg(test)]` or delete.

#### Finding S6 — `purge_expired_jobs` is lazy-only (LOW)

`purge_expired_jobs` (store.rs:1784) is only called from `embed_result`
(lib.rs:5434). Jobs that are never polled by the consumer (e.g. a paused job
whose consumer never calls `embed.result`, or an abandoned job) will never be
purged. The retention TTL is not enforced by a background sweep. This is a
storage-growth concern, not a livelock.

Proposed remedy: add a background sweep (e.g. on a timer or at startup) that
calls `purge_expired_jobs` + `purge_retained_jobs_tx`.

---

## Part 2: unbounded-loop audit

### Table 2: loops → termination → operator-visibility verdict

| Loop | File:line | What terminates it? | Transient vs permanent distinguished? | If condition NEVER clears, what does operator see? | Verdict |
|---|---|---|---|---|---|
| Worker crash-budget respawn | worker_host/mod.rs:677 `record_crash_and_maybe_restart` | `max_crashes` (default 2) within `window` → quarantine (worker_host/mod.rs:689-694) | yes — crash count is per (model,config) key; quarantine is permanent until re-probe | typed `engine_crashed` then permanent `probe_required` after quarantine; `health_snapshot` shows `degraded`/`quarantined_models` (worker_host/mod.rs:472-474) | **OK** — bounded, typed, health-degraded. Matches design-worker-protocol.md:85-88. |
| Gateway provider retry | gateway.rs:478 `for attempt in 0..MAX_ATTEMPTS` | `MAX_ATTEMPTS = 3` (gateway.rs:27) | yes — `FailureClass::Pacing`/`Transient`/`Permanent` (classify.rs); pacing (429) never feeds breaker (runtime.rs:330); permanent exits immediately | typed `provider_unavailable` with `retry_after_ms` or `needs_reauth` (gateway.rs:673-688) | **OK** — bounded (3), typed, matches design-remote-gateway.md:271-274. |
| Vault route bootstrap retry | vault.rs:209 `for attempt in 0..VAULT_ROUTE_ATTEMPTS` | `VAULT_ROUTE_ATTEMPTS = 5`, `VAULT_ROUTE_BACKOFF = 100ms` (vault.rs:14-15) | yes — only retries `CallError::NotSent` (route not yet appeared); all other errors map immediately (vault.rs:215) | typed `VaultError` → `credential_config_invalid` or `provider_unavailable` (runtime.rs:638-647) | **OK** — bounded (5), typed. |
| Credential acquire | runtime.rs:623 `CredentialManager::acquire` | single-shot (no loop) | yes — `VaultLocked`/`NeedsReauth` → `PauseJob`; `NotFound`/`Malformed` → permanent `credential_config_invalid`; `Unreachable` → transient `provider_unavailable` (runtime.rs:634-648) | typed: inline → `needs_reauth`/`credential_config_invalid`; job → paused state | **OK** — no loop, typed. Matches design-remote-gateway.md:223-225. |
| Circuit breaker half-open probe | runtime.rs:256 `CircuitBreaker::admit` | single `probe_in_flight` lease (runtime.rs:280-291); cooldown gate (runtime.rs:272) | yes — only `Transient` feeds breaker (runtime.rs:330); pacing/permanent excluded | `provider_unavailable` with `retry_after_ms = cooldown - elapsed` (runtime.rs:273-276) | **OK** — single half-open lease, matches design-remote-gateway.md:272. |
| Sentinel drift check | runtime.rs:850 `check_profile` | two-run confirmation then quarantine (runtime.rs:908-917) | yes — `DriftState::Quarantined` is sticky; subsequent calls short-circuit (runtime.rs:855-856) | typed `identity_drift` (permanent); cert row stamped `remote_sentinel_quarantined` (runtime.rs:921-924) | **OK** — bounded (2 runs), typed, quarantine is permanent. |
| Remote embed batch job chunk loop | lib.rs:4204 `for chunk in pending.chunks(chunk_size)` | bounded by `pending` item count; each chunk calls `gateway.embed` (bounded retry) | yes — `needs_reauth` → pause; other errors → `fail_job_with_wire_error` (lib.rs:4225-4251) | typed: pause → `paused_needs_reauth`; transient/permanent → job failed with class | **OK** — bounded by item count, typed. |
| Local embed batch job quanta loop | lib.rs:5060 `execute_embedding_quanta` | bounded by scheduler dispatch + token budget | yes — engine errors → `fail_job_with_wire_error` | typed job failure | **OK** — bounded. |
| Startup reconciliation | lib.rs:1186 `fail_prior_generation_incomplete_jobs` | single store call (no loop) | n/a — marks all prior-gen jobs terminal | `module_restarted` error on prior-gen jobs | **OK** — single shot. |
| Pipe reader (worker stdout/stderr) | worker_host/mod.rs:876 `loop { reader.read() }` | EOF (`Ok(0)`) or error (`Err`) (worker_host/mod.rs:878,883) | n/a — terminates on child exit | worker exits → EOF → task ends | **OK** — bounded by child lifetime. |
| **`ensure_model_loaded_for_control` wait loop** | lib.rs:2440 `loop { snapshot.notify.notified().await }` | exits on `Ready`/`Failed`; re-arms on `Resolving`/`Downloading`/`Validating`/`Loading` | **partial — `Failed` exits, but if the background load task is silently cancelled/panics before setting `Failed`, the slot stays in `Loading` forever** | **silent hang — no timeout on the wait; the control op blocks indefinitely with no typed error** | **Finding L1 — per-model silent hang** |
| **`ProviderPool::acquire` wait loop** | runtime.rs:71 `loop { turnover.notified().await }` | permit released via `Drop` (runtime.rs:172) | **no — if all permits are held by hung tasks that never drop, this waits forever; `remaining_deadline_ms` is passed to credential acquire but NOT to pool acquire** | **silent hang — no timeout; the request blocks indefinitely with no typed error** | **Finding L2 — per-provider silent hang** |
| **`acquire_execution_permit` wait** | lib.rs:5603 `runtime.execution.clone().acquire_owned().await` | semaphore permit released via `Drop` (lib.rs:615) | **no — if all execution permits are held by hung tasks, this waits forever; `deadline_ms` is checked at admission but not enforced during the permit wait** | **silent hang — `queue_full` only if the semaphore is closed, not if it's saturated** | **Finding L3 — per-model silent hang** |
| `inline_embed_throughput` poll_job | bin/inline_embed_throughput.rs:296 | `REQUEST_TIMEOUT` deadline (bin:297); exits on done/failed (bin:308-313) | yes — `failed_transient`/`failed_permanent` bail | typed error on timeout | **OK** — bounded by deadline (bench tool, out of scope but correctly bounded). |

### Design-doc-vs-implementation loop bounds

| Design doc | Named bound | Code has it? | Evidence |
|---|---|---|---|
| design-worker-protocol.md:85-88 | "Crash budget: N crashes per (model,config) within window → quarantine; M crashes across all work → lane degraded" | N per (model,config): yes (`CrashBudget::max_crashes` default 2, worker_host/mod.rs:35,689). M across all work: partially — `crash_count_window` is summed across all keys (worker_host/mod.rs:462) and reflected in `degraded` (worker_host/mod.rs:474), but there is no separate "M crashes → lane degraded" threshold; `degraded` is just `crash_count_window > 0`. | worker_host/mod.rs:35,462,474,689 |
| design-worker-protocol.md:66-70 | "PING feeds the background refresher that stamps [health]" | **NO background PING refresher exists.** PING is only called on-demand during ANE placement probes (`ane_placement_share_for_model` lib.rs:6662). The wire contract (wire-contract-v1.md:178) says "ANE worker PING refreshes the last Neural Engine placement share used by explicit certification probes" — which matches the implementation. The design doc's "background refresher" language is aspirational/not implemented. | lib.rs:6662; no `spawn`/`interval` for PING anywhere |
| design-remote-gateway.md:271-274 | "shared retry budget" (no specific number named) | `MAX_ATTEMPTS = 3` (gateway.rs:27) | gateway.rs:27,478 |
| design-remote-gateway.md:273-274 | "429 feeds AIMD pacing only — never increments transport-breaker failure counts" | yes — `record_failure` only records `FailureClass::Transient`; pacing excluded (runtime.rs:330-331). Note: "AIMD" is implemented as simple retry-after honoring (gateway.rs:646 `sleep(retry_after)`), not additive-increase/multiplicative-decrease of a concurrency window. | runtime.rs:330; gateway.rs:646 |
| design-remote-gateway.md:272 | "single half-open lease across tiers" | yes — `probe_in_flight: true` blocks other admits (runtime.rs:280-291) | runtime.rs:280-291 |
| design-synapse-module.md:225 | "On startup, any prior-generation job in queued/running becomes terminal" | yes — `fail_prior_generation_incomplete_jobs` (lib.rs:1186) | lib.rs:1186; store.rs:1757 |

### Findings (loops)

#### Finding L1 — `ensure_model_loaded_for_control` can hang silently (per-model, MEDIUM)

`ensure_model_loaded_for_control` (lib.rs:2440) loops on
`snapshot.notify.notified().await` waiting for the model to leave
`Resolving`/`Downloading`/`Validating`/`Loading`. It exits on `Ready` or
`Failed`. The background load task (`load_catalog_model_task` lib.rs:2465)
normally sets `Ready` or `Failed`. BUT: if the `tokio::spawn` at lib.rs:2430 is
cancelled (runtime shutdown) or the `spawn_blocking` join panics in a way that
bypasses the error handler, the slot stays in `Loading` and the waiter loops
indefinitely with no timeout and no typed error. The control op
(`model.status`/`model.unload`/`probe.start` for that model) hangs silently.

Blast radius: per-model (only control ops on the stuck model hang; embed ops
use `resolve_model_for_request` which has its own path).

Proposed remedy: add a timeout to the `notify.notified().await` wait (e.g.
`tokio::time::timeout`) and return a typed `model_loading` error on expiry, or
have the background task install a `Drop` guard that sets `Failed` if the task is
cancelled.

#### Finding L2 — `ProviderPool::acquire` can hang silently (per-provider, MEDIUM)

`ProviderPool::acquire` (runtime.rs:71) loops on
`self.inner.turnover.notified().await` waiting for a permit. Permits are
released via `Drop` (runtime.rs:172). If all permits are held by tasks that
never complete (e.g. a hung provider request with no timeout), this waits
forever. The outer `execute_embedding_request` receives `remaining_deadline_ms`
but passes it to `credentials.acquire` (gateway.rs:508) — NOT to
`pool.acquire` (gateway.rs:494 has no timeout wrapper).

Blast radius: per-provider (all requests to that provider hang; other
providers unaffected).

Proposed remedy: wrap `pool.acquire` in `tokio::time::timeout(remaining_deadline_ms)`
and return typed `provider_unavailable` on expiry.

#### Finding L3 — `acquire_execution_permit` can hang silently (per-model, MEDIUM)

`acquire_execution_permit` (lib.rs:5603) waits on
`runtime.execution.clone().acquire_owned().await` (a semaphore). If all
execution permits are held by hung inference tasks, this waits forever.
`deadline_ms` is checked at admission (`admit_inline` lib.rs:1339) but not
enforced during the permit wait. `queue_full` is only returned if the semaphore
is closed (lib.rs:5624), not if it's saturated.

Blast radius: per-model (all inline embed/rerank/generate ops on that model
hang; other models unaffected).

Proposed remedy: wrap `acquire_owned` in `tokio::time::timeout` bounded by the
request deadline and return `queue_full` on expiry.

---

## Summary

**Dead surfaces (Part 1):**
- 2 broken wire chains (S1: `job.resume` never re-dispatches; S2: grammar never
  forwarded) — both never-wired, both contract-promised but composition missing.
- 11 dead public surfaces (S3-S5): `load_from_dir`, `embed_tokens`,
  `capabilities()`, `tokenizer_policy_for_package`, `OwnedExecutionConfig`,
  `ExecutionMode`, `Capability`, `LaneScheduler::snapshot`,
  `SchedulerStateSnapshot`, `CircuitBreaker::state`/`BreakerStateSnapshot`/
  `is_half_open_probe`, `ProviderPool::subbatch_tokens` — all never-wired,
  no regressions found.
- 1 lazy-only consumer (S6: `purge_expired_jobs`).
- 0 orphaned-by-refactor findings (all dead surfaces were born dormant).
- 0 e2e-only findings (the e2e suite exercises the wire but every wire op is
  also reachable from the production binary root).

**Unbounded loops (Part 2):**
- 3 silent-hang loops (L1: model-load wait, L2: provider-pool acquire, L3:
  execution-permit acquire) — all per-model/per-provider, all missing timeouts
  on async waits where the design implies bounded wait.
- 0 daemon-wide livelocks (no infinite retry with healthy-looking health found).
- 1 design-vs-implementation gap (background PING refresher described in
  design-worker-protocol.md is not implemented; the wire contract's on-demand
  language is accurate).

**Producer-consumer (Part 1c):**
- 1 producer with no consumer: `resume_paused_job` writes `queued` but nothing
  reads `queued` jobs for re-dispatch (S1).
- 1 lazy-only consumer: `purge_expired_jobs` (S6).

**Zero-risk one-liners fixed inline:** none. All findings require composition
changes (re-spawn after resume, forward grammar, add timeouts) that are beyond
the zero-risk one-liner bar. The audit is findings-only.