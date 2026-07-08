# Synapse module v1 design (Lane 1)

Status: DRAFT for async review. Constraints inherited from: SUBC review rounds
(pm_aa948240, pm_5f7e36ff), MC contract (pm_56c42b11, pm_82691068), AFT riders
(pm_4373925e, pm_74b2aff8), spike evidence (bench/lanes/llama-inproc,
bench/spikes/unified-rt, bench/lanes/candle-embed), Decision #1 doc.

## Shape

One Rust workspace, module template per ai-provider-quota (two-crate split):

- `synapse-core`: pure logic — engine seam, admission, fingerprints/alias table,
  model cache, batching, tokenization. No subc, no I/O beyond what traits inject.
- `synapse-module`: wire binary — subc-client-rs serve(), ModuleHandler, config,
  vault (later, for remote endpoints), storage descriptor consumption.

Pins: subc-protocol 0.7.0, subc-transport 0.3.1 (D-008). Singleton module
(machine-wide admission requires it). ManagementSurface provider role, ops all
kind=Query except explicit mutations (probe.start, model.load, cache.gc).

## Engine seam (the two-lane contract)

```
trait EmbedEngine {
    fn identity(&self) -> EngineIdentity;          // engine + version + build flags
    fn load(&mut self, artifact: &ValidatedArtifact, cfg: &RuntimeConfig) -> Result<LoadedModel, EngineError>;
    fn embed_batch(&self, model, batch: TokenBatch) -> Result<Vectors, EngineError>;
    fn embed_one(&self, model, ids: TokenIds) -> Result<Vector, EngineError>;   // hot query path
    fn unload(&mut self, model);
}
// rerank + microllm get sibling traits (RerankEngine, GenerateEngine) — same
// lifecycle, different call shapes. One engine object may implement several.
```

v1 engines behind the seam: `llama-inproc` (batch workhorse, all platforms),
`ort-inproc` (Apple CPU floor + ONNX/DirectML lane), `llama-child` (retained
ONLY where measured better: Qwen-class hot single-query path, 7.4ms vs 17.7ms —
drops out if the latency spike closes the gap), MLX + ANE join per probe
certification when their lanes pass gates. Lane 2's owned runtime implements the
same traits and graduates per component. Engine assignment is per (machine,
workload) from probe results — never hardcoded.

Crash-domain rules (SUBC): artifact digest+format validation BEFORE any engine
touches a file; engine-load failure = classified permanent error, never retried,
never process-fatal where the library allows. Known live risk: GGML asserts
abort the process (observed: builtin-pooling encode path) — engines carry an
allowlist of validated (model-family, config) combos from the probe, and
anything outside it is rejected at admission, not discovered in ggml.

## Tokenization ownership

Synapse owns tokenization (HF tokenizers crate) for in-process engines; child
engines tokenize server-side (their contract). Hard rule from the padding
incident: published tokenizer.json artifacts are SANITIZED at cache-ingest time
(padding config stripped, truncation ours, recorded in the artifact fingerprint).
Token counts reported to consumers are real-token counts, never padded counts.

## Fingerprints and the alias table

Per the signed contract (decision-1-runtime.md "Fingerprint and equivalence
contract"): strict fingerprint = hash(model digest, quant, engine-lane,
runtime-config-canonical); alias table = Synapse-owned, versioned, epoch-bumped
on any row change; every embed/rerank response carries (fingerprint, table_epoch,
provenance, dims). equivalent_to list inline on responses. Revocation never
retroactive; re-probe triggers explicit (engine version bump, runtime-config
change, model file hash change, OS build change). Day-1 declared pair:
llama-f16 ≡ ort-fp32 (1.00000 full-corpus). MLX bf16, DWQ: distinct fingerprints
always (measured non-members).

## Surface (ops)

Poll-first, job-shaped for anything slow (SUBC: 32-credit route window):

- `embed.batch` — {model_id, input_type query|document, items[{id, text}]} →
  {fingerprint, table_epoch, dims, provenance, vectors[], real_token_counts[]}.
  Token-budget batching internal; length-sorted; order restored.
- `embed.query` — single text, latency path: routed to the hot-query engine
  config (small context, pre-warmed). Same response envelope, one item.
- `rerank.score` — {model_id, query, candidates[]} → per-candidate RAW scores +
  fingerprint. Batch-oriented. (gte-reranker-modernbert default.)
- `microllm.oneshot` — {model_id, prompt, max_tokens<=64, grammar?} → text +
  token counts. Greedy default.
- `model.load` — job: returns {job_id, state: building|ready} immediately;
  `model.status {job_id|model_id}` polls. Load work happens in handle() spawns.
- `models.list` — cached catalog + per-model state + fingerprints + aliases.
- `probe.start` / `probe.status` — explicit op, job-shaped, ~1-2 min micro-bench;
  writes certification rows. NEVER auto-triggered; staleness surfaces as a flag
  in health/status when re-probe triggers fire.
- `cache.pin` / `cache.gc` — model cache management (Synapse owns GC, refcounts).
- `admission.status` — queue depths, per-lane saturation, for consumer fast-fail.

Error contract: every error carries class transient|permanent (MC's hard
requirement) + retry_after_ms for transient. Admission rejection is its own
error (queue_full, fast-fail) — queue latency is NEVER hidden inside a call.

## Admission (machine-wide)

Single process = single authority (the AFT 6-process incident is structurally
impossible through this surface). Two queues per engine lane: interactive
(embed.query, small rerank — bounded latency, shallow queue, fast-fail when
deep) and bulk (embed.batch, corpus work — deep queue, backpressure via typed
queue_position response). GPU lanes get one in-flight batch each; CPU lanes get
a thread-budget. The speed-vs-energy knob maps to engine/config selection +
batch budgets per the probe's measured table, not to scheduling tricks.

## Model cache

~/.local/share/cortexkit/models/ per SUBC convention: content-addressed
(sha256 digest key), tmp+fsync+atomic-rename writes, cortexkit-lease lock
(module=models-cache, scope=digest) for concurrent writers, refcount/pin
metadata beside blobs, Synapse-owns-GC. Artifact record = {digest, source url,
format, sanitized-tokenizer digest, validation state, pins}. synapse-private
state (probe results, alias table, certification rows) in
<data_home>/cortexkit/synapse/store.db via HELLO_ACK storage descriptor.

## Health

health() serves cached in-memory state ONLY (SUBC invariant): model load states,
engine lane states, queue depths — stamped by the serving/loading paths
themselves; liveness ages computed at probe time. Cold load in progress =
degraded + detail "loading <model>". No engine/GPU/child probes on dispatch.

## Config

~/.config/cortexkit/synapse.jsonc (module-owned convention): knob
(performance|balanced|quiet), model pins, remote endpoint list (post-v1),
cache budget. Everything else measured (probe) or contracted (subc).

## v1 cut line

IN: embed (batch+query), rerank.score, microllm.oneshot, model lifecycle +
cache, probe, admission, fingerprint/alias surface, health, llama-inproc +
ort engines, knob mapping on Apple + x86.
OUT (designed-for, post-v1): remote endpoint lane (vault flow), MLX/ANE engine
graduation (Lane 2 + spikes feed this), image/STT/TTS, agentic LLM serving,
DirectML activation, alias events beyond the day-1 pair.

## Open items feeding this design

- Qwen hot-query path: in-process vs slim child — decided by the latency spike.
- ANE spike verdict → quiet knob tier reality on M1-gen hardware.
- probe workload exact contents (small built-in corpus + reference vectors
  shipped in the module — needs a size/licensing pass).
