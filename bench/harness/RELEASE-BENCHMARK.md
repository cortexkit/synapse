# Release benchmark

This runner is separate from the legacy corpus/power matrix. It produces evidence;
it does not certify hardware and never invents measurements when a runtime is
missing. Use Python 3.9 or newer (standard library only).

```
bench/run-matrix.sh release row --source <40-character-S> --config <row-config.json>
bench/run-matrix.sh release assemble --source <40-character-S>
python3 -m unittest discover -s bench/harness -p 'test_release_matrix.py' -v
```

Run `row` on each of the eight certification machines with extracted candidate
assets and the default row fixture. Transfer each `<row>.json` **and its raw/**
subtree into `docs/evidence/benchmark/<S>/` in the evidence checkout. Run
`assemble` there to write `report.json`. Commit that directory in the evidence-only
commit E (whose parent is the candidate source S), including
raw probe stdout/stderr/exit codes and adapter requests. The assembler invokes
the s11 tag-job benchmark validator separately for owned and competitor cells;
owned cells live in `cells`, competitor cells in `competitors.<runtime>` so owned
Qwen3 drop records do not incorrectly label competitor support as dropped.

## Deployment configuration

Runtime deployment, cache control, privileged device telemetry and SUBC discovery
are machine-specific. The harness supplies HTTP inference adapters for `llama.cpp`,
`lm-studio`, `ollama`, `tei`, and a normal-operation adapter for `synapse`. It
requires the operator to supply a **controller executable** on each machine;
there is deliberately no guessed GPU allocation, zero-RSS fallback, fake cold
cache, or fabricated latest-version string. Controllers manage persistent servers
and must not restart them for each sample. Install the latest competitors at run
time and record their actual versions and settings, not a version from this file.
Any missing controller, failed inference, malformed response, authentication or
server fault aborts the row instead of being relabeled as unsupported hardware.

A row config has this shape (repeat the model block for all four manifest slugs):

```json
{
  "source_commit": "<S>",
  "manifest_digest": "<SHA-256 of bench/parity/models.json bytes>",
  "row_id": "metal-m5",
  "machine": {"id": "Mac17,6", "os": "record actual OS", "driver": "record actual driver", "memory_kind": "footprint"},
  "models": {
    "gte-modernbert-base": {
      "certification_record": "docs/evidence/certification/<S>/metal-m5/gte-modernbert-base.json",
      "f16_sha256": "<shared competitor f16 file SHA-256>",
      "inputs": {
        "text": "<shared manifest-reference text>",
        "composed_length": 512
      },
      "adapters": {
        "synapse": {
          "command": ["python3", "bench/harness/synapse_adapter.py", "<synapse-bridge-config.json>"],
          "precision": "f16", "file": "<candidate model file>", "file_sha256": "<SHA-256>"
        },
        "llama.cpp": {
          "command": ["python3", "bench/harness/competitor_adapter.py", "<llama-bridge-config.json>"],
          "precision": "f16", "file": "<shared f16 file>", "file_sha256": "<SHA-256>"
        }
      }
    }
  }
}
```

Add all four competitor keys to every model. Rerank `inputs` instead contains
`query`, exactly 100 `candidates`, and 100 `composed_lengths` from the manifest's
reference tokenizer/grammar (including all template/special tokens). The harness
reproduces the parity evaluator's consecutive-item split with the row's default
64-item/8192-token budgets and singleton 8192-token candidates; the adapter executes
those splits without changing defaults. Embedding admission also accounts for the
total composed token budget; a batch of 32 512-token texts is a job even though it
is below the item ceiling. Every workload has 3 warmups and 20 measured calls.
Nearest-rank p50/p95 are measured indices 9/18; the median is the mean of indices
9/10. Throughput uses items per median elapsed second. Single-query rerank latency
uses one candidate, separately from the 10/100 pools.

All competitors must receive the identical text inputs and the same f16 file
hash. A deployment that must convert a file should prepare and identify that
shared file before the run; never silently substitute a quantized file. Metal GTE
rerank's owned config is `fp32`; its cell carries the precision-mismatch label,
and the runner emits no speed-comparison ratio.

### Bridge and controller protocol

Every adapter invocation receives one JSON request on stdin and writes one JSON
object to stdout. Requests carry source/session/row/model/operation, manifest and
input SHA-256, file path/SHA-256, shared inputs and the default row fixture.
The process is a call bridge to a persistent runtime, not the inference worker.
Timeouts and unclassified nonzero exits are hard errors. The harness saves every
invocation's exact stdout, stderr, exit code, command and request in `raw/`.

Competitor bridge config:

```json
{"endpoint":"http://127.0.0.1:8080", "controller":["/path/to/deployment-controller"], "timeout_seconds":300}
```

The bridge invokes the configured controller with these actions:

* `load`: start or select the exact f16 model, verify file hash and manifest
  inputs, inspect runtime tokenization, and return `running: true`, `model`,
  `manifest_digest`, `input_digest`, `file_sha256`, `precision`, actual `version`,
  `settings` (including launch argv, dtype and hardware),
  `tokenization_differences` (empty array only if verified identical),
  `inline_max_items` (at least 128 for the supplied HTTP adapters), and
  `engine_batch_cap` (actual runtime caps). The bridge then makes one real embed
  or rerank call as the support probe. Ollama has no assumed rerank capability:
  its `/api/rerank` rejection is recorded, not synthesized into a score.
* `memory`: return peak allocation telemetry **covering the preceding request**:
  `kind: "rss_plus_device"`, `process_bytes` and `device_bytes` for discrete GPUs;
  `kind: "footprint"`, `process_bytes` for unified memory (including the Ally X
  integrated Radeon). Set the machine's `memory_kind` accordingly. Include module plus
  worker process memory for Synapse. Device-local memory is not the model-file
  size. Controllers must sample during inference; an idle post-call snapshot is
  not peak memory. Optional `machine_load` carries OS/device utilization counters;
  the harness also records load averages, CPU count and timestamps before/after
  every sample. On Windows, where load averages are null, the controller must
  supply CPU/device load telemetry; null is never claimed to mean zero load.
* `cold_load`: clear caches, unload/reload, and return `caches_cleared: true`,
  `elapsed_ms`, plus cache-clearing command/output evidence. This load is not part
  of the warm series. Permission failure is not a cold-load measurement.
* `parity`: evaluate actual runtime output against the pinned fp32 reference,
  returning `model`, `manifest_digest`, `input_digest`, `gates`, and evaluator
  metrics/fixture identity. This is a benchmark parity result, not a replacement
  for the certification record. Failed gates are retained, not hidden.

A controller that cannot run on this hardware exits nonzero, writes
`{"status":"hardware_unavailable","cause":"<exact stderr line>"}` to stdout
and the cause line to stderr. Include `version`, `settings` and
`tokenization_differences` in that failure object too; version and tokenization
may be null only when unavailable and the settings must explain why.
A running competitor rejecting the model uses
`model_unsupported`. Only those explicit statuses with quoted raw lines are
accepted. HTTP 400/404/422 from a loaded running server are recorded as model or
operation rejection; other HTTP failures abort. Non-measured cells have all
metrics null. Even dropped owned combinations still probe each competitor.

Synapse bridge config adds `model_id`, `fingerprint`, and `rpc_command`, an argv
array containing `{method}` and `{params}` placeholders. The command must invoke
the candidate module's SUBC management route (JSON `{method, params}` requests
addressed to module `synapse`) and return its JSON body
(or a SUBC envelope with a `body` JSON string/byte array). For example, use the
repository's `subc_call --module synapse --method {method} --params {params}` with
an explicit connection file. The adapter issues `embed.batch`, waits for durable
jobs through `embed.result`, fetches every result page, and issues `rerank.score`
for each split. It pins `required_fingerprint`, verifies actual inline/job routing
and scores/vector counts, and includes job completion/delivery in elapsed time.
The controller handles load/cache/parity/telemetry as above. Its `load` response
uses the same binding fields; owned profiles need not have competitor cap fields.

## ANE transition

The `ane-m5` row config additionally has a `transition` object:

```json
{
  "inputs": {"text":"<text verified to compose to exactly 512 tokens>", "composed_length":512},
  "ane-coreml-worker": {"command":["python3","bench/harness/synapse_adapter.py","<coreml-config.json>"]},
  "ane-direct-worker": {"command":["python3","bench/harness/synapse_adapter.py","<direct-config.json>"]}
}
```

Each controller's `prepare_transition` loads the exact lane and verifies the
composed input length, returning `lane` and `composed_length: 512`. The two configs
must pin distinct served fingerprints. Both 23-sample series execute in one row
session on the same machine. `ane-transition.json` records both raw series,
paired per-index machine load records, both medians and `bar_met` using the exact
`direct <= 1.05 * coreml` inequality. ANE burst slowdowns of about 20% are noted in
the measured cells. Dropped Qwen3 cells copy `drop_cause` from the model's
configured `certification_record` and, for a
latency drop, both raw series; their metrics remain null.

No hardware evidence is bundled with these producer tests. Real candidate runs
and the evidence-only commit are release-operator work; run this harness on
the named machines before tagging.
