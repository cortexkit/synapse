# ANE-prefill certification harness

`certify.py` is the machine-facing certification gate for the six Qwen3-0.6B
ANE-prefill arms:

| bucket | f16-step | q8-step |
| --- | --- | --- |
| W128 | required green + 5x worker-TTFT gate | required green + 5x worker-TTFT gate |
| W256 | green or one deterministic ABSENT reason | green or one deterministic ABSENT reason |
| W512 | green or one deterministic ABSENT reason | green or one deterministic ABSENT reason |

The harness has no mock-success mode. It accepts raw observations only from a
machine adapter and computes the certification verdict itself. `test_certify.py`
uses a fake adapter only to prove that the harness rejects a W128 token mismatch,
requires every enumerated reason, and refuses a partial timing path.

## Run

On the certification machine, first build an adapter that drives the production
router/worker and the existing `bench/spikes/ane-prefill-split` CoreML/Metal
comparison tools. The adapter must provide the JSONL protocol below. Then run:

```sh
python3 tests/ane-prefill-certification/certify.py \
  --output evidence/ane-prefill-split/evidence-record-v1.json \
  --driver python3 path/to/machine_driver.py --machine-config path/to/synapse.jsonc
```

The output path is intentionally explicit: an operator cannot accidentally
replace a prior evidence record while only checking the harness. The output is
not a frozen record until the evidence workflow validates and commits its digest.

Run the deterministic harness tests with:

```sh
python3 tests/ane-prefill-certification/test_certify.py
```

## JSONL driver protocol

The driver reads exactly one JSON object per line from stdin and emits exactly
one JSON object per line to stdout. Diagnostic text belongs on stderr. Every
operation is issued by the harness; a driver must never return a self-computed
certification verdict.

All arm objects have this exact shape:

```json
{"machine_profile":"<profile digest>","family":"qwen3-0.6b","bucket":128,"decode_config":"f16-step"}
```

### `metadata`

Return the current profile and source checkpoint digest:

```json
{"status":"ok","machine_profile":"<nonempty profile digest>","source_checkpoint_digest":"<lowercase sha256>"}
```

### `precondition`

The harness supplies `arm`. A machine that cannot execute the required attempt
returns exactly one deterministic absence:

```json
{"status":"absent","absence_reason":"capacity_precondition_unmet","detail":"W512 requires 576 positions; highest exposed q8 bucket is 512"}
```

A ready arm returns the complete digest triple. The harness requires all values
to be SHA-256 strings, requires the source digest to equal `metadata`, and
requires the loaded compiled digest to equal the certification-row digest:

```json
{"status":"ready","artifact_triple":{"source_checkpoint_digest":"<sha256>","derived_or_compiled_artifact_digest":"<sha256>","certification_recorded_artifact_digest":"<sha256>"}}
```

### `generate`

The harness supplies an arm, `engine` (`gpu` or `ane-split`), immutable token
IDs, `max_tokens: 64`, `greedy_top1: true`, and a case ID. It optionally supplies
`grammar: "json-object"` or `chain_k: 16`. Return the raw generation and
fixed-window witnesses:

```json
{"status":"ok","generated_token_ids":[1,2],"padded_width":128,"first_token_index":127,"active_cache_positions":128,"decode_cache_position":128,"cache_handoff":"engine_to_engine"}
```

`generated_token_ids` must contain exactly 64 integers. `cache_handoff` is
required for q8-step and must be `engine_to_engine`; it is ignored for f16-step.
The harness compares split and same-config GPU token IDs itself, checks active
positions, and rejects padded-logit or padded-cache behavior.

### `routing_battery`, `warmup`, and `measure_ttft`

`routing_battery` receives the complete ordered case ID list and returns it in
`executed_case_ids` after executing the production router seams. `warmup` is
called three times per engine before a TTFT run. `measure_ttft` receives
`request_cold: true`, `artifact_warm: true`, and a sample index; return positive
raw `worker_ttft_ms` and `wire_ttft_ms`. The harness collects 20 alternating
samples per engine, requires every green arm to beat GPU p50, and requires both
W128 arms to satisfy `gpu worker p50 / split worker p50 >= 5.0`.

### `exercise`

The adapter must invoke the production router/worker fault hooks and return
raw observed accounting/provenance. The harness calls:

- `kind: "bypass"` once for every authoritative bypass token;
- `kind: "fallback"` once for every deterministic fallback fault key;
- `kind: "state"` for artifact lifecycle, quarantine/probation, and processing
  fingerprint pin cases; and
- `kind: "protocol"` for a CONNECT protocol/engine-identity mismatch.

For bypasses and fallbacks, observations must contain all fields asserted by
`Certifier.run_semantic_exercises`. Artifact lifecycle cases must identify
selection versus bypass/fallback, exact-arm health, and certification-row
preservation. Quarantine cases must identify the exact arm, strike reset, and
probation transition. Pin cases must identify identity-preserving failures and
forbid GPU provenance. The harness rejects a generic pass flag because it cannot
prove those states. The protocol case must show the original exact-arm debit and
a later `quarantined` bypass without a second request debit.

### `worst_case_fallback`

The harness requests `artifact_warm`, `cold_ready_compile_failure`, and
`cold_ready_load_failure`. Return every consumed timing component and the total
TTFT. The warm row must consume guard, prediction, handoff, and GPU prefill; the
cold rows must consume guard, readiness, and GPU prefill. The harness reports
the maximum of the three totals into the evidence record.

## Fixture pinning

Width-exact and variable-length prompts are generated from stable case IDs. Their
canonical JSON SHA-256 is pinned in `certify.py` as `FIXTURE_SHA256`. Updating
that material must change the pin and re-run all affected certification arms.
