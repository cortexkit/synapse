# Certification records

This normal workspace crate implements the schema-1 record, producer decisions,
and the offline validator for matrix coverage, status, drop evidence, artifact
digests and parity identity. `ck-synapse certify validate --assets <dir>
<checkout-root>` validates the 32 canonical records under
`docs/evidence/certification/<candidate-source-commit>/`. The candidate source
commit is embedded when the candidate binary is built, so an evidence-only
commit does not change which records the candidate validates. Artifact paths
are relative to the extracted candidate asset directory, not ZIP paths; the
validator hashes the files themselves. Only passed combinations appear in the
returned `eligible` list.

`ck-synapse certify run --row <row-id> --model <slug>` currently refuses with
`certification_refused: live runner not yet integrated: needs module
sequence_too_long, preload profiles, ane-direct routing and the Metal Qwen
reranker`. It creates no record. The `Runner` trait is the seam for the future
live implementation. The CLI does not accept observations from an external
program or environment-variable-selected evidence provider.

The live-runner follow-up must:

- Spawn the extracted candidate module and sibling workers via a scratch
  in-process daemon and construct preload configs from the manifest, packages,
  and the committed row defaults.
- Obtain final composed input IDs and actual Qwen readout IDs from execution,
  not from the expected fixture IDs or manifest alone, and feed actual outputs
  into the parity evaluator.
- Observe every ADMITTED inventory and per-lane sent-request count; prove that
  8193 returns `sequence_too_long` without any worker request and 8192 is
  processed without truncation or job diversion.
- Probe each non-Metal row on its extracted worker, obtain real machine
  identifiers, hash every executed binary and Windows CUDA runtime DLL, and
  collect same-session 512-token batch-1 latency series where applicable.
- Reap the module, workers and daemon on every exit path, and cover the live
  Metal path with a macOS wire test. Observation mode must remain disabled for
  ordinary consumer traffic.

Records carry only the specified schema-1 fields. Parity is the evaluator's JSON output, with
its identity, fixture set, metrics and boolean gate results. Admission outcomes
are `tokens_8192` and `tokens_8193`, each holding `outcome`, `truncated`,
`diverted`, and the number of `worker_requests`. Artifact roles use the binary
names (`ck-synapse`, `ck-synapse-worker-cuda`, etc.). Optional `raw_series`
holds a shared `session_id` and `ane`/`metal` latency arrays, each containing
three warmups followed by twenty measured samples. A parity-only drop can lack
latency samples when execution failed before measurement; a latency drop must
have both series. The validator disregards the stated drop cause and accepts a Qwen3 ANE drop only for a failing parity gate or a
recomputed ratio strictly greater than three. Producer placement failures never
justify a drop.

Run `cargo test -p synapse-certify` for the synthetic contract fixtures. These
prove decisions and validation, not hardware parity or the unintegrated live
runner.
