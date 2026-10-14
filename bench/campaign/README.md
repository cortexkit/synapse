# Owned-runtime Qwen3 decode campaign harness

`decode-harness.sh` is the trusted controller for the
Qwen3-0.6B f16 single-stream decode campaign. The controller installs an explicitly configured, hash-verified harness blob
before and after execution, and invokes it as:

```text
{harness} {workspace} {candidate_runner} {result}
```

The harness never executes a candidate-tree program directly. Git inspection,
`cargo build`, `cargo test`, and every decode process are argv passed to
`{candidate_runner}`. The candidate never receives `{result}`. Before the first
runner invocation, the controller creates that file with mode `0600`, keeps its
file descriptor open, and rejects replacement or permission changes. The
candidate may write only intermediate files in a temporary candidate-output
directory; the controller validates those files and is the only writer of the
verdict.

## Frozen workload

- Model: `Qwen/Qwen3-0.6B` snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`, f16 safetensors storage.
- Canonical snapshot content SHA-256:
  `0d7d1359007f579fba9f6eceef44c87b947362da893cc565d27656284e4d6f86`.
- Decode: Metal, f16, greedy raw completion, cache bucket 512, top-five tap,
  64 new tokens.

The snapshot digest is SHA-256 over each regular snapshot file in sorted
repository-relative path order. Each entry contributes
`relative_path + NUL + dereferenced_file_bytes + NUL`. This includes model,
tokenizer, configuration, license, and model-card files. The harness refuses to
run if either the registration digest or the installed snapshot digest differs
from its embedded pin. It does not download a model and forces
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for candidate processes.

The committed [`decode-fixtures/`](decode-fixtures/) directory contains the 20
prompts, 20 token-only CPU-fp32 references produced with
`transformers==4.51.0` and `torch==2.13.0` (1,280 tokens), and `SHA256SUMS`.
Because the controller installs one hash-verified harness blob rather than a support
file tree, the same fixture bytes and manifest are embedded in the controller.
At startup it extracts them into a controller-owned temporary directory,
verifies both hashes, checks row IDs and token counts, and makes the files
read-only before any candidate code runs. A candidate-tree fixture is never an
oracle input.

## Gates and measurement

The controller performs these stages in order:

1. Build `spike-unified-rt` in release mode through the runner with
   `cargo build --locked --offline`. The runner creates the temporary output,
   target, and package-cache directories as the candidate identity; the harness
   sets them to mode `0755` so Cargo can write while the controller can read the
   resulting binary. A controller-selected `CARGO_TARGET_DIR` prevents reuse of
   a candidate-supplied binary.
2. Run all 20 prompts against the embedded references. The controller compares
   every token itself; a near-tie exemption, missing row, early stop, malformed
   output, or any token mismatch is a hard failure.
3. Run and positively identify all five intervention tests: token tap,
   pause/resume with cache inspection, forced splice, addressable weight
   regions, and deterministic greedy tie-breaking. A successful zero-test Cargo
   invocation does not pass this gate.
4. Start 12 fresh decode processes. Each gets one distinct fixture selected by
   the fixed stride-seven schedule, so the measured slot cannot replay one
   prompt. Package/executable preparation occurs eagerly at process load and is
    outside `decode_wall_s`, so decode timing excludes process-load preparation.
   The controller recomputes
   each sample as `64 / decode_wall_s`, verifies the binary's reported rate,
   and reports the median of all 12 samples.

The illustrative verdict below shows the field shape, not a measured result:

```json
{
  "gate_passed": true,
  "hooks_passed": true,
  "samples": [40.0],
  "median_tok_s": 40.0,
  "baseline_note": "Configured baseline and pinned fixture protocol.",
  "workspace_commit": "0123456789abcdef0123456789abcdef01234567"
}
```

A correctness or instrumentation failure writes empty `samples` and
`median_tok_s: null`; no speed value can be banked from a failed candidate.
The exit code is the machine-read verdict — the campaign gate extracts only
the numeric `median_tok_s` and the process exit status, not the JSON booleans:
`0` = valid measurement, `3` = candidate rejected (gate or hook failure),
`1` = harness refused to run (environment/integrity problem), `2` = usage
error. The JSON booleans remain as human-readable diagnostics.
Acceptance thresholds, baseline stability, control drift tolerance and the load
admission limit are registration policy. Keep those inputs fixed when comparing
candidate measurements.

## Offline rig preparation

A candidate build has no network egress. Configure all required public
dependencies before applying candidate patches or entering the sandbox, then
prepare the controller image from that hydrated base:

```sh
cargo fetch --locked --manifest-path bench/spikes/unified-rt/Cargo.toml
```

The candidate identity must be able to read the hydrated sibling sources,
prewarmed Rust toolchain, Cargo registry index, crate source/cache, model
snapshot, and controller-created fixture directory. It needs write access only
to the temporary target, package, and intermediate-output directories created
by the harness. Do not run
`cargo fetch` from the harness: it would both touch candidate metadata outside
the runner and make a supposedly offline measurement depend on egress.

## Local checks and invocation

Parsing and integrity checks do not need model weights:

```sh
bash bench/campaign/decode-harness.sh --self-test
```

For a full measurement, provide the pinned snapshot and a configured candidate
runner on compatible hardware with exclusive measurement access. Keep the
controller's measurement locks and load/power admission checks in place; do not
interrupt another hardware user to obtain access. The result path must not exist.

```sh
SYNAPSE_CAMPAIGN_MODEL=/path/to/pinned/qwen3-snapshot \
  bash bench/campaign/decode-harness.sh \
  "$PWD" /path/to/candidate-runner /path/to/new-decode-result.json
```

## ANE direct-API embedding campaign

[`ane-direct-embed-harness.sh`](ane-direct-embed-harness.sh) builds the standalone
probe against `crates/synapse-worker-ane-direct/vendor/ane`, the binding shared
with the direct-ANE worker. The staged workspace needs no external clone or
`SYNAPSE_CAMPAIGN_ANE_BINDING` setting; the binding's `VENDORED.md` records its
license and provenance.

The objective is aggregate active-token throughput at sequence 512; sequences
1024 and 2048 are reported controls. Each row must meet the minimum cosine gate
of 0.999 against the protected in-process fp32 reference and repeat byte-identically.
Aggregate throughput weights rows by their active tokens rather than taking a
median of per-row rates. The controller reconciles reported correctness, row
identity and token accounting before accepting a speed value.

The harness pins `.cargo-checksum.json` and verifies every listed file, the
inventory and the absence of symlinks before admission and after staging,
refusing changes by file name. The binding tree, probe manifest/lockfile, rows,
comparator and gate wiring remain protected. Registration must supply matching
harness and input digests; the template records the vendor checksum as
`SYNAPSE_CAMPAIGN_ANE_VENDOR_SHA256`.

Run `bash bench/campaign/ane-direct-embed-harness.sh --self-test .` without model
weights to check the contract and load gate, named refusals for edited, missing,
added or symlinked binding files, checksum/dependency-path integrity, and
acceptance of the unmodified workspace and staged copy.
