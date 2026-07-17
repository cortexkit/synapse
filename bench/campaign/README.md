# Owned-runtime Qwen3 decode campaign harness

`decode-harness.sh` is the trusted controller for the Athena V3
Qwen3-0.6B f16 single-stream decode campaign. Alfonso installs the exact Git
blob named by `.cortexkit/campaign-lab.jsonc`, verifies its SHA-256 before and
after execution, and invokes it as:

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
- Baseline: **40.55 tok/s** on the locked M1 from
  [`DECODE-WAVE1.md`](../spikes/unified-rt/DECODE-WAVE1.md).
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
Because Alfonso installs one hash-verified harness blob rather than a support
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
   outside `decode_wall_s`, matching decode wave 1. The controller recomputes
   each sample as `64 / decode_wall_s`, verifies the binary's reported rate,
   and reports the median of all 12 samples.

The verdict has this shape:

```json
{
  "gate_passed": true,
  "hooks_passed": true,
  "samples": [40.0],
  "median_tok_s": 40.0,
  "baseline_note": "Frozen master baseline: 40.55 tok/s ...",
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
Acceptance thresholds remain campaign policy. The registration supplies a 5%
baseline stability width and 5% control drift tolerance, plus a one-minute load
threshold of 2.5 that makes Alfonso pause and requeue noisy work.

## Offline rig preparation

A candidate build has no network egress. Preserve Synapse's normal sibling
layout by hydrating the frozen `commons` and `subconscious` revisions beside the
candidate workspace; workspace Cargo metadata refers to `../commons` and
`../subconscious`. Before applying candidate patches or entering the sandbox,
prepare the rig's controller image from that frozen, hydrated base:

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

## Local checks and M1 dry-run

Parsing and integrity checks do not need model weights:

```sh
bench/campaign/decode-harness.sh --self-test
```

A full run is valid only on the M1 while both measurement locks are free. The
result path must not already exist. A pass-through runner can simulate the
campaign argv contract without simulating its sandbox identity:

```sh
test ! -e [bench-user-home]/bench.lock
test ! -e /tmp/aft-measure.lock
! pgrep -f '[R]unner.Worker'
mkdir [bench-user-home]/bench.lock
trap 'rmdir [bench-user-home]/bench.lock' EXIT INT TERM HUP

cat >/tmp/pass-through-runner <<'SH'
#!/bin/sh
exec "$@"
SH
chmod 755 /tmp/pass-through-runner

SYNAPSE_CAMPAIGN_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca" \
  bench/campaign/decode-harness.sh \
  /path/to/pristine/synapse /tmp/pass-through-runner /tmp/decode-result.json
```

### Reference run

On 2026-07-15 the controller ran against pristine commit
`f99215ce199e26c76e9bb5ce911571e847f612ad` on
`[bench-host]` (M1 Max). The two lock paths were absent, no
`Runner.Worker` was active, one-minute load was 1.88 at admission, the Cargo
registry had been prewarmed from that frozen base, and the pass-through runner
above was used. The complete cold-target harness took 146.662 seconds, so the
registered 300-second action timeout is approximately twice the observed pass.
The result file had mode `0600`:

```json
{
  "baseline_note": "Frozen master baseline: 40.55 tok/s on locked M1 (DECODE-WAVE1.md); N=12 fresh processes with varied prompts.",
  "gate_passed": true,
  "hooks_passed": true,
  "median_tok_s": 40.58197703061707,
  "samples": [
    40.61547678127424,
    40.54847727995989,
    40.52563221637361,
    40.798810986991796,
    40.9538110600726,
    40.90184195233265,
    40.236030439075066,
    40.36020644253163,
    40.4882108445747,
    40.36831569341907,
    40.62757006278733,
    40.6594507526227
  ],
  "workspace_commit": "f99215ce199e26c76e9bb5ce911571e847f612ad"
}
```

The median is 0.08% above the frozen 40.55 tok/s reference and therefore within
normal run noise.
