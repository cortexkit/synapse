# Direct-ANE worker acceptance evidence

This is a **verified partial delivery**, not release certification. The parent
stopped the quiet-window queue after5.5hours; no stress request was submitted.
No development stress JSON exists, and no successful report was synthesized.

## Acceptance table

“Verified” names the actual check below; it does not imply an additional hardware
experiment. Subrows separate implemented/unit-checked behavior from unrun
cross-platform or physical certification.

| Acceptance criterion/check | Status | Evidence or missing experiment |
|---|---|---|
| Off-Darwin empty worker stub builds | Verified | Linux-target worker all-target Clippy passed; private binding is Darwin-gated. |
| Ubuntu24.04 `cargo test --workspace` | Not run | This worktree runs on macOS; no Ubuntu runtime is available here. |
| Original `bench/spikes/ane-direct-probe` remains unchanged | Verified | No spike edits in this branch's delivery. |
| Vendored private binding, committed checksums, no sibling checkout | Verified | `vendored_binding_checksums_match` passes; crate-local dependency and provenance committed. |
| HELLO/PING avoid symbol resolution; floor/LOAD report missing API; manifest identity and graph revision | Verified | Worker protocol unit tests, including injected symbol-lookup failure, and real holder handshake. |
| ANE layer placement, fp16 profiles, ModernBERT tanh GELU, fp32 rotations and closed CPU-operation inventory | Verified | Exact-cover/closed-set unit test; four real-weight short/512 forwards; exact matrix CPU projection comparison. Qwen retains its model's SiLU activation. |
| Missing/duplicate layers or unlisted CPU operation rejected | Verified | Placement unit test and prior coverage/CPU-guard mutations. |
| LOAD avoids compilation; on-demand/resident admission; eviction precedes EVICTED; nonresident inference refuses | Verified | Backend tests and source ordering; fresh-process real shape admissions. Physical deallocation timing was not separately instrumented. |
| SHA-256 identity includes model, operation, dtype/profile, rotation, shape, graph/worker revision and OS | Verified | Identity unit test separates rotation/dtype/operation/shape/OS. |
| Physically prepopulated cache differing only in rotation/dtype is not reused | Not run | Unit identity separation is not a physical ANE cache-reuse experiment. |
| Smallest ladder rung and independently committed padded ids/masks | Verified |12 literal cases cover all models and a token equal to pad id; production prologue uses the tested helper. |
| Derived-mask mutation fails | Verified | Only `direct_ane_padding_matches_independent_committed_golden` reddened. |
| Full padded numerical-golden certification | Not run | Real short/512 numerical comparisons exist, but not the complete padding/rung certification. |
| Both embedder/reranker numerical paths and fp32 CPU heads | Verified | Four-model real-weight short/512 comparisons; gte classifier and Qwen yes/no readout are CPU fp32. |
| All four model-backed paths through the live RERANK_SEQUENCES/protocol stress harness | Not run | Full stress never passed its load gate. Numerical model tests are not this wire-level experiment. |
| Unsupported RERANK then PONG | Verified | `unsupported_rerank_drains_raw_frame_then_pongs` passes. |
| Inherited lock and worker exits on supervisor EOF; killed-holder replacement remains busy | Verified | Real debug driver/debug worker: resident128,256 submitted sequences, suspended worker outlives killed holder; replacement reports busy then acquires after worker exits. |
| Transient resource pressure evicts lane-wide LRU unleased victim and retries once; typed250ms refusal | Verified | Supervisor tests and retry/lease mutations; real fourth-executable injected failure leaves shape absent, followed by successful complete admission. |
|37 simultaneous requests, every ACK sample within4/model and8/overall, no leased eviction/resource/nonresident errors | Not run | Parent stopped the eight-hour queue after5.5hours of no qualifying load1<16/load5<20 window. |
| Development stress JSON validated against committed schema | Not run | No model request ran and no JSON was produced; schema rejection unit test passes. |
| Metal reranker ordering and2048-rung reference | Not run | No matching Metal package/lane wired into this isolated harness;2048 parity reference must come from the existing parity generator, not a new worker generator. |
| `cargo fmt --all --check`; `cargo test --workspace` on this Mac | Verified | Full Xcode selected, stale owned/spike caches cleaned:1085 passed,0 failed,64 ignored across64 result summaries. |

## Latency result

Release512-token medians before→after, milliseconds:

| Model | Before | After |
|---|---:|---:|
| gte-modernbert-base |403.621|26.800|
| gte-reranker-modernbert-base |469.290|31.387|
| qwen3-embedding-0.6b |84.769|84.128|
| qwen3-reranker-0.6b |85.062|83.329|

The bottleneck was dense scalar host rotation. Batched fp32 Accelerate GEMM
uses the same committed matrices. No layer moved to CPU and no ANE graph changed.
See `latency-stages.md` for all layer timings and numerical/load qualifications.
The private evaluate API is synchronous; submit and wait cannot be observed
separately through that binding.

## Exact cold-run instructions

Working directory: the repository root containing these files.
Prerequisites: macOS ANE/private framework; full Xcode; four manifest-pinned
converted packages already in `target/ane-direct-packages/<model>.safetensors` for
`gte-modernbert-base`, `gte-reranker-modernbert-base`, `qwen3-embedding-0.6b`, and
`qwen3-reranker-0.6b`. These ignored packages are present in this worktree but not
committed. On a different cold checkout, use the existing parity converter and
its four `<model>.ane-direct-worker` profiles to prepare those exact packages.
For a cold checkout, `$HF_CACHE` must contain the manifest-pinned Hugging Face
snapshots. Prepare all four packages with the existing converter:

```sh
mkdir -p target/ane-direct-packages
for model in gte-modernbert-base gte-reranker-modernbert-base qwen3-embedding-0.6b qwen3-reranker-0.6b; do
  cargo run --release --locked --manifest-path bench/parity/Cargo.toml --bin parity-manifest -- convert \
    --profile "$model.ane-direct-worker" --hf-cache "$HF_CACHE" \
    --out "target/ane-direct-packages/$model.safetensors"
done
```

No other Core ML/ANE workload should run during the experiment.

Prebuild **before** waiting for low load:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
unset TMPDIR SUBC_LAUNCH_NONCE SUBC_LAUNCH_NONCE_FD
cargo clean -p synapse-engine-owned
cargo clean -p spike-unified-rt
cargo build --locked -p synapse-worker-ane-direct
mkdir -p target/ane-experiments
cargo test --locked -p synapse-module --lib --no-run --message-format=json > target/ane-experiments/prebuilt-driver.jsonl
DRIVER=$(python3 - <<'PY'
import json
from pathlib import Path
items = [json.loads(line) for line in Path('target/ane-experiments/prebuilt-driver.jsonl').read_text().splitlines() if line.startswith('{')]
paths = [item['executable'] for item in items if item.get('reason') == 'compiler-artifact' and item.get('target', {}).get('name') == 'synapse_module' and item.get('profile', {}).get('test') and item.get('executable')]
assert len(paths) == 1
print(paths[0])
PY
)
```

Exact stress command, from that root:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
ANE_TEST_WORKER="$PWD/target/debug/ck-synapse-worker-ane-direct" \
ANE_TEST_PACKAGES="$PWD/target/ane-direct-packages" \
python3 crates/synapse-worker-ane-direct/tests/run_stress.py \
  --driver "$DRIVER" --wait-for-load --timeout 28800 \
  --out crates/synapse-worker-ane-direct/evidence/stress-dev.json \
  > target/ane-experiments/stress-dev-eight-hour.log 2>&1
```

`--driver` prevents a Cargo rebuild after the wait. The launcher polls
`sysctl -n vm.loadavg` every60seconds, requires1-minute<16 and5-minute<20, then
runs the prebuilt test from `crates/synapse-module`. It records before/after load
and bounds the whole process group. The Rust harness independently checks the
load gate, validates the committed development schema and writes the requested
JSON only after successful assertions. The resulting file is development
evidence, not a release artifact.
