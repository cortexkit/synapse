# ANE-prefill shared-memory K/V handoff

## Result

The worker/sidecar protocol now uses a worker-owned memory-mapped file instead
of socket data frames. Protocol version 2 is a clean cutover: a version-1
sidecar or worker is rejected during `HELLO` before model installation or an
execution attempt.

The sidecar writes logits and the complete padded Metal cache layout directly
from `MLMultiArray` storage. It publishes a generation-tagged control header
only after storing a SHA-256 digest. The worker verifies the control header,
response layout, and digest, then gives Metal a slice backed by the mapping; it
does not receive a binary socket frame, allocate an active-K/V `Vec`, or expand
active positions after receipt.

## Layout and failure boundary

The mapped file is created by the worker with mode `0600`. The fixed 4 KiB
control page contains the protocol, state, generation, offsets, lengths, and
payload digest. The payload has two regions:

```text
f32 logits: [vocabulary]
f16 K/V:    [layer][key_or_value][head][cache_position][dimension]
```

Before each execution the worker resets the control page to `EMPTY`. After
CoreML prediction the sidecar changes it to `WRITING`, clears the K/V region,
walks the real CoreML strides into active cache positions, writes logits, hashes
both payload regions, and changes the state to `READY`. Timing readback reports
the digest pass separately as `integrity_ms`. The socket then carries
only `EXECUTED`, the exact layout, generation, mapped-file descriptor, and
digest.

Failures stay in the contract's existing vocabulary:

| Failure | Worker classification | Runtime fallback reason |
|---|---|---|
| K/V shape or stride conversion | `kv_conversion` | `kv_conversion_failure` |
| Torn or incomplete publication | `ipc` | `ipc_handoff_failure` |
| Sidecar exit while state is `WRITING` | `ipc` | `ipc_handoff_failure` |
| Header/response/payload digest mismatch | `ipc` | `ipc_handoff_failure` |

A failed request discards the sidecar client and mapping generation, preserving
the existing rule that no late `EXECUTED` response can be consumed by another
request.

## Bit-faithfulness proof

The focused Rust battery constructs the previous active-only socket fixture,
expands it with the retired admission algorithm, writes the resulting full
cache to the v2 mapping, validates the digest, and compares every imported
`u16`. The Swift battery independently writes a non-contiguous strided fixture
through both the active-only reference layout and the direct padded layout.

| Check | Result |
|---|---:|
| Legacy-expanded cache vs mapped cache bit mismatches | 0 |
| Swift active K/V bits changed | 0 |
| Non-zero mapped padding values | 0 |
| Shared publication logits/K/V byte mismatches | 0 |

Commands:

```sh
cargo test -p synapse-worker-decode
cd workers/ane-prefill-sidecar
swift run ane-prefill-sidecar-tests
```

The deterministic six-arm certification harness and contract validator remain
green:

```sh
python3 tests/ane-prefill-certification/test_certify.py
python3 contracts/ane-prefill-split/validate_ane_prefill_split_contract.py --self-test
```

## Stage measurements

The old row is the accepted 20-sample W128 production attribution in
`docs/diag-ane-prefill-ttft.md`. The v2 read-side row uses the checked-in ignored
release probe with the production W128 dimensions (28 layers, 8 K/V heads,
128 dimensions, 512-position decode cache, and 151,936 logits). It isolates the
work that replaced the 123.081 ms socket receive: validate the generation and
layout, hash the mapped regions, and expose the K/V slice. It does not substitute
for the required CoreML/Metal TTFT battery.

| Stage | v1 socket p50 (ms) | v2 mapped p50 (ms) | Samples | Scope |
|---|---:|---:|---:|---|
| Payload handoff / read-side integrity | 123.081 | **18.595** | 20 | W128 production dimensions |
| Active-K/V expansion | 0.467 | 0 (removed) | 20 | Worker admission |

The probe began at load averages `{ 5.74 7.70 9.08 }` and ended at
`{ 5.52 7.62 9.04 }`; the gated one-minute value was below 6 at both boundaries.
Raw v2 samples (ms):

```text
18.307, 18.034, 18.290, 18.033, 17.976,
18.036, 18.317, 18.052, 18.889, 19.391,
19.488, 19.145, 19.103, 18.806, 18.384,
18.290, 19.227, 18.988, 19.086, 19.241
```

Probe command:

```sh
cargo test -p synapse-worker-decode --release \
  shared_memory_handoff_20_sample_w128_timing -- --ignored --nocapture
```

### Hardware battery disposition

The worktree did not contain the Qwen3 checkpoint or compiled CoreML packages,
and `SYNAPSE_OWNED_DECODE_QWEN3_0_6B` was unset. Therefore no new f16-step or
q8-step end-to-end TTFT values, and no real-package six-arm token battery, are
claimed here. Those checks require the operator-owned model assets. The
following command must be run with the worktree-local copies of those assets
before production certification; its output must remain outside the frozen
evidence record unless the evidence workflow separately accepts it:

```sh
python3 tests/ane-prefill-certification/certify.py \
  --output <scratch-output.json> \
  --driver python3 tests/ane-prefill-certification/machine_driver.py \
    --checkpoint <worktree-local-qwen3-checkpoint>
```

No gate constant, evidence record, runtime configuration, manifest, or routing
policy changed.
