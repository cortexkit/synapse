# ANE prefill sidecar

`ane-prefill-sidecar` is the separately supervised Swift/CoreML stage for the
Qwen3 fixed-window ANE prefill path. It owns only CoreML model lifetime,
prediction, cancellation acknowledgement, and the wire payload. The module
continues to own tokenization and prompt templates: `EXECUTE` accepts already
tokenized, already right-padded `input_ids` and `attention_mask` values.

## Connection and failure boundary

The sidecar sends `HELLO` immediately after connecting to the Unix socket. The
host must reply with a strict protocol-v2 `HELLO_ACK` whose version, nonce, and
`expected_engine` exactly match this binary. Unknown handshake fields are
rejected. A mismatch exits before the CoreML stage is created or a request is
accepted; this sidecar does not attach response provenance or charge health, so
the host can classify it as a pre-attempt incompatibility.

After negotiation every JSON command rejects unknown top-level fields:

- `INSTALL` verifies the compiled artifact SHA-256 and loads the model under
  `readiness_timeout_ms` (1 ms through 1 hour). A load that completes after the
  deadline is discarded and is never installed.
- `EXECUTE` requires exactly one installed fixed-width input row and a
  worker-created, mode-`0600` memory-mapped handoff file. It validates right
  padding, runs CoreML with `CPU_AND_NE`, writes logits and K/V directly into
  that mapping, and emits an `EXECUTED` layout header with a SHA-256 digest.
- `ABORT` is handled while prediction is in flight. The sidecar owns the active
  execution ticket and never publishes logits or K/V after observing its
  cancellation. `ABORTED` reports `cancellation_owner: "sidecar"`.
- `TIMING_READBACK` returns bounded, sidecar-local timing for the most recent
  128 executions. It contains readiness, prediction, K/V-layout, logits-copy,
  integrity-digest, and total milliseconds.

CoreML does not expose a force-cancel operation for a synchronous model load or
prediction. The sidecar bounds observable readiness and discards late loads;
the supervising host owns process termination if a platform call hangs.

## Payload layout

The CoreML graph is fixed-window, but `active_tokens` controls the decode
boundary. Logits are read at `active_tokens - 1`, never from a padded slot. The
K/V region uses the Metal import order:

```text
[layer][key_or_value][head][cache_position][dimension]
```

K/V values are f16 little-endian bits. The sidecar walks `MLMultiArray.strides`
directly into the mapped cache layout rather than first materializing Swift
arrays. Active positions retain their exact CoreML bits and every remaining
cache position is zero. The worker validates the generation, layout, completed
publication state, and SHA-256 digest before exposing the mapped K/V slice to
the Metal uploader. The socket carries no logits or K/V data in protocol v2.

## Development

```sh
cd workers/ane-prefill-sidecar
swift run ane-prefill-sidecar-tests
swift build -c release
```
