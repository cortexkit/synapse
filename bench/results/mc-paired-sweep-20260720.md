# MC paired sweep — 2026-07-20

## Status

**Timed sweep not run.** The live daemon passed the lane/certification gates, but the required quiet-box gate never became true during the observation window. The 1-minute load average remained above the required `< 4` threshold (4.92 at 09:26 and 6.15 at 09:30 local; 5-minute averages 6.56 and 6.31). No throughput, latency, queue-during-cell, or sweep power values are reported below. This avoids contaminating MC's comparison with foreign host work.

No daemon was redeployed, restarted, loaded, or probed by this task.

## Daemon fingerprint envelope

| Field | Value |
| --- | --- |
| module_id | `synapse` |
| module_generation | `9` |
| machine_profile_hash | `42a76cdd8dc2e5798629522c63dcfff1e5833ee1bf3c1f8bdb66dc2bbc04500d` |
| chip / architecture | Apple M5 Max / aarch64 |
| os_build | `25F84` |
| current knob | `balanced` |
| certification_stale | `false` |
| performance_stale | `false` |

The preflight `models.list` response showed both requested models `ready`. `probe.report` showed both lanes `certified`, with `certification_stale=false` and `performance_stale=false`:

| Lane | Engine | Fingerprint | Certified | Certified-at (ms) |
| --- | --- | --- | --- | ---: |
| `gte-modernbert-base-f16` | `owned-metal` | `54a62ef80c4f28f6ba765854d81b9ab5e52d4864142cdd81662812465d3003b5` | yes | 1784396455092 |
| `gte-modernbert-base-ane-fp16` | `ane-coreml-worker` | `5a2374bcb587ae22cd7ca93404ee7e89e9889527d15f8671feb0a226625278d8` | yes | 1784396373652 |

`models.list` on this daemon does not serialize the documented `recommended_batch` field (`ModelCatalogEntry` currently contains only `model_id`, `state`, and `fingerprints`), so no recommended row can honestly be marked. The intended client sends `input_type: document`.

## Full sweep matrix

Three requests per cell were planned. An em dash means the cell was not timed.

| Lane | Class | Batch | Recommended | Effective tok/item | Median request ms | Median items/s | Median tok/s | Single p50/p95 ms | Queue p50/p95 + max waiters | GPU W | ANE W | Engine J/item | Status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| Metal f16 | MEMORY | 1 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 2 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 4 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 8 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 16 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 32 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 64 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 128 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | MEMORY | 256 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 1 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 2 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 4 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 8 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 16 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 32 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 64 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 128 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| Metal f16 | CHUNK | 256 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 1 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 2 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 4 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 8 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 16 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 32 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 64 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 128 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | MEMORY | 256 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 1 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 2 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 4 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 8 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 16 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 32 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 64 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 128 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |
| ANE fp16 | CHUNK | 256 | — | — | — | — | — | — | — | — | — | — | not run: loadavg |

## Queue-cleanliness evidence

The last idle preflight `admission.status` (08:57 local) reported:

- `execution_waiters=0`
- `inline_in_flight_executions=0`
- `inline_in_flight_bytes=0`
- rolling acquire wait p50 `0.000166 ms`, p95 `0.000458 ms`
- both lanes `certified=true`, `certification_stale=false`, and `performance_stale=false`

No timed cell was started, so there is no during-cell queue sample and no basis to claim the sweep was queue-clean. Foreign-consumer traffic was therefore neither averaged through nor re-run.

## Power evidence

No macmon samples were taken during a timed batch=64 CHUNK cell. A one-sample host preflight at 09:13 local reported CPU `27.49 W`, GPU `1.13 W`, and ANE `0 W`; these values are explicitly **not** sweep power and are excluded from J/item calculations.

## Honest notes

- The live lane and certification gates passed; the measurement gate did not.
- The host remained materially busy despite waiting: active `opencode`, `rust-analyzer`, WindowServer, storage-management, and other processes were visible during the preflight checks. Swap usage was also non-zero.
- The extended `inline_embed_throughput` client was built but not run against the daemon. It sends `embed.batch` over the `subc` management surface with `input_type: document`, supports the required batch ladder and MEMORY/CHUNK fixtures, performs three repetitions, follows job pages, and samples `admission.status` during requests. It is ready for a quiet-window rerun.
- No throttling, retries, or queue blips can be assessed because no timed request was issued.
