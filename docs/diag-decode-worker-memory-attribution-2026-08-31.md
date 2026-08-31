# Decode-worker static-footprint attribution

## Scope and method

This investigation measured the real `qwen3-0.6b` checkpoint at
`$HOME/.local/share/cortexkit/synapse/certification/qwen3-0.6b` on macOS. The
checkpoint's `model.safetensors` is 1.40 GiB. The measurements use
`proc_pid_rusage(..., RUSAGE_INFO_V4)` for `phys_footprint`, process-tree
aggregation, and `footprint(1)` categories at named worker stages. Raw samples
are checked in beside this report:

- `results/decode-worker-memory-attribution-2026-08-31-f16-gpu.json`
- `results/decode-worker-memory-attribution-2026-08-31-q8-gpu.json`
- `results/decode-worker-memory-attribution-2026-08-31-f16-ane-split.json`
- `results/decode-worker-memory-attribution-2026-08-31-q8-ane-split.json`

The stage tool extends the existing curve harness's worker transport client so
it can sample after the handshake and before `LOAD`:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo build --release -p synapse-worker-decode
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build --package-path workers/ane-prefill-sidecar -c release
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  ./bench/spikes/ane-prefill-split/build_runner.sh

# The locally compiled W128 program was used only to make the ANE sidecar live
# for memory attribution; it is not certification or exactness evidence.
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  python3 tests/decode_worker_stage_attribution.py \
  --checkpoint "$HOME/.local/share/cortexkit/synapse/certification/qwen3-0.6b" \
  --label q8-ane-split --weight-quant q8_0 \
  --compiled bench/spikes/ane-prefill-split/artifacts/memory-attribution/qwen3-prefill-w128.mlmodelc \
  --sidecar workers/ane-prefill-sidecar/.build/release/ane-prefill-sidecar \
  --steady-requests 5 \
  --output results/decode-worker-memory-attribution-2026-08-31-q8-ane-split.json
```

Equivalent f16/q8 GPU and f16 ANE-split commands produced the other three
records. Each arm generated a stable one-token digest for all five requests.
`vmmap -summary <pid>` and `footprint <pid>` were captured at each stage by the
stage tool. The ANE client is intentionally lazy, so its process launch,
`INSTALL`, Core ML program load, and first prefill occur in one request; that
observation is named `post-first-decode/post-ANE-install` rather than claiming
a nonexistent separation.

## Stage-attribution table

All figures are process-tree `phys_footprint` GiB. `Malloc Large` and `graphics`
are `footprint(1)` category totals at steady state. Values vary with macOS
compression, but the stage deltas and category mix were stable enough to answer
attribution.

| arm | post-spawn | post-load | post-first-decode / ANE install | steady | steady Malloc Large | steady graphics | sidecar footprint at steady |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f16 GPU | 0.003 | 6.751 | 5.522 | 5.522 | 3.767 | 1.564 | 0 |
| q8 GPU | 0.003 | 11.465 | 10.497 | 10.702 | 7.771 | 2.499 | 0 |
| f16 ANE split | 0.003 | 7.248 | 6.182 | 6.182 | 4.331 | 1.564 | 0.056 |
| q8 ANE split | 0.003 | 12.064 | 12.366 | 12.366 | 9.480 | 2.443 | 1.453 |

The salient result is before ANE exists: q8 GPU is already 10.70 GiB steady
versus f16 GPU at 5.52 GiB, and q8 has already reached 12.06 GiB at `LOAD` in
the split configuration. The q8 split minus f16 split steady gap is 6.18 GiB.
Therefore the observed 13.8 GiB versus 8.1 GiB population is explained by a
static load-time representation problem, not by request accumulation.

## Where the bytes live

- **Host large allocations are dominant.** q8 GPU has 7.77 GiB in `Malloc
  Large`, 4.00 GiB more than f16 GPU. q8 split has 9.48 GiB there, versus 4.33
  GiB for f16 split. This is the main unexplained-versus-raw-weight component.
- **Metal-owned graphics memory is material but secondary.** The steady q8 GPU
  value is 2.50 GiB, about 0.94 GiB above f16 GPU's 1.56 GiB. It is static after
  the first request. The load observation also briefly shows a roughly 0.60 GiB
  shared upload source which is released after the initial decode.
- **The ANE sidecar is observable but not the initial q8 gap.** In this run the
  q8 sidecar holds 1.45 GiB after install; the f16 sidecar shows 0.056 GiB under
  a different compressed-memory state. The Core ML program is therefore a
  real, variable execution-residency contributor. It cannot explain the
  pre-sidecar q8 GPU gap, and the q8 split tree grows only 0.30 GiB from its
  post-load tree measurement to steady because host compression and graphics
  ownership shift while the sidecar appears.
- **The K/V mmap is small at W128.** The compiled conversion report records
  14,680,064 bytes of f16 K/V output. The sidecar's `IOSurface` category is 24
  MiB. Neither can account for multi-GiB static residency.

## Code attribution and suspect verdicts

| banked suspect | verdict | evidence |
| --- | --- | --- |
| q8 ingest quantizer staging retains f16 source, q8 artifact, and Metal copies | **confirmed, with a wording correction** | `Model::load_with_quant` reads the safetensors into `Tensor.data: Vec<f32>`, keeps each source tensor in every `Weight`, and `load_weight` adds `Q8_0Tensor` beside it. It also retains an f16 Metal mirror. The q8 worker then explicitly loads a second complete f16 model for `f16_prefill`; both models are `Box::leak`ed. These are process-lifetime residents, not a temporary artifact writer buffer. The 4.0+ GiB `Malloc Large` q8 excess is the direct measurement counterpart. |
| ANE sidecar, Core ML program, and K/V mmap | **split** | The sidecar can carry 1.45 GiB after install and is a capacity concern. But q8's multi-GiB excess appears at `LOAD` before a sidecar exists, and q8 GPU alone is already 10.70 GiB. The mmap is only about 14 MiB of K/V plus a 24 MiB IOSurface observation. |
| Metal driver-side wired allocations attributed to the worker | **confirmed secondary contributor** | `Owned physical footprint (unmapped) (graphics)` is 1.56 GiB f16 and 2.50 GiB q8 at steady state. It accounts for about 0.94 GiB of the q8-vs-f16 GPU gap, not the 5.18 GiB total gap. |

The narrow code fact behind the dominant result is in
`crates/synapse-worker-decode/src/runner.rs`: q8 loads a q8 model and then an
entire f16 `f16_prefill` model. In
`crates/synapse-engine-owned/owned-decode-engine/src/qwen3_decode_model.rs`,
q8 quantization preserves `Tensor.data` as f32 and adds q8 bytes rather than
replacing the source. `qwen3_decode_metal_step.m` separately copies the q8
weights into private Metal buffers.

No production memory fix is included. The large resident source and f16
fallback model are currently live inputs to existing decode and fallback paths;
releasing either without redesigning ownership would be unsafe. A safe follow-up
would first prove that the q8 decoder needs only embeddings/norms plus q8
weights after `MetalStepDecoder::new`, then explicitly drop the source and f16
mirrors for uploaded linear weights. That must run the full module battery,
worker transport checkpoint battery, fmt, clippy 1.98 `--all-targets`, and the
banlist, plus this same before/after stage measurement and every exactness gate.

## Fleet exposure

The resident fleet decode worker was inspected without signalling or restarting
it:

```sh
ps -axo pid=,ppid=,rss=,command= | /usr/bin/grep '[c]k-synapse-worker-decode'
vmmap -summary 52672
footprint 52672
```

At measurement time PID 52672 was alive under `ck-synapse` with no child
sidecar. `footprint` reported 11 GiB (`phys_footprint` 11.2 GiB in `vmmap`),
including 8729 MB `Malloc Large` and 2444 MB owned graphics. That closely
matches the measured q8 GPU signature. Thus the fleet daemon's resident decode
worker carries the same static q8 overhead today; the exposure does not require
a growing request curve or a currently live ANE child.
