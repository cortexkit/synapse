# Worker progress liveness hardware evidence

Measured 2026-10-13. The implementation commit is
`97095352fbdb8a7cdd6b41a1b75e9093422de440`. Earlier commit
`4bb88b7723fe5d51024d2e179d7d7117ec54516c` changed a repeatedly faulting catalog
worker's quarantine (temporary refusal of serving) from permanent
`probe_required` to transient `engine_crashed` with a timed cool-down;
see [its separate contract evidence](quarantine-contract.md).
All measurements used new private checkouts/daemons and checksum-pinned assets,
not production discovery. No unrelated process was stopped and no Windows
security policy was changed.

## ROG Ally X: four release Vulkan cells passed

Windows 11 Home 10.0.26200, ROG Ally X RC72LA, AMD Radeon Graphics (vendor
`0x1002`, device `0x15bf`), AMD driver 25.30.27.03 (LLPC). The release module,
Vulkan worker and certification runner were built on the Ally from clean commit
`fcd57b07c50dea752a69e01fece44eee828c9e50` with rustc 1.96.1 and cargo 1.96.1.
That commit adds the private measurement tool/control migration to the same
liveness implementation. Its source tree was checked clean before and after
building; the runner subsequently generated only untracked certification records.
Each cell had a fresh `CORTEXKIT_MODEL_CACHE`.

| Model / operation | Certification | 8,192-token worker exchange | Maximum inference progress gap | LOAD gap |
| --- | --- | ---: | ---: | ---: |
| gte-modernbert-base / embed | passed | 15.2951895 s | 1.1494846 s | 1.0865045 s |
| gte-reranker-modernbert-base / rerank | passed | 15.3906211 s | 1.1627643 s | 0.7981102 s |
| Qwen3-Embedding-0.6B / embed | **passed** | **79.6553142 s** | 4.8727721 s | 1.8118944 s |
| Qwen3-Reranker-0.6B / rerank | **passed** | **94.0049434 s** | 5.7784115 s | 2.1943831 s |

The rerank pool of 100 candidates is split into 64- and 36-candidate RPCs.
The 64-candidate Qwen3 RPC previously exceeded the old 30-second worker cap and
was killed (see the Windows baseline evidence); now it completed in 45.7862241 s,
1,920 completed units. The 36-candidate RPC took 25.7440936 s.
The 8,192-token rows are single-worker exchanges, so these times are unambiguous
worker-host write/progress/full-reply timings, **not** whole-cell/download time.
No fixed 30-second request cap killed them and no timeout quarantine occurred.

Every cell processed 8,192 without truncation/diversion, and refused 8,193 with
`sequence_too_long`, zero worker requests and no truncation/diversion. All
numerical/coverage gates in the unchanged release runner passed. Raw runner
results, executed artifact hashes and event-derived summaries are in
[`ally/summary.json`](ally/summary.json); each cell's stdout/stderr remain in
Git. [`ally/per-request.csv`](ally/per-request.csv) preserves request IDs, final
model-input token totals, total time, largest gap and event counts. The large
raw event streams are archived outside Git as described under Raw evidence.
Largest inference gap
5.778 s is below the 30-second calibration stop threshold (one quarter of the
120-second inference silence window); largest LOAD gap 2.194 s is below 150 s
(one quarter of the 600-second setup window). The actual hang thresholds remain
120/600 seconds, defined beside their reasoning in `worker_host/liveness.rs`.

Release binary hashes:

- module: `c13fb65c6fb733a72e72ae40a5769021c2e0e0a87d8af52dff6b31f607e1305f`
- Vulkan worker: `37840dc1fff0258931762f609d0c2645a302bd044ad046eb6bd8cae7783d3a60`
- runner: `693a90397c66b9c4af6fe6ea164b7f051394be49c448ee626e2710b62e9a888c`

Vulkan manifest digest remains
`6d295cb0a0edf5119fca378ac71aaf48855699e99038594e875944402f80d83c`,
and kernel revision remains
`7a5621d965123c2661ad6a795774d07b81f44a86db4d68167893ad4023e87730`.
The manifest digest identifies the pinned model/tensor definitions; kernel
revision identifies the arithmetic implementation and lane fingerprints identify
its selected numerical options. They match the baseline Windows evidence at
`6fbf7209`; existing lane identifiers were not changed. Passing pinned reference
comparisons is the numerical proof, not equality of identifiers alone.

### Ally reproduction and build-policy notes

The private root is `C:\Users\ufuka\synapse-progress-fcd57b07`; it was cloned
from a Git bundle and checked out detached at the clean commit. VulkanSDK
1.4.357.0 and the mirror of checksum-pinned model/tokenizer/config files under
`synapse-cert` were reused read-only, avoiding a new asset download or edit. Builds used the repository's `build-retry.sh`, which relinks
only Application Control refusals in the owned target directory. Smart App
Control also refused the owned `monostate_impl`/`paste` DLLs and final module;
Code Integrity events 3077/3033 confirmed those paths. Only those owned package
artifacts were cleaned/relinked. No exclusions, signing-policy changes or
foreign process stops were used.

Each model used the release runner command:

```text
CK_LOG=info CORTEXKIT_MODEL_CACHE=<fresh private per-cell path>
target/release/ckdev-synapse-certify.exe run --row vulkan-windows-amd
  --model <bare model id> --assets <private root>\assets
  --checkout <private root>\src --mirror C:\Users\ufuka\synapse-cert\mirror
```

`python scripts/check-release-candidate.py spirv --worker <private worker>`
passed. `ck-synapse.exe certify source` reported the clean source commit.
The runner retains each private session under `crates/synapse-certify-runner/
.live/run-*/data/cortexkit/synapse/logs`; this is where `elapsed_ms` and
`max_gap_ms` were captured. Normal runner result schema was not changed.

## M4 Mac mini: direct ANE compile/inference gaps

The M5's machine-wide ANE lock was held by production. The maintainer explicitly
substituted the slower M4 as a conservative calibration proxy. These are **M4
measurements**, not claimed M5 runs. No ANE work ran on the M5 during its
01:50Z–02:50Z reservation.

`ssh macmini`: Apple M4, 16 GiB, macOS 27.0 build 26A428. `pgrep -fl
"ane|synapse"` was checked before every run; only system services/unrelated UI
matches remained, and none was stopped. The private daemon/worker exited after
measurement. TMPDIR was unset, Xcode selected, and binaries were ad-hoc signed.
The module and direct worker were built clean from the implementation commit;
the private probe was built from fixture commit
`f09beddb1d02a4db75d7238cee43f9474ac91ecf`, which added the catalog's pinned
`config.json` to the local mirror whitelist; production liveness code is identical.

A gap is time from request dispatch or the preceding completed-work frame until
another validated progress frame or the full terminal reply is received, not
layer latency measured on a second timer thread.

| Qwen3 token length | First / repeat inference exchange time (s) | Maximum inference gap | First-time shape compile | Maximum compile gap |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.226598 / 0.022051 s (additional 128-token self-check also recorded) | 0.011769 s | 10.598279 s | 0.632065 s |
| 512 | 0.277115 / 0.104691 s | 0.017290 s | 32.438877 s | 4.748374 s |
| 8,192 | 9.664305 / 8.853995 s | **0.815670 s** | **177.522483 s** | **8.930775 s** |

The first/repeat values are full worker-host exchanges, not end-to-end calls;
compile/resolution costs are separate. The 8,192 compile exceeded the old
30-second request cap and two minutes overall without being a hang, because
all 28 layer completions kept renewing the 600-second silence watchdog.
Its inference/compile gaps are below the 30/150-second calibration headroom
criteria, each one quarter of the corresponding 120/600-second idle window. LOAD gap was 3.432056 s; eviction gap 0.027808 s; PING gap under
0.001 s. Exact token counts, one 1,024-dimensional vector per request and finite
unit norms were verified. The synthetic `hello` workload measures liveness and
shape timing; it is not a replacement for pinned numerical certification.

The retained M4 public-call, request-table and worker-row records are:

- [`m4/calls.json`](m4/calls.json): public request times, token counts and norms;
  cold times include resolution/loading/compilation, unlike inference times above.
- [`m4/summary.json`](m4/summary.json): correlated worker exchanges and gap windows.
- [`m4/per-request.csv`](m4/per-request.csv): every received request's time,
  maximum gap and event count; token/shape attribution uses the row profiles.
  The larger raw event stream is archived outside Git under Raw evidence.
- [`m4/worker-rows.jsonl`](m4/worker-rows.jsonl): synchronous layer/shape profiles
  correlating request IDs with token lengths.

SHA-256 of the signed M4 module, direct worker and probe binaries:

- module: `c8d8528bfbb4545134a54a3b9d9aff468fb70b5958a473f6d9d82b5b24e0e875`
- direct worker: `5f6f9c590edcef55e4b7d174ed82c0d3920e53857275e3650ae67dad1e9cd829`
- probe: `f12c7ee696148d83127e4126a80f0a27b37d875027e81c58609e62e237258053`

The private probe starts its own authenticated daemon and loopback checkpoint
mirror; its store/cache/connection file are under a fresh private root. The
first install attempt refused because the measurement mirror omitted the pinned
`config.json`; no inference ran. The committed measurement-tool fix added that
file to its strict basename whitelist, and the fresh rerun completed all six
requests. The initially attempted `CK_LOG` target filter used unsupported syntax;
`CK_LOG=info` captured the actual retained file events.

## Protocol and final gates

The Linux packaged v3 matrix passed llama/CUDA/Vulkan; the M4 matrix passed
llama, owned decode, Swift Core ML and direct ANE. The M4 owned-decode portion
exposed an existing
fixture nonce bug: the test used a nonhex string, and owned decode correctly
refused it. A separate test-only commit generates a valid eight-byte hex nonce;
worker authentication was not weakened. Test binaries were signed before run.
CMake 3.31.6 was supplied from verified official archives when absent (macOS
archive SHA-256 `330b9514f5112e5ed4fb08b8b05803b776fd9b539a6ae12927d14dcc0ee2ba8d`).

Module lib/it/e2e, frozen fingerprints, strict module and portable-worker clippy,
Windows-target clippy via `scripts/lib/zig-cc-windows.sh`/`zig-ar.sh` (the
repository's train-push preflight wrappers), fmt, the private probe typecheck
and `ckdev-mutate check` passed. Red/green production controls and baseline
native-CUDA-only lint exclusions are documented in
[worker-liveness-tests.md](worker-liveness-tests.md). No new dependency/lockfile,
public error code, numerical kernel revision or in-process Metal behavior was
introduced. The catalog quarantine disposition change and its tests are linked
in [quarantine-contract.md](quarantine-contract.md).

## Raw evidence and compact tables

The five raw completed-work dumps (four Ally cells and the M4 stream, all above
50 KB) are kept at
`~/.local/share/cortexkit/synapse/evidence-raw/worker-progress-liveness/` with
`ally/` and `m4/` subpaths. Their byte-exact checksums are committed in
[`raw-files.sha256`](raw-files.sha256). Archive destination checksums were
confirmed to match the recorded originals before their removal from Git.

The compact [`ally/per-request.csv`](ally/per-request.csv) and
[`m4/per-request.csv`](m4/per-request.csv) were generated from those exact raw
bytes. `tokens` is the sum of final composed model-input tokens; `rows` and
`max_tokens` distinguish a single 8,192-token row from pooled candidates. Blank
token fields mean a control/setup request with no token input; `shape` records
ANE compile/eviction dimensions. `event_count` includes each received progress
event and one terminal drained event; `progress_events` excludes that terminal.

The Ally table maps requests to the sealed profile self-check (four embed or
seven rerank inputs) followed by the runner's ordered, bounded fixture groups.
Fixture hashes and self-check source pins are verified, and the number of
observed inference exchanges must equal that sequence. M4 tokens/shapes come
from the committed worker row profiles, not estimates from text or elapsed time.
The totals/maxima in the README can therefore be checked without loading the
raw streams. To regenerate or verify the tables with archive access:

```sh
python3 docs/evidence/worker-progress-liveness/summarize_raw.py \
  --raw-root "$HOME/.local/share/cortexkit/synapse/evidence-raw/worker-progress-liveness" --check
```

## CUDA progress cost: synchronization reasoning, not a GPU benchmark

No CUDA GPU was available for this work, so the latency/throughput cost has **not
been measured**. Native nvcc compilation and API documentation are not a
performance measurement; the Ally Vulkan and M4 ANE results do not answer it.

The code submits all layers on the context's existing single execution stream
before entering `SynapseLayerProgress::wait`. Graph mode likewise calls
`cudaGraphLaunch` before that loop. Waiting on each completion event blocks the
calling **CPU** thread; it does not insert a host wait between submitting one
layer and submitting the next. Already queued GPU kernels can continue while
the CPU waits and reports, so this is not a new submit-layer/wait/submit-next
scheme, and no cross-stream overlap is removed by those host waits themselves.
The model mutex already serialized requests on that context before progress
instrumentation.

There is nevertheless real added work: one timing-disabled event per layer per
shape plan, an event record per layer (external record nodes in captured graphs),
per-event host synchronization/wake-up and a synchronous JSON progress-frame
write/flush. The GPU/runtime cost of the extra event nodes and the CPU/socket
cost must be measured on CUDA hardware. The output D2H copy is enqueued **after**
the progress wait/report loop, then the original final stream synchronization
still runs. Slow or backpressured reporting can therefore delay D2H submission
and extend the locked operation; it is not zero-cost or guaranteed harmless to
throughput. Without a callback the wait loop returns immediately, but event
creation/records still exist, so that path also has unmeasured instrumentation
overhead. No socket callback runs inside graph capture.

The relevant paths are `src/port/cuda_progress.cuh` (`record`/`wait`), each
family's `ShapePlan::run` and `initialize_and_verify`, `src/cuda.rs`
(`with_progress`/`completed`), and the worker's progress writer. A CUDA GPU
comparison of unchanged arithmetic with reporting disabled/enabled is needed
to put a number on the cost.
