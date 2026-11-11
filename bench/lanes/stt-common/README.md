# Comparable local STT lanes

`stt-bench` verifies all **1,200** prepared mono 16 kHz float32 WAVs against
`WAV_DIRECTORY/inputs.json` (`{id: {path, sha256}}`), checks sample counts and
binds corpus IDs/ground truth to the sealed parity fixtures. Do not edit the WAV
folder during a run. It calls **`synapse_parity::speech::grade_wer`**, not a second
normalizer. Per-split word error rate (WER) is total word edits / total reference words, never a
mean of utterance WERs. Both columns use the evaluator's Unicode compatibility
normalization (NFKC), full case folding,
punctuation deletion and whitespace splitting:

- `ground_truth_wer`: candidate versus original dataset ground truth.
- `nvidia_fp32_wer`: candidate versus the sealed NVIDIA model's `text`.

The parity certification accuracy budget is intentionally not applied to
competitors: the candidate occupies both hypothesis slots in each scoring call.
The normalizer and aggregation are unchanged. WERs are fractions, not percentages.

## Pins and measurements

| Lane | Engine pin | Model pin |
| --- | --- | --- |
| transcribe.cpp | Hex's anomalyco fork **0.4.1**, `bcb639faa4c74a2300ba19e30360cb2355b1663a` | Ultra Q8_0: `handy-computer/parakeet-ultra-gguf@a1539be1b849ea7463b88343e5277b373d0c4ea9` |
| transcribe.cpp | same | Plain v3 Q8_0 and F16: `handy-computer/parakeet-tdt-0.6b-v3-gguf@85ac09ea12fc4b1112fa76810059364bc6adc9de` |
| FluidAudio (Apache-2.0) | `66a239a32609d3f90b50a9634e4b2820ab977011` (source commit, not a release-tag claim) | `FluidInference/parakeet-tdt-0.6b-v3-coreml@7dd20fe6b1797d35f5e3307e8b1732d9a178edfe` |
| Apple SpeechTranscriber | installed macOS 26+ Speech framework; OS version/build recorded, source commit unavailable | Apple-managed installed language assets; weight files/revisions unavailable |
| Synapse Metal F32 | current build commit (`synapse-engine-owned`) | NVIDIA v3 HF snapshot `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` |
| Synapse direct ANE mixed F16/F32 | same, unchanged direct-worker engine modules and vendored `ane` | sealed `parakeet-tdt-0.6b-v3.ane-direct-worker` package |
| Synapse CUDA F16 / F32 | same, `synapse-engine-cuda`; F32 uses its measurement-only loader | sealed owned-CUDA F16 package / unpinned all-F32 conversion of the same checkpoint |
| Synapse Vulkan F16 / F32 | same, `synapse-worker-vulkan`; F32 enables `transcribe-f32-measurement` | sealed owned-Vulkan F16 package / unpinned measurement-F32 conversion of the same checkpoint |

GGUF names and SHA256 pins are in [`models.json`](models.json); mismatches are
refused. CoreML loads **only the supplied local snapshot** with `loadLocal`,
never the library's floating Hub download. Every model file is SHA256-recorded;
Apple's opaque assets are explicitly the exception. `ane-pal6` selects
`Encoder.mlmodelc` (FluidAudio calls this `.int8`, but it is 6-bit palettized).
`ane-int8` selects `Encoder_v2.mlmodelc` (`.int8V2`, per-channel int8).
Both configure `.cpuAndNeuralEngine`, disallowing GPU but allowing CoreML CPU
fallback; these results do not certify exclusive ANE execution.

The transcribe dependency includes Hex's `0004-vulkan-shader-subset.patch` in its
vendored ggml. `build.sh` enables Hex's `f32,f16,q8_0,q5_0` shader subset and
turns off `MUL_MAT_ID`. It retains Rust's C runtime selection rather than Hex's
application-specific static-MSVC override. Each binary requests an explicit
backend and refuses CPU fallback. No sources or models are copied into Git.

Each model/backend gets a **fresh process**, one discarded first-utterance warmup
(per locale for Apple), and one sequential measured corpus pass. Cold time is
model/asset preparation plus warmup, **not** an OS-cache-flushed measurement.
Latency includes WAV reading and frontend/decoding; p50/p95 use nearest rank.
RTF is sum of measured utterance seconds / completed audio duration. Warm wall
and parent-observed process wall are also recorded separately. Synapse uses the
same first-WAV warmup and includes WAV I/O, `synapse-speech` frontend, engine and
detokenization in each latency. Intermediate tensor/decoder-trace capture is
always disabled. No precomputed features from certification fixtures are
substituted for WAVs; the 1,247-file prepared folder is accepted,
but only the same 1,200 corpus IDs are requested and graded. Shape-specific
compilation on a later utterance remains in warm latency, just as any lazy work
in a competitor does; warm means post-first-utterance, not every-shape-precompiled.

Memory covers loading and warmup as well as inference:

- `peak_rss_bytes` is retained unchanged: `getrusage` high-water RSS (bytes on
  macOS, KiB converted to bytes on Linux), or Windows `PeakWorkingSetSize` bytes.
- **`process_peak_bytes`** is the comparison column: on macOS the parent polls
  `proc_pid_rusage(RUSAGE_INFO_V4).ri_lifetime_max_phys_footprint` every 20 ms.
  This lifetime high-water footprint includes charged Metal allocations, unlike
  RSS. It is still worker-only, excluding CoreML, Speech and `aned` service
  processes. On Linux/Windows it uses the worker's high-water RSS/working set.
  The final instant before worker exit can fall between footprint polls.
- **`gpu_peak_bytes`** is sampled NVIDIA memory for the worker PID, summed over
  devices, above its pre-spawn idle baseline of **zero** (not whole-device memory
  minus other users). NVML v2 compute **and graphics** process lists are preferred,
  deduplicated per device, so Vulkan is covered when the driver exposes it. The
  fallback is `nvidia-smi --query-compute-apps=pid,used_gpu_memory`; MiB are
  converted to bytes. The target interval is 100 ms (faster than 200 ms);
  `gpu_peak_method`, `gpu_sample_interval_ms`, `gpu_max_sample_gap_ms` and
  `gpu_samples` record method/cadence/valid samples. Inspect actual gaps: a slow
  command/driver can miss the requested cadence. Peaks shorter than an interval
  can be missed. Missing PID rows, absent drivers and WDDM `N/A` are **null**, not
  zero. The fallback may not expose Vulkan graphics memory on some drivers.
- **`gpu_after_load_idle_bytes`** is a separate resident-memory snapshot after
  model loading, before the first warmup utterance. Synapse and transcribe.cpp
  pause at an optional parent probe barrier; the parent samples the same PID
  counter, then resumes them. The barrier wait is subtracted from worker cold
  time, but remains in parent-observed total wall time. We do **not** subtract
  resident weights from `gpu_peak_bytes`. Mac-only Swift lanes have no NVIDIA
  counter, so their GPU resident-idle value is null, as are old workers/replays
  that do not provide the load boundary.
- **`ane_peak_bytes` is null**: no public per-process Neural Engine memory
  counter exists. Host footprint is not a substitute for ANE resident memory.
- **`model_size_bytes`** sums the files actually opened by a lane, not an entire
  download cache: the GGUF, selected CoreML components plus vocabulary (excluding
  the alternate encoder), Metal config/weights, or the converted Synapse package.
  Apple-managed opaque assets are null. Compiler caches are not model weights.

New counters are null for transcript-only replays when no live parent measurement
exists. Old result JSON still renders in `stt-table` with `NA` for added columns.
`energy_joules` remains null; power needs an external owner-run measurement with
its scope and interval recorded. These are benchmark measurements, not hardware
certification; the F32 CUDA/Vulkan variants are explicitly unpinned measurements.

Each result embeds the existing `LaneResult` and adds STT measurements/provenance,
per-utterance text/latency, hashes and per-split scores. Audio has no input-token
metric: the embedded schema's required token fields are zero and labelled
inapplicable. Apple's unsupported/missing-asset locales carry explicit errors and
completed/requested counts. Partial coverage is **not** full-corpus certification;
WER denominators cover completed utterances only. Fatal errors stop the matrix.

## Fetch models (owner, once; `hf` CLI required)

```sh
export STT_GGUF_DIR="$HOME/stt-models/gguf"
hf download handy-computer/parakeet-ultra-gguf parakeet-ultra-Q8_0.gguf --revision a1539be1b849ea7463b88343e5277b373d0c4ea9 --local-dir "$STT_GGUF_DIR"
hf download handy-computer/parakeet-tdt-0.6b-v3-gguf parakeet-tdt-0.6b-v3-Q8_0.gguf parakeet-tdt-0.6b-v3-F16.gguf --revision 85ac09ea12fc4b1112fa76810059364bc6adc9de --local-dir "$STT_GGUF_DIR"
export STT_COREML_DIR="$(hf download FluidInference/parakeet-tdt-0.6b-v3-coreml --revision 7dd20fe6b1797d35f5e3307e8b1732d9a178edfe --include 'Preprocessor.mlmodelc/*' 'Encoder.mlmodelc/*' 'Encoder_v2.mlmodelc/*' 'Decoder.mlmodelc/*' 'JointDecisionv3.mlmodelc/*' parakeet_vocab.json)"
```

The last command is needed only on the Mac. Preserve that snapshot, not a `main`
checkout. `models.json` lists the complete required component names. Budget about
2.75 GB for the three GGUFs, plus the CoreML bundles and compiler dependencies.

## Build and run: M5 Max

From the repository root, with Rust >=1.89, full Xcode and CMake installed:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_NET_GIT_FETCH_WITH_CLI=true
xcrun --find metal
cargo build --locked --release -p stt-common --bins
bash bench/lanes/stt-transcribe-cpp/build.sh metal
swift build --disable-sandbox -c release --package-path bench/lanes/stt-fluidaudio --scratch-path target/stt-fluidaudio
swift build --disable-sandbox -c release --package-path bench/lanes/stt-apple-speech --scratch-path target/stt-apple-speech
export WAVS="$HOME/Work/Projects/CortexKit-scratch/parity-cache/parakeet-wavs-all"
# Explicit asset installation, separate from benchmark timing; no inference here.
target/release/stt-bench --wavs "$WAVS" --prepare-only --model system --model-revision OS-managed --backend apple-speech --output target/stt-apple-request.json
target/stt-apple-speech/release/stt-apple-speech --request target/stt-apple-request.json --backend apple-speech --install-assets-only
# Complete the Synapse build/package steps below before the default full matrix.
bash bench/stt-matrix.sh "$WAVS" .cortexkit/internal-docs/investigations/stt/m5-run1
```

Keep `STT_GGUF_DIR` and `STT_COREML_DIR` exported from the fetch step. Asset
installation needs network and may be subject to Apple's locale storage limits;
inspect the printed unavailable locales. Model installation is an explicit owner
action, never an implicit benchmark download. Use a new output path for every
run; existing results/requests are never overwritten.

## Build and run: Windows RTX 3080 / Linux NVIDIA rental

Use **Git Bash in an x64 MSVC developer environment** on Windows (native Rust
MSVC, CMake, CUDA toolkit/nvcc and Vulkan SDK/glslc on PATH); on Linux install
C/C++ tools, CMake, CUDA toolkit and Vulkan development headers/glslc plus the
NVIDIA driver. Build only one native backend at a time:

```sh
export CARGO_NET_GIT_FETCH_WITH_CLI=true
cargo build --locked --release -p stt-common --bins
bash bench/lanes/stt-transcribe-cpp/build.sh cuda
bash bench/lanes/stt-transcribe-cpp/build.sh vulkan
# Windows Git Bash example (copy the same prepared corpus including inputs.json):
export STT_GGUF_DIR="$HOME/stt-models/gguf"
bash bench/stt-matrix.sh /c/stt/parakeet-wavs .cortexkit/internal-docs/investigations/stt/rtx3080-run1
# Linux rental, after fetching the three pinned GGUFs there:
bash bench/stt-matrix.sh /data/parakeet-wavs .cortexkit/internal-docs/investigations/stt/linux-run1
```

Run only the command for the actual OS. Windows and Linux both run three models
on CUDA, then three on Vulkan. Index paths must be relative to the WAV folder or
absolute inside it; rebase absolute paths when copying the corpus to another OS,
without changing the WAV bytes or their SHA256 values.
`STT_BACKENDS=cuda` selects just CUDA;
`STT_LANES='fluid apple'` selects only those Mac lanes. Defaults include Synapse
alongside all three competitor Mac lanes on Darwin, and transcribe.cpp plus
Synapse on Windows/Linux. Binaries live in separate
`target/stt-BACKEND/release/` folders to avoid feature/backend confusion.
`STT_BENCH`/`STT_TABLE` can point at separately built portable tools.

`OUTPUT_DIRECTORY/matrix.tsv` retains the per-engine/model/backend/split layout
and existing columns, adding size, precision, hardware identity and memory.
**`comparison.tsv`** has one row per engine/model/precision/machine result with
ground-truth and sealed NVIDIA-reference WER columns per split, plus corpus-wide
p50/p95 and times.
JSON siblings preserve raw transcripts, input requests and provenance.
To combine machines without mixing request/worker files:

```sh
target/release/stt-table /results/m5/transcribe-metal-ultra-q8.json /results/rtx3080/transcribe-cuda-ultra-q8.json > combined.tsv
# One row per engine/precision/hardware, with all split WERs as columns:
target/release/stt-table --wide /results/m5/synapse-metal.json /results/m5/synapse-ane.json /results/rtx3080/synapse-cuda-f16.json /results/rtx3080/synapse-cuda-f32.json /results/rtx3080/synapse-vulkan-f16.json /results/rtx3080/synapse-vulkan-f32.json > comparison.tsv
```

For re-scoring an existing worker JSON, replace `--worker EXE` with
`--transcripts FILE` in a `stt-bench` command and supply the same model/revision/
backend. Replays have null parent-observed wall time, not a fabricated timing.
See [`protocol.md`](protocol.md) for the small worker JSON contract.

The table accepts **result JSON only**, not `.worker.json` or `.request.json`
siblings; enumerate result names instead of globbing all JSON in a run directory.

## Synapse build and run (no certification capture)

All package conversion is an explicit, offline preparation step outside timing.
Fetch the pinned source once (owner action, `hf` CLI), then keep it unchanged:

```sh
export STT_SYNAPSE_METAL_MODEL="$(hf download nvidia/parakeet-tdt-0.6b-v3 --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b --include '*.json' model.safetensors)"
export CHECKPOINT="$STT_SYNAPSE_METAL_MODEL/model.safetensors"
mkdir -p target/stt-models
cargo build --locked --release -p stt-common --bins
cargo build --locked --release -p stt-synapse --bin stt-synapse-prepare
```

### M5: Metal F32 and direct ANE

Full Xcode, from the repository root; compile only, **no inference** in build:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun --find metal
cargo build --locked --release -p stt-synapse --features metal --bin stt-synapse --target-dir target/stt-synapse-metal
cargo build --locked --release -p stt-synapse --features ane --bin stt-synapse --target-dir target/stt-synapse-ane
codesign --force --sign - --identifier ckdev-synapse-stt-ane target/stt-synapse-ane/release/stt-synapse
target/release/stt-synapse-prepare --checkpoint "$CHECKPOINT" --backend ane --output target/stt-models/parakeet-ane.safetensors
export STT_SYNAPSE_ANE_MODEL="$PWD/target/stt-models/parakeet-ane.safetensors"
export WAVS="$HOME/Work/Projects/CortexKit-scratch/parity-cache/parakeet-wavs-all"
unset TMPDIR SYNAPSE_ANE_PROFILE_DIR
# Owner only: arrange exclusive ANE access. The lane refuses a busy shared lock.
STT_LANES=synapse STT_SYNAPSE_BACKENDS='metal ane' bash bench/stt-matrix.sh "$WAVS" .cortexkit/internal-docs/investigations/stt/m5-synapse-run1
# Or use all built competitor + Synapse lanes in a fresh output directory:
bash bench/stt-matrix.sh "$WAVS" .cortexkit/internal-docs/investigations/stt/m5-all-run1
```

The ANE lane holds `~/Library/Caches/ck-synapse/ane-direct.lock` nonblocking for
its full model lifetime, matching the worker/certification process. It requires
`TMPDIR` unset (the default per-user temp directory); never redirect ANE compiler
temporaries. It compiles unchanged direct-worker source modules and the exact
vendored `ane` dependency/default features. Graph constants retain production
identifier order, so the same weights and shapes retain production ANE program
cache identifiers. Cache hits require the compiled programs to remain resident
in Apple's `aned` service; reclaimed programs must compile again. Do not stop
another client to obtain the lock.

### Windows RTX 3080: CUDA and Vulkan F16/F32

In Git Bash inside an x64 MSVC developer environment with CUDA toolkit/nvcc,
CMake and Vulkan SDK/glslc on PATH. Set `CUDA_PATH` and `VULKAN_SDK` to those SDKs.
Use the fetch/portable-build steps above on Windows, then:

```sh
cargo build --locked --release -p stt-synapse --features cuda --bin stt-synapse --target-dir target/stt-synapse-cuda
cargo build --locked --release -p stt-synapse --features vulkan --bin stt-synapse --target-dir target/stt-synapse-vulkan
for backend in cuda-f16 cuda-f32 vulkan-f16 vulkan-f32; do
  target/release/stt-synapse-prepare.exe --checkpoint "$CHECKPOINT" --backend "$backend" --output "target/stt-models/parakeet-$backend.safetensors"
done
export STT_SYNAPSE_CUDA_F16_MODEL="$PWD/target/stt-models/parakeet-cuda-f16.safetensors"
export STT_SYNAPSE_CUDA_F32_MODEL="$PWD/target/stt-models/parakeet-cuda-f32.safetensors"
export STT_SYNAPSE_VULKAN_F16_MODEL="$PWD/target/stt-models/parakeet-vulkan-f16.safetensors"
export STT_SYNAPSE_VULKAN_F32_MODEL="$PWD/target/stt-models/parakeet-vulkan-f32.safetensors"
# Owner only; copy inputs.json and exactly the same WAV bytes to this path.
STT_LANES=synapse bash bench/stt-matrix.sh /c/stt/parakeet-wavs-all .cortexkit/internal-docs/investigations/stt/rtx3080-synapse-run1
# After also building transcribe.cpp competitors above, run both sets:
bash bench/stt-matrix.sh /c/stt/parakeet-wavs-all .cortexkit/internal-docs/investigations/stt/rtx3080-all-run1
```

`STT_SYNAPSE_BACKENDS='cuda-f16 cuda-f32'` selects CUDA only (independent of
competitors' `STT_BACKENDS`). Both precisions use the same per-backend executable.
CUDA F32 uses `load_f32_measurement`, not a production config override; Vulkan
F32 uses its separate measurement shader pipeline and an isolated profile clone.
The preparer checks the source checkpoint hash; F16/ANE outputs must match sealed
package pins. F32 output hashes are recorded but **not production-approved**.
Existing numerical failures in an engine are not hidden or reported as missing
or unsupported utterances: the common evaluator grades whatever transcript it
returns.

Linux NVIDIA owners can use the same commands without `.exe`, with their native
CUDA/Vulkan SDKs. Portable tests do not certify any accelerator or driver.

## Portable checks (no hardware inference)

```sh
cargo test --locked -p stt-common -p stt-transcribe-cpp --features stt-transcribe-cpp/native stt_
cargo clippy --locked -p stt-common -p stt-transcribe-cpp --features stt-transcribe-cpp/native --all-targets -- -D warnings
cargo test --locked -p stt-synapse --bins stt_
cargo clippy --locked -p stt-synapse --all-targets -- -D warnings
python3 -m unittest discover -s bench/lanes/stt-synapse -p test_matrix.py -v
cargo fmt --all --check
```

These use the CPU native build only to check/link the Rust binding; the tests do
not load any model. CUDA/Vulkan runtime behaviour, ANE residency, Apple asset
availability and measured results require the owner's hardware run.
