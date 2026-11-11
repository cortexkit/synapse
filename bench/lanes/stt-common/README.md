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
and parent-observed process wall are also recorded separately. Process peak RSS
includes loading/warmup; on macOS it excludes out-of-process CoreML/Speech service
allocations. It is not total-system or total-accelerator memory. The used APIs do
not expose GPU **peak** allocation or energy counters: `gpu_peak_bytes` and
`energy_joules` are null (TSV `NA`), not zero. Power requires an external owner-run
measurement such as powermetrics/NVML with its scope and interval recorded.

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
export WAVS="$HOME/Work/Projects/CortexKit-scratch/parity-cache/parakeet-wavs"
# Explicit asset installation, separate from benchmark timing; no inference here.
target/release/stt-bench --wavs "$WAVS" --prepare-only --model system --model-revision OS-managed --backend apple-speech --output target/stt-apple-request.json
target/stt-apple-speech/release/stt-apple-speech --request target/stt-apple-request.json --backend apple-speech --install-assets-only
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
`STT_LANES='fluid apple'` selects only those Mac lanes. Defaults are all three Mac
lanes on Darwin and transcribe.cpp on Windows/Linux. Binaries live in separate
`target/stt-BACKEND/release/` folders to avoid feature/backend confusion.
`STT_BENCH`/`STT_TABLE` can point at separately built portable tools.

The final table is `OUTPUT_DIRECTORY/matrix.tsv` (one row per engine/model/backend/
split). JSON siblings preserve raw transcripts, input requests and provenance.
To combine machines without mixing request/worker files:

```sh
target/release/stt-table /results/m5/transcribe-metal-ultra-q8.json /results/rtx3080/transcribe-cuda-ultra-q8.json > combined.tsv
```

For re-scoring an existing worker JSON, replace `--worker EXE` with
`--transcripts FILE` in a `stt-bench` command and supply the same model/revision/
backend. Replays have null parent-observed wall time, not a fabricated timing.
See [`protocol.md`](protocol.md) for the small worker JSON contract.

## Portable checks (no hardware inference)

```sh
cargo test --locked -p stt-common -p stt-transcribe-cpp --features stt-transcribe-cpp/native stt_
cargo clippy --locked -p stt-common -p stt-transcribe-cpp --features stt-transcribe-cpp/native --all-targets -- -D warnings
cargo fmt --all --check
```

These use the CPU native build only to check/link the Rust binding; the tests do
not load any model. CUDA/Vulkan runtime behaviour, ANE residency, Apple asset
availability and measured results require the owner's hardware run.
