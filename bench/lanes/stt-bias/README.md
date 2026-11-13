# Local name-list benchmark (reference runtimes only)

This lane measures Parakeet TDT v3 with transcribe.cpp's existing vocabulary
rescoring against trained context inputs in Qwen3-ASR and Granite Speech.
It does not add an engine to Synapse or modify any reference decoder.
Runtime and model revisions are in `pins.json`; installed Python versions and
model-file hashes are recorded with each run. The aggregate findings are in the
ignored `.cortexkit/internal-docs/investigations/stt-name-list-benchmark.md`.

## Privacy and resources

Run preparation and inference **on the owner's Mac only**. Never pass either
COLLI corpus, a prepared manifest, recordings, or predictions to a remote
build server, a vendor API, or a hosted inference service. The only network
operation below fetches **public** model/dataset assets. Runtime downloads are
disabled during inference. All caches, audio derivatives, per-clip predictions,
resource telemetry, and runtime logs belong under this worktree's
`target/stt-bias/`; the tools reject other prediction/manifest destinations.
Do not commit any of those files. Only `test_score.py`'s synthetic text fixtures
may go to the portable test runner.

Run one GPU model process at a time. The watchdog samples `sysctl vm.swapusage`,
`vm_stat`, `memory_pressure -Q`, process RSS, and host load before model load and
every 30 seconds. MLX provenance also records the kernel's lifetime maximum
`phys_footprint` via `proc_pid_rusage(RUSAGE_INFO_V4)`, which includes peaks
between samples; this is the process-memory comparison, not RSS.
It refuses admission below 25% free; it stops at RSS >20 GiB,
swap growth >1 GiB since admission, free memory <25%, or >25,000 pageouts within
the last two minutes. These are owner-approved shared-host limits, not claims
that all host memory traffic came from the benchmark. MLX additionally caps its
allocator at 16 GiB and its reusable cache at 512 MiB. Downloads require 20 GiB
free and a cumulative 15 GB asset budget including a 1 GB dependency reserve.
FluidAudio requires a **separately scheduled Apple Neural Engine (ANE) window**;
GPU authorization is not ANE authorization.

## Setup (Mac, repository root)

The measured environment used Python 3.12.12, full Xcode, CMake, MLX 0.32.4,
mlx-audio 0.6.5, mlx-lm 0.32.2, and unquantized original HF weights, loaded
directly without an offline conversion. No PyTorch/MPS or cloud inference ran.
Parakeet uses the read-only lane-pinned **F16** GGUF, not Ultra or Q8.

```sh
uv venv --python 3.12 target/stt-bias/venv
uv pip install --python target/stt-bias/venv/bin/python -r bench/lanes/stt-bias/requirements.txt
git clone --filter=blob:none --no-checkout https://github.com/anomalyco/transcribe.cpp target/stt-bias/transcribe.cpp
git -C target/stt-bias/transcribe.cpp checkout bcb639faa4c74a2300ba19e30360cb2355b1663a
git -C target/stt-bias/transcribe.cpp submodule update --init --recursive
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun --find metal
cmake -S target/stt-bias/transcribe.cpp -B target/stt-bias/tc-build \
  -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_SHARED=ON \
  -DTRANSCRIBE_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DTRANSCRIBE_BUILD_TESTS=OFF -DTRANSCRIBE_BUILD_EXAMPLES=ON -DTRANSCRIBE_BUILD_TOOLS=OFF
cmake --build target/stt-bias/tc-build -j 4
uv pip install --python target/stt-bias/venv/bin/python --no-deps ./target/stt-bias/transcribe.cpp/bindings/python
export HF_HOME="$PWD/target/stt-bias/hf-home" HF_HUB_DISABLE_TELEMETRY=1
PY="$PWD/target/stt-bias/venv/bin/python"
for asset in earnings22 qwen_0_6b qwen_1_7b granite_2b; do
  "$PY" bench/lanes/stt-bias/fetch.py "$asset"
done
```

Do not compile portable Rust code locally: this native C++ build needs this
Mac's full Xcode/Metal toolchain. The existing benchmark lanes are unchanged.
`transcribe_adapter.py` uses the pinned Python binding plus its public C
vocabulary extension (absent from that pin's generated Python extension list),
checks the C extension's size and kind written by its native initializer against
the Python structure declaration, and passes boost 2.5 or 4.0. It never uses
the unscored `sounds_like` substitutions. The default 2.5 and stronger 4.0 are
fixed in advance, not tuned on these recordings.

## Prepare and measure

```sh
COLLI="$HOME/Work/Projects/CortexKit/colliculus/.cortexkit/internal-docs/research"
WAVS="$HOME/Work/Projects/CortexKit-scratch/parity-cache/parakeet-wavs-all"
GGUF="$HOME/Work/Projects/CortexKit-scratch/stt-models/gguf/parakeet-tdt-0.6b-v3-F16.gguf"
"$PY" bench/lanes/stt-bias/prepare.py \
  --real "$COLLI/speech-bias-ufuk-v1" --tts "$COLLI/speech-bias-v1" --wavs "$WAVS"
"$PY" bench/lanes/stt-bias/run.py --system transcribe --boost 2.5 --gguf "$GGUF" \
  --output target/stt-bias/transcribe25.jsonl > target/stt-bias/transcribe25.log 2>&1
"$PY" bench/lanes/stt-bias/run.py --system transcribe --boost 4.0 --gguf "$GGUF" \
  --lists local global unrelated --output target/stt-bias/transcribe40.jsonl \
  > target/stt-bias/transcribe40.log 2>&1
for pair in 'qwen_0_6b qwen06' 'qwen_1_7b qwen17' 'granite_2b granite2'; do
  set -- $pair
  "$PY" bench/lanes/stt-bias/run.py --system "$1" --output "target/stt-bias/$2.jsonl" \
    > "target/stt-bias/$2.log" 2>&1 || break
done
```

The loop uses a POSIX shell (`bash`), not fish. Run each command as a background
shell task and wait for its exit before starting another model. Do not
interpret a disconnected shell as successful completion: reconcile provenance
status and the requested ID/list pairs. All normal arms have 1,760 predictions:
720 none, 320 local, 320 global, 400 unrelated. The stronger Parakeet arm has
1,040, because it shares the default run's no-list baseline.
For a watchdog interruption, retain its logs and resume only unfinished pairs
into a **new** output, for example:

```sh
"$PY" bench/lanes/stt-bias/run.py --system qwen_0_6b \
  --resume-from target/stt-bias/qwen06.jsonl --output target/stt-bias/qwen06-complete.jsonl \
  > target/stt-bias/qwen06-complete.log 2>&1
```

Each clip/list timing includes local WAV read, mono conversion/resampling, and
synchronous inference, but excludes model load and one no-list warmup. List
arms are interleaved per clip to limit load drift. Decoder state is reset by
each offline call. Inference is deterministic greedy (MLX temperature zero,
seed zero, 1,024 output tokens; a token-limit hit fails, not a scored partial).
Qwen's comma-joined list is its **system/context** field, not generated text;
English is fixed. Granite's user instruction is exactly
`transcribe the speech to text. Keywords: <comma-joined list>` (without the
`Keywords:` suffix for none). The language selector is not used for Granite,
because its runtime interprets that option as a translation target.

## Corpus and metric contract

- COLLI: 20 real recordings, plus 100 TTS recordings of the same 20 phrases.
  `global` preserves the provided 30-name list/order. `local` is an oracle list
  of the phrase's expected terms. All 34 expected term groups are scored,
  including ones missing from the supplied 30-name list, matching COLLI's bar.
  Preparation loads the supplied **corpus-local** `score.py`, hashes it, and checks normalization
  against it. No private phrase or transcript text is copied into this lane.
- Earnings: pinned `argmaxinc/contextual-earnings22` **test** parquet has 772
  rows, unlike the paper's 760. Select 200 unflagged row indices using
  `sorted(random.Random(20261119).sample(eligible_indices, 200))`. Selection
  indices and parquet hash stay in target provenance. `dictionary` is local;
  `keywords` is call-global, preserving released order. One dictionary term
  has no normalized reference span: it remains in anywhere-recall accounting,
  but cannot enter aligned recall's denominator. All sampled global lists
  have <=46 entries, so a common first-64 slice is identical to global.
  `global64` is prepared for explicit use on another subset but was not
  redundantly executed. No long-list robustness claim follows from this set.
- Controls: all 200 test-clean and 200 test-other clips in the sealed parity
  corpus, against original ground truth, with COLLI's unrelated 30 names.
  Hash-check every supplied WAV. No control reference contains a listed term.

The scorer uses COLLI normalization: ASCII lowercase, hyphens and punctuation
to spaces, and only its four digit substitutions. This intentionally differs
from `synapse-wer-norm-v1`; do not mix these WERs with parity-lane WERs.

Lead metrics are **position-aligned term-occurrence recall and precision**.
Minimum-word-edit alignment breaks ties diagonal, deletion, insertion. Match
whole words, aliases, and longest non-overlapping multiword spans. The evaluation
vocabulary is fixed across arms: COLLI's expected terms plus unused list terms;
Earnings' global/local union; controls' unrelated list. Recall denominator is
every reference occurrence of that vocabulary (including repetitions and
unannotated occurrences). Precision denominator is every predicted occurrence;
an occurrence is right only if the same term occupies exactly the aligned
reference span. False insertions are remaining predicted occurrences, not just
unique term strings. This counts substitutions into names as false insertions.

**COLLI-compatible anywhere recall** is also reported: one hit per expected
annotation if any accepted spoken form appears anywhere. Compare that column,
not occurrence recall, with COLLI's hosted 91–94%. Earnings' annotation-based
anywhere denominator differs from its reference-occurrence denominator.
Full WER is micro-averaged word edits/reference words. **Non-list WER** masks
reference term spans in the same alignment, counting substitution/deletion
errors only on non-term reference words plus insertions not covered by exact
predicted term spans. Thus exact listed-name insertions are excluded from
non-list WER but visible in false insertions and full WER. Baseline and biased
arms use the same masks/vocabulary. Zero denominators are `null`, never 0%.
Partial, duplicate, missing, or invalid measurements fail closed.

## Qwen 1.7B 16-bit versus 8-bit

`qwen_1_7b_8bit` selects the pinned **mlx-community** conversion, not a new
training run. Its model card names mlx-audio 0.3.1 as the converter; its config
specifies standard affine 8-bit quantization, group size 64. The pinned runtime
above loads both models with the same decoder. `run.py` verifies weight SHA256s
and the quantization config, and records all runtime-file hashes and logical
model bytes (excluding environment/cache duplication). Not every tensor is
quantized: convolutions and other unselected tensors retain their stored dtype.
Aggregate findings are in the ignored operational record
`.cortexkit/internal-docs/investigations/stt-qwen-8bit.md`; that private/local
measurement report is not part of the public harness distribution.

Use a fresh target directory for each experiment. Set up Python as above, then:

```sh
for asset in qwen_1_7b qwen_1_7b_8bit earnings22; do
  "$PY" bench/lanes/stt-bias/fetch.py "$asset" --budget-gb 10 || break
done
# Prepare the ordinary 720-clip manifest first with prepare.py, as above.
# Retain all COLLI/control arms and only Earnings global; no clip filtering.
"$PY" - <<'PY'
import json
from pathlib import Path
source = Path('target/stt-bias/manifest.jsonl')
output = Path('target/stt-bias/compare-manifest.jsonl')
with output.open('x') as stream:
    output.chmod(0o600)
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row['dataset'] == 'earnings22-test-200':
            row['lists'] = {'global': row['lists']['global']}
        stream.write(json.dumps(row) + '\n')
PY
for system in qwen_1_7b qwen_1_7b_8bit; do
  "$PY" bench/lanes/stt-bias/run.py --system "$system" \
    --manifest target/stt-bias/compare-manifest.jsonl \
    --output "target/stt-bias/$system-bias.jsonl" || break
done
"$PY" bench/lanes/stt-bias/score.py --manifest target/stt-bias/compare-manifest.jsonl \
  --predictions target/stt-bias/qwen_1_7b-bias.jsonl target/stt-bias/qwen_1_7b_8bit-bias.jsonl \
  --output target/stt-bias/bias-aggregates.json
```

This is **1,360 predictions per model**. Run long measurement loops as
background shell tasks and wait for completion. Use one GPU process at a time;
neither MLX arm uses the ANE. The tighter 10 GB budget includes a 1 GB reserve
for dependencies, in addition to the disk/resource checks above.

For the multilingual matrix comparison, `parity.py` prepares all 1,200 verified
WAVs without exposing ground truth to the decoder. Use **automatic language
detection**, no vocabulary/name list, and no ground-truth language hint. The English bias study
retains its existing forced-English setting. The aggregate language counters
mark a clip as mismatched if any chunk reports the wrong language; absent/empty
labels have a separate unknown count. Unknown and mismatch can overlap if a
multi-chunk clip has both kinds of label.

```sh
"$PY" bench/lanes/stt-bias/parity.py prepare --wavs "$WAVS"
for system in qwen_1_7b qwen_1_7b_8bit; do
  "$PY" bench/lanes/stt-bias/run.py --system "$system" --language auto \
    --manifest target/stt-bias/parity-manifest.jsonl --lists none \
    --output "target/stt-bias/$system-parity.jsonl" || break
done
```

The tiny Rust bridge calls **`stt_common::load_corpus` and `stt_common::score`**,
including `synapse_parity::speech::grade_wer`, rather than approximating Unicode
normalization in Python. It verifies the sealed corpus/WAVs again and emits only
aggregate split scores. Build on the build server normally; a Mac-native scorer
build for local-only transcript grading needs the Mac hardware owner's explicit
exception to the normal build-server policy:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build --locked \
  --manifest-path bench/lanes/stt-bias/parity-score/Cargo.toml \
  --target-dir target/stt-bias/scorer-build
for system in qwen_1_7b qwen_1_7b_8bit; do
  "$PY" bench/lanes/stt-bias/parity.py score --wavs "$WAVS" \
    --predictions "target/stt-bias/$system-parity.jsonl" \
    --scorer target/stt-bias/scorer-build/debug/stt-bias-parity-score \
    --output "target/stt-bias/$system-parity-aggregate.json" || break
done
```

Bias WER still uses COLLI normalization; full-corpus WER uses parity's NFKC,
Unicode case-fold and punctuation-deletion normalization. Do not compare those
two WER columns directly. Median latency is the conventional middle value (or
mean of the middle pair); p95 uses nearest rank `ceil(0.95 * n)`. Model load and
one warmup are excluded, WAV loading/resampling and synchronous GPU inference
included. A single ordered pair on a shared Mac does not isolate quantization's
speed effect from host-load changes.

## Aggregate and check

```sh
"$PY" bench/lanes/stt-bias/score.py --manifest target/stt-bias/manifest.jsonl \
  --predictions target/stt-bias/transcribe25.jsonl target/stt-bias/transcribe40.jsonl \
    target/stt-bias/qwen06-complete.jsonl target/stt-bias/qwen17.jsonl target/stt-bias/granite2.jsonl \
  --output target/stt-bias/aggregates.json
# Portable synthetic-fixture gate only (run on the Linux build server):
python3 -m unittest discover -s bench/lanes/stt-bias -p test_score.py -v
```

The scorer writes aggregate counts/rates/latency only. A fresh run may use
`qwen06.jsonl` instead of the resumed filename. Keep aggregate evidence copied
into the ignored investigation write-up, not solely in the disposable target
directory. No Rust files or Cargo package dependencies are changed, so Rust fmt,
Clippy, and Cargo tests are not applicable to this Python lane.

## FluidAudio CTC reference (separate ANE appointment)

The pinned FluidAudio supports TDT v3 plus its separate 110M CTC spotter.
The new adapter leaves the existing `stt-fluidaudio` lane untouched. It uses
the default size-aware thresholds, per-term weight 10 (the runtime's text-list
loader setting), `.cpuAndNeuralEngine` for encoders and CPU-only preprocessing.
This permits CPU fallback; it is not exclusive-ANE hardware certification.
Native warning logs invalidate the affected benchmark result set if the runtime silently absorbs a CTC
failure, and a dropped vocabulary term or missing token timings fails the run.

Prepare without touching the ANE:

```sh
for asset in fluidaudio_tdt fluidaudio_ctc; do
  "$PY" bench/lanes/stt-bias/fetch.py "$asset"
done
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build \
  --disable-sandbox -c release --package-path bench/lanes/stt-bias/fluid \
  --scratch-path target/stt-bias/fluid-build
```

Only after the hardware owner explicitly authorizes a Neural Engine appointment, run:

```sh
env -u TMPDIR STT_BIAS_ANE=1 "$PY" bench/lanes/stt-bias/run_fluid.py \
  --lane-lock "$HOME/Library/Caches/ck-synapse/ane-direct.lock" \
  --output target/stt-bias/fluid.jsonl > target/stt-bias/fluid-supervisor.log 2>&1
```

The supervisor takes the existing shared lock non-blocking and read-only,
never replacing its inode. If it is held, stop and reschedule; never stop
another process. It holds the lock through model destruction and watches resident set size (RSS)
of itself **and its native child**. A resource stop terminates the child too.
The supervisor preconverts the same audio to raw 16 kHz float32 PCM in target,
so FluidAudio's timed section includes native PCM read, a fresh TDT decode,
vocabulary tokenization/setup, CTC inference and rescoring, **but not Python's
earlier resampling**. This is a timing boundary difference from the GPU arms.
One biased warmup is excluded to warm both models. The FluidAudio run's 1,760 predictions can
be added to `score.py --predictions` with the same manifest. No ANE inference
is implied merely by building this adapter.
