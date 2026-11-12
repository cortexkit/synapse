# STT worker protocol

The `stt-bench` coordinator verifies corpus audio before launching a worker.
Workers are long-lived processes: load a model once, perform an untimed warmup,
then transcribe each requested utterance in order. A worker must not download
models implicitly.

Arguments: `--request FILE --output FILE --model PATH --backend NAME`.
The request is JSON with `utterances`, an array of objects containing `id`,
`path` (absolute verified WAV path), `split`, `locale`, and `duration_seconds`.

The output is JSON with:

- `engine`, `version`, `commit`, `backend`: provenance strings.
- `cold_start_seconds`: model loading, asset preparation and warmup time.
- `warm_wall_seconds`: elapsed time for the measured corpus loop.
- `peak_rss_bytes`: process high-water resident set size.
- `gpu_peak_bytes`, `energy_joules`: nullable counters. Null means unavailable;
  zero is not a substitute for an unavailable counter.
- `transcripts`: one object per requested utterance with `id`, `text`, and
  `seconds`. An unsupported locale has `error` instead of `text` and `seconds`.

The coordinator rejects missing, duplicate, unexpected or invalid measurements.
Failed utterances remain visible as incomplete coverage; they are never scored
as empty predictions. Word error rate (WER) denominators count only completed utterances, so a
partial result is not directly comparable to a full-corpus result.

Cold start means a new process and unloaded model, not a flushed operating-system
file cache. Record wall time includes worker launch and shutdown but excludes
corpus verification and scoring. Warm wall time excludes model loading and warmup;
per-utterance percentiles and real-time factor use measured warm transcriptions.

The coordinator adds (without changing worker-required fields) `model_size_bytes`,
`weight_precision`, `process_peak_bytes`/`process_peak_method`, NVIDIA sampling
method/baseline/cadence/sample-count fields, and nullable `ane_peak_bytes`.
The process comparison column is macOS lifetime maximum **phys_footprint**,
sampled from the parent, or Linux RSS / Windows peak working set. The original
`peak_rss_bytes` remains available with its original units and meaning. NVIDIA
memory is per PID (pre-spawn baseline zero), preferably NVML compute+graphics,
otherwise `nvidia-smi` compute-apps, with 100 ms target interval and observed
maximum gap. When per-PID memory is unavailable (including `N/A` under the Windows
Display Driver Model, WDDM), use device
`memory.used` minus a baseline captured **before worker spawn**, via NVML
`nvmlDeviceGetMemoryInfo` or `nvidia-smi --query-gpu=uuid,memory.used`.
Device snapshots sum the same NVIDIA GPU UUIDs as the baseline; a changed or
missing set is unavailable. Deltas clamp to zero if other users release memory.
`gpu_peak_method` labels `device-delta` and warns that other GPU users are included.
A valid per-PID peak takes priority; it is never combined with device deltas.
`gpu_idle_baseline_bytes` records the chosen peak's baseline, while
`gpu_device_baseline_bytes` separately preserves the pre-spawn device total.
Missing counters/baselines, including Neural Engine memory, stay null.
Live parent counters are not reconstructed during transcript replay.

Optional loaded-memory probe: when `STT_MEMORY_READY` is set to a path, workers
create that file after load, before warmup, and wait (at most 30 seconds) for the
same path with extension `memory-resume`. The parent snapshots
`gpu_after_load_idle_bytes` then creates the resume file. Workers subtract the
barrier's elapsed duration from `cold_start_seconds`; process wall still includes
it. Workers without the extension remain compatible and have a null idle value.
Current NVIDIA-capable lanes implement it; Mac-only Swift GPU counters stay null.
The idle snapshot also uses per-PID memory first, otherwise device used minus
the pre-spawn device baseline. `gpu_after_load_idle_method` records that snapshot's
scope independently of the peak, including the other-GPU-users caveat.

Synapse backends are named `synapse-metal`, `synapse-ane`,
`synapse-cuda-f16`, `synapse-cuda-f32`, `synapse-vulkan-f16` and
`synapse-vulkan-f32`; the prefix distinguishes their package validation from
competitors' GGUF/CoreML inputs. All use WAV input, one discarded first-utterance
warmup, and observation disabled. CUDA/Vulkan F32 packages are measurement-only,
not new production profiles. The lane never invokes certification harnesses.
