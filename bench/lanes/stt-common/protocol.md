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
