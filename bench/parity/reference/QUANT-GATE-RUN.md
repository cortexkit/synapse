# Full public Q8 CPU run kit

This kit runs the 6,559 inferenced public fixtures, not the over-limit refusal or
any private recordings. It never admits engines: the complete public bundle must
be checked locally, followed by the operator's private aggregate and
`bench/parity/reference/seal_qwen3_asr_quant.py`. That sealer already exists at
this path; no alternative layout or compatibility script is needed.

## Upload exactly one archive

From the repository root on the operator's machine:

```sh
KIT="$PWD/.cortexkit/internal-docs/investigations/quant-gate/full-run-kit.tar.gz"
python3 bench/parity/reference/qwen3_asr_quant_run.py pack --output "$KIT"
# Set HOST to the rented machine's SSH destination, for example root@IP.
scp "$KIT" "$HOST:~/full-run-kit.tar.gz"
```

`pack` refuses an existing destination and prints its SHA-256 and byte count.
The archive contains a `quant-gate-kit/` directory with the explicit file/hash
inventory in `quant-gate-kit.json`: parity Cargo.toml/Cargo.lock/models.json and
Rust sources; the six production Rust weight-derivation modules they import;
reference Python/JSON/requirements and these run instructions; sealed fp32 trace
JSON; and the frozen `bench/lanes/stt-bias/score.py`. There are **no model weights,
audio, raw tensors, caches, Git metadata, private inputs or internal reports**.
Check the received archive's SHA-256 against the pack command's output.

## Setup on bare Ubuntu

The instance needs outbound HTTPS, a CPU-only environment, sudo/root for Ubuntu
packages, and enough local disk for original shards, both derived files, public
source archives and decoded audio. Budget at least 50 GB free. Do not run setup
while workers are active.

```sh
# Run on the rented machine. Keep the actual rental billing-start timestamp.
export RENTAL_START="$(date +%s)"  # replace if billing began before this shell
sha256sum ~/full-run-kit.tar.gz
mkdir -p ~/quant-campaign
tar -xzf ~/full-run-kit.tar.gz -C ~/quant-campaign
cd ~/quant-campaign/quant-gate-kit
export WORK="$PWD/quant-work"
set -o pipefail
bash bench/parity/reference/setup_qwen3_asr_quant_run.sh "$WORK" 2>&1 | tee setup.log
```

The single setup script installs Ubuntu prerequisites and pinned Rust **1.100.0**.
It preserves the initial billing timestamp in `quant-work/rental-start`;
an exported `RENTAL_START` overrides its first-invocation timestamp.
It installs pip **25.3**, CPU Torch **2.9.1**, and the exact primary dependency versions in
`requirements-qwen3-asr.txt`. It builds only `q8-derive` and `qwen-asr-wer`, using
the standalone parity Cargo.lock with `--locked` and default features off.
It verifies the uploaded inventory, then fetches every pinned HF file at
`Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`.
The Rust producer derives the Q8 decoder and f16 remainder; both must hash-equal:

- Q8: `c7555f746972f0433121e05ed9accf2f69d8d15cbf32b5ca0b46e9727c1db1e9`
- f16: `a51f2c7bf8ecca63e25d74e5b7044cfaa2cd65d7ecb9a3e480dc2ea7a7f29bf3`

Public audio uses the existing pinned source resolver: source archives/parquets
are SHA-256 verified, then requested members must match the frozen member hash.
Missing single-name/long-form provenance resolves to the sealed clean corpus;
Earnings rows resolve to its pinned parquet. Seeded probes and concatenations
are rebuilt from their frozen recipes. The completed shared cache is read-only.
Any digest mismatch stops setup. A pre-existing verified derivation can be
reused; an incomplete derivation without its receipt must be moved aside before
a setup retry. Nothing silently re-pins an artifact.

## Start or resume

On a 52-thread Xeon with about 1.5 TB RAM, use **28 single-thread processes**.
The observed single-process peak was 17.75 GB. Admission reserves **20 decimal GB
per process**: 28 requires 560 GB of Linux `MemAvailable`. `--processes N` is
explicit; insufficient memory refuses with the safe maximum, never silently
reducing concurrency. This admission estimate is not worst-case hardware
certification. Leave RAM and CPU time for the supervisor, OS and file cache.

```sh
cd ~/quant-campaign/quant-gate-kit
export WORK="$PWD/quant-work"
read -r RENTAL_START <"$WORK/rental-start"
nohup "$WORK/venv/bin/python" -u bench/parity/reference/qwen3_asr_quant_run.py run \
  --cache "$WORK/cache" --state "$WORK/run" \
  --wer-binary "$WORK/bin/qwen-asr-wer" --processes 28 \
  --hourly-rate 0.45 --rental-start "$RENTAL_START" --spent-before 0 \
  >"$WORK/supervisor.log" 2>&1 &
printf '%s\n' "$!" >"$WORK/supervisor.pid"
```

Run the **same command** to resume a stopped job. Finished case archives are
verified and skipped before loading a model; each killed shard loses at most
its case in flight. A case archive atomically publishes teacher/profile JSON,
content hashes, frozen input identity, completion time and timings. Partially
written temporary files never count as finished. A corrupt/foreign archive
fails closed, naming the file; move it aside only after diagnosing the damage.
Definitive gate failures cannot be resumed. Execution interruptions can be
retried; their prior failure record is retained under `run/history/`.

Shards are deterministic longest-processing-time partitions over sealed fp32
lengths, with stable ID tie-breaking. Cost units are
`generated * (prompt + generated) + prompt² // 32`; this is a scheduling proxy,
not a runtime promise. Long expensive fixtures start first. A different explicit
process count or cache mount on restart can re-shard without invalidating
finished cases; numerical code, profile, model/corpus or baseline changes cannot.
One supervisor lock prevents concurrent ownership of the same run directory.

If the machine must be replaced, preserve `quant-work/run/` (especially every
`cases/*.tar.gz` and plan.json) on persistent storage or copy it off the machine
before destruction. Re-upload the **same kit**, rerun setup for a fresh public
cache, restore the run directory and invoke the start command with the new cache
path. Set the new rental start/rate and `--spent-before` to prior rental spend.
Instance destruction without a checkpoint cannot preserve data on an ephemeral
disk; resume does not claim otherwise.

## Progress, budget and stopping

One poll command, from the kit root:

```sh
"$WORK/venv/bin/python" bench/parity/reference/qwen3_asr_quant_run.py progress --state "$WORK/run"
```

It prints JSON: completed/total, cases/hour over the last-hour window, weighted
projected hours remaining, observed-window basis, rental spend so far, projected
total cost against **$10**, stale-status age, and any definitive failure.
The ETA uses completed fp32-cost units per second, not just case count, because
case durations vary enormously. Before any case completes, ETA/cost projection
are unknown. Early rates and estimates after outages are not upper bounds.
Check `status_age_seconds` and supervisor.log if no new case completes.

**The operator polls and enforces the $10 rental cap.** The kit reports budget
risk; it does not control the rental service or silently raise the cap. Include
setup/upload time in the rental-start timestamp. Stop if projected total cost
exceeds the approved cap; then shut down/bill-stop the rental through its owner.
A 28-process run does not inherit single-process pilot throughput: shared memory
bandwidth can dominate. The fp32 CPU run that sealed this corpus measured 399
summed process-hours against a 95-process-hour pilot projection (4.2x); monitor
actual progress rather than trusting that projection.

```sh
read -r SUPERVISOR <"$WORK/supervisor.pid"
kill -TERM "$SUPERVISOR"
# Wait for it to stop before restarting or copying the live run directory.
```

A forbidden listed name on a probe, a new Q8 cap hit where the sealed fp32 trace
stopped normally, or a calibration tolerance above 1.0 logit stops every worker
process group and records failure.json. The tolerance, tau, is the larger of
0.1 logit and twice the largest teacher margin at a teacher/profile top-1
disagreement. Positions with teacher margin above tau are graded; each complete
fixture set must have at least 90% graded positions. Aggregate coverage and
quality checks wait until their whole required set/arm is complete; a definitive
aggregate failure also stops all peers. Completed sets are rechecked when a
larger observed tau changes grading. A failing 30-term tier stops the run; failing 100/300-term tiers are
measurements for choosing the largest passing tier, not engine admission.
Workers also receive Linux parent-death SIGTERM if the supervisor is killed.
The CPU teacher/profile run directly without a Parakeet companion; a companion
`no_speech` result never exempts a probe. The private checker is unchanged and is
never invoked on the rented machine.

## Collect, copy back, verify and seal locally

After successful full public completion, one command builds the transport bundle:

```sh
"$WORK/venv/bin/python" bench/parity/reference/qwen3_asr_quant_run.py collect \
  --state "$WORK/run" --wer-binary "$WORK/bin/qwen-asr-wer" \
  --output "$WORK/q8-public-bundle.tar.gz"
```

Collection re-verifies all 6,559 case archives and reruns complete public grading.
It prints archive SHA-256/bytes and refuses failures, missing cases or a bundle
over **3 GiB**. The bundle contains only `generated/teacher/*.json`,
`generated/forced-profile/*.json`, public grading, plan and a hash manifest.
No raw tensors, model weights or audio are needed by the existing sealer.
If a gate failed, copy failure.json, aggregate.json, logs and case archives for
diagnosis instead; do not claim a sealable passing bundle.

On the operator's machine, use the same checkout files that produced the upload
archive. `verify-bundle` compares the recorded file-hash identity and refuses a
different checkout:

```sh
scp "$HOST:~/quant-campaign/quant-gate-kit/quant-work/q8-public-bundle.tar.gz" .
sha256sum q8-public-bundle.tar.gz  # compare with collect's output
mkdir -p .cortexkit/internal-docs/investigations/quant-gate/full-public
tar -xzf q8-public-bundle.tar.gz -C .cortexkit/internal-docs/investigations/quant-gate/full-public
GENERATED="$PWD/.cortexkit/internal-docs/investigations/quant-gate/full-public/generated"
python3 bench/parity/reference/qwen3_asr_quant_run.py verify-bundle --generated "$GENERATED"
# Run the unchanged local private script as documented in QUANT-GATE.md.
# PRIVATE_AGGREGATE names its aggregate JSON file, never private input data.
cargo build --locked --release --manifest-path bench/parity/Cargo.toml --bin qwen-asr-wer
python3 bench/parity/reference/seal_qwen3_asr_quant.py --generated "$GENERATED" \
  --private-aggregate "$PRIVATE_AGGREGATE" \
  --wer-binary "${CARGO_TARGET_DIR:-bench/parity/target}/release/qwen-asr-wer" \
  --output .cortexkit/internal-docs/investigations/quant-gate/sealed-profile.json
```

Build the WER helper natively for the local machine; the rented machine's Linux
executable cannot run on a Mac. A public-complete bundle still has engine
admission false. Only complete public grading plus the passing private aggregate can seal
a profile. The list-token limit remains provisional until the shared renderer
and trimmer perform their final gates.

## Portable orchestration verification

No model, download, private data or inference is used by these tests:

```sh
PYTHONPATH=bench/parity/reference python3 -m unittest -v test_qwen3_asr_quant_run test_qwen3_asr_quant.Quality test_qwen3_asr_quant.Caps
bash -n bench/parity/reference/setup_qwen3_asr_quant_run.sh
```
