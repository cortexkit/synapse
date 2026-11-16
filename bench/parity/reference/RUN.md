# Run the public Qwen3-ASR reference on one CPU venue

The full run is for a dedicated x86 Linux CPU machine, not a shared build fleet.
All 6,560 fixture rows must be produced on that one machine. The corpus and model
pins are in git; no private audio, private names or private transcripts are inputs.
Pilot outputs are measurements only and cannot be mixed into this reference.

## Provisioning and expected size

Reserve **32 GiB RAM per model process**, plus operating-system and driver
headroom. Processes have one intra-op thread, one inter-op thread, eager attention,
fp32 parameters, deterministic algorithms and MKLDNN disabled. Allocate at least
one physical CPU core per process; simultaneous model loads and memory bandwidth
can limit scaling. Do not use swap as model capacity. Allow at least 30 GB of disk
for original checkpoint/source caches, retained tensors, JSON and temporary files.

The 20-fixture pilot measured 59.65 projected single-thread inference hours for
the whole corpus, before venue differences. Reusing each loaded model removes
the estimated 34.92 hours of per-fixture cold loads. The largest measured process
RSS was 16.44 GB (15.31 GiB) on the 9.5-minute listed clip.

| N | RAM budget before OS headroom | Ideal inference wall time | Planning wall time |
|---:|---:|---:|---:|
| 8 | 256 GiB | 7.46 hours | roughly 8–9 hours |
| 16 | 512 GiB | 3.73 hours | roughly 4–5 hours |

These are projections, not measured full runs. They assume independent-core
scaling; CPU instruction set, memory-bandwidth contention, model-load I/O, network
preparation and shard imbalance can increase time substantially. The two longest
fixtures each took about 45 minutes in the pilot. All CPU models/flags, packages
and per-fixture timings are retained so the venue's actual throughput is auditable.

Expected retained external tensors: 10.392 GB before content-address reuse,
approximately **2.720 GB** after reuse across 1,551 distinct audio inputs. Expected
small committed corpus/traces/receipts: about **44 MB**, with a 30–60 MB planning
range. Tensor bytes stay outside git.

## Install the pinned CPU environment

Use the reviewed checkout and Python 3.11 or newer. Activate a new environment;
the runner records the actual Python version, packages and full `pip freeze`.
The Torch CPU wheel must come from its CPU index, not a CUDA wheel.

```sh
python3 -m venv target/qwen-reference-venv
. target/qwen-reference-venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r bench/parity/reference/requirements-qwen3-asr.txt
```

The runtime checks the pinned inference-package versions. The original checkpoint
is fetched at HF revision `7278e1e70fe206f11671096ffdd38061171dd6e5` and every file
is SHA-256-verified, including both safetensors shards. Public data preparation
must reproduce `qwen3-asr-corpus.json` byte-for-byte before model inference starts.

## Start once

Choose a new or empty output directory on the rented machine:

```sh
bash bench/parity/reference/full_run.sh 8 /data/qwen-reference
```

The only positional arguments remain `N` and `<out>`; `N` was already a parameter,
so no implicit default changes. `PYTHON` can select the active
environment's interpreter. Outputs, cache, logs and receipts stay under `<out>`.
No output is written into the checkout. A nonempty output directory is refused;
there is no implicit resume or merge of partial runs.

Before full shards start, two independent processes each load the model once.
They run the same three target fixtures (LibriSpeech, Greek FLEURS, 300-term
Earnings-22) in opposite orders, after different near-silence/noise fixtures.
All three complete trace hashes must match, including tensor byte hashes. The
matching hashes must also match those fixtures in the eventual full shards.
Any difference stops the run.

The full runner then starts N processes. Shard I receives corpus indices
`I, I+N, I+2N, ...` and runs them in exactly that order, with one model load.
One failed process stops its peers. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`BLIS_NUM_THREADS` are all one; tokenizer parallelism is disabled and the Python
hash seed is fixed. Every trace binds the same run provenance, including CPU model
and flags, source-file hashes, model pins, corpus hash and reference-rules hash.
The original tied-head facts are rechecked on that venue.

Changing `N` changes scheduling, not the reference's single-threaded tensor bytes,
generated IDs, margins or parsed output. The cross-process checks run each target
after different earlier fixtures; matching outputs check that reused model state
does not leak those earlier inputs into the target's result. Provenance records `N`
and a unique run identity, so enclosing JSON hashes from different full runs need
not match even though numerical payloads do. On the 52-core, 1.5-TB dedicated
venue, select 44 shards explicitly and use a **new** directory:

```sh
bash bench/parity/reference/full_run.sh 44 "$HOME/qwen-reference-cap-v2"
```

Generation keeps reference slope 25 in `64 + ceil(slope * duration_s)`. Every
trace stores the exact `cap` and `stop_reason`. A final EOS means `stop`, including
EOS exactly at the cap. Otherwise, generation must fill the cap and records
`cap`, not a fatal error; it is a genuine bounded greedy prefix. Empty, over-budget
or unexplained early non-EOS continuations remain errors. The actual Transformers logits processor lists must be
empty; pinned/effective config receipts attest that increasing this stopping
budget cannot affect logits. `generation-policy.json` preserves that evidence.
Capped trace identity includes the original reference cap. Future serving-cap
choices never change that prefix: engine parity compares the exact recorded IDs.

After complete coverage, the runner measures a lower-bound slope from EOS-terminated
no-list >=2-second speech
fixtures, excluding their generated language prefix through `<asr_text>` but
including their stop token. All LibriSpeech, FLEURS, Earnings-22 and concatenated
LibriSpeech rows are eligible only when their request has empty `terms` and the
trace has `context_tokens == 0`. A `tier` label does not identify list presence;
whole-call requests can have no tier, and `-none` controls retain tier labels.
A measured slope above 25 stops the run for review. A smaller measured slope is
recorded as evidence; **generation and serving slope remain 25**, so the cap
formula matches every recorded cap. Every no-list EOS-terminated fixture needs
headroom of 1.5 times its entire generated length at that fixed slope.
**Both capped traces and name-list requests are exempt from speech-rate slope
measurement and complete-cap headroom.** Capped traces' full
rates are unmeasured, and their identity retains the fixed reference cap rather
than a derived serving cap. `capped-fixtures.json` reports exact counts and IDs
globally, by split, by corpus tier (`none`, `30`, `100`, `300`), and by split/tier.
Here `none` means an absent corpus tier label; whole-call lists and probes can
still have terms. The highest prefix-subtracted rate among stopped listed speech
requests is reported separately, with its fixture ID. Name-list recitation is
preserved as genuine reference behavior, not filtered or reinterpreted as speech.
The final maximum-rate fixture,
ten-minute cap and KV-position counts are written to `calibration.json`.

Key/value-cache sizing reports `7800 audio positions + observed template positions + bias
budget + ten-minute cap`. It records both the bias-free count and a scenario with
the full 2,048-token upper bound. That scenario does not select a serving bias
limit, certify quantized weights against the fp32 teacher, or establish list
accuracy and hallucination safety for the 30/100/300-term tiers.

Successful completion writes `run-complete.json` and finally `SHA256SUMS` for
every retained output except the public download/preparation cache. Exit zero and
those receipts are required. `FAILED.json`, a missing receipt, or a nonzero exit
is not a usable reference. Keep failed outputs for diagnosis; restart into a new
directory only after the failure has been reviewed.

## Return artifacts and seal

The operator returns content-addressed tensor files directly to the Mac's shared
cache, and the rest of the run to the worker checkout. For example, substitute
the actual host, venue output and worker checkout paths:

```sh
rsync -a --checksum HOST:/data/qwen-reference/external/qwen3-asr-1.7b/ "$HOME/Work/Projects/CortexKit-scratch/parity-cache/qwen3-asr-1.7b/"
rsync -a --checksum --exclude='cache/' --exclude='external/' HOST:/data/qwen-reference/ "$WORKTREE/.cortexkit/internal-docs/investigations/ref-corpus/full-run/"
```

Do not return the public checkpoint cache or place tensors in git. Preserve the
venue's complete `SHA256SUMS` and all small output directories, including `checks`,
`shards` and `logs`; the sealer verifies the full retained manifest after relocation.

Only after all returned artifacts are present, from the worker checkout:

```sh
python3 bench/parity/reference/seal_qwen3_asr.py --corpus bench/parity/reference/qwen3-asr-corpus.json --generated .cortexkit/internal-docs/investigations/ref-corpus/full-run --external-cache "$HOME/Work/Projects/CortexKit-scratch/parity-cache"
```

The sealer checks one-venue lineage, source/model/corpus identities, determinism,
processor evidence, every tensor hash and both cap obligations again. It retains
each observed generation cap and stop reason, adds the derived sealed cap only
to EOS-terminated traces, writes fixture receipts
and updates `fixtures/index.json`. Final committed traces and receipts are small;
external f32 tensors remain in the shared cache. No engine work may treat
`full_run.sh` and its Python runner,
pilot or an incomplete returned run as a sealed reference.

## Offline grading of the returned 76cc6a4b run

The dedicated venue completed every trace at revision
`76cc6a4b10f03c40c73ae82baeb55ab83bf793bc`, but its old calibration treated a
stopped 100-name recitation as speech and failed. That returned bundle retains
`FAILED.json` and has no venue `run-complete.json`, `calibration.json` or
`SHA256SUMS`. Do not add or pretend the venue emitted those files.

The explicit recovery mode is limited to that named failure and generation
revision. It matches original source/rules/corpus hashes against
`qwen3-asr-generation-76cc6a4b.json`, reruns all non-inference coverage, shard-order,
greedy-policy, determinism, lineage, tensor and grading checks, and emits a
separate `offline-regrade.json` alongside the sealed reference. That receipt
names the original generation revision, the committed grading revision, the
approved name-list calibration change, the original failure, and checksums for
every returned source artifact and external tensor. Original trace provenance is
retained. Commit the grading code before running, so its identity is verifiable.

Preserve the complete returned bundle unchanged under ignored internal-docs, then:

```sh
python3 bench/parity/reference/seal_qwen3_asr.py --offline-regrade --corpus bench/parity/reference/qwen3-asr-corpus.json --generated .cortexkit/internal-docs/investigations/ref-corpus/full-run --external-cache "$HOME/Work/Projects/CortexKit-scratch/parity-cache"
```

This mode does no generation, does not resume shards and writes nothing into the
returned bundle. A changed generation-input hash, another failure, or any failed
non-inference check is refused rather than silently waived.
