# Qwen3-ASR Q8 CPU quantization gate

This is reference tooling, not a serving engine. **Engine work is not admitted by
the harness or a passing pilot.** `qwen3-asr-quant-profile.json` starts uncalibrated,
with no sealed tau or bias-list token limit. A failed public gate stops the run; a passing
pilot requires the owner's approval for a full-corpus public run and the operator's
separate local private-set check. No private inputs belong on build servers.

## Weights and numerical profile

`q8-derive` is the Rust producer for weight derivation. The Python reference
never quantizes weights. It verifies both complete files against `models.json`:

- `weights.q8_0`: `c7555f746972f0433121e05ed9accf2f69d8d15cbf32b5ca0b46e9727c1db1e9`.
- `remaining.f16.safetensors`: `a51f2c7bf8ecca63e25d74e5b7044cfaa2cd65d7ecb9a3e480dc2ea7a7f29bf3`.

It validates `lineage.json`'s tensor inventory (the derivation sidecar's names,
shapes and byte ranges) against the actual model dimensions
and the prescribed seven-projection-per-layer/head order, not mutable sidecar
offsets. Q8 blocks are dequantized exactly as f32(scale) times signed i8 values.
The f16 remainder supplies encoder, norms and embeddings, expanded exactly to
f32. Head and input embedding are rebound independently, despite the official
loader's tying. CPU reference matrices do not meet the serving engines' resident
memory rule, which forbids keeping dequantized Q8 matrices after load. The
existing fp32 reference and its sealed traces are unchanged.

The Q8 CPU teacher runs without activation rounding. It records its own greedy
ids, including stop, and f32 top-two margins; the existing fp32 traces are quality
baselines, never substituted as Q8 teacher ids. Greedy decoding is explicit
argmax without logits processors or sampling, at the sealed cap of
`64 + ceil(25 * duration_seconds)` tokens.
Reaching the cap is a gate failure unless the same sealed fp32 fixture is marked
capped; only then the Q8 teacher records and forces its exact bounded prefix
without inventing EOS, and the quality and tier gates decide its acceptability.

The separate teacher-forced run uses exactly the boundary list in the profile:
encoder span vectors, non-head `nn.Linear` outputs, norm outputs, positional and
attention/MLP residual adds, softmax probabilities and decoder cache K/V entries
(K after RoPE). All are rounded to binary16 and expanded back to f32. Conv2d,
GELU, activation products and RoPE intermediates stay f32. Dot products, norm and
softmax reductions and logits remain f32; there is no global `.half()`. The
Conv2d outputs stay f32; positional adds round to binary16 and expand back to
f32. Temporary hooks
are removed even after an error. The cache is new for every request. For n teacher
ids, the profile prefills once and consumes exactly the first n-1 teacher ids,
returning n top-1 predictions even if its own predictions diverge or stop early.

Calibration computes `max(0.1, 2 * largest disagreeing teacher margin)`, refuses
tau above 1.0, and requires at least 90% of positions **strictly above** tau in
each pooled fixture set. The stop position is counted. Per-fixture graded and
exempt counts, pooled counts and trace digests go into the sealed profile.

## Public fail-fast pilot

Use the pinned environment in `requirements-qwen3-asr.txt` and the CPU Torch
wheel, on a Linux build server. From the repository root:

```sh
cargo build --locked --release --manifest-path bench/parity/Cargo.toml --no-default-features --features q8-derive --bin q8-derive --bin qwen-asr-wer
python3 bench/parity/reference/run_qwen3_asr_quant.py --cache target/quant-cache --output .cortexkit/internal-docs/investigations/quant-gate/pilot --derive-binary "$CARGO_TARGET_DIR/release/q8-derive"
```

The output directory must be new/empty; the derivation cache must also be empty.
`--start-at ID` permits a separate pilot phase after an execution repair, without
rerunning an unchanged prefix. It lists every omitted case, never merges prior
traces and cannot claim a complete pilot pass or admit engines.
Sources are downloaded only from the frozen corpus/model URLs and SHA-256
verified; public audio members are materialized lazily. All 150 probes run in
corpus order before the remaining 17 representative pilot cases. Every prompt,
span and decoded-audio checksum must match its sealed fp32 baseline. Output holds
small public teacher/profile traces, per-case times and a report, not tensors or
weights. The report binds source hashes, profile, corpus, packages and derived
digests. Model load time and peak process RSS are separate from per-case times.
An atomic sibling `.tar.gz` evidence archive is refreshed after every completed
teacher and profile stage, so a deadline can lose at most the stage in flight.

A listed-term occurrence in parsed probe text, a new cap hit on a fixture whose
fp32 baseline stopped, or tau above 1.0 stops
immediately. A pooled coverage failure stops as soon as that **whole** probe set
has run. A partial speech split's WER or coverage cannot prove the full-split
gate has failed. No companion `no_speech` result can bypass these checks. Exit 2
means an observed gate failure; an exception is an execution error, not evidence
against the model. A pilot pass still leaves `engine_slices_allowed: false`.

The full-corpus evaluator uses the canonical Rust `synapse-wer-norm-v1`
normalizer through `qwen-asr-wer`, accumulating edit/word counts, not mean WER.
It compares every split/list arm separately, long-form included. Name scoring
imports the frozen-digest `bench/lanes/stt-bias/score.py`, not the WER normalizer.
Earnings recall and false occurrences, per-tier 1,200-pair controls, retained-name
recall versus whole-call lists and each control list's unrelated-list WER are
independent gates. Higher tiers may fail; failure of 30 terms stops the gate.
The largest passing tier's longest token cost is rounded up to 64, capped at
2,048, and remains **provisional**, not a serving constant.

Only a complete future run may invoke the offline sealer:

```sh
python3 bench/parity/reference/seal_qwen3_asr_quant.py --generated "$PUBLIC_RUN" --private-aggregate "$LOCAL_AGGREGATE_JSON" --wer-binary "$CARGO_TARGET_DIR/release/qwen-asr-wer" --output "$NEW_PROFILE_REPORT"
```

The sealer requires the entire frozen corpus, exact fp32 baseline bytes and
matched Q8 input identities. It recomputes quality gates and calibration; it
cannot seal a partial pilot. Full public execution is not commissioned by the
pilot command. Its projections show per-case measurements and naive process/
wall hours at four single-threaded processes, plus the previous fp32 run's
399/95 underestimate factor. Neither projection is an upper bound.

## Operator-only private check

The operator runs `run_qwen3_asr_private_quant.py` on the owner's Mac or Windows
machine, never in a worker. It accepts only already-installed local model and
derived artifacts; downloads and HF telemetry are disabled. Model/diagnostic
output (Python and native file descriptors) is suppressed without saving a log.
Only aggregate counts are printed, including names-right percentage and whether
it reaches 90%. Errors print no exception payload, input path or private text.

The local manifest is a JSON array. Each row has `audio` (a relative path to a
mono 16-kHz audio file), `terms` (the private name-list array), `reference` (the
private phrase text), and `expected` (the names spoken in that recording, for
names-right scoring). Keep this manifest and every input outside git;
no real private example belongs in documentation or a worker brief.

```sh
python3 bench/parity/reference/run_qwen3_asr_private_quant.py --manifest "$PRIVATE_MANIFEST" --model "$LOCAL_PINNED_MODEL" --artifacts "$LOCAL_DERIVED_ARTIFACT_DIRECTORY"
```

No per-clip transcript, name, identifier or audio is written. The printed JSON
can be saved locally and handed to the full public evaluator as aggregates only.
This script does not run the private set automatically and does not grant a
worker access to it.

## Scoped verification

```sh
python3 -m unittest discover -s bench/parity/reference -p test_qwen3_asr_quant.py -v
cargo test --locked --manifest-path bench/parity/Cargo.toml --test qwen_asr
cargo clippy --locked --manifest-path bench/parity/Cargo.toml --all-targets -- -D warnings
cargo fmt --all --check
cargo fmt --manifest-path bench/parity/Cargo.toml --check
```

Tests include a deliberately vacuous *input profile* that must be refused,
strict tau/count boundaries, divergent forced prefixes, exact WER limits,
unpooled split failures, parsed-text probe matching, tier selection, signed Q8
block decoding, explicit round/no-round boundaries, digest-mismatched objects,
lazy public
member preparation and aggregate-only private output. They do not mutate
production code or certify hardware.
