# Qwen3-ASR public CPU reference

This reference is not a serving backend. `models.json` pins the original
`Qwen/Qwen3-ASR-1.7B` files at revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`; engine profiles are added separately.
The official `qwen-asr==0.0.6` Transformers implementation runs in fp32 on CPU,
with eager attention, deterministic algorithms, one intra-op thread and one
inter-op thread. MKLDNN is disabled. `qwen3-asr-reference.json` records the
numerical and front-end rules. The Python environment is separate from the
Parakeet reference's environment because their Transformers versions differ.

## Checkpoint identities

The model's `files` map includes both original shards, their HF weight map,
configs, chat template, preprocessor config, and all three tokenizer components
(the repository has no `tokenizer.json`). Each SHA-256 identifies original bytes.
The aggregate checkpoint digest hashes the canonical JSON map of shard file
names to SHA-256 strings. The aggregate tokenizer digest similarly hashes the
map for `merges.txt`, `tokenizer_config.json`, and `vocab.json`. Canonical JSON
uses lexicographically sorted keys, UTF-8 and no insignificant whitespace.

`checkpoints/qwen3-asr-1.7b/tensor-index.json` is the canonical merge of both
safetensors headers, including their identical metadata. Duplicate tensor names
are refused. Offsets stay shard-local; `model.safetensors.index.json` maps tensor
names to shards. The independent architecture table checks all 708 tensor
names/shapes, including the encoder's 2,048-wide final projection. Downloaded
checkpoint checking also verifies the weight map agrees with the actual shards.
Adding this model changes no existing profile or grammar digest.

## Public corpus

`qwen3-asr-corpus.json` is the immutable public plan. Its recording and term-array
tables intern repeated values; `qwen3_asr_corpus.expand` reconstructs complete
per-fixture rows without fetching anything. Every input source is a pinned public
HF object; every recording has an original member checksum or a deterministic
recipe. No caller audio, private name list, phrase text or private transcript is
an input to these tools.

The plan contains 6,560 fixtures:

- The same 1,200 Parakeet corpus members: 200 each of LibriSpeech test-clean/other,
  and 100 each of the eight FLEURS test splits.
- The pinned 200 Earnings-22 rows with `is_flagged == false`, sampled with seed
  `20261119`. Excluding dataset-flagged rows matches the existing public STT-bias
  subset, so the two benchmarks use the same recordings:
  no list, whole-call global list, and three tier arms with matched baselines.
  The tiers are 30, 100 and 300 terms. A global list is cut in its original order
  or padded from other calls' global terms. Padding excludes any alias whose
  scorer-normalized whole-token form occurs in the reference. Both the tier arm
  and its baseline share the extended global/local/padding scoring vocabulary.
- Three unrelated public Earnings-name lists per tier, each applied to all 400
  LibriSpeech controls: 1,200 recording/list pairs per tier. Terms occurring in
  any control reference are excluded, so all three lists are unrelated to both
  splits. The unlisted baselines are the original 400 corpus fixtures.
- Fifty near-silence and fifty noise fixtures (seeded PCG64 Gaussian samples,
  amplitudes `1e-6` and `0.05`), each with a public 30-term list.
- Fifty single-name test-clean probes. `qwen3-asr-name-curation.json` records the
  complete population, seed, all 209 reviewed IDs/references, 50 accepted names
  with half-open whitespace word spans, and every rejection (`none`, `multiple`
  or `ambiguous`). Review follows the shuffled ID order and stops at the 50th
  acceptance. A name is one person, place or organisation occurrence; titles
  without names and common nouns do not count. No tagging model is used. Each
  accepted name is listed with 29 Earnings distractors absent from its reference.
- Whole LibriSpeech utterances in chapter/utterance order concatenated to 2, 5
  and 9.5 minutes, both unlisted and with a 300-term list. The last complete
  utterance is followed by silence to the target duration; speech is never cut
  or repeated. References concatenate exactly the retained utterance references.
- A 600.01-second over-limit recipe, plus 0.1-second, 137-frame and 31.37-second
  front-end fixtures. The over-limit row records a refusal, not an engine run.

List sampling uses `Random(int.from_bytes(SHA256("20261119:" + label)[:8],
"big"))` over sorted candidate strings, with labels identifying the row/tier or
control-list number. This avoids dependence on process/hash-table iteration order.
Name scoring follows `bench/lanes/stt-bias/score.py`: lowercase, replace hyphens
with spaces, replace non-ASCII-letter/digit/space characters with spaces, collapse
whitespace, and expand the numeric tokens `1`, `4`, `28`, `20` to `one`, `four`,
`twenty eight`, `twenty`. A `|` separates alternative aliases; any alias matching
a whole-token span counts as occurring. This is not `synapse-wer-norm-v1`;
parity WER continues to use its existing normalizer.

## Reproduction and pilot

Run these commands on a public-data CPU build machine from the repository root:

```sh
python3 -m venv target/qwen-reference-venv
target/qwen-reference-venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
target/qwen-reference-venv/bin/python -m pip install -r bench/parity/reference/requirements-qwen3-asr.txt
target/qwen-reference-venv/bin/python bench/parity/reference/prepare_qwen3_asr.py --cache target/qwen-reference-cache --output target/qwen-reference-cache/corpus.json --annotations bench/parity/reference/qwen3-asr-name-curation.json
target/qwen-reference-venv/bin/python bench/parity/reference/generate_qwen3_asr.py --cache target/qwen-reference-cache --corpus bench/parity/reference/qwen3-asr-corpus.json --output target/qwen-reference-generated --pilot
```

Preparation must reproduce the committed corpus byte-for-byte before inference.
The pilot selects 20 representative cases, including every speech split, all
three front-end edges, all long-form durations, a 300-term Earnings row, and one
of each probe. Each fixture runs in a fresh process. Its sidecar records total
wall time, model-load time, peak process RSS (Linux), committed trace bytes and
external tensor bytes. Two cases run twice; their complete deterministic trace
hashes (which bind tensor hashes) must match. Timing/RSS sidecars are not fixtures.

For a full run, use `full_run.sh <N> <out>` on a single x86 Linux machine, as
described in [RUN.md](RUN.md). Each of its N single-threaded processes loads the
model once and runs its shard in sealed order. Three fixtures must match in two
processes after different preceding fixtures before the full shards start.
CPU model, instruction flags, installed packages and all generation inputs are
recorded. Never combine venues, pilot traces or independent partial runs.

Each continuation records `stop_reason` (`stop` or `cap`), the exact `cap` used,
and one top-two logit margin per step. An EOS-terminated continuation includes
the stop token, even when it occurs at the last allowed position. A capped
continuation preserves exactly the generated prefix without inventing EOS.
Prompt IDs and the exact contiguous audio-placeholder span are recorded. Caller
text is encoded with special-token parsing disabled. Parsed text/language use
the pinned official parser, decoding only the continuation. Mel tensors are
`[mel_frames,128]` and encoder tensors are `[audio_tokens,2048]`, both row-major
little-endian fp32. They remain external, content-addressed by SHA-256 under
`external/qwen3-asr-1.7b/`; only their checksums/sizes and small traces enter git.

After all partitions are gathered, seal with:

```sh
python3 bench/parity/reference/seal_qwen3_asr.py --corpus bench/parity/reference/qwen3-asr-corpus.json --generated "$RETURNED_FULL_RUN" --external-cache "$HOME/Work/Projects/CortexKit-scratch/parity-cache"
```

The sealer refuses incomplete coverage, missing/corrupt tensors, lineage changes,
an invalid termination reason, a generated-ID count different from the per-step
top-two margin count, or insufficient headroom on an EOS-terminated trace. The
full run generates with fixed reference slope 25. A non-EOS continuation must
fill that exact cap and is recorded rather than treated as a harness failure.
It captures the actual Transformers logits
processor list and refuses any nonempty list. The pinned and effective generation
configs therefore attest that a larger stopping budget did not alter logits.

The measured slope is `max(6, ceil(1.5 * highest_speech_tokens_per_second))`, measured
over every no-list EOS-terminated speech fixture of at least two seconds: LibriSpeech (including the
concatenated chapters), all FLEURS splits, and Earnings-22. The generated language
prefix through `<asr_text>` is subtracted; the stop token remains. Probes and
front-end clips do not set that slope. No-list means empty request `terms` and
recorded `context_tokens == 0`, not any particular tier label. Independently,
**every no-list EOS-terminated** fixture, including short clips and probes, must
have a complete sealed cap at least
`ceil(1.5 * generated_length)`. A required slope above 25 or any headroom failure
stops the run before sealing. Generation and serving slope stay 25; a measured
slope below 25 is recorded but does not lower any cap. Both name-list requests
and capped prefixes are exempt from speech-rate
calibration and complete-cap headroom: their complete rate is unmeasured and
their exact reference cap remains their identity regardless of a later serving
cap choice. Name-list recitations, stopped or capped, remain genuine reference
outputs; their highest stopped speech-fixture rate is reported separately.
Engine parity compares each capped trace's exact recorded generated-ID prefix.
`capped-fixtures.json` records all cap-hit counts and IDs, by split, by tier and
by split/tier. Tier `none` means no corpus tier label, not necessarily no list.
Stopped greedy traces retain both their observed
generation cap and derived sealed cap; they are not regenerated merely to lower
an unused budget. The maximum-rate fixture, ten-minute cap and key/value-cache sizing with
audio/template/bias/cap positions are recorded. The final slope comes from the
full corpus, never the pilot. The explicit offline-regrade mode for the completed
76cc6a4b run preserves its failed calibration evidence and records separate
generation/grading identities; see [RUN.md](RUN.md). No engine work may rely on
an unsealed reference.

The original BF16 head and embedding byte ranges are identical, and the official
loader shares their fp32 storage. `qwen3-asr-reference.json` records both original
range hashes and the loaded fp32 hashes. The full venue rechecks these facts;
engines must use the same logits matrix, not infer a different head from the
presence of two tensor names.

Offline checks (no model download or inference):

```sh
python3 -m unittest discover -s bench/parity/reference -p 'test_qwen3_asr*.py' -v
cargo test --locked --manifest-path bench/parity/Cargo.toml --test qwen_asr
```

## Sealed public reference

`fixtures/qwen3-asr-1.7b/reference.json` registers all 6,560 venue-generated
cases. Its separate offline-grading receipt binds the original generation at
`76cc6a4b` to grading at `21ffed30`. The venue generated every case, but its old
final calibration counted name-list recitation as speech and exceeded the
slope-25 guard. The offline receipt retains that failure and grades no-list
speech separately; it does not pretend the venue emitted the missing successful
calibration, completion record or `SHA256SUMS`. The numerical traces and the
3,022 unique external tensor receipts are unchanged by that regrade.

The slope is the tokens-per-second coefficient in
`64 + ceil(slope * duration_s)`. The measured no-list requirement is **24**,
the ceiling of 1.5 times the maximum observed speech rate, set by Greek fixture
`fleurs-el-test-1102888255749710863` at 15.8986 tokens/s after language-prefix
subtraction. The fixed generation/serving slope remains **25**. Headroom passes
for 2,006 no-list stopped fixtures. The highest listed-speech rate, 22.4667
tokens/s at `earnings22-0024-tier-100`, remains visible as name-list recitation,
not a claim about speech throughput.

Exactly two traces are capped: `earnings22-0106-tier-300` and
`earnings22-0290-tier-300`; both remain exact bounded teacher prefixes. The
ten-minute cap is **15,064** tokens. With 7,800 audio positions and the observed
15-token template, key/value-cache sizing is **22,879 + bias tokens** positions,
or **24,927** at the 2,048-token upper-bound bias scenario. Bias tokens are the
tokenized name-list context added to the system prompt; this scenario reserves
capacity for 2,048 such tokens. This sizing neither selects a serving name-list
limit nor certifies hallucination safety.
