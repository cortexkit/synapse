# Parakeet fp32 reference

The speech reference is independent of the token-input embedding/reranking
fixtures. Its contract is the optional `transcribe` block on each Parakeet profile
in `../models.json`; no speech setting belongs to the shared admission block.
`../fixtures/index.json` seals the reference document and each tensor/trace.
The input corpus and attribution are committed here; corpus audio is not.

## Reproducing the reference

Use Linux and Python 3.13. Install the pinned CPU torch wheel and the remaining
requirements in an isolated environment:

```sh
python3 -m venv bench/parity/target/parakeet-env
bench/parity/target/parakeet-env/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
bench/parity/target/parakeet-env/bin/pip install -r bench/parity/reference/requirements-parakeet.txt
bench/parity/target/parakeet-env/bin/python -m unittest discover -s bench/parity/reference -p 'test_parakeet_reference.py' -v
bench/parity/target/parakeet-env/bin/python bench/parity/reference/generate_parakeet.py --cache bench/parity/target/parakeet-cache --output bench/parity/target/parakeet-generated --group corpus --workers 4
```

Run ladder generation with `--group ladder --partition N --partitions 4`, for
each N from 0 through 3, and merge the output directories. Each partition is
independent; the index order is sorted by case id, not process completion order.
`run_ladder.sh N` installs the pinned environment, runs that partition, and
keeps the remote caller connected with a 20-second progress line. Transport
archives normalize names, timestamps, owners and groups. Wall-clock timing is
written beside the archive, never inside it. The job prints every artifact's
name, size and SHA-256 and also writes a separate `.seals.json` inventory. Verify
the returned archive and inventory hashes against the log, then verify every
extracted artifact against the inventory before merging the output directories.
For the remote build service's return-size limit, return each partition's tar separately. Reusing a
host cache is optional; every source and audio member is verified on every read.
Each inference process uses fp32 CPU, one intra-op thread, one inter-op thread,
deterministic algorithms, seed 0 and eager attention. Multiprocessing only
distributes disjoint case lists, never an individual inference computation.

The corpus selection is fixed before inference: 200 sorted utterance ids from
each LibriSpeech test split, and 100 sorted `(id, filename)` pairs from each
FLEURS test language. Four shortest corpus clips supply the resampling fixtures;
the 32 shortest supply committed corpus tensors. Ladder recipes concatenate
four fixed test-clean clips and repeat/trim to the committed exact sample count.
The FLOAT WAV writer's PEAK timestamp is fixed to zero, so container hashes do
not depend on wall time. This changes no encoded samples or feature values.

After all summaries and artifacts are present, run `seal_parakeet.py
--generated <merged-output> --prepare`. Convert the three worker profiles with
the manifest converter and put their packages in the merged output directory,
named `<profile-id>.safetensors`. Alternatively pass `--package-seals` with the
SHA-256/byte-size receipt printed by a remote run of that same converter; its
checkpoint SHA-256 and conversion rule must match. This avoids transporting
gigabyte-sized converted packages merely to pin their hashes.
Run the sealer again without `--prepare`, then
`parity-manifest seal`. The preparation output has no fabricated package pins
and must not be committed. Final sealing refuses missing packages, missing
cases, bad seals, a cap-reaching natural fixture, a weak self-check margin, or
an exceeded fixture/subset byte budget.

## Two explicitly pinned upstream differences

Transformers 5.16.1 TDT skips the recurrent neural network transducer (RNNT)
per-frame symbol counter. Its value 10 sizes
a global output buffer instead. NeMo 2.4.0 applies a per-frame bound but adds +1
even after a positive duration at that bound. Synapse's bounded adaptation uses
K=10 and forces +1 **only** after a zero-advance K-th decision. Both stub-joint
tests drive the same independent loop as real inference. Every natural fixture
must stay strictly below K emissions at a frame and reproduce HF greedy tokens
and durations.

The pinned eager attention path produces NaN for fully masked padded query rows.
`synapse-parakeet-mask-fix-v1` substitutes a finite -10000 sentinel before
softmax and zeroes masked probabilities afterwards, following NeMo. The installed
Transformers package is never edited. The generator temporarily replaces only
`modeling_parakeet.eager_attention_forward`; the profile pins the patch's SHA-256.
The existing mask after the gated linear unit (GLU) and before the depthwise
convolution already matches NeMo and is
unchanged. On unpadded inputs the patched encoder must be bit-identical to the
unpatched encoder. On padded inputs the generator first reproduces upstream NaN,
then requires finite valid encoder frames and independent-loop equality against
patched HF greedy output. The reference document lists every such exception id.

## Artifact mirror and grading

Committed f32 tensors are little-endian and row-major: mel `[mel_frames, 128]`,
encoder `[valid_encoder_frames, 1024]`. Traces use the 40-byte little-endian wire
record layout. `reference.json` records valid counts, triples, words and text;
the self-check seal records both head margins after PCM16 encoding.
The local fixture attributes mark WAV, tensor and trace files as binary; their
seals must not be changed by Git's text newline conversion on any platform.

Large ladder outputs have no public URL. Their seal contains `sha256`, `bytes`
and `cache_path`, relative to a caller-configured private mirror, for example:

```
<mirror>/parakeet-tdt-0.6b-v3/<sha256>.f32le
```

Generation writes these below `external/`. Sealing copies them to the ignored
`../.cache-out/` handoff directory. Copy that directory's contents to the durable
mirror on each certification host; verify size and SHA-256 against the index.
`speech::Artifact::read` takes the mirror root explicitly and refuses absent,
corrupt or traversing paths. Input audio continues to use immutable source URLs
and SHA-256, fetched by `prepare_parakeet.py` into a separate local cache.

`speech::load_transcribe_fixtures` verifies the index's reference seal and model
binding. `speech::evaluate_transcribe` grades every fixture whose input duration
and resampled valid encoder frame count fit the profile's certified maximum.
It requires exact pre-engine refusals for above-limit ladder cases and a minimum
per-frame encoder cosine of 0.999 on every stored encoder reference. Refusal
codes are checked against the pinned media-contract code supplied by
the certification caller; every in-limit encoder observation must be present,
correctly shaped and finite, even when no encoder tensor is stored. Trace
decisions are compared against the reference in order. At the first differing
frame index or head argmax, an argmax difference is exempt only if the frame
index agrees and each differing reference head's top-1 minus top-2 logit margin
is below `tau_margin`. A difference only in frame index or trace length fails.
The engine's triples must always equal the token, frame and duration of every
non-blank step in its whole trace, even after an exemption. The evaluator also
computes word-weighted per-split WER using `synapse-wer-norm-v1`: NFKC,
Unicode case folding, deletion of Unicode punctuation-category characters,
whitespace collapsing, and whitespace splitting.
Profiles without any of the three positive finite tolerances are refused.
The 600000-ms envelope is the reference target, not hardware certification;
each backend must certify its own declared maximum before it is runnable.
