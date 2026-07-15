# Native constrained decoding in the owned decode loop

## Status

The shared decode controller in `src/qwen3_decode.rs` now supports an optional
`DecodeConstraint` between fp32 logit readback and greedy token selection. The
same controller drives Qwen3 Metal decode and LFM2 CPU/Metal decode, so both
families use the same masking and commit order:

1. read the next-token logits;
2. ask the constraint for a token mask;
3. select top-k from allowed ids only (equivalent to setting every other logit
   to negative infinity);
4. expose the selected token to `TokenStreamTap::before_commit`;
5. advance the constraint;
6. commit the token to the sequence and model cache.

The unconstrained branch still calls the original `top_logits` implementation.
It does not build a vocabulary index or execute constraint code.

This change is spike-first. `crates/synapse-engine-owned` currently contains
encoder/reranker inference but no causal decode controller, KV generation loop,
or CPU-visible pre-commit logits surface to port. The production copy therefore
has not diverged from this decode path; a later production generation API can
move the controller and constraint module together.

## CLI

`--decode-json` attaches the unrestricted JSON recognizer. Supplying
`--decode-json-schema PATH` attaches the same recognizer with the documented
schema subset and implies JSON mode. Each constrained result includes decoded
`text`; the runner reparses it with `serde_json` and revalidates the schema
before reporting success.

Example:

```sh
MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
target/debug/spike-unified-rt \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --generate-prompts bench/spikes/unified-rt/constrained-decode-prompts.jsonl \
  --decode-json-schema bench/spikes/unified-rt/constrained-decode-schema.json \
  --max-new-tokens 128 --decode-cache-bucket 512 --decode-top-k 5 \
  --device metal --dtype f16 --execution explicit \
  --package-cache target/qwen3-decode-packages \
  --out target/qwen3-constrained-15.json
```

Use the same prompt/schema files with an LFM2 model and its existing decode
flags for the second-family gate.

## Constraint interface

The backend-facing interface is intentionally grammar-neutral:

```rust
trait DecodeConstraint {
    fn allowed(&mut self) -> Result<Arc<TokenMask>>;
    fn advance(&mut self, token_id: u32) -> Result<()>;
    fn is_complete(&self) -> bool;
    fn describe(&self) -> String;
}
```

`TokenMask` is a dense token-id bitset. It can either be iterated by the current
greedy/top-k path or applied to a mutable logit vector for a future temperature,
top-p, or seeded sampler. `describe` supplies state and prefix diagnostics when
a generation budget is exhausted; the decode controller otherwise knows
nothing about JSON.

`DecodeSession::generate_constrained` is separate from `generate`, preserving a
minimal and token-identical unconstrained branch. Pause/resume works by retaining
the same constraint beside the paused session. LFM2's full-reprefill cache check
also has a constrained variant, so `--verify-decode-cache` can compare the
cached and full-reprefill paths under the same mask decisions.

## Token bytes and vocabulary indexing

Constraints run on emitted bytes, never vocabulary display strings or
individually decoded Unicode strings.

At tokenizer load, `TokenVocabulary`:

- verifies that the tokenizer has a ByteLevel decoder (directly or as the sole
  decoder in a sequence), which covers the pinned Qwen3 and LFM2 tokenizers;
- enumerates the actual vocabulary, including sparse/added ids;
- excludes tokens marked special from ordinary byte transitions;
- inverts the GPT-2/Hugging Face ByteLevel byte-to-Unicode alphabet;
- follows ByteLevel's all-or-nothing rule: if a token contains a character
  outside that alphabet, the token's UTF-8 bytes are used as a literal added
  token;
- stores `token id -> Vec<u8>` and inserts every nonempty piece into a byte trie.

This preserves tokens containing only part of a UTF-8 scalar. For example, a
token ending in `E2 82` remains an incomplete but viable prefix, while a token
that follows it with a non-continuation byte is masked. Unit tests cover the
split-multibyte case and verify all 256 inverse ByteLevel mappings.

A tokenizer with Metaspace, WordPiece cleanup, Replace, Strip, ByteFallback, or
another context-sensitive decoder is rejected in v1 rather than assigned an
incorrect per-token byte sequence. ByteFallback can be added later by mapping
exact `<0xHH>` pieces before trie construction.

## JSON recognizer and mask automaton

`JsonParser` is a cloneable, hashable incremental byte recognizer. Its state
contains:

- a stack of object/array frames;
- the expected structural phase (value, key, colon, comma/end, or root end);
- literal and number substates;
- string escape, `\uXXXX`, surrogate-pair, and partial-UTF-8 substates;
- schema object properties already seen and the candidate key/enum prefixes.

A token is allowed only when every byte transition in that token succeeds. This
handles tokens that open a string, finish an escaped key, emit a value, and
close an object in one token.

For each parser state, the mask builder walks the vocabulary trie while cloning
only viable parser branches. Shared token prefixes are interpreted once rather
than rescanning every full token piece. The resulting bitset is cached by parser
state. Repeated states inside unconstrained strings and structural phases then
reuse an `Arc<TokenMask>`.

Two progress restrictions prevent a greedy model from spending the entire
budget on optional syntax while preserving valid output:

- pure-whitespace tokens are suppressed between JSON symbols (whitespace bytes
  inside a string, or tokens combining whitespace with a symbol, remain valid);
- number literals are limited to 32 bytes, and the last byte must leave a
  terminal number state.

Once the root value is complete, only configured EOS/stop ids are exposed. EOS
is never in a mask for an incomplete root. A generation that reaches its token
budget while incomplete returns an error instead of publishing an invalid JSON
prefix.

## Supported schema subset

The input is JSON Schema-shaped but intentionally smaller than a general JSON
Schema evaluator.

| Feature | v1 behavior |
|---|---|
| Root/value types | `object`, `array`, `string`, `number`, `boolean`, `null`, or `{}` as any JSON value |
| Objects | `properties`, `required`, and key order freedom |
| Additional keys | `additionalProperties: false`, `true`, omitted (standard `true`), or a supported schema |
| Arrays | homogeneous `items` schema is required |
| Enum | nonempty string enums only; may be paired with `type: "string"` |
| Metadata | `$schema`, `title`, and `description` are accepted and ignored |
| Object keys | known-key duplicates are rejected; escaped and UTF-8 key spelling is recognized incrementally |
| Numbers | JSON number syntax, limited to 32 encoded bytes for progress |

Unsupported keywords or combinations are rejected at schema load. This includes
`integer`, numeric/string bounds, tuple arrays, non-string enums, `const`,
`oneOf`/`anyOf`/`allOf`, references, patterns, conditionals, and dependent
schemas. The schema used by the gate is a closed object with required `result`
and `score` keys, a hostile string enum, and a number field.

## Correctness gates

Measurements were taken on 2026-07-15/16 on a contended Apple M5 Max MacBook
Pro (128 GB), using the debug owned-runtime binary and local model snapshots.
The checked-in 15 prompts explicitly ask the model to refuse, emit prose/XML or
Markdown, omit required keys, use the wrong enum/type, add fields, or leave JSON
unterminated. Running them once on each family gives 30 adversarial generations.

| Family and local weights | Prompts | Generated tokens | JSON parsed | Schema valid |
|---|---:|---:|---:|---:|
| Qwen3-0.6B, Metal f16 explicit | 15 | 647 | 15/15 | 15/15 |
| LFM2-350M, Metal f16 lazy | 15 | 678 | 15/15 | 15/15 |
| **Total** | **30** | **1,325** | **30/30** | **30/30** |

The runner performs the parse and schema checks before writing each successful
result; `constraint_valid_prompts` was 15 in both result files.

### Unconstrained token-exact regression

The pinned campaign fixtures were not changed. Running Qwen3 without either
constraint flag against
`bench/campaign/decode-fixtures/{decode-prompts,reference-tokens}.jsonl`
produced:

- exact prompts: **20/20**;
- exact generated tokens: **1,280/1,280**;
- accepted near ties: **0**.

This gate was rerun after the final decode-loop changes.

### Unit and instrumentation hooks

`cargo test -p spike-unified-rt --bin spike-unified-rt` passed **49 tests** with
4 pre-existing ignored tests and no failures. The five decode instrumentation
hooks all passed:

- `token_stream_tap_observes_before_commit_without_changing_tokens`;
- `paused_state_resumes_to_uninterrupted_tokens`;
- `splice_matches_prefilling_the_concatenated_sequence`;
- `greedy_argmax_uses_lowest_token_id_for_exact_ties`;
- `addressable_weight_regions_are_byte_identical_across_loads`.

A new controller test makes an invalid prose token the highest logit and proves
that masking selects `{}` before the pre-commit tap, then gates EOS. JSON/vocab
unit tests cover token boundary straddling, split UTF-8, escapes and surrogate
pairs, schema key/enum prefixes, key order and required fields, number progress,
negative-infinity mask application, and EOS gating. A one-prompt LFM2 run with
both `--decode-json-schema` and `--verify-decode-cache` also passed, proving that
cached and full-reprefill decode choose the same constrained tokens.

## Performance

The same 15 hostile prompts and `max_new_tokens=128` were run with and without
the schema constraint. Constrained generations stop at EOS as soon as the JSON
root is complete, so token counts and cache-position distributions differ; the
tok/s comparison is an end-to-end serving observation, not an isolated claim
that masking accelerates the model.

| Family | Constraint | Generated tokens | Decode wall | Decode tok/s | Measured constraint path |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | none | 1,920 | 33.767 s | 56.86 | — |
| Qwen3-0.6B | JSON schema | 647 | 9.468 s | 68.33 | **0.245 ms/token** |
| LFM2-350M | none | 1,736 | 131.981 s | 13.15 | — |
| LFM2-350M | JSON schema | 678 | 45.913 s | 14.77 | **0.157 ms/token** |

`constraint_wall_s` times mask lookup/construction, masked top-k selection, and
constraint state advance. It is therefore a conservative upper bound on mask
machinery rather than an incremental subtraction from unconstrained sampling.
At Qwen3's **151,669-id** actual tokenizer vocabulary, the measured aggregate is
0.245 ms/token, below the 0.5 ms/token target and well below the 2 ms stop point.
The LFM2 vocabulary contains 64,400 ids. Results are contended development-host
numbers; locked-hardware serving measurements remain future work.

## Adding regex or CFG constraints

The decode loop, mask type, byte vocabulary, trie walker, cache, EOS policy, and
sampler integration are reusable. A new grammar class needs only an incremental
state with three operations: consume one byte, report whether the prefix remains
viable, and report whether the root is complete.

- A regex implementation can compile to a byte DFA, use the DFA state as the
  mask-cache key, and mark accepting states complete.
- A GBNF-like grammar needs a deterministic or set-valued pushdown state (for
  example LR items or compact Earley sets), explicit nullable/EOF handling, and
  a hashable canonical state for mask caching.
- Character-oriented grammars must define UTF-8 decoding across token pieces;
  byte-oriented grammars can consume trie edges directly.
- Ambiguous CFG states need deduplication and bounded-state safeguards so one
  mask request cannot grow without limit.

No decode-controller or backend change is required for those implementations.
