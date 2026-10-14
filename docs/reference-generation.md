# CPU fp32 reference generation

The reference generator is `bench/parity/reference/generate_reference.py`.
It uses Transformers 5.16.1, Torch 2.14.0, seed 0, CPU fp32, eager attention,
single-threaded reductions and deterministic algorithms. Thread settings are
established before importing the numerical libraries and recorded in fixture
metadata. Model revisions and file digests come from `bench/parity/models.json`;
each declared SHA-256 is checked before loading.

## Stage two independent generations

Use a cache containing the pinned checkpoint files and run from the repository
root. Offline settings prohibit model downloads, not installation of the pinned
Python tools themselves. Write into fresh staging directories rather than
replacing committed references:

```sh
: "${HF_CACHE:?set the pinned Hugging Face cache directory}"
first=$(mktemp -d)
second=$(mktemp -d)
for output in "$first" "$second"; do
  for model in gte-modernbert-base gte-reranker-modernbert-base \
    qwen3-embedding-0.6b qwen3-reranker-0.6b; do
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      uv run --no-project --python 3.12 \
      --with transformers==5.16.1 --with torch==2.14.0 \
      python bench/parity/reference/generate_reference.py \
      --hf-cache "$HF_CACHE" --model "$model" --output-dir "$output"
  done
done
diff -r "$first/fixtures" "$second/fixtures"
```

Verify complete fixture files and `fixtures/index.json`, not only aggregate
cosine values. Review input IDs, case coverage, readout and ranking order as well
as output values. Never derive a candidate's expected values with the same
unproved implementation being checked.

## Stage catalog references

```sh
catalog_output=$(mktemp -d)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-project --python 3.12 \
  --with transformers==5.16.1 --with torch==2.14.0 \
  python bench/parity/reference/generate_reference.py \
  --hf-cache "$HF_CACHE" --catalog --output-dir "$catalog_output"
```

Catalog mode preserves committed input texts, query/candidate order and unrelated
metadata, writing the same relative catalog path beneath the staging root. It
does not add a self-check to an entry that has none. Qwen3 catalog inputs include
the required terminal EOS; parity cases keep their existing token IDs. GTE rerank
scores use the double-precision sigmoid of the fp32 classifier logit.

## Verify derived references before publication

Replacing full fixtures requires regenerating their digest index, the sealed
certification self-check subsets, and direct-ANE padding provenance with their
existing generators. Verify that the subsets retain exact source cases and seals:

```sh
cargo test --locked -p synapse-certify --lib
cargo test --locked --manifest-path bench/parity/Cargo.toml
```

`self_check::tests::subsets_are_exact_copies_of_the_sealed_source_cases` and
`self_check::tests::tampered_subset_digest_refuses_and_names_profile` guard the
copy and seal boundaries. A reference refresh is not permission to widen
numerical tolerances or omit expensive certification cases.

## Reverse a staged run

Staging does not modify tracked fixtures. Once comparison is complete, remove
only the temporary directories created above:

```sh
: "${first:?}" "${second:?}"
rm -rf "$first" "$second"
if [ -n "${catalog_output:-}" ]; then
  rm -rf "$catalog_output"
fi
unset first second catalog_output
```

Omit `--output-dir` only after reviewing and authorizing replacement of the
committed references and all derived seals.
