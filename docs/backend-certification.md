# Backend certification

Certification binds exact candidate artifacts, model/profile identity, hardware
row, fixture corpus and source commit. A worker build or hosted protocol smoke is
not a numerical hardware certification. Source records and offline validation
live in `synapse-certify`; the non-shipped `ckdev-synapse-certify` runner produces
hardware evidence using an isolated daemon and the extracted candidate assets.

## 1. Prepare a matching candidate and runner

Obtain the immutable release-candidate assets and extract their platform archives
under one aggregate root, using their published digests. Work from the exact
candidate source commit in a clean checkout; do not rebuild the extracted assets.
Build only the runner:

```sh
cargo build --release --locked -p synapse-certify-runner
```

Keep assets and scratch state on one filesystem: execution uses development-named
hard links, and a cross-filesystem copy would no longer execute the attested file.
The candidate's `certify source` stamp must match the runner's clean source.
Dirty or missing provenance refuses validation.

Checks: `validate_dispatch_uses_supplied_clean_candidate_source`,
`validate_dispatch_without_clean_source_refuses_and_writes_nothing`, and
`executed_digest_is_extracted_file_not_zip_sidecar` in
`crates/synapse-certify/tests/records.rs`. These check source binding and execution
of the exact extracted bytes, rather than a rebuilt or copied candidate.

## 2. Produce a hardware record

Set `ROW` to a hardware row declared by `ROW_BACKENDS` in `synapse-certify`
(for example `metal-m5` or `vulkan-windows-amd`), and `MODEL` to the catalog ID
reported by `models.catalog`. `ASSETS` is the aggregate directory containing
extracted platform archives. `MODEL_MIRROR` is a local directory containing the
checkpoint files at the catalog's pinned revisions and digests, laid out as
`<hf_repo>/resolve/<revision>/<file>`. Supply these inputs explicitly:

```sh
: "${ROW:?set the declared hardware row}"
: "${MODEL:?set the catalog model ID}"
: "${ASSETS:?set the aggregate extracted candidate root}"
: "${MODEL_MIRROR:?set the directory containing pinned model files}"
target/release/ckdev-synapse-certify run \
  --row "$ROW" --model "$MODEL" --assets "$ASSETS" \
  --checkout "$PWD" --mirror "$MODEL_MIRROR"
```

Use the row's supported hardware and required native toolchain. For ANE, obtain an
exclusive slot and unset a custom `TMPDIR`; never bypass another process's lock.
The mirror layout is `<hf_repo>/resolve/<revision>/<file>`. The runner downloads
through the candidate's catalog surface, waits for committed installation, and
requires its self-check and fingerprint before grading the original fixtures.
Failed parity, admission, hardware or placement gates must not write a passing
record. Non-Apple collection uses `SYNAPSE_CERTIFY_INSTANCE_ID` as private run
metadata. Preserve an unredacted original privately and remove `machine.instance_id`
from the public record before publication. Machine UUIDs remain domain-separated
SHA-256 digests; local paths and operator identifiers are not public evidence.

For ANE, every admitted transformer-layer executable must appear in its placement
inventory, and inventory count must agree with independently observed admissions.
An omitted inventory cannot be treated as complete placement evidence.
Checks: `producer_refuses_each_failed_gate_without_writing`,
`serialized_inventory_omission_fails_placement_gate`, and
`records_with_a_raw_hardware_uuid_or_no_digest_are_refused`.

## 3. Validate the canonical evidence tree

Records belong at `docs/evidence/certification/<source>/<row>/<model>.json`.
Validation enumerates all declared cells and rejects missing, extra, mismatched
or failed records. It hashes extracted executable files, verifies sealed fixture
bytes and admission/placement claims, and does not trust a checkout's modified
fixture index. Schema-2 records remain historical and cannot certify a schema-3
candidate.

Use a same-filesystem development link to the candidate executable:

```sh
: "${OS_ARCH:?set the platform directory under ASSETS}"
scratch=$(mktemp -d "$ASSETS/.validate.XXXXXX")
ln "$ASSETS/$OS_ARCH/ck-synapse" "$scratch/ckdev-synapse"
"$scratch/ckdev-synapse" certify source
"$scratch/ckdev-synapse" certify validate --assets "$ASSETS" "$PWD"
rm "$scratch/ckdev-synapse"
rmdir "$scratch"
```

Checks: `canonical_checkout_round_trip_and_unknown_fields_rejected`,
`shipped_validate_entry_point_accepts_a_complete_synthetic_matrix`,
`validator_matrix_refuses_missing_declared_cell`,
`validator_matrix_refuses_undeclared_record_file`, and `validator_refuses_schema_two`.

## 4. Verify release evidence and reverse safely

Portable validator fixtures:

```sh
cargo test --locked -p synapse-certify --lib --test records
python3 -m unittest discover -s crates/synapse-release-checks -v
python3 scripts/check-release-candidate.py --self-test
python3 -m unittest discover -s bench/harness -p test_release_matrix.py -v
```

The tag gate also checks benchmark/stress/transition evidence, release notes,
artifact inventory and the evidence commit's relationship to its candidate.
`test_source_parent_and_prior_evidence`, `test_raw_hardware_uuid_refused`, and
`test_stress_transition_and_release_notes_required` prove these release refusals.
Synthetic fixtures prove validator decisions, not a real device's parity.

To reverse an unpublished evidence attempt, remove only that attempt's records,
links and scratch state. A changed backend or artifact requires a new source and
complete derived matrix; never relabel old evidence or lower a tolerance to make
a release pass. Keep unredacted operational run records privately.
