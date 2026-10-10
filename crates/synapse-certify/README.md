# Certification records

This workspace crate implements schema-3 records, producer gates, and offline
validation. `ck-synapse certify validate --assets <dir> <checkout-root>` validates
the catalog-derived cells under
`docs/evidence/certification/<candidate-source-commit>/`. The candidate source
commit is supplied from the module's existing build stamp only when its source
tree was clean. A dirty build or a build without Git provenance refuses
validation because its evidence cannot be bound to a commit. An evidence-only
commit does not change which records an already-built candidate validates.
Artifact paths are `<os_arch>/<file>` under the aggregate extraction root, not
ZIP paths; the validator hashes the extracted files themselves. Only `passed`
records are accepted and appear in the returned `eligible` list. Every declared
backend requires a record on each hardware row mapped to that backend. Extra
records for undeclared cells are refused as well as missing records.

Build the separate, non-shipped runner from the candidate's clean source commit:
`cargo build --locked --release -p synapse-certify-runner`. Do not rebuild the
extracted candidate assets. `ck-synapse certify source` prints its embedded commit
only for a clean build; dirty or missing stamps refuse. Before producing evidence,
the runner probes that candidate and requires the same clean source commit as its
own build. It logs its own executable path, commit and SHA-256 to stderr; the
record describes only the candidate assets.

`ckdev-synapse-certify run --row <row-id> --model <catalog-id> --assets <aggregate-root>
--checkout <root> --mirror <dir>` executes the extracted candidate's `ck-synapse`
and row worker, never a Cargo-built fallback. Assets are extracted beneath
`<aggregate-root>/<os_arch>/`. The mirror holds the pinned files at
`<hf_repo>/resolve/<revision>/<file>` and is served locally through `hf_endpoint`.
Candidate and worker execution use hard links under `ckdev-*` names, never copies;
keep the assets and scratch checkout on the same filesystem. Their release bytes
remain the artifacts hashed in the record. Only the runner links the private
in-process daemon; the shipped module retains offline validation and self-checks.
The runner starts the daemon with isolated config, leases and store.
The runner checks the cell and its pin through `models.catalog`, downloads it
through `models.download`, and polls the job to `committed`. Requests use the
catalog lane id `<catalog-id>-<backend>`. The first fixture request loads and
self-checks the lane; `model_loading` replies are retried until success or the
runner deadline. Grading begins only after successful loading and a
`models.list` check of the lane's self-check and fingerprint. Every successful
response must carry the same catalog pin. The existing `certify_observation`
flag enables admission and placement observation during the run.

The runner sends the committed fixture source text through module tokenization,
using batch-one requests (also consecutive single-candidate chunks for ranking
pools), and grades observed outputs with the existing parity evaluator. It
collects engine-bound IDs, readout IDs, all ADMITTED inventories, an independent
supervisor admission count and actual worker request counts. Inventory omissions
fail the producer placement gate. Metal never invokes a floor probe; other rows
execute their candidate worker's probe. A floor, parity, admission or placement
gate failure prevents writing a certification record. Passing records are
written atomically at their canonical paths. A lane that cannot pass must
be removed from the catalog in a new source commit and the new candidate's
entire derived matrix must be rerun.

Non-Apple machine identification uses `nvidia-smi` for CUDA or the selected
adapter in `vulkaninfo --summary` for Vulkan. Rented-machine operators must set
`SYNAPSE_CERTIFY_INSTANCE_ID`. Missing identifiers refuse rather than invent
identity. Hardware UUIDs never enter a record: Apple records carry
`machine.platform_uuid_sha256`, the lowercase hex SHA-256 of
`synapse-certify/machine/v1` plus one NUL byte followed by the
`system_profiler` platform UUID, and CUDA and Vulkan records carry
`machine.gpu.uuid_sha256`, the same over `synapse-certify/gpu/v1` plus NUL and
the GPU UUID. The same machine always produces the same digest, so its records
still link. The validator refuses a record with a raw `machine.platform_uuid`
or `machine.gpu.uuid`, or without the digest for its platform.
Generated run state and hardware-test output live in ignored `.live/`.
The ignored Mac hardware test accepts `SYNAPSE_CERTIFY_CANDIDATE`, the release
candidate `ck-synapse` path, and prints the canonical record verbatim.

Records carry only the specified schema-3 fields. Schema-2 records are refused.
Parity is the evaluator's JSON output, with
its identity, fixture set, metrics and boolean gate results. Admission outcomes
are `tokens_8192` and `tokens_8193`, each holding `outcome`, `truncated`,
`diverted`, and the number of `worker_requests`. Artifact roles use the binary
names (`ck-synapse`, `ck-synapse-worker-cuda`, etc.). ANE records require a
`placement` block containing `layer_count`, independent `admitted_count`, and
`inventories`. Offline validation requires the embedded architecture's layer
count, at least one admission, exactly one inventory per admission, and passing
inventory placement for every transformer layer.

The validator rechecks all parity gates and admission outcomes. It binds fixture
identity and both completion counts to fixture bytes verified with the embedded
index digest, fingerprint and operation to the compiled catalog, and profile
and tolerance class to the embedded manifest. It reads neither a checkout
manifest nor a checkout fixture index and runs no Git commands. Changing a
fixture corpus and checkout index together cannot change the embedded digest.

Run `cargo test -p synapse-certify -p synapse-certify-runner` for the synthetic contract fixtures. These
prove decisions and validation, not hardware parity. Run the ignored hardware
test explicitly on the named machine to certify a real candidate.
