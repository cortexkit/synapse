# Certification records

This normal workspace crate implements the schema-2 record, producer decisions,
and the offline validator for matrix coverage, status, drop evidence, artifact
digests and parity identity. `ck-synapse certify validate --assets <dir>
<checkout-root>` validates the 32 canonical records under
`docs/evidence/certification/<candidate-source-commit>/`. The candidate source
commit is supplied from the module's existing build stamp only when its source
tree was clean. A dirty build or a build without Git provenance refuses
validation because its evidence cannot be bound to a commit. An evidence-only
commit does not change which records an already-built candidate validates.
Artifact paths are relative to the extracted candidate asset directory, not ZIP paths; the
validator hashes the files themselves. Only passed combinations appear in the
returned `eligible` list.

`ck-synapse certify run --row <row-id> --model <slug> --assets <extracted-dir>
--checkout <root> --weights <pinned-model-dir>` executes the named candidate
asset directory's `ck-synapse` and row worker, never a Cargo-built fallback.
It starts a private in-process daemon with isolated config, leases and store.
The generated config selects one `<slug>.<lane>` preload profile from
`bench/parity/models.json`, pinning the model's numeric execution settings for
the requested hardware lane, and enables the
certification-only observation surface. Normal consumer configurations leave it
inactive. The weights directory contains the original pinned checkpoint files;
the runner checks them and converts worker profiles with the existing converter.

The runner sends the committed fixture source text through module tokenization,
using batch-one requests (also consecutive single-candidate chunks for ranking
pools), and grades observed outputs with the existing parity evaluator. It
collects engine-bound IDs, readout IDs, all ADMITTED inventories, an independent
supervisor admission count and actual worker request counts. Inventory omissions
fail the producer placement gate. Metal never invokes a floor probe; other rows
execute their candidate worker's probe. Qwen ANE latency series use three
warmups and twenty measured 512-token batch-one calls in the same session as the
Metal reference series. A gate failure writes nothing except for the authorized
Qwen ANE drop cases. Records are written atomically at their canonical paths.

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

Records carry only the specified schema-2 fields. Parity is the evaluator's JSON output, with
its identity, fixture set, metrics and boolean gate results. Admission outcomes
are `tokens_8192` and `tokens_8193`, each holding `outcome`, `truncated`,
`diverted`, and the number of `worker_requests`. Artifact roles use the binary
names (`ck-synapse`, `ck-synapse-worker-cuda`, etc.). Optional `raw_series`
holds a shared `session_id` and `ane`/`metal` latency arrays, each containing
three warmups followed by twenty measured samples. A parity-only drop can lack
latency samples when execution failed before measurement; a latency drop must
have both series. The validator disregards the stated drop cause and accepts a Qwen3 ANE drop only for a failing parity gate or a
recomputed ratio strictly greater than three. Producer placement failures never
justify a drop.

Run `cargo test -p synapse-certify` for the synthetic contract fixtures. These
prove decisions and validation, not hardware parity. Run the ignored hardware
test explicitly on the named machine to certify a real candidate.
