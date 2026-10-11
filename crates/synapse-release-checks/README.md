# Tag-job gates

These standard-library Python tools run on release runners without building any
code. `check.py validate` checks the evidence commit, downloads' sidecars,
inventory bindings and separate release gates, then runs the extracted Linux
candidate's `ck-synapse certify validate` over the catalog-derived records. Artifact
paths in records are relative to the extraction root, including the `os_arch`
subdirectory. `smoke.py` checks exact refusal exits; `promote.py` rechecks the
sidecars after artifact transport and publishes the original archive bytes.

Run fixtures with `python3 -m unittest discover -s crates/synapse-release-checks -v`.

The separate evidence formats are JSON objects with the following fields:

- Stress uses the field names defined in
  `crates/synapse-worker-ane-direct/tests/fixtures/ane-direct-stress.schema.json`.
  Every request must complete: `completed_count == request_count`, with empty
  `request_errors` and `no_ane_resources.count == 0`. Both `reranker_pool_check`
  results (`finite` and `byte_identical_repeats`) must be `passed`. Residency
  maxima are `max_resident_per_model` (at most 4) and `max_resident_total` (at
  most 8); leased evictions and shape-not-admitted events must remain zero.
- Benchmark report: `cells` contains exactly the row/model combinations derived
  from the checkout's `crates/synapse-module/src/catalog/models.json`, with `row_id`,
  `model`, `status`, `metrics`, `probe: {exit_code, stdout, stderr}` and a `cause`
  quoted verbatim from the probe for unavailable/unsupported cells. `embedding`
  entries carry `batch_size` and `request_path`; `inline_max_items` is recorded
  on the cell. Non-measured cells have null metrics. Competitor cells may repeat a
  combination. Hugging Face Text Embeddings Inference (TEI) may report hardware
  unavailable on AMD when its probe cannot run there, or measured on CUDA when
  its probe succeeds. These observations do not block release.
- Transition: `session_id`, 23-element `coreml` and `direct` series,
  23-element `machine_load`, `coreml_median`, `direct_median`, `bar_met`.

`RELEASE-NOTES.md` must exist in the candidate's certification directory.
Certification directories keyed by other source commits are
ignored because their binary hashes do not certify this candidate.

The CUDA boundary comes from `cuda_min_driver_api` in
`bench/parity/models.json`. Profiles for models declaring a CUDA backend must
agree, and each such model must have a passed CUDA record at exactly that driver
API, not merely a newer one. Removing a catalog backend removes its cells and
their evidence obligations; a leftover record for an undeclared cell is refused.

Produce each declared cell with `ckdev-synapse-certify run --row <row> --model
<catalog_id> --assets <aggregate-dir> --checkout <root> --mirror <dir>`. The mirror
contains `<hf_repo>/resolve/<revision>/<file>` paths. The runner exercises catalog
download and first-request lane loading, then requires a passing self-check and
the catalog fingerprint throughout the fixture run. Only passed records ship.

The evidence checkout must be the repository root and have a clean working tree.
Its sole parent is the candidate's source commit; the evidence commit may change
only `docs/evidence/`. The candidate CLI validates numerical parity against
reference outputs, admission and ANE placement; these scripts do not infer
hardware success from hosted no-GPU probes. A missing candidate release or any
missing evidence is a hard failure, including docs-only tags.
