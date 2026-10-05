# Tag-job gates

These standard-library Python tools run on release runners without building any
code. `check.py validate` checks the evidence commit, downloads' sidecars,
inventory bindings and separate release gates, then runs the extracted Linux
candidate's `ck-synapse certify validate` over the canonical 32 records. Artifact
paths in records are relative to the extraction root, including the `os_arch`
subdirectory. `smoke.py` checks exact refusal exits; `promote.py` rechecks the
sidecars after artifact transport and publishes the original archive bytes.

Run fixtures with `python3 -m unittest discover -s crates/synapse-release-checks -v`.

The separate evidence formats are JSON objects with the following fields:

- Stress: `request_count`, `sample_count`, `max_resident_per_model`,
  `max_resident_overall`, `shape_not_admitted_count`, `leased_evict_count`.
- Benchmark report: `cells` contains every row/model combination, with `row_id`,
  `model`, `status`, `metrics`, `probe: {exit_code, stdout, stderr}` and a `cause`
  quoted verbatim from the probe for unavailable/unsupported cells. `embedding`
  entries carry `batch_size` and `request_path`; `inline_max_items` is recorded
  on the cell. A dropped cell has null metrics and `drop_cause`, plus
  `raw_series: {ane, metal}` for a latency drop. Competitor cells may repeat a
  combination. Hugging Face Text Embeddings Inference (TEI) may report hardware
  unavailable on AMD when its probe cannot run there, or measured on CUDA when
  its probe succeeds. These observations do not block release.
- Transition: `session_id`, 23-element `coreml` and `direct` series,
  23-element `machine_load`, `coreml_median`, `direct_median`, `bar_met`.

Metal notes must name the model and both the catalog record fingerprint and each
committed preload fingerprint. Models without a committed preload have no pair
to enumerate. Certification directories keyed by other source commits are
ignored because their binary hashes do not certify this candidate.

The candidate CLI validates numerical parity against reference outputs and the
permitted Qwen3 ANE drop decisions; these scripts do not infer
hardware success from hosted no-GPU probes. A missing candidate release or any
missing evidence is a hard failure, including docs-only tags.
