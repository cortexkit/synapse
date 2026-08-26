# gte-modernbert Vulkan Ally baseline

Run this single command from the repository root only after an SSH idle probe shows
no wave tenant on the Ally. The harness repeats the probe before every timed cell,
synchronizes the candidate checkout with a Git bundle, uploads the pinned UTF-8
fixture, builds with Vulkan under `%USERPROFILE%\target`, and writes the gate
verdict plus `median_tok_s` to the result file.

```sh
SYNAPSE_CAMPAIGN_BASELINE_TOK_S=pending \
SYNAPSE_CAMPAIGN_REMOTE_TARGET=<bench-host> \
SYNAPSE_CAMPAIGN_FIXTURES=/path/to/licensed-fixtures \
  bench/campaign/vulkan-embed-harness.sh "$PWD" /bin/sh /tmp/gte-modernbert-vulkan-baseline.json
```

The command is intentionally allowed to run with a pending baseline. On success,
record `median_tok_s`, `fixture`, `parity_passed`, and `determinism_passed` from
`/tmp/gte-modernbert-vulkan-baseline.json` in the
`gte-modernbert-vulkan-embed` registration, then replace `pending` in both
registration command environments with that measured number.

The harness probes the explicitly configured remote target before each timed
cell. `cargo.exe`, `unified-rt.exe`, or `spike-unified-rt.exe` output is a hard
refusal; do not wait for a busy tenant or kill another worker. Supply a
separately licensed fixture directory through `SYNAPSE_CAMPAIGN_FIXTURES`; the
harness verifies its manifest before using it.
