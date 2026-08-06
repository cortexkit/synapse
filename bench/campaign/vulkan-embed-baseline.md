# gte-modernbert Vulkan Ally baseline

Run this single command from the repository root only after an SSH idle probe shows
no wave tenant on the Ally. The harness repeats the probe before every timed cell,
synchronizes the candidate checkout with a Git bundle, uploads the pinned UTF-8
fixture, builds with Vulkan under `%USERPROFILE%\target`, and writes the gate
verdict plus `median_tok_s` to the result file.

```sh
SYNAPSE_CAMPAIGN_BASELINE_TOK_S=pending \
  bench/campaign/vulkan-embed-harness.sh "$PWD" /bin/sh /tmp/gte-modernbert-vulkan-baseline.json
```

The command is intentionally allowed to run with a pending baseline. On success,
record `median_tok_s`, `fixture`, `parity_passed`, and `determinism_passed` from
`/tmp/gte-modernbert-vulkan-baseline.json` in the
`gte-modernbert-vulkan-embed` registration, then replace `pending` in both
registration command environments with that measured number.

The controller's first probe is:

```text
ssh -T -o BatchMode=yes ufuka@[lan-ip] "cmd /c tasklist /FI \"IMAGENAME eq cargo.exe\" /FO CSV /NH & tasklist /FI \"IMAGENAME eq unified-rt.exe\" /FO CSV /NH & tasklist /FI \"IMAGENAME eq spike-unified-rt.exe\" /FO CSV /NH"
```

`cargo.exe`, `unified-rt.exe`, or `spike-unified-rt.exe` output is a hard refusal; do not wait for a
busy tenant or kill another worker. The pinned fixture is the checked-in
`bench/campaign/metal-embed-fixtures/` corpus (2,000 rows, SHA-256
`25d1d54427030d94c882dd96a5f5d26bfda426d902028e75aa8c3d527e34a7a7`) and the
reference vector digest is
`d55221d41098aa293507c734ebedbf2df7f095c5e7c767943167403bbb520afd`.
