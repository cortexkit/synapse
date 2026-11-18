# Configuration

## Change Synapse settings

The CortexKit daemon manages Synapse settings. All fields are user-layer only
and take effect when the supervised module restarts. Synapse does not read
project settings, including `.cortexkit/synapse.jsonc`. Only one instance may
hold the singleton lease on a machine; do not launch a second instance to apply
a change.

The daemon's user-layer file is `cortexkit/synapse.jsonc` under its config home:

| Platform | Default user-layer file |
| --- | --- |
| Linux | `~/.config/cortexkit/synapse.jsonc` |
| macOS | `~/.config/cortexkit/synapse.jsonc` |
| Windows | `%APPDATA%\cortexkit\synapse.jsonc` (`%USERPROFILE%\AppData\Roaming\cortexkit\synapse.jsonc` if `APPDATA` is unset) |

An absolute `XDG_CONFIG_HOME` overrides the config home on all three platforms.
Use the daemon's settings review and approval flow after editing the user file.
File edits alone do not change a running Synapse. If approval is required and
the daemon has not delivered approved settings, Synapse waits and refuses
route binds with `module_warming`. If the daemon is not yet enforcing settings
approvals, it delivers the current user settings with defaults for unspecified
fields, including when the user file does not exist.

`SYNAPSE_CONFIG_PATH` is not a managed settings override. It is used only with
daemons that do not advertise settings support, such as the isolated release
certification daemon. With such a daemon and no override, Synapse reads the
same user path directly.

## Example user settings

The following JSON has no preloads or credentials. Merge only the desired
fields into the existing user file, rather than overwriting it:

```json
{
  "knob": "balanced",
  "microllm_max_tokens": 512,
  "cache_max_bytes": 34359738368,
  "worker": { "load_timeout_ms": 900000 },
  "preload_models": [],
  "inline": {
    "max_items": 64,
    "max_tokens": 8192,
    "byte_budget": 67108864,
    "max_queue_ms": 5000,
    "deadline_ms": 30000,
    "estimated_execution_ms": 25,
    "max_concurrent_workers": 2
  },
  "jobs": { "execution_ttl_ms": 86400000, "result_page_bytes": 524288, "bulk_quantum_tokens": 3072 }
}
```

Use `task` for preloads; the older `kind` and `capability` spellings are not
accepted. Each embed or rerank preload must specify a manifest `profile`. Llama
generation preloads may omit it. Remove `owned-metal-decode` preloads before
upgrading; owned decode serving was retired. `certify_observation` is reserved
for the release certification runner.

Remove the retired keys `alias_admin_enabled` (also `dev_alias_admin` and
`enable_alias_admin`), `grammar_enabled`, `sidecar_spec`, `decode_chain_k`, and
`remote_providers`. They are invalid even at their former default values.
Modules that need remote providers should call those providers directly.

## Verify the behavior, not only the file syntax

After review and restart, use Synapse's management surface to inspect the
active model catalog, health and admission status. [The wire contract](wire-contract-v1.md)
defines those operations and typed refusals. Configuration does not bypass
hardware or self-check gates. For changes that affect numerical results, see
[backend certification](backend-certification.md) before serving the new settings.
