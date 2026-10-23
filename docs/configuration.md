# Configuration

## Prepare a configuration example

Synapse reads `SYNAPSE_CONFIG_PATH` when supplied. Otherwise it uses project
`.cortexkit/synapse.jsonc`, falling back to user `cortexkit/synapse.jsonc` only when
no project file exists. Only one module instance may hold the singleton lease on
a machine. `remote_providers` is rejected: remote providers were removed; modules
should call providers directly.

Create a new example file rather than overwriting an existing configuration:

```sh
config=$(mktemp)
cat > "$config" <<'JSON'
{
  "knob": "balanced",
  "microllm_max_tokens": 512,
  "grammar_enabled": false,
  "decode_chain_k": 1,
  "cache_max_bytes": 34359738368,
  "alias_admin_enabled": false,
  "worker": { "load_timeout_ms": 180000 },
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
  "jobs": { "ttl_ms": 86400000, "result_page_bytes": 524288, "bulk_quantum_tokens": 3072 },
  "probe": { "mean_cosine_threshold": 0.999, "worst_decile_rank_overlap_threshold": 0.9, "ane_placement_threshold": 0.9 }
}
JSON
python3 -m json.tool "$config" >/dev/null
```

This preparation does not launch or reconfigure a module. When using a reviewed
configuration, pass `SYNAPSE_CONFIG_PATH="$config"` through the normal supervised
launch configuration. Do not start a second instance to change a running module.
The example has no preloads or credentials and is not a certified model installation.
`decode_chain_k` is user-tier only, ranges from 1 through 16, and remains one under
grammar; larger free-text spans require matching certification coverage.
Project configurations must omit that user-only setting.

## Verify the behavior, not only the file syntax

The JSON check above verifies syntax only. The module validates configuration at
startup; use its management surface to inspect the active model catalog, health
and admission status. [The wire contract](wire-contract-v1.md) defines those
operations and typed refusals. A backend advertised in configuration is not
permission to bypass hardware, self-check or approval gates.

## Reverse the example

After the run using the example has stopped, remove only the temporary file:

```sh
rm "$config"
unset config
```

Do not delete installed model data, leases, approvals or an existing user config
to reverse this example. Changes to numerical policy need identity and
certification handling, not just a configuration edit.
