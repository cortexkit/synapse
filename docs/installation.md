# Installation support check

Before enabling Synapse in `ck setup`, run the support check from the unpacked
release. Keep the worker binaries beside `ck-synapse`; the check looks there,
not in the supervised module's configuration.

| Platform | Command | Backends checked on that platform |
| --- | --- | --- |
| macOS | `./ck-synapse --check-backend` | Metal, direct ANE |
| Linux | `./ck-synapse --check-backend` | CUDA, Vulkan |
| Windows (PowerShell) | `.\ck-synapse.exe --check-backend` | CUDA, Vulkan |

The check uses the same backend detection as `models.catalog` for embedding.
It does not start or contact the daemon, open a store, read module configuration,
write Synapse data, or download or load a model. A positive result means you can
enable the module and offer to download the embedding model; it does not mean
the model is installed or that every model meets its own serving requirements.

On success, stdout contains one usable backend name per line (`metal`, `ane`,
`cuda`, or `vulkan`). If none is usable, stdout shows one
`backend_unavailable: <backend>: <reason>` line per backend, using the same
reason codes as `models.catalog`.

For automation, append `--json`. Stdout then has a `usable` boolean and a
`backends` array with one row per backend, including those that are not supported
on this platform. Each row has `backend`, `runnable`, and `reason`; the reason
is `null` for a runnable backend.

| Exit code | Meaning |
| --- | --- |
| `0` | At least one backend is runnable. |
| `1` | None is runnable, and at least one probe returned `probe_failed`. Support is uncertain. |
| `2` | None is runnable, and all refusal reasons are definite. |
| `3` | The check could not run. See stderr for the error. |

Run `ck-synapse --check-backend --help` for the command's help.
