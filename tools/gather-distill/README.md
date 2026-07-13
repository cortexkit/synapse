# External gather-distillation harness

Standalone Bun/TypeScript data generation for the production gatherer contract. It calls `https://api.anthropic.com/v1/messages` directly and proxies every repository tool through a pinned AFT binary, so trajectories preserve production-ranked search and canonical server-side formatting. It only reads SHA-pinned clones below `~/Work/OSS/gather-corpus`.

## Contract provenance

- System prompt: `CortexKit/alfonso/crates/alfonso-core-module/src/gather_prompt.rs`, `GATHER_CONTEXT_SYSTEM_PROMPT_V1`, commit `3ff7970e723e3c228c6efa4c61d27092db42d078`. The original project brief called this contract “v10”; `prompts/gather-system-v10.ts` retains that filename while identifying the code constant precisely.
- Budget control: `gather_tool_call_budget_thresholds`, `gather_tool_call_budget`, and `GATHER_BUDGET_FINALIZE_TEXT` in `crates/alfonso-core-module/src/manager_runtime.rs` at the same commit. With `maxSteps=40`, nudges fire at tool calls 20 and 25 and finalization fires at 30.
- Final JSON and snippet-pointer ingestion: `crates/alfonso-core/src/evidence/render.rs`, `crates/alfonso-core/src/evidence/types.rs`, and the active prompt above.
- The older salvage section in `.cortexkit/alfonso/docs/gather-context-request-flow.md` is stale. This harness keeps the tool declaration array byte-identical on the forced final turn and sends Anthropic `tool_choice: {"type":"none"}`. Selecting `tools_empty` is an error because it forks the student contract.

## AFT NDJSON integration

The harness starts `bin/aft-v0.46.0` directly with no command-line arguments and uses newline-delimited JSON on stdin/stdout. It configures each repository before use with `harness: "opencode"`, `session_id: "trainer"`, `semantic_search: false`, `search_index: true`, and `storage_dir: "/tmp/gather-campaign-aft"`.

The model receives the same bare production declarations as the gather sender at commit `3ff7970`: `search`, `outline`, `zoom`, `callgraph`, `read`, `grep`, `glob`, `inspect`, and `conflicts`. `src/aft-tool-catalog.ts` is the verbatim v0.46.0 schema catalog for those declarations. The v0.46.0 NDJSON wire manifest uses those same bare names. AFT response `text` is sent to the model verbatim; the harness never JSON-wraps, trims, or drops a trailing AFT status line. In particular, lexical-only search disclosures remain in trajectories.

### Pinned binary provenance

The campaign binary is the Darwin arm64 asset from [`cortexkit/aft` v0.46.0](https://github.com/cortexkit/aft/releases/tag/v0.46.0), not the fleet-staged `~/.local/share/cortexkit/aft/bin/aft-stable` path. It is intentionally ignored by Git so a fleet restage cannot change an in-progress campaign.

```sh
cd tools/gather-distill
mkdir -p bin
curl --fail --location https://github.com/cortexkit/aft/releases/download/v0.46.0/aft-darwin-arm64 \
  -o bin/aft-v0.46.0
chmod 755 bin/aft-v0.46.0
shasum -a 256 bin/aft-v0.46.0
# expected: e8aef37ba914f8760110c7feabddc19da67e936b9645d303a6c843dcc8557e2d
bin/aft-v0.46.0 --version
# expected: aft 0.46.0
```

A small AFT process pool is bounded by gather concurrency. It keeps one process for each active repository and reconfigures the least-recently-used idle process when work moves to another clone. The harness starts the AFT binary directly, then best-effort lowers only that child process's priority with `renice 19`; platforms that deny `renice` continue without changing the harness process priority.

Before the queue opens, a scratch-repository canary sends `search` and requires `Semantic search unavailable; returning lexical-only fallback results.` in the returned text. Before the first job for each corpus repository, the worker configures AFT and warms it with `glob README*`. A warm-up that reaches five minutes is ledgered as a failed, retryable job and the repository is skipped with a queue note. Callgraph storage is never pre-warmed: giant repositories build it only if a trajectory actually calls `callgraph`.

A request timeout resets a wedged child and retries once after respawn. If the retry cannot recover, the failed row is ledgered and the gather queue retries that job once without ending the campaign.

## Authentication and safety

For a cheap-key dry run, set `GATHER_DISTILL_API_KEY`; requests use `x-api-key`. For OAuth generation, copy `accounts.json.example` to the ignored `accounts.json`, set `GATHER_DISTILL_ACCOUNTS_FILE` if it is elsewhere, and let the operator refresh tokens externally. OAuth requests use `Authorization: Bearer`, `anthropic-beta: oauth-2025-04-20`, and `anthropic-version: 2023-06-01`.

The pool reparses changed credentials (with a 30-second stat cache), rotates healthy accounts round-robin, enforces a per-account in-flight cap, and cools an account after a 401 or quota response. Tokens are held only in memory and are never logged or written to rows.

Every corpus directory must contain:

```json
{"fullName":"owner/repo","sha":"40-character-sha","language":"TypeScript","size_mb":12.3}
```

The harness verifies `git rev-parse HEAD` against that manifest before gathering or validating. Tool paths reject absolute paths, `..`, and symlinks escaping the clone.

## Install and test

No package install is required after placing the pinned AFT binary above.

```sh
cd tools/gather-distill
bun test
```

## Required API-key dry run (before OAuth/Opus)

Use one tranche-1 clone and cheap model overrides. The commands are intentionally separate so qgen can later be run fleet-side without changing gather execution.

```sh
export GATHER_DISTILL_API_KEY='...'

bun run src/cli.ts qgen \
  --repo ~/Work/OSS/gather-corpus/OWNER__REPO \
  --count 1 \
  --model claude-3-5-haiku-latest \
  --output data/dry-jobs.jsonl

bun run src/cli.ts gather \
  --jobs data/dry-jobs.jsonl \
  --model claude-3-5-haiku-latest \
  --concurrency 1 \
  --inline-validate \
  --rows data/dry-rows.jsonl \
  --ledger data/dry-ledger.jsonl \
  --status data/dry-status.json

bun run src/cli.ts validate \
  --rows data/dry-rows.jsonl \
  --output data/dry-validated.jsonl
```

The gather command prints each complete banked row, including the trajectory and inline validity. The validate command prints valid/rejected counts and exits nonzero when any row is rejected. Inspect `data/dry-rows.jsonl` and `data/dry-validated.jsonl` before supplying OAuth credentials.

## Production lanes

### QGEN

```sh
bun run src/cli.ts qgen \
  --corpus-root ~/Work/OSS/gather-corpus \
  --model claude-sonnet-5-0 \
  --count 20 \
  --output data/jobs.jsonl
```

QGEN grounds on the manifest, bounded AFT file listing, README, and a few entry files. It accepts only a strict JSON array of code-answerable questions tagged with request class, difficulty 1–5, and specificity.

### GATHER

```sh
export GATHER_DISTILL_ACCOUNTS_FILE="$PWD/accounts.json"
bun run src/cli.ts gather \
  --jobs data/jobs.jsonl \
  --model claude-opus-4-8 \
  --inline-validate \
  --rows data/rows.jsonl \
  --ledger data/ledger.jsonl \
  --status data/status.json
```

Jobs are interleaved round-robin by language × request class × repository, with high-difficulty/high-specificity questions first inside a cell. The append-only ledger skips banked and mechanically rejected `(dir, request)` jobs on resume; failed API jobs remain retryable. Rows include every model, tool-call, and tool-result turn, final pointer JSON, token usage, budget outcome, model, non-secret account name, timestamp, and validation result.

The status file starts at row one and reports rolling tokens/hour, trajectories/hour, and per-account totals.

### Validation

```sh
bun run src/cli.ts validate --rows data/rows.jsonl --output data/validated.jsonl
```

Validation is reject-only: it schema-checks final JSON, confirms manifest SHA equals clone HEAD, resolves every path and line range, and compares any cited range backed by trajectory source bytes with the pinned file. It writes copied rows with `valid` and `reason` to a new file; it never edits citations or repairs model output.

## Useful controls

- `--account-inflight 2` and `--auth-cooldown-ms 300000`
- `--max-steps 40` (production thresholds become 20/25/30)
- `--token-ceiling 200000` total API tokens per trajectory
- `--max-response-tokens 8000`
- `--finalize-mode tool_choice_none_full_toolset` (the only accepted mode)
- `--concurrency N`

Generated JSONL, live status, and `accounts.json` are ignored by git.
