# External gather-distillation harness

Standalone Bun/TypeScript data generation for the production gatherer contract. The default Anthropic lane calls `https://api.anthropic.com/v1/messages`; the local OpenAI-compatible lane calls a llama-server endpoint. Both proxy every repository tool through a pinned AFT binary, so trajectories preserve production-ranked search and canonical server-side formatting. The harness only reads SHA-pinned clones below `~/Work/OSS/gather-corpus`.

## Contract provenance

- System prompt: `CortexKit/alfonso/crates/alfonso-core-module/src/gather_prompt.rs`, `GATHER_CONTEXT_SYSTEM_PROMPT_V1`, commit `3ff7970e723e3c228c6efa4c61d27092db42d078`. The original project brief called this contract “v10”; `prompts/gather-system-v10.ts` retains that filename while identifying the code constant precisely.
- Budget control: `gather_tool_call_budget_thresholds`, `gather_tool_call_budget`, and `GATHER_BUDGET_FINALIZE_TEXT` in `crates/alfonso-core-module/src/manager_runtime.rs` at the same commit. With `maxSteps=40`, nudges fire at tool calls 20 and 25 and finalization fires at 30.
- Final JSON and snippet-pointer ingestion: `crates/alfonso-core/src/evidence/render.rs`, `crates/alfonso-core/src/evidence/types.rs`, and the active prompt above.
- The older salvage section in `.cortexkit/alfonso/docs/gather-context-request-flow.md` is stale. This harness keeps the tool declaration array byte-identical on the forced final turn. Anthropic sends `tool_choice: {"type":"none"}`; the local OpenAI-compatible lane sends the same schemas as function definitions with `tool_choice: "none"`. Selecting `tools_empty` is an error because it forks the student contract.

## AFT NDJSON integration

The harness starts `bin/aft-v0.46.0` directly with no command-line arguments and uses newline-delimited JSON on stdin/stdout. It configures each repository before use with `harness: "opencode"` and `session_id: "trainer"`. The core flags `semantic_search: false` and `search_index: true` are encoded in v0.46.0's inline user-tier config document, while `storage_dir: "/tmp/gather-campaign-aft"` is a top-level configure field.

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

Before the queue opens, a scratch-repository canary sends `search` and requires `Semantic search unavailable; returning lexical-only fallback results.` in the returned text. Before the first job for each corpus repository, the worker configures AFT and retries a real `search` with exponential backoff until the trigram index stops reporting a fully degraded fallback. The default wait is bounded at 60 seconds; a repository that remains cold is logged loudly and its first job ledger entry records a warning, but gathering proceeds so the campaign cannot hang. Callgraph storage is never explicitly pre-warmed: giant repositories build it only through AFT's own configure maintenance or if a trajectory calls `callgraph`.

A request timeout resets a wedged child and retries once after respawn. If the retry cannot recover, the failed row is ledgered and the gather queue retries that job once without ending the campaign.

## Authentication and safety

For a cheap-key dry run, set `GATHER_DISTILL_API_KEY`; requests use `x-api-key`. For OAuth generation, copy `accounts.json.example` to the ignored `accounts.json`, set `GATHER_DISTILL_ACCOUNTS_FILE` if it is elsewhere, and let the operator refresh tokens externally. OAuth subscription requests are assembled, signed, and headed as Claude Code requests; API-key requests remain plain.

The pool reparses changed credentials (with a 30-second stat cache), rotates healthy accounts round-robin, enforces a per-account in-flight cap, and cools an account after a 401 or quota response. Tokens are held only in memory and are never logged or written to rows.

`gather --backend openai` is deliberately separate: it does not construct an account pool, read `accounts.json`, send credentials, use prompt caching, or add Claude Code metadata. Its `--request-timeout` value is in seconds and defaults to 300 for slow local models.

Every corpus directory must contain:

```json
{"fullName":"owner/repo","sha":"40-character-sha","language":"TypeScript","size_mb":12.3}
```

The harness verifies `git rev-parse HEAD` against that manifest before gathering or validating. Tool paths reject absolute paths, `..`, and symlinks escaping the clone.

## Install and test

Install the local authentication dependency after placing the pinned AFT binary above.

```sh
cd tools/gather-distill
bun install
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

### Local OpenAI-compatible smoke lane

The local lane uses the identical system prompt, user prompt, budget text, and AFT catalog. It translates the canonical trajectory to `POST /v1/chat/completions`: the system prompt becomes a `system` message, `tool_use` blocks become assistant `tool_calls`, and `tool_result` blocks become `tool` messages. No local credentials are needed.

Download the official Q4_K_M artifact into the ignored model directory and start a current llama.cpp build with the model's Jinja template:

```sh
cd tools/gather-distill
mkdir -p bin/models
curl --fail --location \
  'https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q4_K_M.gguf?download=true' \
  -o bin/models/MiniCPM5-1B-Q4_K_M.gguf
llama-server --version
llama-server -m bin/models/MiniCPM5-1B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 --n-gpu-layers 99 --ctx-size 8192 --jinja
```

If the installed server cannot load the MiniCPM5 template or return OpenAI `tool_calls`, build current llama.cpp master before running the smoke:

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git bin/llama.cpp
cmake -S bin/llama.cpp -B bin/llama.cpp/build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build bin/llama.cpp/build --target llama-server -j 8
bin/llama.cpp/build/bin/llama-server -m bin/models/MiniCPM5-1B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 --jinja
```

`tool_choice: "none"` is retained with the nonempty tool catalog on the forced final turn to preserve the gather contract. A server or model that still emits a structured tool call or unparsed function markup is recorded as an honest `budget_finalize` failure and the harness does not execute another tool turn.

With the server running, create one job for an existing pinned clone, then bank the ignored smoke row:

```sh
cat > data/minicpm5-smoke-jobs.jsonl <<EOF
{"dir":"$HOME/Work/OSS/gather-corpus/ollama__ollama","request":"Where is the HTTP API server configured and started?","tags":{"request_class":"feature_orientation","expected_difficulty":1,"specificity":"high"}}
EOF

bun run src/cli.ts gather \
  --backend openai \
  --base-url http://127.0.0.1:8080/v1 \
  --model MiniCPM5-1B-Q4_K_M \
  --request-timeout 300 \
  --jobs data/minicpm5-smoke-jobs.jsonl \
  --concurrency 1 \
  --max-steps 4 \
  --max-response-tokens 768 \
  --rows data/minicpm5-smoke-rows.jsonl \
  --ledger data/minicpm5-smoke-ledger.jsonl \
  --status data/minicpm5-smoke-status.json
```

Rows record `thinking_tokens`. When llama-server reports a separate reasoning-token field it is used; otherwise `<think>...</think>` text is removed before final JSON parsing and counted with a deterministic whitespace-token estimate.

### Gold-overlap scoring

```sh
bun run src/cli.ts score \
  --candidate data/minicpm5-smoke-rows.jsonl \
  --gold data/eval-gold.jsonl \
  --output data/minicpm5-scores.json \
  --corpus-root ~/Work/OSS/gather-corpus
```

The scorer re-runs `validateBankedRow` for every latest candidate row, then pairs rows by repository, pinned SHA, and request. It uses only `final_json.snippets[].path` for file F1; `scope` is context rather than a citation. It reports inclusive line-range Jaccard overlap for shared files, clamped to the pinned file length, candidate-to-gold tool-call ratio, budget outcomes, output and thinking-token statistics, and a flattened `summary_row` for comparison tables. It never calls an LLM.

### Student checkpoint evaluation

`./scripts/eval-student.sh CHECKPOINT_OR_GGUF MODEL_LABEL` turns a fine-tuned checkpoint into one comparable ladder row. An HF safetensors directory is converted through the text-only `convert_hf_to_gguf.py` path, quantized to Q8_0, served with `-ngl 99 --jinja -fa on`, gathered over all 40 fixed jobs, and mechanically scored against Opus gold. The script is pinned to llama.cpp build `9580` / revision `b4e3dc613`; keep that revision or an equivalent current build because Qwen3.5 and Gemma 4 text conversion support is required.

For a standalone GGUF, provide its adjacent `config.json` or set `EVAL_CONFIG_JSON`; the script clamps the normal `131072` context request to the trained maximum in that config. A file named like `*Q8_0.gguf` is used directly; use an F16/BF16 GGUF for fresh quantization. Results are written under ignored `data/students/` as `<label>-scores.json`, resumable rows/ledger/status files, and `LADDER.md`. The ladder uses the same columns as the zero-shot bake-off, including natural-only F1/Jaccard and whole-run validity, API, tool, budget, context, and timing measurements.

```sh
cd tools/gather-distill
./scripts/eval-student.sh ~/checkpoints/student-merged student-sft-v1

# Existing Q8 GGUF: config is needed only to apply the trained-context clamp.
EVAL_CONFIG_JSON=~/checkpoints/LFM2.5/config.json \
  ./scripts/eval-student.sh ~/models/LFM2.5-Q8_0.gguf student-sft-v1
```

The local default is a Metal server at `127.0.0.1:8090`. On a CUDA host, run the same command there with `LLAMA_CPP_DIR` (or explicit `LLAMA_SERVER` and `LLAMA_QUANTIZE`) set to that host's llama.cpp build. For the intended GPU-host server / Mac harness topology, start the remote server first and run the harness with a tunnel:

```sh
# The remote model is already converted, quantized, and listening on port 8090.
EVAL_REMOTE_ENDPOINT=root@gpu.example EVAL_REMOTE_SSH_PORT=22 \
  ./scripts/eval-student.sh remote-serving-checkpoint student-sft-v1
```

`EVAL_REMOTE_ENDPOINT` makes the script open `ssh -L` to the remote `llama-server`; it does not inspect or copy the positional checkpoint path in tunnel mode. The harness machine must have the ignored eval data, `~/Work/OSS/gather-corpus-eval` clones, and `GATHER_DISTILL_AFT_BINARY` (normally `bin/aft-dev-7cabfdd0`). A remote all-in-one run needs those same assets staged on the GPU host.

### TRACE reward bridge

`train_reward/reward.py` exposes `reward(trajectory_or_final_package, job_id) -> {"reward": float, "diagnostics": {...}}` without depending on a training framework. It writes the candidate to a temporary JSON file and calls the reusable score-one lane:

```sh
bun run src/cli.ts score-one \
  --job JOB_ID \
  --candidate-file candidate.json \
  --gold data/eval-gold-rows.jsonl
```

`candidate.json` may be a final package, a BankedRow, or a trajectory whose final assistant text contains the package. Score-one uses the selected gold row's identity and returns exactly one JSON verdict line. Reward shaping v1 is intentionally narrow: a **natural**, schema-valid completion receives its cited-file F1; an invalid or non-natural completion receives `0`. Diagnostics include `line_jaccard`, `contract_valid`, and candidate `tool_calls` for later TRACE shaping work. The committed TRACE fixtures retain the final packages and tool-call counts from two scrubbed real Opus eval rows while omitting prompts, tool results, account data, and timing metadata.

## Useful controls

- `--account-inflight 2` and `--auth-cooldown-ms 300000` for the Anthropic lane
- `--backend openai --base-url http://127.0.0.1:8080/v1 --request-timeout 300` for a local server
- `--max-steps 40` (production thresholds become 20/25/30)
- `--token-ceiling 200000` total API tokens per trajectory
- `--max-response-tokens 8000`
- `--finalize-mode tool_choice_none_full_toolset` (the only accepted mode)
- `--concurrency N`

Generated JSONL, live status, `accounts.json`, and `bin/models/` are ignored by git.
