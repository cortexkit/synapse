# Public-release audit

## Scope and reading guide

This is an inventory of the **1,909 tracked files** at the audit commit. It is
an inventory only: no source, data, or configuration was removed by this audit.
The scan covered the supplied seeds plus SSH syntax, non-loopback IPs, personal
paths/names, credential-file references, account aliases, session and campaign
identifiers, private sibling repositories, and workflow secrets. `docs/synapse-performance-matrix.xlsx` is a binary workbook and is called out for manual review.

Severity means:

- **CRITICAL** — an actual secret/credential value, a private IP/addressable
  endpoint, or private third-party source/data.
- **HIGH** — infrastructure metadata, identity, internal fleet/process material,
  or an unreleased sibling-project dependency.
- **MEDIUM** — personal-path residue, generated evidence, or operational noise.

The report deliberately does **not** repeat the discovered password, host public
keys, or any candidate token material. A path that is a directory means every
tracked file below it. `DELETE` is the preferred public-tree disposition;
`REDACT` means retain only after removing the cited material; `REVIEW` is a
release blocker until an owner explicitly approves a safe replacement. `KEEP`
is used only where the audited item did not itself expose private material.

## Disposition 1 — DELETE from the public tree

The following is a copy-paste-ready path list for `git rm -r` (directories are
intentional recursive targets). These are high-confidence private artifacts, not
source files that merely need a configurable default.

```text
.cortexkit
evidence
operations
.alfonso
corpus/aft-chunks.jsonl
tools/classify-distill
tools/gather-distill/data/students
bench/campaign/metal-embed-fixtures
bench/spikes/unified-rt/results
bench/spikes/unified-rt/f16-evidence
bench/spikes/ane-minilm/results
bench/spikes/ane-spec-decode/results
bench/eval-coir/results
RIG-CUDA-SHOOTOUT.txt
RIG-VLLM-CELL.txt
docs/campaign-context-repro.md
docs/deployment-owned-decode-approvals.md
docs/diag-ane-prefill-divergence.md
docs/diag-ane-prefill-m1-authority.md
docs/dogfood-ane-lane.md
docs/dogfood-p0.md
docs/eval-lfm25-vl-3b.md
docs/lab-roadmap-orx-findings.md
docs/mtp-challenge-m1-loop.md
docs/prod-nonmac-lanes.md
docs/prod-rerank-lane.md
```

| Path | Category | Severity | Why this must be deleted |
| --- | --- | --- | --- |
| `.cortexkit/` (359 tracked files) | Process artifacts; infrastructure; credentials; identity; session/fleet artifacts | **CRITICAL** | No tracked content in this tree is public-appropriate. It includes campaign registration, two SSH host public keys, SSH identity paths, LAN/cloud endpoints and ports, user names, private sibling paths, 76 tracked task-output files, raw consult content, controller PIDs, launch nonces, campaign/consult IDs, and 11 prompts containing an actual M1 password. Do not attempt line-by-line sanitization. |
| `.cortexkit/alfonso/task-outputs/` (76 files, included above) | Session/fleet artifacts | **HIGH** | Raw consultation reports and patches contain campaign IDs, request/send/session IDs, roster/member model names, private worktree paths, controller process identities, and embedded implementation discussion. |
| `evidence/` (119 tracked files) | Process artifacts; private IP; identity; private source | **CRITICAL** | Captured production/test evidence includes vast.ai/TensorDock endpoints, provider requests, machine profiles, Windows user paths, personal paths, campaign IDs, and AFT/subconscious-derived corpus text. The subtrees `nonmac/`, `ane-prefill-split/`, and `semantic-sidecar-v1/` are all private operational records. |
| `operations/` (2 tracked files) | Infrastructure; process artifacts | **HIGH** | `operations/ane-prefill-split/stage_m1_cert.sh` hard-codes the `[bench-host-alias]` route, `[bench-user-home]` layout, cache snapshot, and private `commons`/`subconscious` siblings. `ENABLEMENT.md` is an internal fleet enablement procedure tied to private evidence. |
| `.alfonso/spikes/coreml_spike.rs` | Personal identity; private IP | **CRITICAL** | Throwaway spike with personal default paths into the private magic-context model cache and embedded snippets from another private project. |
| `corpus/aft-chunks.jsonl` | Third-party private IP | **CRITICAL** | 15,271 chunks of the private AFT repository, including source paths, identifiers, and embedded source text. This is not a redistributable benchmark fixture. |
| `tools/classify-distill/` | Third-party private IP; credentials/process | **CRITICAL** | Contains line-preserving vendored ALF/Alfonso contract excerpts. `contract/PROVENANCE.md` identifies the private source checkout and commit; package metadata also points at a private auth package. Remove the whole tool rather than separating derived ports from the private contract. |
| `tools/gather-distill/data/students/` | Third-party private IP; session artifacts | **CRITICAL** | `codegraph-explore-raw.jsonl` and derived rows/packages reproduce private `cortexkit/subconscious` exploration requests, source excerpts, repository SHAs, scratch paths, and model/account run metadata. The accompanying scores/ladder are derived from the same private set. |
| `bench/campaign/metal-embed-fixtures/` | Third-party private IP | **CRITICAL** | `embedding-corpus.jsonl` is AFT/codegraph source text; `master-reference-vectors.bin.gz`, hashes, selection, and metadata are derived from it. The entire fixture set must go together. |
| `bench/spikes/unified-rt/results/` and `bench/spikes/unified-rt/f16-evidence/` | Infrastructure; process artifacts; identity | **HIGH** | Locked-M1, Ally, and rented-rig measurement evidence exposes host names, local paths, hardware fingerprints, campaign IDs, and deployment procedures. Preserve privately if needed, not in the public tree. |
| `bench/spikes/ane-minilm/results/`, `bench/spikes/ane-spec-decode/results/`, and `bench/eval-coir/results/` | Personal identity | **HIGH** | Generated result JSON embeds `/Users/[owner]` cache/worktree paths and private runner/artifact locations. Do not publish raw results; regenerate sanitized summaries if the metrics matter. |
| `RIG-CUDA-SHOOTOUT.txt`, `RIG-VLLM-CELL.txt` | Infrastructure secrets | **CRITICAL** | Each gives a live-style vast.ai SSH endpoint, port, root user, and key-attachment status. |
| Deleted docs listed above | Docs; infrastructure; fleet process | **HIGH** | These are runbooks, campaign reproduction records, operator transcripts, private fleet status, or UI-capture evaluation notes. They are private-by-nature rather than a few redactable lines. |

## Disposition 2 — REDACT before release

These items can remain only after the cited values are replaced by neutral,
configurable examples. Redacting a path does not authorize publishing an actual
credential; rotate the credential first if a value was exposed.

| Path(s) | Category | Severity | Required redaction |
| --- | --- | --- | --- |
| `.github/workflows/ci.yml` | Workflow/CI | **HIGH** | Replace `CK_CI_APP_ID`/`CK_CI_APP_PRIVATE_KEY`, `cortexkit` owner, private `subconscious` and `commons` checkouts, sibling-layout comments, and retired self-hosted `macos-metal`/M1 references with a public CI design. No secret values are committed here, but their names and use prove private cross-repository infrastructure. |
| `crates/synapse-opctl/src/main.rs`; `crates/synapse-module/src/bin/subc_call.rs`; `crates/synapse-module/src/bin/inline_embed_throughput.rs` | Personal identity; credential reference | **HIGH** | Replace the `/Users/[owner]/.../subc-connection.json` default with an explicit argument, XDG/config lookup, or a generic placeholder. |
| `crates/synapse-module/src/owned_decode_contracts/mod.rs` | Personal-path noise | **MEDIUM** | Remove the absolute `/Users/.../worktree` example from the comment. |
| `crates/synapse-module/owned-decode-manifests/slice-plan-v1.json`; `crates/synapse-module/owned-decode-manifests/runtime-bound-decode-cutover-epic-manifest-v1.json` | Session/fleet artifacts | **HIGH** | Remove `ct_...` work-item identifiers and campaign provenance from shipped manifests, or replace with neutral release metadata. |
| `Cargo.toml` | Private project dependency | **HIGH** | Public release cannot retain `../subconscious` and `../commons` path dependencies unless those repositories are simultaneously public, versioned, and licensed. Publish dependencies, vendor with permission, or split/remove the dependent features. |
| `ARCHITECTURE.md`; `FOUNDING.md` | Private project/credential internals | **HIGH** | Replace references to `@cortexkit/anthropic-auth-core`, the private vault/claustrum implementation, and internal credential flow with public interfaces and security documentation. |
| `bench/campaign/vulkan-embed-harness.sh`; `bench/campaign/vulkan-embed-baseline.md`; `bench/campaign/README.md` | Infrastructure secrets | **CRITICAL** | Remove `ufuka@[ally-host]`, `[lan-ip]`, host-specific probing, and M1 lock/path details. Make the remote target an uncommitted environment/config input. |
| `bench/campaign/provision-cuda-rig.sh`; `bench/campaign/{cuda-quant-harness.sh,decode-harness.sh,lfm2-cuda-harness.sh,lfm2-decode-harness.sh,metal-step-harness.sh}` | Infrastructure/process; private dependencies | **HIGH** | Remove campaign-controller assumptions and private sibling repo URLs/paths. Retain only after a public harness is independently specified and tested. |
| `bench/spikes/unified-rt/{BATCHED-VERIFY.md,CAMPAIGN-1-BRIEF.md,DECODE-WAVE1.md,F16-SERVING.md,GRADUATION-PROBE.md,LFM2-DECODE-BASELINES.md,LFM2-METAL-STEP.md,M1-BUCKET-MATRIX.md,M1-SERVING-MATRIX.md,METAL-STEP.md,MODERNBERT.md,QWEN3.md,RERANK.md,VULKAN-DECODE.md}` | Infrastructure; session/fleet artifacts | **HIGH** | Remove `[bench-host]`, `[bench-host-alias]`, `[ally-host]`, Windows user paths, campaign `ct_...` IDs, and private lock/deployment commands. Publish sanitized benchmark methodology instead of the operational transcripts. |
| `bench/spikes/unified-rt/{summarize-m1-rerank.py,summarize-m1-bucket-matrix.py,summarize-graduation-results.py,summarize-graduation-probe.py,src/lfm2_decode_metal_step.rs}` | Infrastructure | **HIGH** | Replace embedded M1 host labels in emitted JSON and code comments with generic machine-profile fields. |
| `bench/spikes/ane-minilm/{SPIKE.md,README.md,ANE-WAVE1.md,ANE-WAVE2-QWEN3.md,ANE-QWEN3-RETRIEVAL.md}`; `bench/spikes/ane-prefill-split/{README.md,ANE-PREFILL-SPLIT.md,results/locked-m1.json}` | Infrastructure | **HIGH** | Remove M1 aliases, host names, runner labels, and `[bench-user-home]` commands; keep only reproducible public benchmark instructions. |
| `bench/spikes/stt-bias/{STT-BIAS.md,evalkit/README.md,evalkit/terms.jsonl,evalkit/utterances.jsonl}` | Private project internals | **MEDIUM** | Replace `CortexKit sibling` source labels and internal vocabulary/test sentences with independently licensed public examples. |
| `bench/NOTES.md`; `bench/lanes/candle-embed/SPIKE.md`; `bench/eval-coir/MIXED-SPACE-AB.md` | Private IP; personal identity; process noise | **HIGH** | Remove AFT-corpus provenance, `bg_...` worker IDs, the personal path, and private fleet connection file. |
| `docs/design-remote-gateway.md`; `docs/design-synapse-module.md`; `docs/design-stt-bias.md`; `docs/study-2026-08-24-speculative-serving-sweep.md`; `docs/decision-1-runtime.md` | Docs; private project/process internals | **MEDIUM** | Redact `vault-handles.json`, sibling `commons`/`subconscious`/`.cortexkit` implementation references, Mason/agent work labels, draft paths, campaign/lab claims, and the private 15,271-chunk corpus provenance. These documents otherwise contain reusable design material. |
| `tools/gather-distill/accounts.json.example`; `tools/gather-distill/tests/queue.test.ts` | Credential references | **MEDIUM** | Rename `acct1`/`acct2` aliases to clearly synthetic names (`test-account-1`, etc.) and keep placeholders only. The supplied `acct3` seed was not present in tracked content. |
| `tools/gather-distill/src/{auth.ts,openai-oauth.ts,cli.ts}`; `tools/gather-distill/train/DEEPSEEK-FLASH-COMPARISON.md` | Credential references | **HIGH** | Retain only after the OAuth/documented auth-file flow is security-reviewed and all references to a locally managed OpenCode account are made generic. The scanned references are file paths/flow descriptions, not token values. |
| `README.md`; `ARCHITECTURE.md` | Process/config noise | **MEDIUM** | Keep public configuration examples only if they are presented as generic XDG paths and do not imply access to private CortexKit services or packages. |

## Disposition 3 — REVIEW before release (and items that can remain)

| Path(s) | Category | Severity | Disposition and reason |
| --- | --- | --- | --- |
| `results/vulkan-wave2/` (7 files) | Process artifacts | **MEDIUM** | **REVIEW.** The tracked JSON/fingerprint files did not match the private-host, personal-path, or credential indicators used here. Confirm hardware/driver fingerprints are intentionally public; then **KEEP**. |
| `docs/synapse-performance-matrix.xlsx` | Docs | **MEDIUM** | **REVIEW.** Binary workbook; inspect sheet cells, comments, hidden sheets, external links, author metadata, and custom properties before deciding **KEEP** or delete. |
| `tools/gather-distill/` outside `data/students/` | OAuth and private dependency review | **HIGH** | **REVIEW.** Source is not a credential leak by itself, but it depends on `@cortexkit/anthropic-auth-core` and implements personal-account OAuth flows. Public release requires a public dependency and a threat-model review. |
| `scripts/provision-campaign-machine.sh` | Infrastructure seed | **MEDIUM** | **KEEP (not present).** This supplied seed is not a tracked path in this checkout. The related tracked provisioning script is `bench/campaign/provision-cuda-rig.sh`, which is separately marked for redaction/review. |
| `bench/campaign/{decode-fixtures,lfm2-decode-fixtures}/` | Fixtures | **MEDIUM** | **KEEP after provenance review.** These contain synthetic prompts/reference token IDs, not the private AFT corpus. Keep only if model and fixture licenses permit redistribution. |
| `tools/stt-voice-test/` | Local service | **MEDIUM** | **KEEP.** It binds to loopback (`127.0.0.1`) only; loopback examples are not private LAN exposure. |
| Public Rust/Swift engine source not listed above | Product source | **MEDIUM** | **KEEP after normal license/dependency review.** No actual API key, SSH private-key block, GitHub PAT, AWS key, or OAuth access/refresh-token value was found by the token-pattern scan. This does not remove the need to rotate the exposed password. |

## Per-category detail

### 1. Infrastructure secrets and private endpoints

| Finding path(s) | Severity | Disposition | Detail |
| --- | --- | --- | --- |
| `.cortexkit/campaign-lab.jsonc` | **CRITICAL** | DELETE | Contains direct host/IP configuration, SSH identity-file paths, two host public keys, `[ally-host]`, private sibling source paths, and campaign remote-target variables. |
| `.cortexkit/alfonso/prompts/` | **CRITICAL** | DELETE | Many prompts name `[bench-host]`, `[bench-host-alias]`, `[ally-host]`, `192.168.*`, vast.ai endpoints/ports, TensorDock, SSH users, and private rig paths. This includes all specifically seeded prompt material. |
| `RIG-CUDA-SHOOTOUT.txt`; `RIG-VLLM-CELL.txt` | **CRITICAL** | DELETE | Direct root SSH coordinates for cloud instances. |
| `evidence/nonmac/{linux-nvidia,windows-nvidia,windows-ally}/` | **CRITICAL** | DELETE | Provider API calls, cloud endpoint data, direct/indirect SSH data, machine profiles, and Windows user paths. `windows-nvidia/transcripts/provider-diagnostics.txt` references `~/.config/[gpu-provider].key` and `[gpu-provider].authid`; `verification-v1.json` and `deployment-record-v1.json` retain TensorDock API details. |
| `bench/campaign/vulkan-embed-baseline.md`; `bench/campaign/vulkan-embed-harness.sh` | **CRITICAL** | REDACT | Private LAN IP/user and named Ally host are committed outside `.cortexkit`. |
| `bench/spikes/unified-rt/VULKAN-DECODE.md` | **HIGH** | REDACT | Retains the Ally hostname, user, and Windows user directory. |
| `operations/ane-prefill-split/stage_m1_cert.sh` | **HIGH** | DELETE | Named SSH alias plus fixed remote filesystem/cache layout and sibling clones. |
| `.github/workflows/ci.yml` | **HIGH** | REDACT | Uses Blacksmith runner labels, a retired self-hosted M1 label, app-token secret references, and private organization/sibling repository names. |

### 2. Credential references and actual credentials

| Finding path(s) | Severity | Disposition | Detail |
| --- | --- | --- | --- |
| `.cortexkit/alfonso/prompts/{mtp-challenge-m1-baseline.md,m1-shipout-prep.md,m1-llama-reference-cells.md,harness-candidate-dest.md,harness-runner-cp-diag.md,harness-siblings.md,embed-o1-attribution.md,decode5-winner-integrate.md,campaign-context-simulator.md,ane-prefill-m1-full-certification.md,ane-prefill-m1-authority-battery.md}` | **CRITICAL** | DELETE and rotate | Eleven tracked prompts contain an actual M1 SSH/sudo password. The value is intentionally omitted here. Treat the value as compromised even if the M1 was decommissioned; rotate or disable the account and remove it from all reachable history. |
| `.cortexkit/alfonso/prompts/gather-distill-oauth-impersonation.md`; `.cortexkit/alfonso/prompts/judge-openai-oauth.md`; `.cortexkit/alfonso/prompts/deepseek-flash-comparison.md`; `.cortexkit/alfonso/prompts/s7-[gpu-provider]-windows-nvidia.md` | **HIGH** | DELETE | Private OAuth account flow, `[oauth-account]` account name, OpenCode `auth.json` locations, and TensorDock credential paths. |
| `tools/gather-distill/accounts.json.example`; `tools/gather-distill/tests/queue.test.ts` | **MEDIUM** | REDACT | `acct1` and `acct2` are retained account aliases. They are placeholders, not token values, but should not look like a real account inventory. No tracked `acct3` match was found. |
| `tools/gather-distill/src/openai-oauth.ts`; `tools/gather-distill/src/auth.ts`; `tools/gather-distill/src/cli.ts` | **HIGH** | REVIEW | Reads `auth.json` and handles access-token records. These files do not contain actual token values, but require a public security/design review. |
| `docs/design-remote-gateway.md`; `FOUNDING.md` | **HIGH** | REDACT | Refer to `vault-handles.json` and private credential/vault implementation details. |
| `evidence/nonmac/windows-nvidia/transcripts/provider-diagnostics.txt` | **HIGH** | DELETE | Uses credential-file substitutions for `[gpu-provider].key` and `[gpu-provider].authid`; no value was printed in the tracked transcript. |

### 3. Personal identity

| Finding path(s) | Severity | Disposition | Detail |
| --- | --- | --- | --- |
| All `/Users/[owner]/...` findings, principally `.cortexkit/`, `evidence/`, `.alfonso/`, and the result artifacts named in the delete list | **HIGH** | DELETE | The supplied audit seed identifies 115 personal-path hits. The high-volume trees are deleted wholesale rather than leaving a brittle per-line scrub list. |
| `crates/synapse-opctl/src/main.rs`; `crates/synapse-module/src/bin/{subc_call.rs,inline_embed_throughput.rs}` | **HIGH** | REDACT | Product defaults embed the personal account name and private connection path. |
| `bench/spikes/unified-rt/f16-evidence/EVIDENCE.md`; `bench/spikes/ane-spec-decode/results/phase-c-raw.json`; `bench/spikes/ane-spec-decode/results/packages/convert-report.json`; `bench/spikes/ane-minilm/results/`; `bench/eval-coir/{MIXED-SPACE-AB.md,results/}`; `bench/lanes/candle-embed/SPIKE.md` | **HIGH** | DELETE/REDACT as listed | These are the material non-`.cortexkit` personal path findings in benchmark documentation/results. |
| `evidence/nonmac/preparation/nonmac-cert-policy-v2.json`; `evidence/nonmac/linux-nvidia/recalibration-20260807/recalibration-v1.json` | **HIGH** | DELETE | Identify the owner by personal name (`Ufuk`). |
| `.cortexkit/alfonso/task-outputs/`; `tools/classify-distill/contract/PROVENANCE.md` | **HIGH** | DELETE | Worktree paths identify the account and private repository location. |

### 4. Session, fleet, and consultation artifacts

| Finding path(s) | Severity | Disposition | Detail |
| --- | --- | --- | --- |
| `.cortexkit/alfonso/task-outputs/` | **HIGH** | DELETE | Contains consult/campaign `ct_...` IDs, report/manifest records, request/send/session IDs, PID/PGID/start-time/launch-nonce fields, raw plans, patches, and panel/roster member identifiers. The report directory is tracked (76 files), not merely ignored local output. |
| `.cortexkit/vulkan-consult-verdicts-1740.json` | **HIGH** | DELETE | Raw consult material with a `consult_id` and member verdict content. |
| `.cortexkit/alfonso/{drafts,context,prompts,patches}/` | **HIGH** | DELETE | Draft/refire lineage, private sibling source reads, campaign decisions, patch provenance, and peer-agent instructions. |
| `bench/spikes/unified-rt/{QUANT-DECODE.md,METAL-STEP.md,F16-SERVING.md,DECODE-WAVE1.md}`; `bench/NOTES.md` | **HIGH** | REDACT | Retain campaign `ct_...` and worker `bg_...` identities and descriptions of private campaign outcomes. |
| `docs/campaign-context-repro.md`; `docs/design-stt-bias.md`; `docs/study-2026-08-24-speculative-serving-sweep.md` | **HIGH** | DELETE/REDACT as listed | Include controller campaign paths/IDs or named Mason/agent process details. |
| `crates/synapse-module/owned-decode-manifests/{slice-plan-v1.json,runtime-bound-decode-cutover-epic-manifest-v1.json}` | **HIGH** | REDACT | Shipped manifest data contains a private `work_item` campaign identity. |

### 5. Third-party private IP and private sibling projects

| Finding path(s) | Severity | Disposition | Detail |
| --- | --- | --- | --- |
| `corpus/aft-chunks.jsonl` | **CRITICAL** | DELETE | Complete AFT chunk corpus (15,271 chunks), not just a reference to AFT. |
| `bench/campaign/metal-embed-fixtures/` | **CRITICAL** | DELETE | Derived AFT/codegraph corpus and vector/reference fixture set. |
| `evidence/nonmac/preparation/staged/refs/{qwen3-corpus-400.jsonl,modernbert-corpus-400.jsonl,minilm-corpus-1000.jsonl,modernbert-ort-400-vectors.jsonl}`; `evidence/nonmac/**/rerank-input-v1.json` | **CRITICAL** | DELETE | AFT source identifiers/text/vectors retained in release evidence. Covered by the `evidence/` delete. |
| `tools/gather-distill/data/students/` | **CRITICAL** | DELETE | `codegraph-explore-raw.jsonl` identifies `cortexkit/subconscious` and includes private exploration/source outputs; derived `rows`, package metrics, scores, and ladder must be removed with it. |
| `tools/classify-distill/contract/` and its consumers | **CRITICAL** | DELETE | Vendored line-preserving ALF contract code from private Alfonso source. |
| `Cargo.toml`; `.github/workflows/ci.yml`; `operations/ane-prefill-split/stage_m1_cert.sh`; `bench/campaign/provision-cuda-rig.sh` | **HIGH** | REDACT/REVIEW | Direct private dependency/checkout references to `subconscious`, `commons`, and related CortexKit packages must be replaced with public releases or removed. |
| `bench/spikes/stt-bias/evalkit/{terms.jsonl,utterances.jsonl}` | **MEDIUM** | REDACT | Internal sibling vocabulary labels/source text require provenance cleanup even though they are not a source dump. |

### 6. Process-artifact disposition

| Tree | Severity | Disposition | Assessment |
| --- | --- | --- | --- |
| `.cortexkit/` | **CRITICAL** | DELETE | **No file is public-appropriate.** Campaign control/configuration, prompts, context, drafts, outputs, and patches are a private operational system. |
| `evidence/` | **CRITICAL** | DELETE | **No file is public-appropriate as raw evidence.** Publish a freshly written, sanitized benchmark report only if needed. |
| `operations/` | **HIGH** | DELETE | Internal remote enablement/staging runbooks; re-author public documentation from first principles if necessary. |
| `.alfonso/` | **CRITICAL** | DELETE | Private-spike source and personal model-cache defaults. |
| `results/vulkan-wave2/` | **MEDIUM** | REVIEW then KEEP | The seven files did not match the sensitive indicator set; validate metadata before retaining. |

### 7. Documentation audit

**Private-by-nature: delete.** The delete list covers these files because their
value is an internal operation/campaign record, not reusable public documentation:

- `docs/campaign-context-repro.md` — exact M1 controller/reproduction paths,
  `ct_...` identity, remote invocation, and fleet process behavior.
- `docs/deployment-owned-decode-approvals.md` — private fleet deployment and
  release-operator procedure.
- `docs/diag-ane-prefill-divergence.md` and
  `docs/diag-ane-prefill-m1-authority.md` — named M1 host, runner, lock, sibling
  checkout, and certification evidence.
- `docs/dogfood-ane-lane.md` and `docs/dogfood-p0.md` — real fleet connection,
  worktree path, operator transcript, and deployment state.
- `docs/eval-lfm25-vl-3b.md` — internal application/UI capture observations
  including Magic Context and a private PR/workflow context.
- `docs/lab-roadmap-orx-findings.md` — internal ALF/Athena campaign process and
  private operating strategy.
- `docs/mtp-challenge-m1-loop.md` — SSH host/alias, `[bench-user-home]` layout, and
  private staged model location.
- `docs/prod-nonmac-lanes.md` and `docs/prod-rerank-lane.md` — exact rented/Ally
  profiles, fleet daemon operations, personal local paths, and operator output.

**Redactable design material.** Retain only after the specific private internals
listed in the disposition table are removed:

- `docs/design-remote-gateway.md` — `vault-handles.json`/private base reference.
- `docs/design-synapse-module.md` — `commons`, `.cortexkit`, and internal release
  history references.
- `docs/design-stt-bias.md` — Mason-in-flight/peer process references and sibling
  vocabulary provenance.
- `docs/study-2026-08-24-speculative-serving-sweep.md` — private Alfonso draft
  link and agent-process claims.
- `docs/decision-1-runtime.md` — campaign/lab results and private corpus
  provenance should be replaced by publicly reproducible evidence.

`docs/audit-surfaces-loops.md`, `docs/design-worker-protocol.md`,
`docs/diag-ane-prefill-direct-kv-layout.md`, `docs/diag-ane-prefill-shm-handoff.md`,
`docs/diag-ane-prefill-ttft.md`, `docs/eval-minicpm-v46.md`,
`docs/inline-embed-batch-throughput.md`, `docs/leap-finetune-assessment.md`,
`docs/plan-ane-prefill-split-production.md`, `docs/reference-mtp-depth-controller.swift`,
`docs/study-mtplx.md`, `docs/study-tokenspeed.md`, `docs/vlm-grounder-acceptance.md`,
`docs/wave-11-windows-transport-notes.md`, and `docs/wire-contract-v1.md` did not
match the named machine/IP/personal-path/private-sibling/peer-process indicators
used for this audit. They are **KEEP pending normal publication and license
review**. `docs/synapse-performance-matrix.xlsx` remains the separate manual-review
exception.

### 8. Workflow and CI audit

`.github/workflows/ci.yml` has no committed secret value, but it is a public
release blocker: it requests an app token using `CK_CI_APP_ID` and
`CK_CI_APP_PRIVATE_KEY`, sets the private `cortexkit` owner, checks out
`cortexkit/subconscious` and `cortexkit/commons`, documents sibling path
requirements, and mentions the former self-hosted `macos-metal` M1 runner. Use a
new public workflow with repository-scoped permissions and public/package
Dependencies; do not merely rename the secrets.

## History note

A targeted history spot-check found **three current high-risk findings with verified history presence**: the literal M1 password appears in 11 commits (`git log --all -S`), the `[lan-ip]` LAN endpoint appears in 12 commits, and `aft-chunks.jsonl` appears in 11 commits. The requested deleted-name check, `git log --all --diff-filter=D --name-only | grep -iE 'keepalive|\.ssh|password'`, returned **0** paths, so there is no known-deleted `[bench-host-alias]` keepalive/SSH/password-named artifact to count from that probe. This is a lower bound, not a full historical secret scan: the three-for-three positive sample means a fresh public history (or a complete, independently verified history rewrite) is required rather than a tip-only scrub.
