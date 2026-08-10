# Blind audit: runtime-bound decode cutover records

## Scope and verdict

Audited `11d4a15..HEAD` in the requested records-epic surfaces, with the pinned contracts in `crates/synapse-module/owned-decode-manifests/` treated as normative. I found four S1 correctness findings and five S2 robustness/test-integrity findings. The serving path is store-resident and fail-closed on ordinary store errors, but approval and evidence TOCTOUs, cross-class certification reads, and migration validation gaps prevent the epic claims from being accepted as written.

## Findings

### S1 — A committed disable or emergency rollback can lose the race to an already-resolved request

`owned_decode_admission_matching` reads and validates the enabled approval with the current cert row in one fenced transaction (`crates/synapse-module/src/store.rs:2790-2869`), but `owned_decode_environment` reduces that result to booleans before queueing and worker preparation (`crates/synapse-module/src/lib.rs:6375-6418`). At the final dispatch boundary, routing re-reads only `profile_activation_epoch` (`crates/synapse-module/owned-decode-routing/mod.rs:661-680`); it does not re-read the approval identity, enabled bit, semantic digest, or generation. `disable_approval` and `emergency_rollback` can therefore commit after admission resolution but before the first worker frame without invalidating the stamped request (`crates/synapse-module/src/store.rs:2197-2214`, `2231-2277`). This is not the documented disabled-versus-`NotPreferred` diagnostic collapse: the owned worker can actually dispatch after the row is durably disabled.

**Reproduction:** Hold a request after `owned_decode_environment` returns serving=true, commit `approvals.disable` (or emergency rollback) without rotating the profile, then release the request; the epoch-only check still passes and the owned dispatch runs.

### S1 — The shared certification reader defeats embedding/rerank class isolation

Probe writes correctly select `CertificationClass::Embedding` or `CertificationClass::Rerank` and refuse generation on that path (`crates/synapse-module/src/lib.rs:9858-9885`). The serving reader does not accept an expected class, however: `get_cert_row` selects any certified row whose class is in `('measured_owned_decode', 'embedding', 'rerank')` using only `(key_hash, fingerprint)` (`crates/synapse-module/src/store.rs:2981-2999`). `ensure_model_certified` then uses that unscoped result for the requested model (`crates/synapse-module/src/lib.rs:8400-8432`). Thus an embedding row can certify a rerank lane (or vice versa) when their fingerprints coincide, contrary to the class-specific identities frozen in `cert-row-schema-manifest-v1`.

**Reproduction:** Store a certified embedding row for machine key `H` and fingerprint `F`, configure a rerank model with fingerprint `F`, and call rerank; `get_cert_row(H,F)` returns the embedding row and the rerank path passes certification without a rerank probe.

### S1 — Constrained-runtime evidence is matched as a subset instead of byte-exact equality

The pinned admission/probe contract requires byte-exact comparison of ordered constrained identity sets. The implementation instead returns true when every requested identity is present in the certified row, allowing extra certified identities (`crates/synapse-module/src/store.rs:5130-5136`); owned admission relies on that predicate (`crates/synapse-module/src/store.rs:2851-2857`). The unit test explicitly blesses an empty runtime set against a row containing `constraint-a` (`crates/synapse-module/src/store.rs:6642-6647`). This lets a row survive a runtime-set contraction or otherwise certify a tuple different from the captured tuple.

**Reproduction:** Store a cert row with constrained identities `['A','B']` and resolve admission with `['A']` (or `[]`); `owned_decode_admission_matching` returns `Some` instead of rejecting the unequal set.

### S1 — Migration's “all-four catalog” check validates names, not canonical owned-decode entries

The migration contract requires each source identity to resolve to exactly one validated production `CatalogEntry`. Migration only executes `SELECT COUNT(*) FROM models WHERE model_id = ?` (`crates/synapse-module/src/store.rs:1811-1823`); it does not validate `engine`, `task`, canonical fields, aliases, or the catalog-derived decode fingerprint. The direct enable path demonstrates the missing minimum check by requiring `engine = 'owned-metal-decode' AND task = 'generate'` (`crates/synapse-module/src/store.rs:1535-1545`). Because `upsert_model` can later replace the configuration behind a stable `model_id` (`crates/synapse-module/src/store.rs:1350-1376`), a non-owned placeholder can acquire a migrated approval that a later owned entry inherits.

**Reproduction:** Insert four `models` rows under the pinned seed IDs with `engine='ort'` and `task='embed'`, run migration, and observe `outcome=applied` with enabled approvals instead of `unmappable_identity`/`invalid_seed`; then upsert the real owned configurations under those IDs.

### S2 — Terminal probe identities are resolved before, not inside, the fenced write transaction

At probe completion, the code reads profile state and re-derives catalog, decode, processing, runtime, constraint, and worker identities before calling the store (`crates/synapse-module/src/lib.rs:9786-9825`; the mutable catalog read is at `6444-6487`). `store_owned_decode_cert_row_if_current` compares the two caller-supplied tuples before opening the fenced transaction, then re-resolves only the persisted hash and epoch inside it (`crates/synapse-module/src/store.rs:2483-2512`, `2513-2537`). A catalog/runtime mutation can therefore land after the “terminal” resolution and before the cert insert. This violates `admission-probe-boundary-contract-v1`, which requires every runtime-derived member to be re-resolved in the terminal fenced transaction, and creates an A→B→A stale-evidence window.

**Reproduction:** Pause after `owned_decode_probe_match_inputs` returns, replace the runtime catalog entry, then let the cert transaction commit; the old terminal tuple is written because only profile hash/epoch are re-read.

### S2 — The revisioned owned hash silently replaced the legacy hash for non-owned state

Initialization assigns `machine_profile_hash`—the field used by legacy embedding/rerank certification, performance rows, and knob selection—to `revisioned_machine_profile_hash`, passes the same revisioned value into remote runtime compatibility, and computes the legacy hash only for health serialization (`crates/synapse-module/src/lib.rs:1405-1432`). Non-owned lookups continue to consume `state.machine_profile_hash`, for example knob selection at `crates/synapse-module/src/lib.rs:2896-2918` and certification at `8400-8429`. This contradicts `machine-profile-hash-boundary-manifest-v1`, which freezes `MachineProfile::hash()` for non-owned historical state and changes only the owned-decode boundary. Existing non-owned rows become invisible after upgrade, albeit fail-closed.

**Reproduction:** Create an embedding/rerank certification or knob row under `MachineProfile::hash()` with the pre-epic binary, upgrade without changing the machine profile, and observe the new binary query the revisioned hash and return probe-required/not-certified.

### S2 — Operator health and probe report can call rejected owned evidence “serving” or “certified”

`storage_health_inputs` labels an enabled approval `serving` whenever a current identity row exists, without checking complete G-DEC evidence, processing/runtime/worker equality, constrained identities, or the semantic serving predicate (`crates/synapse-module/src/store.rs:3606-3625`). Per-lane module health likewise marks owned lanes certified from the raw identity lookup (`crates/synapse-module/src/lib.rs:11067-11083`). Separately, `probe.report` obtains rows through `lane_measurement_rows` (`crates/synapse-module/src/lib.rs:10655-10675`), whose generic store query omits owned `profile_activation_epoch` and `model_id` (`crates/synapse-module/src/store.rs:2981-2999`), so an A/epoch-1 row can appear current after A→B→A reaches epoch 3. Serving itself still uses the stricter predicate; this is an operator-diagnostics defect, not a demonstrated gate bypass.

**Reproduction:** Insert a `status='certified'` owned row at the current identity with incomplete G-DEC entries and read health; `approval_certification_outcomes[].admission` is `serving` while `owned_decode_admission_matching` returns `None` (or rotate A→B→A and observe epoch-1 evidence reported current at epoch 3).

### S2 — Migration does not bind the parsed seed bytes to the recorded seed digest or exact four entries

The current checked-in seed's SHA-256 does equal `APPROVAL_MIGRATION_SEED_DIGEST`, but migration never computes that hash. It performs broad shape/revision checks (`crates/synapse-module/src/store.rs:1649-1677`), accepts any four structurally plausible entries and any numeric D-009 source indexes (`1686-1809`), then records the hard-coded digest (`1834-1835`, `1936-1946`). Consequently a changed seed with the same revision can be applied to a fresh store while the marker falsely records the old pinned digest. This makes the contract's byte-identical rerun rule and “verify the four expanded entries” guard vacuous.

**Reproduction:** Change an entry's semantic value (for example `grammar_enabled`) without changing the revision or hard-coded digest, rebuild, and migrate a fresh store; migration returns `applied` and records the unchanged digest instead of `invalid_seed`.

### S2 — Mandatory concurrency and pre-enable proofs are serial or absent

The frame-boundary test starts with the epoch reader already set to 2 rather than using a barrier to rotate between admission and dispatch (`crates/synapse-module/owned-decode-routing/mod.rs:967-986`). The stale-probe store test directly supplies a changed terminal tuple in one thread (`crates/synapse-module/src/store.rs:6868-6873`), and the rotation test invokes observers serially (`6749-6784`); none proves two observers, rotation-during-probe, or approval mutation at the dispatch boundary. The real checkpoint battery enables the approval before probing and makes its first serving request only afterward (`crates/synapse-module/tests/skeleton_e2e.rs:3480-3500`, `3512-3526`), so it cannot detect a cert-only serving bypass before enablement. These fall short of the barrier-controlled and negative-mutation families required by `owned-decode-test-manifest-v1` and `admission-probe-boundary-contract-v1`.

**Reproduction:** Remove/neutralize the approval-enabled gate and run only the checkpoint helper; it still enables before its sole serve assertion, so the missing pre-enable refusal never turns the battery red.

## Verified claims and negative checks

- No current Rust serving/runtime source contains or loads `load_checked_in_cutover_records`, `D009CutoverRecords`, `D009CutoverRecord`, `CutoverRecord`, `disable_profile`, or `owned_decode_cutover_for_test`. The retained D-009 JSON is referenced only as provenance by the migration seed; runtime migration embeds `migration-seed-manifest-v1.json`, not the retired records.
- Approval rows are store-resident. `ApprovalRow::expected_digest` covers all nine semantic fields in the pinned order (`crates/synapse-module/src/store.rs:771-795`), and ordinary approval loads/admission reject digest mismatch.
- Ordinary store failures in approval/certification resolution collapse to non-serving (`.ok().flatten()`), and the pre-frame persisted epoch recheck also fails closed on read errors. The known disabled→`NotPreferred`→`owned_decode_not_certified` collapse is diagnostic/refusal fidelity, not the gate bypass described above.
- Epoch rotation uses the required hash+epoch predicates and writes profile state plus parent/children in one transaction (`crates/synapse-module/src/store.rs:3250-3358`). The supported engine-identity trigger's pinned legacy and revisioned hashes reproduce exactly.
- Class-scoped probe writes use embedding/rerank classes and explicitly refuse generation on that helper path; owned generation uses the dedicated epoch-scoped write.
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo test -p synapse-module --lib` passed: 317 passed, 3 ignored.
- Targeted profile-rotation, dual-profile-guard/stale-probe, and migration store tests passed. The ignored checkpoint battery was not run, as requested.
