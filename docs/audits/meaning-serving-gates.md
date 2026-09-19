# What the serving-gate values MEAN

An audit of the code that decides whether a request may execute on a lane, asking
of each value not *what is it* but *what must a reader know to act on it
correctly*, and — the actual question — **where that knowledge comes from**.

I did not write this code and did not read the design documents before reading
the source, on the theory that a reader who already knows what a value is *for*
cannot see which values fail to say so. Where a document did turn out to state a
meaning, I cite it; in two cases the document and the code disagree, and those
are the most interesting entries here.

Ordered by **how confident I am that I read the meaning correctly**, not by
severity. Confidence drops as you go down. Everything carries `file:line`.
Claims I could not support are labelled *suspicion*, not finding.

---

## Index

| # | Value | Confidence | Where its meaning comes from | Anything break? |
|---|---|---|---|---|
| 1 | `certified` in `models.list` | High | Doc **contradicts** code | Yes — documented guarantee is false |
| 2 | `evidence["g_dec"]` → `gates_complete` | High | Inferred from the one producer | Yes — 12 names, 1 measurement |
| 3 | `serving_admission` (`probe.report`) | High | Inferred from two disjoint tables | Yes — reports a gate it does not read |
| 4 | `certification_stale` on owned lanes | High | Inferred; contradicted by aliasing | Yes — constant `false` |
| 5 | `SYNAPSE_OS_BUILD_OVERRIDE` | High | Nowhere; inferred from one test | Yes — can re-open a fail-closed lane |
| 6 | `revoked` / `ServingBoundaryOutcome` | High | Module doc, minus one wrong comment | Partly — weaker than the word |
| 7 | `admission_boundary_is_current()` | High | Stated in code, upheld non-locally | No, today |
| 8 | `scheduler_evidence_committed` | High | Doc comment overstates by one word | No |
| 9 | `current_epoch_valid`, `wire_bindings_literal` | High | Inferred; tautologies at the call site | No |
| 10 | `ane_subtype: None` | Medium-high | Doc comment **and** code disagree | No |
| 11 | `current_unified_memory_bytes() -> None` | Medium-high | Inferred from the one consumer | No (meaning); suspicion on hangs |
| 12 | `certified: false` on store error | Medium | Doc licenses it; sibling comment forbids it | No |
| 13 | `minimum_unified_memory_bytes` | Medium | Inferred; producer field is named differently | Suspicion only |
| 14 | `DECODE_WORKER_ENGINE` vs the literal | Medium | Inferred | No, today |

Then: **[values whose meaning I could not establish](#values-whose-meaning-i-could-not-establish)**.

---

## 1. `certified` in the `models.list` descriptor

**What it is.** `Option<bool>` — `Some(true)`, `Some(false)`, or omitted
(`crates/synapse-module/src/lib.rs:14745-14779`).

**What it means (as documented).** `docs/wire-contract-v1.md:140-144`:

> `certified` (boolean, optional) — serving certification flag **sourced directly
> from the serving admission predicate**. A consumer will never see
> `certified: true` for a lane that would refuse to serve due to missing or
> uncertified machine profile evidence.

**What it means (as implemented).** For an owned-decode lane it means: *a
`measured_owned_decode` certification row exists for the current revisioned
profile hash, the current activation epoch, this model id, this decode
fingerprint, the current evidence-schema revision, and the **empty** constraint
identity set, and that row's status is `certified`*
(`crates/synapse-module/src/lib.rs:14747-14760`, executing the query at
`crates/synapse-module/src/store.rs:4378-4415`).

**Where the meaning comes from.** The documented meaning is stated. The
implemented meaning had to be inferred from the query, and the two are not the
same value. The serving admission predicate is
`owned_decode_routing::lane::serving_predicate`
(`crates/synapse-module/owned-decode-routing/lane.rs:298-322`); `models.list`
never calls it, and never reads the approval row that the predicate's first two
arms depend on (`lane.rs:299-301`).

**What breaks if a reader assumes the documented meaning.** The guarantee
"a consumer will never see `certified: true` for a lane that would refuse to
serve" is false in at least two ways:

- *Approval absent or disabled.* `serving_predicate` returns
  `ServingRefusal::CutoverDisabled` when no approval row exists or it is
  disabled (`lane.rs:299-301`), which is exactly the state a lane is in after a
  successful probe and before `approvals.enable`
  (`crates/synapse-module/src/lib.rs:13684-13718`). Certification and approval
  are independent writes. What happens next depends on the request shape, and
  neither outcome matches `certified: true`: an `owned_only` request — which is
  every `owned_decode.decode` (`lib.rs:4050`) — or a constrained request is
  refused outright (`lane.rs:191-211`, `215-227`), while a substitutable
  unconstrained `microllm.oneshot` with a llama lane configured is quietly
  served by **llama instead** (`lane.rs:229-240`). So `certified: true` means
  either "this lane refuses" or "a different lane answers", and the descriptor
  cannot distinguish them.
  **Reproduction:** probe an owned-decode lane to `certified`, do not call
  `approvals.enable`; `models.list` reports `certified: true` while
  `owned_decode.decode` on that session refuses. `probe.report` on the same lane
  correctly reports `serving_admission: "disabled"`,
  `serving_admission_reason: "approval_absent"` (`lib.rs:14049`), so the two
  surfaces of the same daemon disagree about the same lane in the same second.
- *Constrained requests.* The lookup passes `&[]` for the constraint runtime
  identities (`lib.rs:14755`), which hashes to the digest of the empty list
  (`store.rs:6891-6895`) and therefore selects only the **unconstrained** row.
  A grammar-constrained request is matched against a different digest
  (`store.rs:4639-4648`, `4706`), so `certified: true` does not mean the lane is
  certified for the constrained shape a caller may be about to request.

**Confidence.** High on the mechanism; I traced the producer and both consumers.
I have not executed the reproduction.

---

## 2. `evidence["g_dec"]`, consumed as `gates_complete`

**What it is.** A JSON array of twelve objects `{id, status, manifest_revision}`
with `id` in `G-DEC-01 … G-DEC-12` and `status` the literal string `"passed"` —
constructed unconditionally, from a range, with no gate outcome as input
(`crates/synapse-module/src/lib.rs:9322-9334`).

**What it means (as a consumer must read it).** The store reads it back and
requires all twelve ids present, each `"passed"`, each stamped with the current
manifest revision (`crates/synapse-module/src/store.rs:7621-7648`); the result
becomes the final conjunct of the fenced admission match
(`store.rs:4738-4741`) and surfaces in the serving predicate as `gates_complete`
(`lib.rs:9270`, consumed at `lane.rs:307`). Every name in that path says the same
thing: *the twelve acceptance gates passed for this row*.

**What it actually means.** *The generate structural-band probe passed.* The
array is attached only inside `if passed` (`lib.rs:12765-12771`), where `passed`
is one boolean: fixture token-exactness plus one constrained-schema check
(`lib.rs:12726-12729`). Twelve gate identities carry the information content of
one measurement.

**Where the meaning comes from.** Inferred. There is no comment at
`lib.rs:9322` and none at the field. The real gate runner exists —
`GateRunner::run_all` evaluates twelve genuinely different gates
(`crates/synapse-module/owned-decode-certification/gates.rs:226-310`) — and is
called from nowhere outside its own test module (`gates.rs:1458`); `release_ready`
(`gates.rs:186-203`) likewise.

**What breaks if a reader assumes the wrong meaning.** Two concrete things:

- The gate module's own documentation states that G-DEC-11 and the
  scheduler-dependent part of G-DEC-12 *cannot* be `Passed` — they report
  `Blocked` until a named external commitment lands, and `release_ready` is false
  while any gate is blocked (`gates.rs:1-9`,
  `crates/synapse-module/owned-decode-certification/mod.rs:24-27`). The evidence
  the serving path writes and then re-reads says `G-DEC-11: passed`. The same
  identifier carries opposite claims in two files.
- An operator or auditor reading a certification row's evidence, or the
  `complete_g_dec_evidence` check, will believe twelve independent gates were
  evaluated on this machine for this artifact. Re-running the eleven unrelated
  gates cannot change the row.

**Reproduction.** Read any `measured_owned_decode` row written by a successful
probe: its `evidence.g_dec` is the constant array regardless of anything
`GateRunner` would have concluded. I assert the constancy from the source
(`lib.rs:9322-9334` has no input parameter); I did not run a probe.

**Confidence.** High. The producer is a pure function of a range and a constant.

---

## 3. `serving_admission` in `probe.report`

**What it is.** `Option<&'static str>` — `Some("enabled")`, `Some("disabled")`,
or `None` (`crates/synapse-module/src/lib.rs:14034-14051`), paired with
`serving_admission_reason`. Documented at `docs/wire-contract-v1.md:221-225` as
"`enabled` or `disabled` for owned-decode lanes, and `null` for lanes without
approval-backed admission".

**What it means.** It means: *the row in the `approvals` table keyed by
`(model_id, decode_fingerprint)` is enabled, and an unconstrained certification
row exists for the current profile* (`lib.rs:14213-14221`, feeding
`evidence_certified` from `measurements.current_certification.is_some()`).

**Where the meaning comes from.** Inferred from the call site. The projection
function carries a good doc comment about what it deliberately *excludes*
(worker residency, `lib.rs:14025-14033`) — a well-documented negative — but
nothing says which of the daemon's two approval tables it reads.

**What breaks if a reader assumes the wrong meaning.** There are two approval
systems, with different keys, different state vocabularies, and disjoint control
surfaces:

| | `approvals` | `serving_approvals` |
|---|---|---|
| Key | `(model_id, decode_fingerprint)` | `catalog_fingerprint` |
| State | `enabled INTEGER` (2 values) | `state TEXT` in `enabled`/`disabled`/`revoked` (3 values) — `store.rs:651` |
| DDL | `store.rs:515-531` | `store.rs:646-665` |
| Operator writes | `approvals.enable` / `approvals.disable` / `approvals.emergency_rollback` (`lib.rs:13684`, `13720`, `13747`) | `owned_decode.disable` / `owned_decode.revoke` (`lib.rs:4543`, `4599`) |
| Gates | `serving_predicate` → `microllm.oneshot` and `owned_decode.decode` (`lib.rs:9257-9283`, reached from `8152` and `4052`) | `serving_admission_material` → `owned_decode.admit_session` (`lib.rs:3271-3393`, called at `3420`) |
| Reported by `serving_admission` | **yes** | **no** |

So `serving_admission` is named after the gate it does *not* read. Concretely:

- `owned_decode.revoke(catalog_fingerprint)` sets `serving_approvals.state =
  'revoked'` (`store.rs:4003-4060` via `lib.rs:4609`), after which
  `owned_decode.admit_session` refuses with `ArtifactRevoked`
  (`lib.rs:3302-3307`). `probe.report` keeps reporting
  `serving_admission: "enabled"`, because nothing in the projection can express
  `revoked` — its input type is `Option<(bool, Option<String>)>`
  (`lib.rs:14037`), a two-state value standing in for a three-state one.
- Symmetrically, `approvals.disable` flips `serving_admission` to `"disabled"`
  while `owned_decode.admit_session` continues to admit sessions.

**Reproduction.** Enable an approval, run `probe.report` (expect
`serving_admission: "enabled"`), call `owned_decode.revoke` with the lane's
catalog fingerprint, run `probe.report` again. I did not execute this; the claim
rests on `owned_decode_revoke` (`lib.rs:4599-4646`) writing only
`serving_approvals` and `serving_admission_projection` (`lib.rs:14034-14051`)
reading only `get_approval`.

**A caveat I want to be explicit about.** I could not establish whether a
`catalog_fingerprint` and a `decode_fingerprint` ever denote the same artifact
in a live store — they are built by different constructions (`catalog_fingerprint`
is the base+MTP-head+depth-gate unit identity,
`crates/synapse-module/owned-decode-certification/certification_unit.rs:90-100`;
`decode_fingerprint` is the decode identity). If they never coincide, the two
systems govern disjoint populations and the surface is merely misnamed rather
than misreporting. That distinction changes how bad this is and I could not
settle it from the code.

**Confidence.** High that the two tables are disjoint and that
`serving_admission` reads only one of them. Medium on the operational impact,
for the caveat above.

---

## 4. `certification_stale` for an owned-decode lane

**What it is.** `bool`, computed as `current_certification.is_none() &&
latest_certification.is_some()` (`crates/synapse-module/src/lib.rs:13943`).

**What it means.** *Evidence for this lane exists, but not for the machine
profile the module is currently running under* — i.e. the lane was certified and
then the profile rotated. That is the distinction an operator needs to tell
"never probed" from "probed, then the machine changed".

**Where the meaning comes from.** Inferred from the expression and from
`probe.report`'s use of it (`lib.rs:14289`).

**What breaks.** For owned-decode lanes the value is a constant `false`. The
decode branch assigns the same `Option` to both slots:

```
(current_certification.clone(), current_certification, current_probe.clone(), current_probe)
```
(`lib.rs:13880-13885`), so `certification_stale` evaluates `x.is_none() &&
x.is_some()` — unsatisfiable. The same aliasing makes `probe_stale`
(`lib.rs:14196-14197`) a constant `false`, which in turn makes the `"stale"`
field of the per-row report (`lib.rs:14139`) a constant `false` for decode lanes.
After a rotation the decode lane reports `certification: null` (every arm of the
`.or()` chain at `lib.rs:14198-14204` is `None`) — which is also what a
never-probed lane reports. The two states are indistinguishable on this surface
for exactly the lane class where rotation matters most.

For embedding and rerank lanes the same field is meaningful, because that branch
fetches `latest_*` from a genuinely different query (`lib.rs:13896-13904`). The
end-to-end test that covers staleness
(`crates/synapse-module/tests/skeleton_e2e.rs:2537-2595`) exercises a MiniLM
embedding lane, so it passes through the meaningful branch and cannot observe
the decode branch's contradiction.

**Reproduction.** Certify an owned-decode lane, restart with
`SYNAPSE_OS_BUILD_OVERRIDE` set to rotate the profile, call `probe.report`: the
decode lane shows `certification_stale: false` and `certification: null`, where
the embedding lane in the same report shows `certification_stale: true`.

**Confidence.** High — it is a syntactic contradiction in one expression, and I
read both branches.

---

## 5. `SYNAPSE_OS_BUILD_OVERRIDE`

**What it is.** An environment variable read in the production boot path; when
set to a non-empty trimmed string it replaces `MachineProfile::os_build` before
any hash is computed (`crates/synapse-module/src/lib.rs:192`, `2979-2987`,
applied at `1972-1974`).

**What it means.** I could not establish this from the code, and I want to say so
plainly. Two candidate meanings:

- *a test-only seam*, which is how the only in-repo use reads
  (`crates/synapse-module/tests/skeleton_e2e.rs:2554`, named
  `os_build_override_marks_probe_rows_stale_in_report_and_status`); or
- *a supported operator knob* for pinning identity across an OS update.

Nothing distinguishes them. The constant has no doc comment
(`lib.rs:192`), the function has none (`lib.rs:2979`), it is not `#[cfg(test)]`,
and it is not routed through the config's `dev` section, which is where this
codebase puts deliberately unsafe switches (`config.dev.alias_admin_enabled`,
`lib.rs:2049`).

**What breaks if a reader assumes the wrong meaning.** This override substitutes
precisely the value that `synapse-core` refuses to substitute. That refusal is
the most carefully argued comment in the subsystem
(`crates/synapse-core/src/machine_profile.rs:17-24`): a placeholder `os_build`
"rotates the profile, fails every certified lane closed, and rotates BACK on the
next boot that happens to succeed", and four tests exist to keep it a refusal
(`machine_profile.rs:412-428`, `432-460`, `473-536`, `597-608`). The refusal
holds inside `collect_base_profile`; `machine_profile_with_overrides` then runs
on its result.

The direction that matters is not the one the test exercises. Setting the
override to a *novel* string rotates the hash and fails closed — annoying, safe,
and visible. Setting it to the *previous* machine's real `os_build` after an OS
update restores the old hash, and every certification row recorded before the
update matches again: `serving_admission_material`'s machine-tuple check
(`lib.rs:3332-3339`) compares against `state.machine_profile.os_build`, which is
the overridden value, and the routing path's fenced match compares
`state.revisioned_machine_profile_hash` (`lib.rs:9169`, `store.rs:4656-4657`),
which is derived from the overridden profile. An environment variable can
therefore re-open lanes that the identity gate had closed, silently, with the
evidence rows still claiming they were measured on this machine.

**Reproduction.** Certify a lane; change `os_build` (an OS update, or a first run
with the override set to value A); observe the lane fail closed; restart with the
override set to the original build string; the lane serves again from the
pre-update evidence. I did not perform this.

**Confidence.** High that the override is applied to the hashed profile in
production and has no stated meaning. The "operator pins the old build" story is
a scenario, not an observed incident.

---

## 6. `revoked`, and `ServingBoundaryOutcome::Terminated`

**What it is.** `ServingApprovalState::Revoked`
(`crates/synapse-module/src/store.rs:1238-1242`) and the boundary outcome
`Terminated { terminal_reason, tokens_emitted, unload_artifact }`
(`store.rs:1426-1438`).

**What it means.** Stated well, in `crates/synapse-module/src/rollback.rs:40-41`:
emergency revoke "fences admissions, invalidates retained states, and requests
terminal accounting from active sessions **at their next committed boundary**".

**Where the meaning comes from.** Stated in the module doc comment. This is one of
the places where the code does tell you.

**What breaks if a reader assumes the stronger, natural-language meaning** — "a
revoked artifact stops executing":

- Revocation fences *admission* and truncates *emission*. It does not prevent one
  further complete worker execution. `owned_decode.decode` generates the whole
  response before any boundary is consulted: `route_owned_decode_wire` returns a
  finished `worker_stream.generated_token_ids` (`lib.rs:4052-4059`, length-checked
  at `4073`), and only then is the first `commit_serving_session_boundary` issued
  (`lib.rs:4178`). A session that was **idle** when revoke ran is not touched by
  revoke's scheduler sweep, which only visits sessions with an `active_request`
  (`lib.rs:4623-4639`), and its in-memory record is not marked closed, so its next
  `owned_decode.decode` passes the idle check at `lib.rs:3992` and runs.
- `tokens_emitted` does not mean "tokens committed before the revoke". The
  terminating transaction reads the approval, sees `Revoked`, sets the session
  terminal, **and writes the new committed count in the same statement**
  (`store.rs:3905-3929`). So the first quantum generated after revocation is
  committed and published as a `Progress` frame (`lib.rs:4190-4200`).

The comment at `lib.rs:4187-4189` states the opposite of what the store does:

> // The store committed this exact prefix before revocation.

It did not. It committed the prefix in the transaction that *observed* the
revocation. The behaviour is the documented one; the comment explaining it is
wrong, which is worse than no comment, because a later reader will use it to
reason about what is safe to publish.

**Confidence.** High. `store.rs:3915` writes `committed_token_count` on both
branches, before the `Terminated` return at `3933`.

---

## 7. `admission_boundary_is_current() -> bool`

**What it is.** `bool`
(`crates/synapse-module/owned-decode-routing/mod.rs:437-447`).

**What it means.** `false` means: *the persisted approval digest, generation, or
profile epoch no longer matches the snapshot taken at the fenced admission read,
or the read failed.* `true` means: *that comparison succeeded* — **or** *no
snapshot was ever attached*.

**Where the meaning comes from.** Stated in code, and stated honestly: the doc
comment says "Missing or failed reads fail closed" (`mod.rs:434-435`) and the
early return carries its own explanation — "Legacy/test environments have no
persisted admission snapshot; production environments that select owned always
attach one" (`mod.rs:438-441`).

**What breaks.** Today, nothing. This is the *partially sound check* shape: the
answer `false` is reliable, the answer `true` is ambiguous, and the calling
sentence acts on `false` — `if !env.admission_boundary_is_current() { return
Err(...) }` (`mod.rs:719-724`). Consuming the sound direction is the correct
choice and it was made.

What holds the ambiguous direction safe is an argument spanning two files, not a
type. The boundary is attached only inside `if let Some(admission) = admission`
(`lib.rs:9297-9310`), and seven arms of the serving predicate are set from
`admission.is_some()` (`lib.rs:9264`, `9266`, `9269-9274`), so a routing
environment cannot select the owned lane without a snapshot. That is true, and it
is nowhere checked. A future constructor of `RoutingEnvironment` that sets
`serving_enabled` by another route gets a silently passing gate. The type
`Option<AdmissionBoundarySnapshot>` can express "not checked"; the return type
`bool` cannot.

**Confidence.** High. I read the only two call sites.

---

## 8. `scheduler_evidence_committed`

**What it is.** `bool`, `matches!(status, Committed { .. })`
(`crates/synapse-module/owned-decode-certification/scheduler_evidence.rs:29-31`),
consumed as a serving-predicate arm (`lib.rs:9281`, `lane.rs:316`).

**What it means.** Its doc comment says: "Whether the numeric scheduler manifest
is **committed and executed**" (`scheduler_evidence.rs:27-28`).

**What it actually means.** *The checked-in manifest file declares a complete
evidence record.* `ingest_scheduler_evidence` validates a `SchedulerManifest`
value: production N is one of the candidates, `committed_n == production_n`,
`observed_n` recorded and equal, and several vectors non-empty
(`scheduler_evidence.rs:41-107`). Nothing in it observes a scheduler. The
manifest reaches it through `include_str!`
(`lib.rs:9226-9232`), so at this call site the arm is a compile-time constant of
the binary.

**Where the meaning comes from.** The doc comment — and it is one word wrong.
"Committed" is accurate. "Executed" is not: the function cannot distinguish a
manifest recording a real scheduler run from a manifest recording plausible
numbers. The emptiness checks (`sequence_traces`, `permit_events`,
`queue_depth`, …) are checks that the *file is filled in*, not that the runtime
did anything.

**What breaks if a reader assumes "executed".** Nothing at runtime — the arm is
`true` for every build shipping this manifest either way. It breaks a reviewer's
model of what the serving predicate is checking: one of its inputs is a property
of a file in the repository, not of this machine.

**Confidence.** High on the mechanism. I am reading "executed" strictly; the
author may have meant "the committed N is the one the runtime uses", which
`committed_n == runtime.production_n` (`scheduler_evidence.rs:59-62`) does check
for the manifest's own declared runtime record.

---

## 9. Two predicate arms that cannot be false

`ServingPredicateInputs` has seventeen fields
(`crates/synapse-module/owned-decode-routing/lane.rs:273-292`) — two decide the
cutover refusal (`lane.rs:299-301`) and the remaining fifteen form one
conjunction (`lane.rs:302-317`) — and a doc comment explaining why they live in one pure
function: "Keeping these checks in one pure function prevents a caller from
treating a partial match as admission" (`lane.rs:269-271`). At the one production
call site, two of those fifteen are constants.

**`current_epoch_valid`** — `state.profile_activation_epoch > 0`
(`lib.rs:9265`). The name says *the current epoch is valid*. What it checks is
that a value captured at boot is positive. It cannot be false: boot refuses when
the store's epoch is absent (`lib.rs:1976-1983`), the store rejects any
persisted state whose epoch is not positive (`store.rs:7740-7761`, and the DDL
`CHECK (new_profile_activation_epoch > 0)` at `store.rs:564`), and
`observe_profile` only ever writes `1` or `old + 1` (`store.rs:5120`, `5164`).
The check a reader would expect from the name — *the store's epoch still matches
the one this request was admitted under* — is real, but it is somewhere else:
inside the fenced read (`store.rs:4656-4657`) and at the dispatch boundary
(`store.rs:4520-4526`). The same dead comparison appears again at
`store.rs:4630` and `store.rs:4496`.

**`wire_bindings_literal`** — `wire_bindings_are_literal(&wire_bindings)`
(`lib.rs:9278`), where `wire_bindings` was constructed from string literals six
lines earlier (`lib.rs:9233-9240`) and the predicate only rejects two named
placeholder strings and the empty string
(`crates/synapse-module/owned-decode-certification/migration.rs:11-17`). The
function is correct and well documented for a manifest that is *loaded*; here it
is applied to a value the same function body just wrote.

**What breaks.** Nothing. This is a real answer and worth recording: the
predicate reads as fifteen independent facts, and a reader estimating "how much
evidence stands behind an owned-decode admission" from the struct will
over-count. Below the two constants, seven more fields
(`current_profile_matches`, `evidence_revisions_compatible`, `gates_complete`,
`processing_fingerprint_matches`, `runtime_config_digest_matches`,
`worker_path_matches`, `constrained_identities_match`) are all set from the
single expression `admission.is_some()` (`lib.rs:9264-9274`). That collapse is
*sound* — the fenced query really does compare all seven
(`store.rs:4727-4741`) — but it means a failed admission cannot say which arm
failed, and the struct's per-arm shape promises a granularity the producer does
not have.

**Confidence.** High.

---

## 10. `ane_subtype: Option<String>`

**What it is.** `Option<String>`; `Some("h17(map)")` for `apple m5 max`,
`Some("h16(map)")` for `apple m5` / `apple m4` / `apple m4 *`, `None` otherwise
(`crates/synapse-core/src/machine_profile.rs:222-234`). It is
`skip_serializing_if = "Option::is_none"` (`machine_profile.rs:63-64`,
`107-108`), so `None` leaves the field out of the hashed bytes entirely
(asserted at `machine_profile.rs:646-648`).

**What it means (as documented).** `machine_profile.rs:95-97`: "Derived from the
chip model rather than probed, and **legitimately absent on hardware without a
Neural Engine**."

**What it means (as implemented).** *This chip model is not in a three-entry
map.* Every Apple silicon Mac has a Neural Engine; `mapped_ane_subtype` returns
`None` for M1, M2, and M3 families, and the test pins exactly that behaviour for
`"Apple M3 Max"` (`machine_profile.rs:621`). So `None` conflates "no ANE
hardware" with "ANE present, subtype unknown to this build" — and the second is
the common case.

**Where the meaning comes from.** A doc comment, which states one of the two
meanings. The code states the other. The longer comment just above
(`machine_profile.rs:208-211`) explains *why* the value is mapped rather than
probed, and the `(map)` suffix is a provenance marker whose meaning is recorded
only in a test's name,
`ane_subtype_mapping_marks_chip_identity_provenance`
(`machine_profile.rs:611`).

**What breaks.** Nothing, and I checked rather than assumed. `ane_subtype` has
exactly two readers: the serialized profile that feeds the hash, and the rotation
field-diff (`crates/synapse-module/src/store.rs:7831-7833`). No gate branches on
whether an ANE exists. The consequence is confined to a future reader: adding an
M6 entry to the map rotates the identity hash of every M6 machine and fails its
certified lanes closed until a re-probe — correct behaviour, but a consequence
that the field's stated meaning ("absent = no Neural Engine") does not prepare
anyone for.

**Confidence.** Medium-high. I enumerated the readers with a repo-wide search
(37 matches across four files, all either the definition, the hash, the diff, or
tests).

---

## 11. `current_unified_memory_bytes() -> Option<u64>`

**What it is.** `Option<u64>` (`crates/synapse-module/src/lib.rs:3254-3269`).

**What `None` means.** The one consumer reads it as a single fact — "the module
cannot read the machine unified-memory capacity", refusing with
`UnsupportedMachine` (`lib.rs:3340-3345`). The producer folds five distinct
causes into that one `None`: not macOS (`lib.rs:3265-3268`), `sysctl` could not
be spawned or run (`lib.rs:3257-3260`), it exited non-zero (`3261`), its output
was not UTF-8 (`3262`), or it did not parse as a `u64` (`3263`).

**Where the meaning comes from.** Inferred entirely from the consumer. The
function has no doc comment.

**What breaks.** For the gate: nothing. It fails closed in every case, and the
serving-side comparison is `>=` against the certified floor
(`crates/synapse-module/owned-decode-routing/admission.rs:194-204`), so a
too-small or missing reading cannot admit. On non-macOS the whole
`owned_decode.admit_session` route refuses with a message that says the module
*cannot read* a value that platform does not have — a diagnosis that will send
someone looking for a broken `sysctl`.

**Suspicion (not a finding): this probe has no deadline.** It uses
`Command::output()` (`lib.rs:3259`), an unbounded wait. The same command,
`sysctl -n hw.memsize`, is run under a 2-second deadline in
`synapse-core`, with a comment giving the reason: "`Command::output()` is an
unbounded wait on a process this module does not own, and on macOS 27 a program
that touches a path under policy evaluation can block in the kernel
indefinitely" (`crates/synapse-core/src/machine_profile.rs:253-260`, budget at
`14`, test at `393-409`). That reasoning applies to `lib.rs:3254` unchanged, and
it runs on the serving-admission path, once per `owned_decode.admit_session`.
I label this a suspicion because I cannot reproduce the kernel condition; the
structural claim — same program, same platform, one caller deadlined and one not
— is citable and is itself a meaning defect: the knowledge that this probe can
hang lives in `synapse-core`'s comment and did not travel with the call.

**Confidence.** Medium-high on the meaning; the deadline gap is a suspicion with
a stated reason.

---

## 12. `certified: false` when the store read fails

`module_catalog_entries` computes the flag with `.ok().flatten().is_some_and(…)`
(`crates/synapse-module/src/lib.rs:14757-14759`), so a `SynapseStoreError`
becomes `Some(false)` — "this lane is not certified" — rather than "I could not
tell".

What makes this worth listing is that the last branch of the same `if`/`else if`
chain refuses to do exactly this, with a comment:

```
// Lane classes with no certification concept (such as worker-backed llama generate)
// omit the field rather than fabricating true or false.
```
(`lib.rs:14776-14778`)

The principle is stated, and then not applied to the store-error case in the same
expression. `docs/wire-contract-v1.md:143-144` absorbs it — "`true` only when
verified evidence exists for the current machine profile, and `false`
otherwise" — so a consumer following the document is not misled, but a consumer
following the code comment is. Nothing unsafe follows: the direction is
fail-closed, and the cost is an operator who cannot distinguish "not certified"
from "database unreadable". The same swallow appears throughout
`lane_measurement_rows` (`lib.rs:13873`, `13894`, `13902`, `13912`, `13920`,
`13935`, `13940`).

**Confidence.** Medium — high on the mechanism, medium that anyone is harmed.

---

## 13. `minimum_unified_memory_bytes`

**What it is.** `u64`, a private field of `PlatformEnvelope` with a `const`
accessor (`crates/synapse-module/owned-decode-routing/admission.rs:94`,
`137-140`). Its name means *the least unified memory a machine must have to run
this envelope*, and `validate_machine_tuple` uses it that way — a serving
machine must have `>=` it (`admission.rs:194-204`).

**What is put in it.** The certifying machine's *observed* capacity.
`serving_admission_material` passes `platform.unified_memory_bytes`
(`crates/synapse-module/src/lib.rs:3370-3376`), and that field is validated at
certification time to equal the machine tuple's actual memory:
`self.unified_memory_bytes != machine.unified_memory_bytes` is a rejection
(`crates/synapse-module/owned-decode-certification/certification_unit.rs:612`).
So a value meaning "what this box had" is stored in a field meaning "what any box
must have".

**Where the meaning comes from.** Both names are clear; nothing says they are the
same number, and the rename happens silently at the constructor call.

**What breaks — suspicion only.** On one machine, nothing: the certifying and
serving machine are the same box and `hw.memsize` does not move. The conflation
becomes reachable only if a certification record can be evaluated on a *different*
machine with the *same* revisioned profile hash. That is possible in principle
because `ram_class` is bucketed: `ram_class_from_bytes` maps everything in
`(64, 128]` GiB to `le_128_gib` (`crates/synapse-core/src/machine_profile.rs:237-246`),
so a 96 GiB and a 128 GiB machine can share a profile hash. Records could travel
between stores through the restore path, which keeps `serving_approvals` and
imports `serving_certification_records`
(`crates/synapse-module/src/store.rs:719-749`). I did **not** verify that a
restore onto a different host is a supported or reachable operation, so this is a
suspicion. If it is reachable, the direction is fail-closed anyway (the
lower-memory machine is refused with `UnsupportedPlatformTuple`), so the cost
would be an unexplained refusal, not an unsafe admission.

**Confidence.** Medium on the naming observation, low on reachability — which is
why it is a suspicion.

---

## 14. `DECODE_WORKER_ENGINE` and the literal beside it

`DECODE_WORKER_ENGINE` is `"owned-metal-decode"`
(`crates/synapse-core/src/worker_engine_names.rs:14`). Two conditions in the
reporting path read:

```
if spec.engine == DECODE_WORKER_ENGINE || spec.engine == "owned-metal-decode"
```
(`crates/synapse-module/src/lib.rs:14746`)

Today the disjunction is a duplicate of itself: the second arm is unreachable. A
reader cannot tell whether it is defensive (catching legacy rows if the constant's
*value* is ever changed) or an oversight. The same engine is spelled three
different ways within the file — the constant alone at `lib.rs:13862` and
`14210`, the bare literal alone at `lib.rs:8143-8144` and `14059-14065`, and the
disjunction at `14746`.

The distinction matters because `lane_requires_certification` uses **only
literals** — `"owned-metal" | "owned-metal-decode"` (`lib.rs:14059-14065`) — so
changing `DECODE_WORKER_ENGINE`'s value would keep `certified` and
`serving_admission` working for the new name while silently flipping
`certification_required` to `false` and `certification_status` to
`"not_required"` for that engine (`lib.rs:14012-14022`). The two spellings are
not interchangeable and nothing says which one is authoritative.

**Confidence.** Medium. The coupling is real; whether anyone would change the
constant's value is speculation.

---

## Values whose meaning I could NOT establish

These are the ones I want on the record. I did not resolve them by guessing.

### `identities_installed` (serving-predicate arm)

`pub identities_installed: bool`
(`crates/synapse-module/owned-decode-routing/lane.rs:288`), required true by
`serving_predicate` (`lane.rs:313`).

**What I tried.** Repo-wide search for every occurrence. There are four
construction sites and every one of them is the literal `true`:
`crates/synapse-module/src/lib.rs:9276` (the production call site),
`crates/synapse-module/src/store.rs:10865`,
`crates/synapse-module/owned-decode-certification/gates.rs:688`, and
`crates/synapse-module/owned-decode-routing/lane.rs:415` — the last three in
tests. There is no producer anywhere. The field has no doc comment, though its
neighbours in the same struct have none either, so that is not a signal.

**Candidate meanings.** (a) *The constrained-decoding runtime identities named by
the certification row are installed in this worker build* — plausible because the
adjacent arm is `constrained_identities_match`. (b) *The model's tokenizer /
grammar automaton identities are materialised on disk*. (c) A placeholder for a
check that was planned and never written, kept in the struct so the arm count
matches a contract document.

I cannot choose. The distinction matters: under (a) or (b) this is an unwired
gate and something real is unchecked; under (c) it is a harmless stub and
removing it would be the clean fix. A reader who assumes (c) and deletes the arm
may be deleting a commitment; a reader who assumes (a) may believe a check is
running that is not.

### The `"gates"` key accepted by `complete_g_dec_evidence`

```
evidence.get("g_dec").or_else(|| evidence.get("gates"))
```
(`crates/synapse-module/src/store.rs:7622-7625`)

**What I tried.** Searched every `.rs` and every checked-in manifest under
`crates/` for a writer of a `"gates"` key in certification evidence. The only
writer I found is `evidence["g_dec"]` (`crates/synapse-module/src/lib.rs:12767`).
The three other places that build `G-DEC-nn` arrays (`store.rs:10197`, `10501`,
`10806`) are tests.

**Candidate meanings.** (a) *Backward compatibility* — rows written by an older
module used `"gates"`, and this alias keeps them admissible. (b) *Forward
compatibility* — a planned external producer will use `"gates"`. (c) Dead
tolerance copied from a draft.

Under (a) it is load-bearing and removing it would fail-closed a population of
existing rows; under (c) it is an unnecessary second name for the value that
decides `gates_complete`. I could not tell which, and the difference is exactly
whether there exist rows in the field whose gate evidence lives under a key the
producer never writes.

### `ram_class = "unknown"` on non-macOS

`ram_class` returns the literal string `"unknown"` on every non-macOS build
(`crates/synapse-core/src/machine_profile.rs:196-202`). The comment states the
*design intent* clearly — "A constant on this platform rather than a probed
value, so it is identical on every run and cannot rotate the profile" — so the
meaning is stated for a reader of that function.

What I could not establish is what it means to a reader *downstream*. The string
is hashed into the machine profile alongside genuine values, and it is
indistinguishable from a probe that answered "unknown". The sibling field
`macos_build` has the same shape: on Linux it holds `uname -sr` output
(`machine_profile.rs:162-165`) under a name that says macOS, and that name
propagates all the way into the admission tuple
(`crates/synapse-module/owned-decode-routing/admission.rs:27`) and the
certification tuple (`certification_unit.rs:123`). Whether non-macOS is a
supported serving platform at all — in which case these are live values under
wrong names — or unreachable because `current_unified_memory_bytes()` refuses
everything off macOS (`crates/synapse-module/src/lib.rs:3265-3268`), I could not
determine. The two answers give the fields completely different meanings.

### `"h16(map)"` / `"h17(map)"`

The `(map)` suffix is data, inside a value that feeds a hash. It plainly encodes
provenance — *this subtype came from the static chip map, not from a device
probe* — and the comment above explains why a probe was not used
(`crates/synapse-core/src/machine_profile.rs:208-211`). What I could not
establish is what a consumer is supposed to *do* with it, because there is no
consumer: nothing parses the suffix, and the only test that mentions the idea
does so in its name (`machine_profile.rs:611`). If a probed subtype is ever added
alongside the mapped one, whether `"h16"` and `"h16(map)"` are the same machine
or different ones is a decision nobody has written down, and it is a decision
that rotates identity hashes.

### `generation = 0` on a first serving approval (checked, and clean)

`approve_serving_catalog` assigns `generation` as `previous + 1` if a row exists
and `0` if it does not (`crates/synapse-module/src/store.rs:3657-3659`), so `0`
means "first ever approval for this catalog fingerprint". I chased it because `0`
is the classic absent-sentinel and I wanted to check rather than assume. Nothing
breaks, and the reason is worth stating because it is not the obvious one.

The two `generation` fields with the same name play different roles:

- `approvals.generation` **is** a fence. The dispatch-boundary read compares it
  for exact equality alongside the semantic digest and the epoch, in one fenced
  statement (`store.rs:4520-4526`), which is what makes
  `admission_boundary_is_current()` meaningful (#7).
- `serving_approvals.generation` is **not** a fence. It is stamped onto the
  session row at admission (`store.rs:3780-3791`) and echoed to the caller
  (`store.rs:3792-3796`, surfacing at `lib.rs:3476-3480` and `3524`), but I found
  no statement anywhere that compares the stamped value against the live one.
  Freshness on that path comes from re-reading the approval at every boundary
  instead (`store.rs:3823`, `3868`, `3905-3910`) — a stronger design than a
  stamp comparison, but it means `approval_generation` on this surface is a
  reporting value, and a reader who expects it to behave like its namesake will
  look for a fence that is not there.

So `0` is never read as "absent" on either side. Recording the negative result,
because "I checked and it is fine" is worth as much here as a finding.

Related and also clean: neither `expected_digest` covers `generation` or
`updated_at_ms` (`ApprovalRow` at `store.rs:1043-1068`, `ServingApprovalRecord`
at `store.rs:1284-1296`). So `semantic_digest` means *the decision-bearing fields
are unmodified*, not *the row is unmodified*. On the `approvals` side the gap is
covered because generation is compared separately (`store.rs:4524-4525`). That
meaning is stated only by the function body — there is no comment naming which
fields are in scope — but it is correct.

---

## Two cross-cutting shapes

Two failure shapes are worth pulling out across the whole path, because both
produce well-formed values that no test can fail on.

### Where a value can be absent *and* zero/empty, does the code keep them apart?

Mostly yes, and in one place unusually well. `profile_activation_epoch` is
`Option<u64>` at the store boundary and `validate_profile_state` makes `(None,
None, None)` — never observed — a distinct legal state from any `Some(epoch)`,
which must additionally be `> 0`
(`crates/synapse-module/src/store.rs:7740-7761`). The DDL enforces the same
(`store.rs:564`). Boot then converts the `Option` into a plain `u64` by refusing
when it is absent (`lib.rs:1976-1983`). Absent and zero are kept apart by the
producer, and the consumer never has to tell them apart because one of the two
cannot reach it. That is why `current_epoch_valid` is dead rather than dangerous
(#9).

The certification types follow the same discipline explicitly:
`unified_memory_bytes == 0` is `MissingEvidence`, not a small machine
(`crates/synapse-module/owned-decode-certification/certification_unit.rs:131-133`);
so are `observed_at_ms == 0` (`certification_unit.rs:151-153`) and the three zero
checks in `ArtifactReservation::validate`
(`crates/synapse-module/owned-decode-routing/admission.rs:73-84`).

Where it is not kept apart: `ane_subtype: None` (#10), where absent means two
different physical facts; `current_unified_memory_bytes() -> None` (#11), where
absent means five different causes; and `certified: false` (#12), where a store
error is reported as a measured negative.

One more, in the rotation ledger. When the previous profile snapshot is missing
— which happens after a trust-only restore, a case `observe_profile` explicitly
anticipates (`store.rs:5099-5102`) — `changed_fields` becomes the single-element
list `["unknown_previous_snapshot"]` (`store.rs:5155-5163`). That sentinel sits
in a list whose other inhabitants are field names (`store.rs:7817-7839`). The
consumer immediately below asks `changed_fields.iter().any(|field| field ==
"os_build")` to decide whether to log the rotation at `warn` or `info`
(`store.rs:5280-5297`), and the sentinel is not `"os_build"`, so a rotation that
really did change the OS build is logged at `info` whenever the previous snapshot
was absent. Small consequence, exact shape: a reader treating "empty/unknown" as
"no os_build change".

### Where a check is partially sound, which direction does the caller act on?

Three cases, and they split.

- `admission_boundary_is_current()` (#7): `false` is sound, `true` is ambiguous,
  and the caller acts on `false` (`mod.rs:719`). **Correct.**
- `stale_os_build` (`lib.rs:14140`, `14160`): `true` is sound — the recorded
  build genuinely differs. `false` is not: equal `os_build` strings do not mean
  the row is current, because the profile hash also covers `arch`, `chip_model`,
  `ram_class`, `ane_subtype`, and `engine_identities`
  (`crates/synapse-core/src/machine_profile.rs:101-111`). No gate consumes it; it
  is display-only on `probe.report`. **Safe, because nobody acts on it.**
- `serving_admission` (#3): `"disabled"` is sound — the `approvals` row really is
  absent or off, so the owned lane will not execute (the request is refused, or
  substituted to llama, `lane.rs:215-240`). `"enabled"` is the ambiguous
  direction, because it says nothing about the `serving_approvals` row that gates
  `owned_decode.admit_session`. And `"enabled"` is precisely the answer an
  operator acts on when deciding whether a lane is ready to take traffic.
  **This is the one where the ambiguous direction is the one being consumed.**

---

## What a fix would touch

This document changes no code. For the record, in confidence order:
`serving_admission` should either read both approval tables
or be renamed to say which one it reads (#3); `models.list`'s `certified` should
either call the serving predicate as its contract claims or the contract should
be rewritten (#1); `evidence["g_dec"]` should carry the gate outcomes it names,
or carry one honest field (#2); `certification_stale` on the decode branch is a
one-line contradiction (#4); `identities_installed` should be wired or removed,
and someone who knows which should say so (unresolved section);
`SYNAPSE_OS_BUILD_OVERRIDE` needs a doc comment stating whether it is supported
(#5); and the comment at `lib.rs:4187-4189` states the opposite of what the store
does (#6).
