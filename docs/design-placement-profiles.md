# Design note: placement profiles (`@auto` model aliases)

Status: banked design, not scheduled. Trigger for implementation: a consumer
(AFT is the expected first) outgrows single-lane placement — typically when
bulk-refresh latency on the all-ANE configuration starts to matter.

## Problem

A model family can be certified on several lanes of one machine (gte-modernbert
on Metal f16 and ANE fp16 today). The lanes are distinct vector spaces with
distinct fingerprints, but measured equivalence can hold in one direction
(certified 2026-07: ANE-embedded corpus + Metal-embedded queries is
retrieval-clean; the reverse direction fails). Consumers who want the good
mixed placement today must juggle two model ids, two fingerprints in their
index headers, and their own direction discipline — bad DX, easy to hold
wrong.

## Shape

One alias, module-side routing, honest envelopes:

- Consumer configures a single placement alias, e.g.
  `gte-modernbert-base@auto`.
- The module routes per request class: job-tier bulk (`embed.batch`) executes
  on the quiet lane (ANE); interactive single embeds execute on the fast lane
  (Metal). The routing table is part of machine capability state, derived from
  which lanes are certified — never configured ad hoc per request.
- Direction enforcement is structural: corpus-write traffic cannot reach the
  fast lane and query traffic cannot reach the quiet lane while the alias is
  active, because the certified equivalence only holds in one direction.
- Envelopes keep telling the truth: every response carries the fingerprint of
  the lane that actually executed. Nothing is silently substituted.
- The alias table carries the certified pair as an explicit equivalence row
  (corpus-fingerprint, query-fingerprint, direction, evidence reference), so
  `check_index` blesses a mixed-space index without consumer-side logic.

## Relationship to existing machinery

This is the speed-vs-energy knob (onboarding-bench era) applied per request
class instead of per machine. It reuses: lane certification records, the alias
table with validity intervals, `check_index` verdicts, and the 3-class
scheduler's existing request classification. The new pieces are small: the
alias resolution step in embed dispatch, the direction guard, and one new
alias-row kind carrying a direction plus evidence pointer.

## Contract requirements

1. A placement alias may only bind lane pairs whose mixed direction has
   certification evidence (the ANE-corpus/Metal-query A/B is the template:
   paired NDCG deltas, tie counts, broken-query count on a pinned eval set).
2. Index headers written by consumers store the corpus-lane fingerprint plus
   the alias identity; `check_index` resolves the pair through the alias row.
3. Revoking either lane's certification revokes the alias row (fail-closed,
   same epoch semantics as every other approval).
4. The alias never changes an already-written index's identity: adopting or
   abandoning a placement alias is an explicit consumer migration, not a
   silent re-route.
