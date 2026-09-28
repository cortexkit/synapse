# Search reranking evaluation — stopped at fidelity gate

## Result

**There is no measured answer yet to whether reranking improves either tool.**
The required exact-order fidelity gate failed for Qwen3-Reranker-0.6B, so the
experiment stopped before production-sized scoring or judging. No serving code
changed. Near agreement is not presented as a passing gate.

On 20 real AFT query/candidate pairs, the official Transformers reference and an
officially converted F16 GGUF differed by one pairwise inversion. Absolute score
agreement passed the preselected tolerance, but exact ordering did not:

| Fidelity measure | Qwen3-Reranker-0.6B |
|---|---:|
| Pairs | 20 |
| Distinct queries represented | 6 |
| Maximum absolute probability error | 0.0032245964 |
| Mean absolute probability error | 0.0003744475 |
| Allowed maximum absolute error | 0.005 |
| Inverted pairs / all pairs | 1 / 190 |
| Spearman correlation | 0.9984962406 |
| Kendall tau | 0.9894736842 |
| Identical descending order | **No — fail** |

The inversion was between the two lowest-scoring items. The reference assigned
0.0018913834 and 0.0018942355; llama.cpp assigned 0.0019104981 and 0.0019013400,
respectively. These are not exact ties, and no tie tolerance was introduced after
seeing the result. The other 189 pairwise comparisons agreed. This is not evidence
of the missing-classifier-tensor failure: scores spanned approximately 0.002–0.986.
The cause of the small numerical discrepancy has not been established.

| Model | Fidelity status |
|---|---|
| plumb-4b | Not reached |
| Qwen3-Reranker-0.6B | Failed exact order; passed absolute-error tolerance |
| Qwen3-Reranker-8B | Not reached; conversion interrupted after the failed gate |

A follow-up must investigate the numerical difference and pass the originally
specified gate, or obtain an explicit methodological decision about near ties.
This run did neither, and did not continue scoring with the failed model.

## Inputs and replay coverage

The supplied extractions each contained 100 real calls. Exact query strings were
deduplicated within each tool, keeping the first occurrence in the supplied file;
options and session provenance came from that occurrence. No calls were
re-extracted.

| Tool | Supplied calls | Distinct queries | Completed candidate replay | Queries scored for quality |
|---|---:|---:|---:|---:|
| aft_search | 100 | 97 | 1 project-bound query before stop | 0 |
| ctx_search | 100 | 92 | 92 | 0 |

These are input/replay counts, **not a final eligible evaluation sample**. The
AFT replay was interrupted while warming the next project. There is consequently
no completed per-project exclusion census or final AFT evaluation n.

### AFT

The prescribed `subc_call` management-surface invocation was rejected with
`unknown_management_op: management routes accept only declared operation envelopes`.
With explicit approval, an isolated evaluation client instead used the declared
`ToolProvider` route and `ToolCallRequest { name: "search", arguments: ... }`.
The observed AFT catalog version was 0.58.0.

An initial cross-project replay collected 85 responses, including 21 missing-
directory mappings, but frequently entered the bounded lexical fallback rather
than a ready index. Those responses are not an eligible baseline corpus. They
were retained privately for diagnostics, not scored as a quality evaluation.
The fidelity sample used 20 real pairs from six of those responses that did not
mark themselves degraded or incomplete; the fidelity sample is not a quality
sample.

The approved correction binds the client to each resolved original project root
and still passes that same `path`, requests `topK: 50`, and preserves
`includeTests`. Missing worktrees map to an existing repository directory sharing
the original input's `project_id`. The revised collector records every response
and status, warms each project with a bounded retry window of approximately three
minutes, and excludes responses still degraded or partial. Only one project-bound
query completed before the fidelity stop; it had one candidate and a full status.

Three explicit-root parity probes in this repository matched the assistant's
`aft_search` rendered top-10 output exactly, including order and candidate text.
The probes included both values of `includeTests`. Three earlier implicit-root
probes also matched. Cross-project tool parity is not claimed: a cross-project
assistant call can itself use the borrowed-index fallback.

### Magic Context

An isolated MC clone at `fc106720c5e2c77e241404b0930b567948686671` supplied
`packages/plugin`'s TypeScript `unifiedSearch`. Its dependencies were installed
with Bun 1.4.2. The live database and its WAL were copied into private evaluation
storage; only the copy was opened read-write. SQLite `quick_check` passed.

Only 22 of the 92 original session IDs had a `session_projects` row. With explicit
approval, the other 70 project identities were derived using MC's own
`resolveProjectIdentityForSession`, not an invented identity algorithm. Before
using the fallback, directory resolution matched **all 22/22** available stored
identities. Missing directories were mapped by original project identity to an
existing repository: 23 of the 92 queries needed this directory mapping. All 92
identities resolved; no MC query was dropped for missing project attribution.
Per-query stored/resolved provenance remains in the private artifacts.

Replay requested `limit: 50`, preserved `sources`, enabled explicit search, and
verified configured embedding identities against indexed registrations. It
refused null query embeddings rather than silently accepting an embedding
fallback. Retrieval counters and production measurement writes were disabled.
**Visible-memory filtering was disabled for every query** (`visibleMemoryIds:
null), so today's visible set was not mistaken for the historical set. No message
ordinal cutoff was supplied; this was a current-index replay, not a reconstruction
of historical visibility. Git-commit search followed the loaded configuration.

The replay returned fewer than 50 candidates when fewer were available:

| Candidates returned | MC queries |
|---|---:|
| 0 | 2 |
| 1 | 5 |
| 2 | 8 |
| 3 | 3 |
| 50 | 74 |

These counts describe retrieval only, not relevance or successful judging.

## Fidelity implementation

Hardware: Apple M5 Max, 128 GiB unified memory; the machine was shared. Python
3.14.2, PyTorch 2.11.0, Transformers 4.57.6, NumPy 2.2.6. The reference used FP32
on MPS, evaluation mode, and Qwen's model-card Transformers next-token yes/no
log-softmax readout. The candidate path used a locally built Metal llama-server
with `--reranking`, F16 weights, context 4096, batch/ubatch 2048, and GPU offload.

llama.cpp was pinned to **`680a036285273a3ff56032ec5d7f3352609eba4f`**.
Its unmodified `convert_hf_to_gguf.py` converted the official safetensors,
extracted the yes/no `cls.output.weight`, selected rank pooling, and emitted the
rerank template. No community reranker GGUF was used.

The converter's default web-search instruction was replaced, using the same
commit's official `gguf_new_metadata.py`, with this code-search instruction:

> Given an agent search query, retrieve relevant code or documentation in the repository that helps answer the query.

All other template text followed Qwen's official Transformers example, including
the non-thinking assistant suffix. A first server launch with a string metadata
CLI override failed because that interface limits strings to 127 characters;
the successful launch used the metadata-rewritten GGUF instead. That launch
failure was not counted as a fidelity comparison.

For this gate, each candidate was truncated to **512 tokens** with the official
Qwen3-Reranker-0.6B tokenizer, without special tokens, then decoded. Both paths
received the identical resulting text and original query. The planned quality
experiment would freeze these same text bytes across every arm and the judge;
that shared quality corpus was not completed. Gate pair selection was deterministic:
scan the saved input order, skip incomplete/degraded responses, and take up to
four rendered candidates per query until 20 pairs are collected. No relevance
labels were used to select them.

## Pinned model artifacts

| Model | Official revision |
|---|---|
| crh225/plumb-4b | `1c5f4483addb049476ac33107796d217a1dc089d` |
| Qwen/Qwen3-Reranker-0.6B | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| Qwen/Qwen3-Reranker-8B | `77d193c791ed757ca307ee72715aa132723da912` |

SHA-256 hashes of downloaded official weight files:

| Model / file | SHA-256 |
|---|---|
| plumb-4b / model.safetensors | `89e119ea07f4c5b4b6715560c7de6694ec3b777dfd0e62351da0b33986d2e1e1` |
| Reranker-0.6B / model.safetensors | `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b` |
| Reranker-8B / shard 1 of 5 | `22cdfea4a13b7b3e866573800eeeb638fc38962940adf631d06dc03befed047a` |
| Reranker-8B / shard 2 of 5 | `d2163b74137e35b4614bd2aa5bf27bcb07de4ca61c6962495feb968385eb0df8` |
| Reranker-8B / shard 3 of 5 | `a5038caa78c817e8acce6806104869675938a33fd4e60ed038e9931d390d6989` |
| Reranker-8B / shard 4 of 5 | `247f85538c5996d4c296291b0e4004f618c9b17ca8cdc25d1fc726567eb15803` |
| Reranker-8B / shard 5 of 5 | `8ba41b93c2e4ec8339ad16b000bc977fde196aeac054956cbfc8c0186ee6d4cf` |
| Reranker-0.6B / official conversion F16 GGUF | `42b24043fc2315f1244bd01607a4f60a2c23ccc7fc6b019babd580122b941344` |
| Reranker-0.6B / code-instruction F16 GGUF used in gate | `61e3a4616f957a078f0b09fcdfc57847182fe3c98dba366d4c4bc7c47e9b8ee2` |

The plumb card names jevk5 **v0.2.0**, pinned here to
`85238d7be5527370c43206fe54cd752eb3134c1b`. Its `runtime.py` was read for the
actual `noul` prompt: evidence/criterion/options JSON, options A/B described as
`true: The proposition is true.` and `false: The proposition is false.`, with
`add_generation_prompt=True` and `enable_thinking=False`. The proposed relevance
criterion, not yet scored, was:

> Is the supplied evidence relevant to answering the agent's search query: {query}?

SemIf source was fetched at `23cf1f39fc9534fe81437200959b6dfc7106e45a`, but neither
its backend nor plumb inference was exercised. The plumb GGUF was not downloaded.
No claim of fidelity is made for plumb or the 8B reranker.

## Quality metrics and judge status

The authorized judge channel was BROCA `session.send` using
`openai/gpt-6-luna`, variant `low`, temperature 0, and 512 maximum output tokens.
The reserved session prefix was **`synapse-judge-rerank2609-`**. **No judge session
was sent**, including the two-query dry run, because the fidelity gate failed.
There are no judge charges or usage facts attributable to this run's prefix.

| Tool | Arm | nDCG@10 | P@10, grade ≥1 | P@10, grade 2 | MRR, grade 2 |
|---|---|---|---|---|---|
| aft_search | baseline | Not measured | Not measured | Not measured | Not measured |
| aft_search | plumb-4b | Not measured | Not measured | Not measured | Not measured |
| aft_search | Qwen3-Reranker-0.6B | Not measured | Not measured | Not measured | Not measured |
| ctx_search | baseline | Not measured | Not measured | Not measured | Not measured |
| ctx_search | plumb-4b | Not measured | Not measured | Not measured | Not measured |
| ctx_search | Qwen3-Reranker-8B | Not measured | Not measured | Not measured | Not measured |

Paired deltas, bootstrap intervals, top-10 grade-2 gains/losses, per-query union
sizes, and the permutation noise floor are **not measured**, not zero. The
optional full-50 judging audit and cross-size reranker runs were not performed.
No validated ambient per-candidate scoring latency is reported.

The intended, unexecuted judge protocol is the arm-blind union of each arm's top
10, deduplicated by candidate identity, sorted by identity, and split into fixed
batches before any permutation control. Labels would be 0/1/2 for irrelevant,
partial, and relevant. Ten percent of batches would be re-judged with permuted
row order. Any future query-bootstrap interval holding labels fixed must be
identified as too narrow because it omits judge uncertainty.

## What was unexpected

- A numerically close, correctly headed official conversion still failed the
  strict ordering requirement on two very low-scoring candidates.
- Most historical MC sessions no longer had a project-attribution row, although
  the production identity resolver recovered every project and validated exactly
  against the surviving rows.
- AFT's cross-project `path` replay can use a different, degraded search path
  from a client actually bound to that project. A successful tool response alone
  is not enough to certify an indexed baseline.

All query text, candidate text, session identifiers, project paths, replay
responses, and gate pair scores remain in private evaluation storage. This
public document contains only methodology, artifact provenance, and aggregates.
