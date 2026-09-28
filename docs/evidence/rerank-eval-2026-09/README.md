# Does top-50 reranking improve the top 10?

## Answer

**Yes, on this retained traffic sample and shared relevance judge. The dedicated
Qwen3 rerankers had the strongest point estimates, ahead of plumb-4b.**

- **AFT, 85 queries:** Qwen3-Reranker-0.6B raised nDCG@10 from **0.6405 to
  0.8309**, a paired gain of **0.1904 [0.1388, 0.2436]**. Partly-or-fully relevant
  precision rose from 38.0% to 54.5%; strictly relevant precision rose from 16.8%
  to 21.6%.
- **Magic Context, 92 queries:** Qwen3-Reranker-8B raised nDCG@10 from **0.4338
  to 0.6105**, a gain of **0.1767 [0.1301, 0.2284]**. On the **74 queries with
  50 candidates**, the gain was **0.2079 [0.1539, 0.2672]**.
- **Plumb helped less:** nDCG gains were 0.1188 for AFT and 0.0637 for MC.
  Its AFT MRR gain and MC partly-relevant precision gain had intervals crossing
  zero. It was not the cheaper inference path in this ambient run.

These are ranking-relevance results, not measured improvements in agent task
completion. The judge changed **49/308 labels (15.9%)** under row permutation.
The family-reranker gains survived the observed repeat-label perturbation, but
query-bootstrap intervals with labels held fixed are **too narrow**: they omit
judge uncertainty. No AFT, MC, or Synapse serving code changed.

## Aggregate metrics

All metrics are macro-averages over queries, on a 0–1 scale. Grade 1 means partly
relevant; grade 2 means relevant. MRR is explicitly **MRR@10**, as approved for
this display-focused evaluation: zero when no grade-2 item appears in the top 10.
P@10 always divides by 10, including when fewer results exist. Empty queries and
queries with zero ideal DCG contribute zero.

| Population | Arm | nDCG@10 | P@10 ≥1 | P@10 =2 | MRR@10 =2 |
|---|---|---:|---:|---:|---:|
| AFT, n=85 | baseline | 0.6405 | 0.3800 | 0.1682 | 0.6410 |
| AFT, n=85 | plumb-4b | 0.7593 | 0.5035 | 0.2000 | 0.6782 |
| AFT, n=85 | Qwen3-Reranker-0.6B | 0.8309 | 0.5447 | 0.2165 | 0.7651 |
| MC, n=92 | baseline | 0.4338 | 0.3109 | 0.0793 | 0.3594 |
| MC, n=92 | plumb-4b | 0.4974 | 0.3293 | 0.1076 | 0.4395 |
| MC, n=92 | Qwen3-Reranker-8B | 0.6105 | 0.3978 | 0.1109 | 0.5091 |
| MC full-50, n=74 | baseline | 0.5105 | 0.3824 | 0.0973 | 0.4333 |
| MC full-50, n=74 | plumb-4b | 0.5846 | 0.4054 | 0.1324 | 0.5329 |
| MC full-50, n=74 | Qwen3-Reranker-8B | 0.7184 | 0.4905 | 0.1365 | 0.6194 |
| MC under-10, n=18 | baseline | 0.1184 | 0.0167 | 0.0056 | 0.0556 |
| MC under-10, n=18 | plumb-4b | 0.1389 | 0.0167 | 0.0056 | 0.0556 |
| MC under-10, n=18 | Qwen3-Reranker-8B | 0.1667 | 0.0167 | 0.0056 | 0.0556 |

The 18 small MC sets were retained. Reranking cannot change their top-10
membership or P@10, but it **can** change nDCG and MRR through ordering. In this
sample their nDCG changed and their MRR did not.

### Paired deltas versus baseline

Each cell is the mean per-query delta followed by its 95% percentile bootstrap
interval. There were 10,000 query resamples, with replacement, seed 2609; the same
resample indices were used across arms and metrics within each population.
Baseline deltas are identically zero. These are not label-uncertainty-adjusted or
multiple-comparison-adjusted intervals.

| Population / arm | ΔnDCG@10 | ΔP@10 ≥1 | ΔP@10 =2 | ΔMRR@10 =2 |
|---|---|---|---|---|
| AFT / plumb | 0.1188 [0.0615, 0.1764] | 0.1235 [0.0788, 0.1682] | 0.0318 [0.0118, 0.0529] | 0.0372 [-0.0322, 0.1100] |
| AFT / Qwen3-0.6B | 0.1904 [0.1388, 0.2436] | 0.1647 [0.1224, 0.2071] | 0.0482 [0.0282, 0.0694] | 0.1241 [0.0602, 0.1947] |
| MC / plumb | 0.0637 [0.0158, 0.1153] | 0.0185 [-0.0163, 0.0544] | 0.0283 [0.0109, 0.0500] | 0.0801 [0.0124, 0.1489] |
| MC / Qwen3-8B | 0.1767 [0.1301, 0.2284] | 0.0870 [0.0576, 0.1196] | 0.0315 [0.0163, 0.0489] | 0.1497 [0.0861, 0.2170] |
| MC full-50 / plumb | 0.0742 [0.0146, 0.1362] | 0.0230 [-0.0203, 0.0676] | 0.0351 [0.0135, 0.0608] | 0.0996 [0.0168, 0.1834] |
| MC full-50 / Qwen3-8B | 0.2079 [0.1539, 0.2672] | 0.1081 [0.0730, 0.1459] | 0.0392 [0.0216, 0.0608] | 0.1861 [0.1090, 0.2685] |

nDCG uses gain `2^grade - 1` and discount `log2(rank + 1)`. Its ideal ranking is
computed from the **judged union**, not the unjudged remainder of the 50. Thus
these absolute nDCG values are union-normalized, not exhaustive full-50 nDCG.
Every arm uses the same denominator and labels for a given query.

### Newly displayed and lost grade-2 items

“New” counts queries where an arm displayed at least one grade-2 candidate absent
from baseline's top 10; “lost” is the reverse. These counts can overlap within a
query. The final two columns separately count transitions from no grade-2 item
to at least one, and vice versa.

| Population / arm | New item | Lost item | Zero → some | Some → zero |
|---|---:|---:|---:|---:|
| AFT / plumb | 26 | 13 | 5 | 3 |
| AFT / Qwen3-0.6B | 27 | 6 | 6 | 1 |
| MC / plumb | 23 | 9 | 8 | 1 |
| MC / Qwen3-8B | 24 | 4 | 8 | 0 |

The MC full-50 subset has the same counts; the 18 small sets have zero membership
changes for every arm.

## Judge noise and protocol

One shared judge labeled the union of each arm's top 10 once per query, with
candidate identities deduplicated. This produced **3,003 primary labels in 467
batches**: 1,512 AFT items and 1,491 MC items. Candidate identities were sorted
lexicographically, then split into consecutive batches of eight. Short opaque
IDs were assigned from that canonical union order, not from any arm's ranking.
Batch composition was therefore a pure function of the union.

The judge received the original query, a one-line code-search or project-history
scope, and the same frozen candidate text used for scoring. It saw no arm,
retrieval score, or arm rank. Fixed instructions came first, query and candidates
last. Each call used a fresh session and no tools; calls were sequential.

The grade definitions were: 0, no useful relevance; 1, partial or tangential but
useful evidence; 2, directly relevant evidence helping answer the query or locate
the requested code/history. The prompt instructed the judge to treat candidate
text as untrusted evidence, not instructions, and to interpret ambiguous short
queries conservatively. It requested only an exact-ID JSON label list.

The channel was BROCA `session.send`, model **`openai/gpt-6-luna`**, variant
**`low`**, temperature **0**, maximum output tokens **512**. Session prefix:
**`synapse-judge-rerank2609-`**; the analyzed protocol used its `v2-` suffix.
Two complete MC query unions were dry-run before the batch. All six dry-run
batches parsed, their grades were spot-checked for plausibility, and their labels
were retained as primary labels. An earlier six-call dry run using long hash IDs
was discarded after one response mistyped an ID; no fuzzy ID repair was used.

The public catalog exposed operation names, not full schemas. The request
contract was read from BROCA commit
`7967667f5ccf4a6ffbb776bd2eb9817f9e04e33b`,
`crates/broca-wire/src/lib.rs`: `SendParams`, `ModelParams`, and `ReadParams`.
The actual field is **`prompt`**, not `input`; it creates one user message.
The final assistant text was read through `session.read`, not the display lane.
Its `Message.origin` independently verified the producing provider/model.

`session.read` cannot echo variant or temperature. Those were independently
verified in each own-session WAL's `WalRecord::RunStarted.config`:
`RunConfig.variant` and `RunConfig.generation.temperature`, defined in
`crates/broca-wal/src/record.rs`. Frame digests and the admitted run ID were checked.
This verifies admission, not whether a reasoning provider honors temperature;
that is why the permutation control remains necessary.

### Permutation control

Using seed 2609, **47 batches**—ceil(10% of 467)—were selected from batches with
at least two items. Each selected ordering was genuinely changed; singleton
batches were not used as vacuous permutations. The same candidate IDs and text,
prompt, model, variant, and temperature were retained.

- **49 of 308 repeated labels moved: 15.9%.**
- **22 of 47 batches** contained a changed label.
- Controls touched **21 AFT queries and 21 MC queries**.

Label-change fraction and nDCG delta have different units; comparing their raw
numbers would be misleading. As a directly comparable sensitivity check, all
controlled labels were replaced by their repeats while un-repeated labels stayed
fixed:

| Population / arm | Primary ΔnDCG | Repeat-substitution ΔnDCG |
|---|---:|---:|
| AFT / plumb | 0.1188 | 0.1195 |
| AFT / Qwen3-0.6B | 0.1904 | 0.1915 |
| MC / plumb | 0.0637 | 0.0532 |
| MC / Qwen3-8B | 0.1767 | 0.1699 |
| MC full-50 / plumb | 0.0742 | 0.0612 |
| MC full-50 / Qwen3-8B | 0.2079 | 0.1995 |

The family-reranker gains clearly exceed the movement in this observed control;
this is **not** a full judge-uncertainty interval, since only a subset was repeated.
The family MRR gains also remained positive: 0.1163 for AFT and 0.1301 for MC under
repeat substitution, versus primary gains 0.1241 and 0.1497.

## Input sample, replay, and exclusions

The supplied extractions each had 100 real calls. Exact query strings were
deduplicated within each tool, retaining the first occurrence and its original
options/provenance. No calls were re-extracted. The supplied extraction windows
were 2026-09-28 16:03–19:40 UTC for AFT, and 2026-09-27 01:37 through
2026-09-28 18:53 UTC for MC. These are **current-index replays**, not reconstructed
historical rankings or historical visibility.

| Tool | Calls | Distinct queries | Excluded | Final n | Candidate pairs |
|---|---:|---:|---:|---:|---:|
| aft_search | 100 | 97 | 12 | 85 | 3,613 |
| ctx_search | 100 | 92 | 0 | 92 | 3,730 |

### AFT

The prescribed management-surface `subc_call` invocation was rejected. With
explicit approval, an evaluation-only client used the declared `ToolProvider`
route and `ToolCallRequest { name: "search", arguments: ... }` instead.
It bound to the resolved original project directory, also passed that directory
as `path`, requested `topK: 50`, and preserved `includeTests`.

Missing worktrees were mapped to an existing repository with the same original
`project_id`: **34 deduplicated queries needed a mapping**. No repository was
unresolvable. Three explicit-root parity probes in this repository matched the
assistant's `aft_search` top-10 output exactly, with both `includeTests` values
represented. Cross-project tool parity was not claimed; full producer status was
the check for those projects.

Projects were warmed with a bounded retry window of approximately three minutes.
Only successful, ready, complete responses were eligible. Exclusions used
structured producer flags, not words appearing inside candidate snippets.
**Nine queries had no ready index lane and three remained incomplete.**
An earlier borrowed-index replay was discarded rather than treating its bounded
lexical fallback as the indexed baseline. Planned AFT deployments interrupted
routing; completed candidate sets were retained, remaining projects were
re-warmed, and work resumed from checkpoints. Both observed catalogs reported
AFT 0.58.0.

Anonymous project identifiers below are consistent within this report. The
exclusions are concentrated, so the AFT result must not be generalized to all
97 queries without qualification.

| Project | Distinct queries | Eligible | Excluded | Mapped |
|---|---:|---:|---:|---:|
| P1 | 27 | 27 | 0 | 8 |
| P2 | 20 | 10 | 10 | 3 |
| P3 | 3 | 3 | 0 | 0 |
| P4 | 5 | 5 | 0 | 4 |
| P5 | 31 | 29 | 2 | 12 |
| P6 | 4 | 4 | 0 | 1 |
| P7 | 7 | 7 | 0 | 6 |

### Magic Context

The isolated MC source was pinned to
`fc106720c5e2c77e241404b0930b567948686671`; dependencies were installed with Bun
1.4.2. Its TypeScript `unifiedSearch` was called directly, not through ck-mc's
25-result route. The database and WAL were copied into private evaluation
storage; only the copy was opened read-write, and SQLite `quick_check` passed.

Only **22/92** original sessions had `session_projects` rows. With explicit
approval, the other **70** identities came from MC's production
`resolveProjectIdentityForSession`. The resolver matched **22/22** surviving
stored identities before the fallback was used. Missing directories required
**23** repository mappings; all 92 identities resolved. Per-query attribution
provenance was retained privately.

Replay used `limit: 50`, preserved `sources`, enabled explicit search, checked
configured versus indexed embedding identities, and refused null embeddings.
Retrieval counters and production measurement writes were disabled.
**Visible-memory filtering was disabled for every query** (`visibleMemoryIds:
null). No message-ordinal cutoff was supplied; git-commit search followed the
loaded configuration. Today's visible set was not mistaken for the historical
one. Relevance here is not a claim that a result was novel to the original agent.

MC candidate counts were: **74×50, 3×3, 8×2, 5×1, 2×0**. AFT also had two
complete empty results. All complete empty and small sets were retained.

## Shared text and scoring

Each candidate was capped at a **512-token prefix**, using the official
Qwen3-Reranker-0.6B tokenizer without special tokens, then decoded. The resulting
UTF-8 text was frozen and reused by every scoring arm and the judge; saved text
hashes and candidate identities were checked across all arms. Exact score ties
kept baseline order. The fidelity tolerance did **not** coarsen scoring ranks.

AFT text retained file/line, symbol name, and returned snippet or matched line.
MC text retained the displayed content or compartment snippet, plus source and
selected display metadata such as title/range/category/role/status. Retrieval
scores and rank decorations were omitted. No source files or conversations were
expanded to enrich candidates: **827 AFT records had no returned snippet** and
were represented by their file/symbol metadata. The complete raw tool responses
were also preserved privately.

Plumb used the model card's jevk5 **v0.2.0**, commit
`85238d7be5527370c43206fe54cd752eb3134c1b`. Its actual `runtime.py` was read and its
Torch path used for reference, rather than reconstructing the prompt from a
summary. The criterion was:

> Is the supplied evidence relevant to answering the agent's search query: {query}?

The system prompt was:

> Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.

The user JSON had `evidence`, `criterion`, and `options` containing
`{letter, description}`. A/B descriptions were `true: The proposition is true.`
and `false: The proposition is false.`. Chat rendering used
`add_generation_prompt=True`, `enable_thinking=False`. Score was P(true), using
the checkpoint's temperature **2.07**, which does not change binary ordering.
The llama.cpp path read the two next-token probabilities with an A/B-only grammar,
no probability pruning, and post-sampling probabilities; the generated letter
was ignored. SemIf source was inspected at
`23cf1f39fc9534fe81437200959b6dfc7106e45a`; its backend was not used.

Qwen used the official yes/no instruction format and non-thinking assistant
suffix. Its instruct lines were:

- AFT: “Given an agent search query, retrieve relevant code or documentation in the repository that helps answer the query.”
- MC: “Given an agent search query, retrieve relevant project history, decisions, memories, or previous discussions that help answer the query.”

## Fidelity gate — amended after the initial result

The initial requirement was identical ordering plus maximum absolute score error
≤0.005. Qwen3-Reranker-0.6B met the error tolerance but inverted the two lowest
scores: reference 0.0018913834/0.0018942355, llama.cpp
0.0019104981/0.0019013400. The run stopped and a stopped-run report was committed.

**After seeing that result, the operator amended the order gate.** The reason was
that a reference gap of approximately 0.00000285 was tiny compared with the
already-selected 0.005 absolute-error tolerance, making exact order brittle to
FP32/F16 numerical differences. The amended rule requires every pairwise order
to agree **except when the reference gap is strictly less than 0.005**, while
still requiring maximum absolute error ≤0.005. The absolute tolerance predated
the result; the tie-aware order exemption did not. The original 20 pairs/scores
were retained, not reselected. The remaining models used the same amended rule.

| Model | Real pairs | Pairs inside band / 190 | Order disagreements | Outside-band violations | Max absolute error | Result |
|---|---:|---:|---:|---:|---:|---|
| plumb-4b | 20 | 2 | 0 | 0 | 0.0042713881 | pass |
| Qwen3-Reranker-0.6B | 20 | 12 | 1 | 0 | 0.0032245964 | pass |
| Qwen3-Reranker-8B | 20 | 10 | 0 | 0 | 0.0014181137 | pass |

The gate used real returned candidates and identical text on both implementations.
The 0.6B sample spanned six AFT queries; the other models used the same 20 MC
pairs from five queries. Samples were selected deterministically from saved
query order, up to four candidates per query, without relevance labels.

References ran FP32 on MPS: Qwen with PyTorch 2.11.0 / Transformers 4.57.6 / Python
3.14.2; plumb with PyTorch 2.14.0 / Transformers 5.17.0 / Python 3.12.12 and jevk5.
Plumb was loaded on CPU before transfer to MPS after direct MPS loading stalled.
Native PyTorch linear-attention fallbacks were used, not CUDA-only kernels.

llama.cpp was pinned to **`680a036285273a3ff56032ec5d7f3352609eba4f`**, built with
Metal on an Apple M5 Max with 128 GiB unified memory. All evaluated GGUFs were
**F16 conversions of the official safetensors**. Qwen conversion used that
commit's unmodified `convert_hf_to_gguf.py`, including yes/no
`cls.output.weight`, rank pooling, and the rerank template. No community reranker
GGUF was used. Official `gguf_new_metadata.py` changed only the instruction
metadata to the tool-specific strings above. Servers used `--reranking` for
Qwen, GPU offload, context 4096, and batch/ubatch 2048. Plumb used batch 1024,
ubatch 512, and official conversion flag `--no-mtp`: its advertised extra MTP
layer had no corresponding tensor and the default conversion could not load.
All three fidelity gates passed before the full scoring runs.

### Pinned model revisions and hashes

| Model | Official revision |
|---|---|
| crh225/plumb-4b | `1c5f4483addb049476ac33107796d217a1dc089d` |
| Qwen/Qwen3-Reranker-0.6B | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| Qwen/Qwen3-Reranker-8B | `77d193c791ed757ca307ee72715aa132723da912` |

SHA-256 of official weight files and the exact GGUFs used:

| Artifact | SHA-256 |
|---|---|
| plumb model.safetensors | `89e119ea07f4c5b4b6715560c7de6694ec3b777dfd0e62351da0b33986d2e1e1` |
| Qwen 0.6B model.safetensors | `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b` |
| Qwen 8B safetensors shard 1/5 | `22cdfea4a13b7b3e866573800eeeb638fc38962940adf631d06dc03befed047a` |
| Qwen 8B safetensors shard 2/5 | `d2163b74137e35b4614bd2aa5bf27bcb07de4ca61c6962495feb968385eb0df8` |
| Qwen 8B safetensors shard 3/5 | `a5038caa78c817e8acce6806104869675938a33fd4e60ed038e9931d390d6989` |
| Qwen 8B safetensors shard 4/5 | `247f85538c5996d4c296291b0e4004f618c9b17ca8cdc25d1fc726567eb15803` |
| Qwen 8B safetensors shard 5/5 | `8ba41b93c2e4ec8339ad16b000bc977fde196aeac054956cbfc8c0186ee6d4cf` |
| plumb F16 GGUF, no MTP | `c710d60dc8b07f1577e9a5911cb26bccff3108cc516d840f22075e5dac497f63` |
| Qwen 0.6B F16 GGUF, code instruction | `61e3a4616f957a078f0b09fcdfc57847182fe3c98dba366d4c4bc7c47e9b8ee2` |
| Qwen 8B F16 GGUF, history instruction | `a38d0365b85abbdad57e306137143f4dee225afa9c87be3d60f51a9a0a1f0555` |

## Ambient scoring latency

**Shared-machine observations, for scale only.** Sequential per-candidate elapsed
time includes request overhead and prompt preparation; model loading is excluded.
There was no exclusive-machine reservation. Scoring processes continued through
planned periods when AFT was unavailable; a timed-out wrapper was resumed from
completed-query checkpoints.

| Tool / model | Candidates | Mean ms | Median ms | p95 ms | Mean ×50, seconds |
|---|---:|---:|---:|---:|---:|
| AFT / plumb-4b | 3,613 | 633.2 | 594.0 | 1,119.2 | 31.7 |
| AFT / Qwen3-0.6B | 3,613 | 40.2 | 29.6 | 93.3 | 2.0 |
| MC / plumb-4b | 3,730 | 488.1 | 460.8 | 832.6 | 24.4 |
| MC / Qwen3-8B | 3,730 | 262.3 | 271.1 | 342.8 | 13.1 |

The final column is a serial scale estimate, not a measured batch endpoint or a
production latency promise. Different execution periods and prompt lengths
prevent treating this as a controlled cross-model speed benchmark.

## Per-query union sizes

Counts below follow retained, deduplicated input order within each tool; private
artifacts map these positions to query IDs and the original queries. Zero denotes
a complete empty search.
Medians were 19 for both tools. These are union counts, not full-50 relevance
counts.

AFT, 85 queries:

```text
1,15,18,15,20,19,20,13,19,22,24,23,17,17,17,1,1,20,14,21,24,20,17,20,23,18,21,17,24,18,21,22,23,26,7,21,22,19,14,18,1,21,19,20,20,19,21,18,20,20,17,17,23,0,2,24,19,18,21,0,16,21,17,20,15,19,19,22,18,21,21,19,20,20,21,22,20,20,19,19,12,21,19,21,18
```

MC, 92 queries:

```text
19,20,23,19,19,16,20,22,21,19,19,19,1,2,1,19,19,21,1,25,19,1,20,23,19,20,21,15,19,16,20,22,2,18,2,22,2,1,22,21,17,20,24,18,2,2,18,19,17,22,20,17,19,0,20,21,19,25,21,18,19,22,20,20,3,19,20,25,22,22,19,21,0,3,2,18,18,18,18,17,15,21,20,20,2,3,20,21,18,18,20,18
```

## Limits and what surprised us

- **The dedicated rerankers improved ordering and displayed relevance substantially**, even
  though AFT queries were often short identifiers and many candidates had no snippet.
  The results favor them as the next evaluation candidates, not an immediate serving change.
- **Judge order sensitivity was material: 15.9% of repeated labels moved.** Shared labels
  eliminate between-arm judging differences within the primary run, not systematic judge
  error. Replacing controlled labels with their repeat judgments did not erase the
  family-reranker gains; confidence intervals still omit label uncertainty and possible
  within-project query dependence.
- The strict initial fidelity gate rejected a numerically close conversion over a near tie.
  The amendment is disclosed above; only 12, 10, and 2 of 190 pairs were exempt, and only one
  actually disagreed. No missing-classifier-tensor workaround or substitute model was used.
- Missing historical project-attribution rows and borrowed-index fallback were substantial
  replay hazards. MC attribution was validated against surviving rows; AFT conclusions are
  conditional on the 85 complete-index queries, with concentrated exclusions documented.
- The optional **full-50 relevance audit was skipped**, so full-pool recall and exhaustive
  ideal DCG are unknown. The optional cross-size reranker runs were also skipped to bound
  shared-machine runtime. Do not attribute differences between AFT and MC to model size
  alone: tool, corpus, and model size all differ.

All raw queries, candidate text, session identifiers, project paths, scores, and
labels remain in private evaluation artifacts and the authorized judge channel.
This public report contains methodology, provenance, counts, and aggregate results only.
