# Does top-50 reranking improve the top 10?

## Erratum: the original plumb probability readout

The first report's plumb arm did **not consistently rank by
P(A | A or B)**. Its grammar-constrained, post-sampling readout sometimes
returned raw P(A) over the whole vocabulary instead. For one reproduced
candidate, **P(A) = 0.8966 and P(B) = 0.0288**, summing to **0.9254**;
the saved score was 0.8966, rather than the conditional probability 0.9689.
The pinned sampler applied the grammar only when the initially sampled token
required it, so the defect was conditional, not a constant rescaling.

The original 20-pair plumb fidelity gate missed this case: every saved response
had A/B probability mass within **2.98 × 10⁻⁸ of 1**. Its passing error bound
therefore did not validate the readout on the remaining candidates. The first
five queries per tool reproduced bit-for-bit after rebuilding the exact GGUF
(166 AFT and 250 MC pairs), including the defective score above.

**The Qwen arms are unaffected:** they used `--reranking` rank pooling, not
this next-token readout. Historical tables below are retained as reported;
the Jev / sol follow-up distinguishes corrected plumb results from the original
unnormalized comparator. No additional luna judging is used for the correction.

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

**After seeing that result, the order gate was amended.** The reason was
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

## Follow-up: cross-size rerankers and MC head-15 variants

This follow-up uses the same frozen text, candidate identities, query populations,
official weight revisions, instruction lines, and llama.cpp pin as the original
run. No retrieval was rerun and no serving code changed. The new full-pool arms
are **0.6B on MC** and **8B on AFT**. Both MC head-15 variants are included:
they rerank only baseline positions 1–15, leave positions 16–50 in baseline
order, and display the first 10. They reuse full-pool scores; exact ties retain
baseline order. Small and empty sets are retained.

**Takeaways on the expanded common denominator:**

- **MC 0.6B retains a substantial lift:** nDCG@10 rises from 0.4244 to 0.5669,
  a gain of 0.1425 [0.0974, 0.1893], versus 0.5953 for the full-pool 8B.
- **MC 0.6B head-15 trades relevance for cost:** nDCG@10 is 0.5229, a gain
  of 0.0986 [0.0657, 0.1336]. Its serial cost estimate is about 0.40 seconds
  per 15 candidates, versus 1.34 seconds per 50 for the same new scoring run.
- **AFT 8B adds little to the 0.6B point estimate:** nDCG@10 is 0.8338
  versus 0.8273, while strict P@10 is slightly lower (0.2118 versus 0.2165).
  These baseline-relative intervals do not establish an 8B-versus-0.6B win.
- **Judge uncertainty remains material:** 13/60 repeated new labels moved
  (21.7%). The reported bootstrap intervals hold those labels fixed.

### Reproduction before new scoring

All six original nDCG@10 results were recomputed from primary v2 labels and
saved orders, matching the original report to four decimals. The other three
metrics and all reported paired bootstrap intervals for AFT, MC, and MC
full-50 also reproduced to four decimals. Recovered bootstrap details are
Python `random.Random(2609)`, 10,000 resamples using `randrange(n)`, reset per
population, and linear percentile endpoints; each population shares resamples
across arms and metrics.

| Population | Baseline | plumb-4b | Original Qwen arm |
|---|---:|---:|---:|
| AFT | 0.6405 | 0.7593 | 0.8309 (0.6B) |
| MC | 0.4338 | 0.4974 | 0.6105 (8B) |

Both original instructed GGUFs were rebuilt from the pinned official weights.
Their SHA-256 hashes exactly matched `61e3a461…` (0.6B/code) and
`a38d0365…` (8B/history), including the full hashes recorded above.
Re-scoring the first five retained query sets per tool produced **bit-identical
numeric scores**: AFT 166 pairs and MC 250 pairs, maximum absolute error **0**
for each. These are exact reproduction results, not merely tolerance passes.

### New conversions and fidelity gates

The new files are F16, with only the tool-specific rerank instruction changed
by the pinned official metadata utility. The 8B/code file reuses the freshly
rebuilt, hash-verified 8B/history tensors. References used FP32 MPS, PyTorch
2.11.0, Transformers 4.57.6, and Python 3.14.2. The amended gate remained
maximum absolute error ≤0.005 and no pairwise order violation outside a
reference gap strictly below 0.005. Neither new gate required an amendment.

| New arm | Real pairs / queries | Inside-band pairs / 190 | Order disagreements | Outside-band violations | Max absolute error | Result |
|---|---:|---:|---:|---:|---:|---|
| MC / 0.6B | 20 / 5 | 6 | 0 | 0 | 0.0041511059 | pass |
| AFT / 8B | 20 / 6 | 6 | 0 | 0 | 0.0006102622 | pass |

The MC gate retained the previous 20 real MC pairs. The AFT gate selected the
first 20 pairs from the final frozen AFT corpus, taking up to four candidates
per query in saved order, without consulting grades. Every gate and scoring
input was checked against the corpus text and SHA-256 digests. Both gates
passed before their respective full scoring runs.

| New GGUF | SHA-256 |
|---|---|
| Qwen 0.6B F16, history instruction | `0de02616633bb00fa2b6782dcf000c3f2b1a70906280c7945cbc4ba04b9dc349` |
| Qwen 8B F16, code instruction | `d774e75121988287b21524944c79e9943a13920680ce9b82ff22e41ca243d410` |

### Shared expanded-union metrics

**All rows below, including old arms, use the same expanded judged union per
query.** Additional judged items can increase ideal DCG, so old-arm nDCG
values here may be lower than the original table even though their ranking
and labels did not change. Precision and MRR retain their original values.
Do not compare a new arm here to an old-arm nDCG denominator from above.
The union is still not an exhaustive full-pool relevance audit.

| Population | Arm | nDCG@10 | P@10 ≥1 | P@10 =2 | MRR@10 =2 |
|---|---|---:|---:|---:|---:|
| AFT, n=85 | baseline | 0.6367 | 0.3800 | 0.1682 | 0.6410 |
| AFT, n=85 | plumb-4b | 0.7554 | 0.5035 | 0.2000 | 0.6782 |
| AFT, n=85 | Qwen3-Reranker-0.6B, full pool | 0.8273 | 0.5447 | 0.2165 | 0.7651 |
| AFT, n=85 | Qwen3-Reranker-8B, full pool | 0.8338 | 0.5435 | 0.2118 | 0.7735 |
| MC, n=92 | baseline | 0.4244 | 0.3109 | 0.0793 | 0.3594 |
| MC, n=92 | plumb-4b | 0.4852 | 0.3293 | 0.1076 | 0.4395 |
| MC, n=92 | Qwen3-Reranker-8B, full pool | 0.5953 | 0.3978 | 0.1109 | 0.5091 |
| MC, n=92 | Qwen3-Reranker-0.6B, full pool | 0.5669 | 0.3870 | 0.1098 | 0.5216 |
| MC, n=92 | Qwen3-Reranker-0.6B, head 15 | 0.5229 | 0.3489 | 0.0902 | 0.5149 |
| MC, n=92 | Qwen3-Reranker-8B, head 15 | 0.5342 | 0.3641 | 0.0946 | 0.4841 |
| MC full-50, n=74 | baseline | 0.4988 | 0.3824 | 0.0973 | 0.4333 |
| MC full-50, n=74 | plumb-4b | 0.5695 | 0.4054 | 0.1324 | 0.5329 |
| MC full-50, n=74 | Qwen3-Reranker-8B, full pool | 0.6995 | 0.4905 | 0.1365 | 0.6194 |
| MC full-50, n=74 | Qwen3-Reranker-0.6B, full pool | 0.6642 | 0.4770 | 0.1351 | 0.6349 |
| MC full-50, n=74 | Qwen3-Reranker-0.6B, head 15 | 0.6096 | 0.4297 | 0.1108 | 0.6267 |
| MC full-50, n=74 | Qwen3-Reranker-8B, head 15 | 0.6236 | 0.4486 | 0.1162 | 0.5883 |

### Paired deltas versus baseline

These use the reproduced seed and percentile bootstrap method above, holding
all v2 and v3 labels fixed. Intervals omit judge uncertainty, within-project
dependence, and multiple-comparison adjustment. Baseline deltas are zero.

| Population / arm | ΔnDCG@10 | ΔP@10 ≥1 | ΔP@10 =2 | ΔMRR@10 =2 |
|---|---|---|---|---|
| AFT, n=85 / plumb-4b | 0.1187 [0.0628, 0.1750] | 0.1235 [0.0788, 0.1682] | 0.0318 [0.0118, 0.0529] | 0.0372 [-0.0322, 0.1100] |
| AFT, n=85 / Qwen3-Reranker-0.6B, full pool | 0.1906 [0.1394, 0.2433] | 0.1647 [0.1224, 0.2071] | 0.0482 [0.0282, 0.0694] | 0.1241 [0.0602, 0.1947] |
| AFT, n=85 / Qwen3-Reranker-8B, full pool | 0.1971 [0.1483, 0.2489] | 0.1635 [0.1235, 0.2047] | 0.0435 [0.0247, 0.0647] | 0.1325 [0.0692, 0.2008] |
| MC, n=92 / plumb-4b | 0.0609 [0.0143, 0.1109] | 0.0185 [-0.0163, 0.0544] | 0.0283 [0.0109, 0.0500] | 0.0801 [0.0124, 0.1489] |
| MC, n=92 / Qwen3-Reranker-8B, full pool | 0.1709 [0.1255, 0.2212] | 0.0870 [0.0576, 0.1196] | 0.0315 [0.0163, 0.0489] | 0.1497 [0.0861, 0.2170] |
| MC, n=92 / Qwen3-Reranker-0.6B, full pool | 0.1425 [0.0974, 0.1893] | 0.0761 [0.0402, 0.1130] | 0.0304 [0.0152, 0.0500] | 0.1622 [0.1039, 0.2250] |
| MC, n=92 / Qwen3-Reranker-0.6B, head 15 | 0.0986 [0.0657, 0.1336] | 0.0380 [0.0185, 0.0587] | 0.0109 [0.0033, 0.0207] | 0.1555 [0.0980, 0.2173] |
| MC, n=92 / Qwen3-Reranker-8B, head 15 | 0.1099 [0.0769, 0.1445] | 0.0533 [0.0348, 0.0728] | 0.0152 [0.0076, 0.0239] | 0.1247 [0.0673, 0.1868] |
| MC full-50, n=74 / plumb-4b | 0.0707 [0.0127, 0.1307] | 0.0230 [-0.0203, 0.0676] | 0.0351 [0.0135, 0.0608] | 0.0996 [0.0168, 0.1834] |
| MC full-50, n=74 / Qwen3-Reranker-8B, full pool | 0.2007 [0.1476, 0.2588] | 0.1081 [0.0730, 0.1459] | 0.0392 [0.0216, 0.0608] | 0.1861 [0.1090, 0.2685] |
| MC full-50, n=74 / Qwen3-Reranker-0.6B, full pool | 0.1654 [0.1121, 0.2216] | 0.0946 [0.0514, 0.1405] | 0.0378 [0.0189, 0.0608] | 0.2016 [0.1310, 0.2772] |
| MC full-50, n=74 / Qwen3-Reranker-0.6B, head 15 | 0.1108 [0.0737, 0.1514] | 0.0473 [0.0230, 0.0743] | 0.0135 [0.0027, 0.0257] | 0.1934 [0.1256, 0.2687] |
| MC full-50, n=74 / Qwen3-Reranker-8B, head 15 | 0.1248 [0.0881, 0.1632] | 0.0662 [0.0432, 0.0892] | 0.0189 [0.0095, 0.0297] | 0.1550 [0.0849, 0.2322] |

### Newly displayed and lost grade-2 items

As in the original report, these are **query counts**, not total item counts:
new/lost means at least one such membership change, and both can occur for
one query. Zero/some columns count changes in having any grade-2 top-10 item.

| Population / arm | New item | Lost item | Zero → some | Some → zero |
|---|---:|---:|---:|---:|
| AFT, n=85 / plumb-4b | 26 | 13 | 5 | 3 |
| AFT, n=85 / Qwen3-Reranker-0.6B, full pool | 27 | 6 | 6 | 1 |
| AFT, n=85 / Qwen3-Reranker-8B, full pool | 25 | 6 | 5 | 2 |
| MC, n=92 / plumb-4b | 23 | 9 | 8 | 1 |
| MC, n=92 / Qwen3-Reranker-8B, full pool | 24 | 4 | 8 | 0 |
| MC, n=92 / Qwen3-Reranker-0.6B, full pool | 22 | 5 | 7 | 0 |
| MC, n=92 / Qwen3-Reranker-0.6B, head 15 | 12 | 4 | 5 | 0 |
| MC, n=92 / Qwen3-Reranker-8B, head 15 | 12 | 0 | 5 | 0 |

MC full-50 has the same membership-change counts as MC overall; small sets
cannot change top-10 membership.

### New-only judging and permutation control

Only **418 previously unlabelled candidates** were judged: 129 AFT and
289 MC, in **140 new primary batches** (66 AFT, 74 MC).
No v2 label was replaced or re-judged. New identities were sorted per query,
split into batches of at most eight, and assigned short opaque IDs from that
canonical order. Every call used the unchanged judge prompt and original
one-line scope, `openai/gpt-6-luna`, variant `low`, temperature 0, output cap
512, no tools, and a fresh `synapse-judge-rerank2609-v3-` session.
The producing model was verified through `session.read` origin; variant and
temperature were independently verified through digest-checked own-session
`RunStarted.config` records matching each admitted run ID.

**New items were labelled in batches containing only other new items, so their
batch context differs from v2.** The permutation rate on v3 batches is the
check on that sensitivity, not proof that mixed v2/v3 batch contexts are
equivalent or that judge bias is absent.

With seed 2609, **14 batches** (ceil of 10% of new primary batches) were
selected among batches with at least two items. Every repeated order genuinely
changed; IDs, text, prompt, and generation configuration stayed fixed.
**13/60 labels moved (21.7%)**, across
**6/14 repeated batches**. The primary labels, not repeats,
were used for the metric tables.

### Ambient latency and head-15 scale

Shared-Mac sequential per-candidate observations include request overhead and
exclude model load. No exclusive reservation was made. The ×50 and ×15
columns are serial scale estimates from mean candidate latency, **not measured
batch-endpoint or production head-15 latency**. Head-only variants reused
full-pool scores and were not separately timed.

| Tool / model | Candidates | Mean ms / candidate | Median ms | p95 ms | Mean ×50, seconds | Mean ×15, seconds |
|---|---:|---:|---:|---:|---:|---:|
| AFT / 8B (new) | 3,613 | 138.0 | 117.9 | 189.8 | 6.90 | 2.07 |
| MC / 0.6B (new) | 3,730 | 26.9 | 21.4 | 58.3 | 1.34 | 0.40 |
| MC / 8B (original) | 3,730 | 262.3 | 271.1 | 342.8 | 13.11 | 3.93 |

Different execution periods, prompt lengths, and cache behavior prevent a
controlled speed comparison. Reranking 15 limits which evidence can enter the
display, not just inference cost. These are relevance and ambient-cost results,
not measurements of agent task completion. All scripts, raw outputs, frozen
inputs, and judge evidence for this follow-up are retained in private evaluation
storage; this report contains only aggregate results and model provenance.

## Jev follow-up: fixed full-pool sol references

This follow-up replaces new-item union judging with a **fixed, full-candidate
reference**. Adding an arm no longer expands the judge pool or moves another
arm's denominator. All earlier sections are retained, subject to the plumb
readout erratum above. Before new scoring, all **16** cross-size follow-up
expanded-union nDCG values were recomputed from saved orders and primary v2+
v3 labels and matched to four decimals. No v4 luna judgments were run.

**Primary-metric point estimates (mean of the two references):**

- AFT: baseline **0.6386**; leader **Qwen3-Reranker-8B 0.7633**; best Jev point estimate **Jev noul 0.7319**.
- MC: baseline **0.5141**; leader **Jev noul 0.6055**; best Jev point estimate **Jev noul 0.6055**.

Point-estimate leaders are not automatically distinguishable: see the paired
intervals and judge-rerun flags below. The failed plumb score gates leave
the requested Jev-versus-plumb ordinal comparison unresolved.

### Scoring and privacy

The same 85 AFT and 92 MC queries, including empty and small sets, and all
3,613 AFT / 3,730 MC frozen candidate texts were reused without retrieval or
text enrichment. Exact score ties preserve baseline order. MC head-15 arms
rerank only baseline positions 1–15 and leave the rest untouched.

**Jev is a hosted third-party API: frozen candidate texts and search queries
were sent to TypeSafe.** Every request had one candidate as its unchanged
`state` and one decision, with the original plumb relevance criterion.
The noul arm ranks by the returned probability of true. The score decision
uses the documented ordered `criteria` list with three levels, copied
verbatim from the existing luna judge's grade definitions:
- 0: not relevant or no useful evidence for the query.
- 1: partly relevant, tangential but useful, or incomplete evidence.
- 2: directly relevant evidence that helps answer the query or locate the requested code/history.

**Sharing the ranker/evaluator definition of relevance was intentional.**
Expected grade **E = P(1) + 2·P(2)** was fixed before results; P(2)-only
ordering is a separately labelled **secondary** arm. These are two orders
from one score response, not two scoring calls. The API documentation was
read before execution; score decisions accept 2–10 levels. Only one decision
was sent per request, with at most four Jev requests in flight. Responses
were checkpointed; transient failures used exponential backoff.

Resolved Jev version(s): **jev-1.13.0**. Primary
sample responses were reused in the full run rather than rescoring them.

A client defect rejected **21 MC score responses** for not summing to one
within an undocumented 0.001 tolerance before saving their bodies. Those
bodies and usage counts were lost. With explicit authorization, **only those
21 responses were requested once again**, adding 21 calls; the corrected
client saves raw responses before validation and uses returned probabilities
as-is, without renormalizing Jev E. Jev’s repeat spread means these 21
replacements need not match what the lost originals said.

### Plumb readout correction and failed score gates

The official plumb revision and llama.cpp pin above were retained. Rebuilding
with `--outtype f16 --no-mtp` reproduced SHA-256
`c710d60dc8b07f1577e9a5911cb26bccff3108cc516d840f22075e5dac497f63`.
The historical readout reproduced all 416 first-five-query scores exactly.

The corrected readout uses **pre-sampling, pre-grammar log probabilities**,
requires every declared option-letter token to be present, divides their
log probabilities by the checkpoint temperature 2.07, and normalizes only
over A/B (noul) or A/B/C (score). The shared vocabulary normalizer cancels.
A grammar constrains only the unused generated token to ASCII, avoiding an
incomplete UTF-8 token suppressing the probability payload; probabilities
are captured before that grammar. No absent token is imputed.

Both noul and score were checked on the same fixed 20 AFT pairs, selected
without grades (up to four per query). Reference was pinned jevk5 v0.2.0,
FP32 Torch on MPS. Every component and ranking statistic had to satisfy
maximum absolute error ≤0.005 and no pairwise order violation when the
reference gap was at least 0.005. The tolerance and pairs were not changed.

| Readout | Max P(0)/P(true) error | Max P(1)/P(false) error | Max P(2) error | Max E error | Outside-band order violations | Gate |
|---|---:|---:|---:|---:|---:|---|
| noul F16 | 0.0046361 | 0.0046361 | — | — | 0 | pass |
| score F16 | 0.0041070 | 0.0011069 | 0.0052139 | 0.0093209 | 0 | fail |
| score F32 | 0.0032637 | 0.0008598 | 0.0041234 | 0.0073871 | 0 | fail |

F32 was converted from the same official safetensors with `--outtype f32
--no-mtp`, SHA-256 `ce64c779598089710e7a8edce1affeb67584d3553dacced5a6df4c8843ef185e`. It reduced error but did **not**
pass the unchanged expected-grade gate. The discrepancy therefore cannot
be attributed solely to F16 weight rounding; **its cause is not established**.
Consequently **no full plumb score
arm was run**, including no P(2)-only or head-15 score arm. This is an
explicit missing comparison, not a zero result or a relaxed fidelity claim.
Corrected plumb noul passed and was rescored over every candidate. The old
unnormalized noul ranking remains a separate comparator. In AFT, correction
changed six complete query orders, four top-10 orders and one top-10
membership set; **all MC orders were unchanged**.

Maximum F16–F32 differences on those same 20 score pairs: P(0)=0.0009197, P(1)=0.0002472, P(2)=0.0010905, E=0.0019338.

### Full-pool reference and repeatability

Judge: **`openai/gpt-6-sol`, variant `medium`, temperature 0**, output cap
2,048, no tools, through BROCA `session.send`. Each call used a fresh
`synapse-judge-rerank2609-sol-` session. Fixed instructions preceded the
original query, original one-line scope, and all frozen candidate texts.
Opaque IDs came from canonical candidate identity order, not any arm rank.
Presentation was shuffled with seed 2609 + query index for A and 9062 +
query index for B (zero-based global order: AFT then MC). Every query was
submitted in both passes, including empty sets. Singleton/empty sets cannot
have a genuinely different row permutation. The exact seeded shuffles also
coincided for **three two-candidate MC queries**. These same-order repeats
are excluded from the sol B-versus-A repeatability mean (MC n=89), but
retained in every arm metric (MC n=92); no additional calls were made.
For a two-item pool it is impossible for both passes to avoid baseline
order and also differ from each other. We retained the specified random
seeds rather than forcing an order. Calls ran with at most three
query workers; pass B started only after all pass-A checkpoints existed.

The judge returned `{"relevant":[ids, most relevant first],
"not_relevant":[ids]}`. Relevant means helping answer the query or locate
what it asks for. Every ID had to appear exactly once across the two lists;
invalid responses were logged and retried in fresh sessions. Model origin
was independently checked through `session.read`; medium, temperature, output
cap and no-tools admission were verified from digest-checked WAL records
matching the run ID. Account-limit exhaustion would stop further submissions,
not trigger retries around the limit.

Accepted references: **354**; invalid-response retries: **3**.
Dry-run input/output tokens were **2,215 / 431** and **4,708 / 786**.
Before the full run, their mean projected **1,225,371 input / 215,409 output
tokens** for both passes; this was an uncertain two-query extrapolation.

Jev repeat probes were **not deterministic**. For 20 candidates scored twice
per decision type, noul had 7/20 identical probabilities, maximum absolute
spread **0.04**, mean **0.013**. Score had 16/60 identical probability
components, maximum spread **0.09**, mean **0.018**. Full-run candidates
were otherwise scored once per decision type. This is a property of the
hosted model, not a local numerical-conversion fidelity pass.

Sol B-versus-A agreement is an **empirical repeatability benchmark, not a
mathematical upper bound**: an arm can agree with A more than B happens to.
For overlap and precision, the B arm displays only its ordered relevant
list (up to 10); its irrelevant items enter only the tied-tail RBO.

| Tool | Sol B vs A overlap@10 | RBO p=0.9 | P@10 |
|---|---:|---:|---:|
| AFT, n=85 | 0.7704 | 0.5644 | 0.3153 |
| MC, n=89 | 0.6201 | 0.5191 | 0.1787 |

### Metric definitions and results

**Primary: overlap@10** = arm/sol top-10 intersection divided by
min(10, number of sol-relevant items, candidate count), or 0 if none are
relevant. **P@10** counts displayed sol-relevant items and always divides
by 10, including small sets. Empty queries contribute zero.

**RBO uses finite extrapolated expected-prefix-overlap tied-tail RBO**,
p=0.9. Sol-relevant items have strict ranks; all not-relevant items form
one tied tail. At depth d, each tail member has inclusion probability
(d − relevant-count)/tail-size, clipped to [0,1]. Expected overlap is the
sum of products of the two lists’ inclusion probabilities. RBO is
`(1−p) Σ[d=1..n] p^(d−1) overlap(d)/d + p^n`, since both full lists
contain the same candidate set. B’s tail is tied too. For an empty pool
RBO is 0. Thus RBO can be positive even when sol marks nothing relevant.

Every arm is scored against **both** fixed references. Each cell below is
**A / B / mean(A,B)**. Qwen orders are reused, not rescored.

| Tool / arm | overlap@10 A / B / mean | RBO A / B / mean | P@10 A / B / mean |
|---|---|---|---|
| AFT / baseline | 0.6378 / 0.6394 / 0.6386 | 0.4426 / 0.4448 / 0.4437 | 0.2294 / 0.2306 / 0.2300 |
| AFT / Qwen3-Reranker-0.6B | 0.7485 / 0.7350 / 0.7417 | 0.4988 / 0.5000 / 0.4994 | 0.2859 / 0.2871 / 0.2865 |
| AFT / Qwen3-Reranker-8B | 0.7710 / 0.7556 / 0.7633 | 0.5197 / 0.5147 / 0.5172 | 0.3000 / 0.2941 / 0.2971 |
| AFT / plumb noul, unnormalized readout (as first reported) | 0.7048 / 0.6861 / 0.6955 | 0.4665 / 0.4567 / 0.4616 | 0.2612 / 0.2529 / 0.2571 |
| AFT / plumb-4b noul, corrected F16 | 0.7068 / 0.6878 / 0.6973 | 0.4714 / 0.4618 / 0.4666 | 0.2624 / 0.2541 / 0.2582 |
| AFT / Jev noul | 0.7358 / 0.7279 / 0.7319 | 0.4839 / 0.4845 / 0.4842 | 0.2800 / 0.2776 / 0.2788 |
| AFT / Jev score, E | 0.7319 / 0.7262 / 0.7290 | 0.4750 / 0.4842 / 0.4796 | 0.2788 / 0.2753 / 0.2771 |
| AFT / Jev score, P(2), secondary | 0.7276 / 0.7237 / 0.7256 | 0.4758 / 0.4845 / 0.4802 | 0.2753 / 0.2729 / 0.2741 |
| MC / baseline | 0.5116 / 0.5167 / 0.5141 | 0.4231 / 0.4190 / 0.4210 | 0.1152 / 0.1141 / 0.1147 |
| MC / Qwen3-Reranker-8B | 0.5689 / 0.5646 / 0.5667 | 0.4876 / 0.4788 / 0.4832 | 0.1576 / 0.1489 / 0.1533 |
| MC / Qwen3-Reranker-0.6B | 0.5487 / 0.5604 / 0.5545 | 0.4798 / 0.4792 / 0.4795 | 0.1522 / 0.1489 / 0.1505 |
| MC / Qwen3-Reranker-0.6B, head 15 | 0.5222 / 0.5256 / 0.5239 | 0.4620 / 0.4617 / 0.4618 | 0.1272 / 0.1217 / 0.1245 |
| MC / Qwen3-Reranker-8B, head 15 | 0.5391 / 0.5298 / 0.5345 | 0.4649 / 0.4595 / 0.4622 | 0.1293 / 0.1217 / 0.1255 |
| MC / plumb noul, unnormalized readout (as first reported) | 0.5349 / 0.5307 / 0.5328 | 0.4516 / 0.4519 / 0.4518 | 0.1391 / 0.1391 / 0.1391 |
| MC / plumb-4b noul, corrected F16 | 0.5349 / 0.5307 / 0.5328 | 0.4516 / 0.4519 / 0.4518 | 0.1391 / 0.1391 / 0.1391 |
| MC / plumb-4b noul, corrected F16, head 15 | 0.5296 / 0.5268 / 0.5282 | 0.4470 / 0.4473 / 0.4472 | 0.1250 / 0.1196 / 0.1223 |
| MC / Jev noul | 0.6118 / 0.5991 / 0.6055 | 0.4905 / 0.4877 / 0.4891 | 0.1565 / 0.1522 / 0.1543 |
| MC / Jev noul, head 15 | 0.5385 / 0.5316 / 0.5351 | 0.4661 / 0.4639 / 0.4650 | 0.1293 / 0.1250 / 0.1272 |
| MC / Jev score, E | 0.6023 / 0.5956 / 0.5989 | 0.4819 / 0.4769 / 0.4794 | 0.1554 / 0.1500 / 0.1527 |
| MC / Jev score, E, head 15 | 0.5366 / 0.5300 / 0.5333 | 0.4638 / 0.4610 / 0.4624 | 0.1283 / 0.1228 / 0.1255 |
| MC / Jev score, P(2), secondary | 0.5888 / 0.5810 / 0.5849 | 0.4746 / 0.4688 / 0.4717 | 0.1511 / 0.1446 / 0.1478 |
| MC / Jev score, P(2), secondary, head 15 | 0.5361 / 0.5295 / 0.5328 | 0.4629 / 0.4598 / 0.4614 | 0.1272 / 0.1228 / 0.1250 |
| MC / plumb noul, unnormalized readout (as first reported), head 15 | 0.5296 / 0.5268 / 0.5282 | 0.4470 / 0.4473 / 0.4472 | 0.1250 / 0.1196 / 0.1223 |

### Paired deltas versus baseline

10,000 paired query bootstrap resamples, `random.Random(2609)`, reset per
tool, linear percentile endpoints. Resample indices are shared across arms,
metrics and references within a tool. Each cell gives **A; B** as mean
delta [95% interval]. Baseline deltas are zero. These intervals omit
systematic judge error, within-project dependence and multiplicity adjustment.

| Tool / arm | Δoverlap@10 A; B | ΔRBO A; B | ΔP@10 A; B |
|---|---|---|---|
| AFT / Qwen3-Reranker-0.6B | 0.1106 [0.0572, 0.1684]; 0.0956 [0.0492, 0.1457] | 0.0562 [0.0303, 0.0823]; 0.0551 [0.0303, 0.0803] | 0.0565 [0.0294, 0.0859]; 0.0565 [0.0294, 0.0847] |
| AFT / Qwen3-Reranker-8B | 0.1331 [0.0783, 0.1932]; 0.1162 [0.0662, 0.1710] | 0.0771 [0.0525, 0.1035]; 0.0699 [0.0442, 0.0979] | 0.0706 [0.0435, 0.1012]; 0.0635 [0.0341, 0.0941] |
| AFT / plumb noul, unnormalized readout (as first reported) | 0.0670 [0.0020, 0.1306]; 0.0467 [-0.0112, 0.1031] | 0.0240 [-0.0011, 0.0488]; 0.0119 [-0.0121, 0.0367] | 0.0318 [0.0082, 0.0576]; 0.0224 [-0.0047, 0.0494] |
| AFT / plumb-4b noul, corrected F16 | 0.0689 [0.0040, 0.1327]; 0.0484 [-0.0100, 0.1051] | 0.0288 [0.0041, 0.0532]; 0.0170 [-0.0068, 0.0414] | 0.0329 [0.0082, 0.0588]; 0.0235 [-0.0035, 0.0506] |
| AFT / Jev noul | 0.0980 [0.0488, 0.1527]; 0.0885 [0.0380, 0.1423] | 0.0413 [0.0184, 0.0651]; 0.0396 [0.0163, 0.0647] | 0.0506 [0.0235, 0.0788]; 0.0471 [0.0188, 0.0776] |
| AFT / Jev score, E | 0.0941 [0.0421, 0.1513]; 0.0867 [0.0364, 0.1405] | 0.0324 [0.0090, 0.0564]; 0.0394 [0.0147, 0.0645] | 0.0494 [0.0235, 0.0776]; 0.0447 [0.0165, 0.0741] |
| AFT / Jev score, P(2), secondary | 0.0897 [0.0373, 0.1474]; 0.0843 [0.0348, 0.1377] | 0.0333 [0.0093, 0.0578]; 0.0397 [0.0144, 0.0655] | 0.0459 [0.0188, 0.0753]; 0.0424 [0.0141, 0.0729] |
| MC / Qwen3-Reranker-8B | 0.0573 [0.0120, 0.1003]; 0.0480 [-0.0082, 0.1031] | 0.0645 [0.0415, 0.0901]; 0.0598 [0.0388, 0.0837] | 0.0424 [0.0185, 0.0717]; 0.0348 [0.0120, 0.0609] |
| MC / Qwen3-Reranker-0.6B | 0.0371 [-0.0199, 0.0918]; 0.0437 [-0.0091, 0.0948] | 0.0567 [0.0341, 0.0812]; 0.0602 [0.0374, 0.0859] | 0.0370 [0.0141, 0.0641]; 0.0348 [0.0141, 0.0598] |
| MC / Qwen3-Reranker-0.6B, head 15 | 0.0106 [-0.0199, 0.0364]; 0.0090 [-0.0222, 0.0359] | 0.0389 [0.0227, 0.0566]; 0.0427 [0.0271, 0.0595] | 0.0120 [0.0033, 0.0217]; 0.0076 [-0.0022, 0.0185] |
| MC / Qwen3-Reranker-8B, head 15 | 0.0275 [0.0094, 0.0498]; 0.0131 [-0.0196, 0.0424] | 0.0419 [0.0256, 0.0598]; 0.0406 [0.0250, 0.0580] | 0.0141 [0.0054, 0.0239]; 0.0076 [-0.0022, 0.0174] |
| MC / plumb noul, unnormalized readout (as first reported) | 0.0233 [-0.0329, 0.0790]; 0.0140 [-0.0447, 0.0694] | 0.0285 [0.0045, 0.0546]; 0.0330 [0.0084, 0.0594] | 0.0239 [0.0011, 0.0489]; 0.0250 [0.0043, 0.0489] |
| MC / plumb-4b noul, corrected F16 | 0.0233 [-0.0329, 0.0790]; 0.0140 [-0.0447, 0.0694] | 0.0285 [0.0045, 0.0546]; 0.0330 [0.0084, 0.0594] | 0.0239 [0.0011, 0.0489]; 0.0250 [0.0043, 0.0489] |
| MC / plumb-4b noul, corrected F16, head 15 | 0.0180 [0.0018, 0.0382]; 0.0101 [-0.0226, 0.0397] | 0.0240 [0.0059, 0.0436]; 0.0283 [0.0098, 0.0486] | 0.0098 [0.0022, 0.0185]; 0.0054 [-0.0054, 0.0174] |
| MC / Jev noul | 0.1002 [0.0564, 0.1482]; 0.0825 [0.0278, 0.1374] | 0.0675 [0.0427, 0.0946]; 0.0688 [0.0440, 0.0962] | 0.0413 [0.0217, 0.0631]; 0.0380 [0.0174, 0.0620] |
| MC / Jev noul, head 15 | 0.0269 [0.0087, 0.0490]; 0.0150 [-0.0176, 0.0446] | 0.0431 [0.0255, 0.0624]; 0.0449 [0.0279, 0.0636] | 0.0141 [0.0065, 0.0239]; 0.0109 [0.0011, 0.0217] |
| MC / Jev score, E | 0.0907 [0.0490, 0.1360]; 0.0789 [0.0274, 0.1315] | 0.0588 [0.0359, 0.0847]; 0.0579 [0.0348, 0.0837] | 0.0402 [0.0207, 0.0620]; 0.0359 [0.0163, 0.0598] |
| MC / Jev score, E, head 15 | 0.0250 [0.0048, 0.0484]; 0.0133 [-0.0204, 0.0440] | 0.0407 [0.0235, 0.0595]; 0.0421 [0.0251, 0.0607] | 0.0130 [0.0043, 0.0228]; 0.0087 [-0.0022, 0.0207] |
| MC / Jev score, P(2), secondary | 0.0772 [0.0358, 0.1220]; 0.0643 [0.0119, 0.1178] | 0.0515 [0.0295, 0.0759]; 0.0498 [0.0277, 0.0746] | 0.0359 [0.0163, 0.0576]; 0.0304 [0.0109, 0.0533] |
| MC / Jev score, P(2), secondary, head 15 | 0.0245 [0.0045, 0.0473]; 0.0129 [-0.0210, 0.0431] | 0.0398 [0.0230, 0.0584]; 0.0409 [0.0241, 0.0597] | 0.0120 [0.0033, 0.0207]; 0.0087 [-0.0000, 0.0185] |
| MC / plumb noul, unnormalized readout (as first reported), head 15 | 0.0180 [0.0018, 0.0382]; 0.0101 [-0.0226, 0.0397] | 0.0240 [0.0059, 0.0436]; 0.0283 [0.0098, 0.0486] | 0.0098 [0.0022, 0.0185]; 0.0054 [-0.0054, 0.0174] |

### Leader gaps sensitive to the judge rerun

Leaders are selected separately per metric by mean(A,B). An arm is flagged
when the paired leader-minus-arm interval includes zero under A **or** B,
or when their point-estimate ordering swaps between A and B. This is a
repeatability warning, not an equivalence test. It does not compare a gap
to one minus an agreement score. Full paired gap intervals are retained
privately with the per-query results.

| Tool / metric | Mean-reference leader | Flagged comparators |
|---|---|---|
| AFT / overlap10 | Qwen3-Reranker-8B | Qwen3-Reranker-0.6B |
| AFT / rbo09 | Qwen3-Reranker-8B | none |
| AFT / p10 | Qwen3-Reranker-8B | Qwen3-Reranker-0.6B; Jev noul; Jev score, E; Jev score, P(2), secondary |
| MC / overlap10 | Jev noul | Qwen3-Reranker-8B; Qwen3-Reranker-0.6B; Jev score, E; Jev score, P(2), secondary |
| MC / rbo09 | Jev noul | Qwen3-Reranker-8B; Qwen3-Reranker-0.6B; Jev score, E |
| MC / p10 | Jev noul | Qwen3-Reranker-8B; Qwen3-Reranker-0.6B; Jev score, E; Jev score, P(2), secondary |

### Link to existing luna labels

Luna v2/v3 labelled only the old top-10 unions, not the full pool. Unknown
luna grades are **not** treated as irrelevant. The first fraction below is
the requested share of all sol-relevant items known to have luna grade ≥1;
the second restricts its denominator to sol-relevant items that luna judged.
The third is the share of all existing luna grade-2 items sol marked relevant.

| Tool / reference | Luna ≥1 / all sol-relevant | Luna ≥1 / luna-judged sol-relevant | Sol-relevant / luna grade-2 |
|---|---|---|---|
| AFT / A | 295/370 (79.7%) | 295/312 (94.6%) | 186/212 (87.7%) |
| AFT / B | 286/365 (78.4%) | 286/304 (94.1%) | 190/212 (89.6%) |
| MC / A | 174/216 (80.6%) | 174/186 (93.5%) | 87/126 (69.0%) |
| MC / B | 161/196 (82.1%) | 161/175 (92.0%) | 83/126 (65.9%) |

Existing luna grade-2 membership changes versus baseline, as **query counts**.
New and lost may overlap. Because new full-pool items lack luna labels,
these count **known** grade-2 items only, not exhaustive grade-2 relevance.

| Tool / arm | Known new | Known lost | Known zero → some | Known some → zero |
|---|---:|---:|---:|---:|
| AFT / Qwen3-Reranker-0.6B | 27 | 6 | 6 | 1 |
| AFT / Qwen3-Reranker-8B | 25 | 6 | 5 | 2 |
| AFT / plumb noul, unnormalized readout (as first reported) | 26 | 13 | 5 | 3 |
| AFT / plumb-4b noul, corrected F16 | 26 | 13 | 5 | 3 |
| AFT / Jev noul | 27 | 11 | 6 | 2 |
| AFT / Jev score, E | 26 | 11 | 6 | 1 |
| AFT / Jev score, P(2), secondary | 26 | 10 | 6 | 0 |
| MC / Qwen3-Reranker-8B | 24 | 4 | 8 | 0 |
| MC / Qwen3-Reranker-0.6B | 22 | 5 | 7 | 0 |
| MC / Qwen3-Reranker-0.6B, head 15 | 12 | 4 | 5 | 0 |
| MC / Qwen3-Reranker-8B, head 15 | 12 | 0 | 5 | 0 |
| MC / plumb noul, unnormalized readout (as first reported) | 23 | 9 | 8 | 1 |
| MC / plumb-4b noul, corrected F16 | 23 | 9 | 8 | 1 |
| MC / plumb-4b noul, corrected F16, head 15 | 11 | 4 | 5 | 1 |
| MC / Jev noul | 25 | 6 | 7 | 1 |
| MC / Jev noul, head 15 | 11 | 2 | 5 | 1 |
| MC / Jev score, E | 24 | 5 | 7 | 0 |
| MC / Jev score, E, head 15 | 12 | 1 | 5 | 0 |
| MC / Jev score, P(2), secondary | 24 | 7 | 7 | 1 |
| MC / Jev score, P(2), secondary, head 15 | 11 | 1 | 5 | 0 |
| MC / plumb noul, unnormalized readout (as first reported), head 15 | 11 | 4 | 5 | 1 |

### Latency and accounting

Ambient wall-clock call latency, including client/network overhead; no
exclusive hardware reservation. ×50 and ×15 are **serial scale estimates**,
not measured batch/head-15 endpoints or production latency. Jev had up to
four calls in flight; local plumb was sequential. Head-15 arms reuse scores.

| Tool / arm | Calls | Mean ms | Median ms | p95 ms | Mean ×50 s | Mean ×15 s |
|---|---:|---:|---:|---:|---:|---:|
| AFT / jev-noul | 3613 | 323.83 | 296.65 | 464.89 | 16.19 | 4.86 |
| AFT / jev-score | 3613 | 325.15 | 297.47 | 445.46 | 16.26 | 4.88 |
| AFT / plumb-noul | 3613 | 157.11 | 142.64 | 256.46 | 7.86 | 2.36 |
| MC / jev-noul | 3730 | 341.29 | 306.45 | 520.40 | 17.06 | 5.12 |
| MC / jev-score | 3730 | 341.93 | 305.47 | 533.07 | 17.10 | 5.13 |
| MC / plumb-noul | 3730 | 145.82 | 138.81 | 198.09 | 7.29 | 2.19 |

Successfully checkpointed Jev calls (including 40 repeat probes): **14726**.
Reported Jev usage: **5,573,636 input_tokens**, **301,883 output_tokens**.
Completed/transcribed sol runs: **357**; reported usage:
**1,016,116 input_tokens**, **0 cached_input_tokens**, **0 cache_write_tokens**, **104,341 output_tokens**, **54,235 reasoning_tokens**.
Tokens are reported in the provider fields as returned; reasoning tokens
are not added again to output tokens. No monetary charge was returned in
the recorded Jev responses or judge usage, so total cost is **unknown**, not
zero. Pauses resumed from checkpoints; an interrupted in-flight Jev request
can have been billed without a saved response (at most four at the pause),
so successful-call/token totals are not an exact billing reconciliation.
The 21 lost original score responses are additional successful HTTP calls
whose token usage is unavailable; they are not included in checkpointed usage.

### Per-query sol self-agreement

Anonymous triples are **overlap@10 / RBO / P@10**, in the retained corpus
query order within each tool. An asterisk marks the three same-order
two-item repeats excluded from the repeatability mean, not from arm metrics.
These include small and empty sets; no query
or candidate text, session identity, or project path is published.

AFT, 85 queries:

```text
1.0000/1.0000/0.1000, 0.8333/0.7064/0.5000, 1.0000/0.7616/0.6000, 1.0000/0.6709/0.1000, 1.0000/0.6063/0.3000
1.0000/0.4233/0.1000, 1.0000/0.7808/0.8000, 1.0000/0.7239/1.0000, 0.8333/0.7517/0.5000, 0.7778/0.6953/0.7000
0.9000/0.8524/0.9000, 0.9000/0.8599/1.0000, 0.9000/0.8587/1.0000, 1.0000/0.8365/0.7000, 1.0000/0.5461/0.3000
0.0000/1.0000/0.0000, 1.0000/1.0000/0.1000, 1.0000/0.4267/0.1000, 1.0000/0.6908/0.7000, 1.0000/0.4233/0.1000
0.0000/0.1990/0.0000, 0.9000/0.8253/0.9000, 1.0000/0.5342/0.1000, 1.0000/0.6037/0.3000, 1.0000/0.2276/0.2000
1.0000/0.6513/0.3000, 1.0000/0.4267/0.1000, 1.0000/0.5578/0.2000, 0.2000/0.2079/1.0000, 1.0000/0.6487/0.3000
1.0000/0.5113/0.1000, 1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 0.0000/0.1990/0.0000, 1.0000/0.8735/0.1000
0.0000/0.1990/0.0000, 0.6000/0.5563/0.3000, 0.0000/0.1990/0.0000, 1.0000/0.8220/0.3000, 0.7000/0.8236/0.7000
1.0000/1.0000/0.1000, 1.0000/0.6684/1.0000, 1.0000/0.5461/0.3000, 0.0000/0.1990/0.0000, 1.0000/0.4549/0.2000
0.6667/0.5578/0.2000, 0.0000/0.1990/0.0000, 1.0000/0.5578/0.2000, 1.0000/0.6011/0.3000, 1.0000/0.7975/0.8000
1.0000/0.6037/0.3000, 1.0000/0.4660/0.2000, 0.0000/0.1990/0.0000, 0.0000/0.0000/0.0000, 1.0000/1.0000/0.2000
1.0000/0.4233/0.1000, 0.0000/0.1990/0.0000, 0.8750/0.7866/0.7000, 1.0000/0.4233/0.1000, 0.0000/0.0000/0.0000
1.0000/0.4233/0.1000, 0.8889/0.8350/0.8000, 1.0000/0.6011/0.3000, 1.0000/0.6011/0.3000, 1.0000/0.6867/0.1000
1.0000/0.5186/0.1000, 0.0000/0.1990/0.0000, 1.0000/0.5578/0.2000, 0.8000/0.9346/1.0000, 1.0000/0.4543/0.1000
1.0000/0.7587/0.9000, 0.3333/0.4233/0.1000, 1.0000/0.6487/0.3000, 0.7143/0.7657/0.5000, 1.0000/0.7025/0.5000
0.0000/0.1990/0.0000, 0.6667/0.5374/0.4000, 0.8571/0.6071/0.6000, 0.3333/0.5549/0.2000, 1.0000/0.4233/0.1000
1.0000/0.7429/0.2000, 0.0000/0.1990/0.0000, 1.0000/0.4233/0.1000, 0.8000/0.5406/0.4000, 1.0000/0.4233/0.1000
```

MC, 92 queries:

```text
0.9000/0.6681/1.0000, 0.6667/0.4549/0.2000, 0.3333/0.4233/0.1000, 0.5000/0.6827/0.4000, 0.8571/0.5207/0.6000
1.0000/0.5549/0.2000, 1.0000/0.5549/0.2000, 1.0000/0.5549/0.2000, 0.1000/0.4233/0.1000, 1.0000/0.4233/0.1000
1.0000/0.7415/1.0000, 1.0000/0.4233/0.1000, 0.0000/1.0000/0.0000, 0.0000/0.9500/0.0000*, 0.0000/1.0000/0.0000
0.0000/0.1990/0.0000, 1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 0.0000/1.0000/0.0000, 0.0000/0.1990/0.0000
1.0000/0.4233/0.1000, 0.0000/1.0000/0.0000, 1.0000/0.8030/1.0000, 1.0000/0.7745/1.0000, 1.0000/0.5549/0.2000
0.8333/0.7064/0.5000, 0.8000/0.8049/0.9000, 1.0000/0.8394/0.7000, 1.0000/0.7796/0.6000, 0.9000/0.8187/1.0000
1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 0.0000/0.9500/0.0000*, 0.5000/0.3233/0.1000, 0.0000/0.9500/0.0000
1.0000/0.5549/0.2000, 0.0000/0.9500/0.0000, 0.0000/1.0000/0.0000, 0.0000/0.1990/0.0000, 0.0000/0.1990/0.0000
0.5000/0.4233/0.1000, 1.0000/0.5549/0.2000, 0.0000/0.1990/0.0000, 0.5000/0.4233/0.1000, 0.0000/0.9500/0.0000*
0.0000/0.9500/0.0000, 1.0000/0.5549/0.2000, 1.0000/0.5549/0.2000, 1.0000/0.4233/0.1000, 1.0000/0.4549/0.2000
1.0000/0.5549/0.2000, 1.0000/0.4233/0.1000, 0.0000/0.1990/0.0000, 0.0000/0.0000/0.0000, 1.0000/0.4233/0.1000
1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 1.0000/0.6461/0.3000, 0.0000/0.1990/0.0000, 1.0000/0.4233/0.1000
0.0000/0.1990/0.0000, 1.0000/0.5549/0.2000, 1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 0.0000/0.9033/0.0000
1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 0.0000/0.1990/0.0000, 1.0000/0.4233/0.1000, 0.0000/0.1990/0.0000
1.0000/0.4233/0.1000, 0.0000/0.1990/0.0000, 0.0000/0.0000/0.0000, 1.0000/0.9775/0.1000, 0.0000/0.9500/0.0000
0.0000/0.1990/0.0000, 1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 0.8000/0.5505/0.4000, 1.0000/0.4233/0.1000
0.0000/0.1990/0.0000, 1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 1.0000/0.4233/0.1000, 0.0000/0.9500/0.0000
0.0000/0.9033/0.0000, 0.5000/0.4233/0.1000, 0.0000/0.1990/0.0000, 0.5000/0.4233/0.1000, 1.0000/0.4233/0.1000
1.0000/0.7657/0.5000, 1.0000/0.4549/0.2000
```

These results measure agreement with a relevance-ranking judge, not agent
task completion or exhaustive human ground truth. Sol can be systematically
wrong in both passes. The fixed-reference method prevents new arms moving
the reference pool; it does not eliminate judge bias or sampling uncertainty.
Raw prompts, responses, per-query metrics, retry records, script snapshots,
fidelity evidence and call accounting remain in private evaluation storage.

## Neural Engine reranker follow-up: gte-reranker-modernbert on ANE (direct API)

**Stopped at the fidelity gate; no full-pool ANE arm was scored.** The direct
fp16 encoder with an fp32 CPU classification head exceeded the unchanged
maximum sigmoid-score error of 0.005. Consequently there are no new sol A/B
metrics, paired baseline deltas, leader-gap flags, MC head-15 result, or luna
grade-2 counts for this arm. Existing saved orders were first rerun through the
existing fixed-reference metric script: every arm and its aggregate statistics
reproduced exactly, hence also to the published four decimals.

This is a **spike path, not a served lane**. The new `modernbert_rerank` binary in
`bench/spikes/ane-direct-probe` uses the same `_ANEInMemoryModel` graph builders
as the successful embedding probe, including per-token centred-channel
rescaling before LayerNorm squaring. No Core ML conversion, precision change,
Hadamard rotation, or relaxed gate was used.

### Checkpoint, architecture and pair boundary

- Checkpoint: `Alibaba-NLP/gte-reranker-modernbert-base`, revision
  `f7481e6055501a30fb19d090657df9ec1f79ab2c`.
- Reference: Transformers **5.17.0**, PyTorch **2.14.0**, CPU fp32, eager
  attention, evaluation mode. The reference's classification forward and its
  explicit pooling/head decomposition agreed within 1e-6 on all 20 pairs.
- Binding pin: `ec54af9501d4bfd0cf3a4b162e59022dee2118cb`; macOS 27.0, arm64.
  Compilation ran with `TMPDIR` unset. The existing Cargo manifest and lockfile
  were unchanged; an ignored local build mirror resolved the external binding.
- The config matches the embedding port's architecture: 22 layers, hidden 768,
  intermediate 1152, 12 heads, global attention every third layer (starting at
  layer zero), local window 128, global/local RoPE bases 160000/10000, norm
  epsilon 1e-5, maximum 8192 positions. Encoder weights have the additional
  `model.` prefix; the loader now handles it without changing graph arithmetic.
- The config specifies **mean pooling**. Following
  `ModernBertForSequenceClassification` and `ModernBertPredictionHead`, the CPU
  path computes the attention-masked mean of final normalized token states,
  head dense → exact-erf GELU → LayerNorm → classifier. Dropout is disabled for
  evaluation. `classifier_bias=false` applies to the head dense, **not** the
  final classifier: its checkpoint bias is loaded and applied.
- The frozen candidate strings were used unchanged, with their saved SHA-256
  values checked. Hugging Face encoded `(query, document)` pairs, including
  special tokens, with **no truncation**. All 3,613 AFT and 3,730 MC pairs were
  tokenized, but only the fidelity sample received ANE scores. The longest pair
  is **694 tokens**. Required widths across the full input are
  **64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704**; padding is to the
  smallest multiple of 64, with a minimum width of 64.

### Fidelity gate and error isolation

Selection read no grades: among queries with at least four candidates, choose
those with shortest, median, and three longest maximum pair lengths; choose
four candidate-length quantiles within each query, then restore saved corpus
order. The resulting 20 pairs span **22–694 tokens**, including **641, 664 and
694** above 512. Each of the five queries contributes exactly four pairs.

The maximum absolute error on `sigmoid(logit)` is **0.01037967**, exceeding
**0.005**; pairs 3 and 20 fail. No pairwise order violation has a reference gap
≥0.005, even checking all 190 pairs rather than only within-query comparisons.
One inversion (pairs 11/12) has reference gap **0.00043529**, inside the exempt
band. Passing the ordering condition does not rescue the failed error gate.

| Pair (saved sample order) | Tokens | HF fp32 logit | ANE + CPU logit | Absolute sigmoid error |
| --- | ---: | ---: | ---: | ---: |
| 1 | 641 | 2.223608 | 2.239576 | 0.00139801 |
| 2 | 83 | 0.332835 | 0.350631 | 0.00432146 |
| 3 | 47 | -0.271247 | -0.229074 | **0.01037967** |
| 4 | 30 | 1.128618 | 1.134704 | 0.00112222 |
| 5 | 54 | 1.270585 | 1.277332 | 0.00115239 |
| 6 | 45 | 1.301642 | 1.299789 | 0.00031173 |
| 7 | 694 | 0.829474 | 0.839307 | 0.00207568 |
| 8 | 37 | 1.305662 | 1.318152 | 0.00208770 |
| 9 | 664 | 3.371196 | 3.373440 | 0.00007198 |
| 10 | 26 | 1.229646 | 1.228110 | 0.00026886 |
| 11 | 30 | 1.337598 | 1.339258 | 0.00027333 |
| 12 | 34 | 1.334957 | 1.340008 | 0.00083196 |
| 13 | 50 | 1.199571 | 1.207120 | 0.00134058 |
| 14 | 22 | 0.817416 | 0.799724 | 0.00377206 |
| 15 | 30 | 0.785636 | 0.772817 | 0.00276361 |
| 16 | 36 | 0.963942 | 0.955209 | 0.00174876 |
| 17 | 155 | -0.122301 | -0.122783 | 0.00012015 |
| 18 | 94 | -0.015225 | -0.017021 | 0.00044914 |
| 19 | 75 | -1.005514 | -0.986368 | 0.00377135 |
| 20 | 113 | -0.528034 | -0.504456 | **0.00551837** |

Feeding the HF pooled fp32 states into the **same Rust CPU head** limits maximum
absolute logit error to **0.00000406**. The divergence therefore arises before
the head, in the ANE encoder/pooling values; the maximum pooled-channel error
is **0.03276828**. This isolates the failing stage, not an individual layer or
operator. The inherited encoder includes fp16 arithmetic and tanh-approximate
GELU; this run does not distinguish their individual contributions. Embedding
cosine ≥0.999 is not sufficient evidence of calibrated classification-score
fidelity. No corrective precision/path experiment or full scoring followed the
failed gate.

### Ambient timing, not a full-pool latency result

The read-only campaign rig claim was checked before starting and before every
sample pair (stronger than between queries); all 21 checks found no claim.
Observed host load averages ranged **9.99–14.77 / 16.97–17.33 / 17.39–17.51**
(1/5/15 minutes). These are **ambient wall-clock** measurements, with first-use
execution included and no claim of isolated or steady-state performance.

| Fidelity sample timing | Median | Minimum–maximum |
| --- | ---: | ---: |
| Encoder, including embedding gather and surface transfers | 8.032 ms | 5.298–72.885 ms |
| CPU mean pooling and classification head | 0.407 ms | 0.305–0.677 ms |
| Total scoring, excluding compilation and JSON I/O | 8.442 ms | 5.715–73.546 ms |

Four widths were compiled once each and cached for the process: **64: 4.289 s;
128: 6.619 s; 192: 7.155 s; 704: 11.231 s**. These are full-encoder bundle
compilation times (embedding norm, 22 individual layers, final norm), not the
approximately 0.2-second single-graph shape-compilation measurement. The other
seven input widths were not compiled because full scoring was prohibited by
the failed fidelity gate.

Private storage retains the tokenizer IDs, reference states, per-pair logits
and timing, per-shape compile timing, rig/load observations, exact model hashes,
selection rule and metric-reproduction artifacts. No query or candidate text
is included here.

### Authorized diagnosis: formula error versus ANE numerical error

After the initial stop, the same retained 20 pairs were rerun for diagnosis,
without changing the gate, model, tokenizer IDs, padding, or CPU head. An
important correction to the initial hypothesis: the embedding probe's
**in-process CPU reference already uses exact-erf GELU**. Only its ANE graph
uses tanh GELU. The diagnostic adds a CPU tanh option matching the graph's
activation formula, while preserving the original exact-erf CPU default.
Both CPU variants use fp32 encoder weights and arithmetic; LayerNorm uses its
ordinary fp32 expression rather than the graph's algebraically equivalent
rescaled expression. This is a formula comparison, not an emulation of every
fp16 compiler operation or intermediate rounding.

| Encoder, same fp32 CPU head | Maximum absolute sigmoid error vs HF | Maximum absolute logit error | ≥0.005-gap order violations | Unchanged gate |
| --- | ---: | ---: | ---: | --- |
| Rust fp32, tanh GELU | 0.0026112680 | 0.0111742020 | 0 | Pass |
| Rust fp32, exact-erf GELU | 0.0000018849 | 0.0000075400 | 0 | Pass |
| ANE fp16, unchanged tanh graph | 0.0103796673 | 0.0421731174 | 0 | **Fail** |

All 20 ANE logits reproduced their initial values. The near-tie inversion
remains confined to pairs 11/12; neither CPU variant inverted a pair. The
original two failures show both a real formula contribution and a larger
additional device-path discrepancy:

| Pair | HF fp32 logit | Rust fp32 tanh logit | Rust fp32 erf logit | ANE logit | Tanh sigmoid error | Erf sigmoid error | ANE sigmoid error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | -0.27124730 | -0.26167122 | -0.27124774 | -0.22907418 | 0.00235202 | 0.00000011 | 0.01037967 |
| 20 | -0.52803361 | -0.51685941 | -0.52803248 | -0.50445628 | 0.00261127 | 0.00000026 | 0.00551837 |

**Best-supported explanation:** tanh GELU introduces measurable cumulative
formula error, but it does **not** by itself fail the gate in fp32. Exact erf
reduces CPU-vs-HF score error to roughly two millionths. The much larger ANE
error therefore cannot be explained by that formula difference alone; it
includes device-path numerical effects absent from the fp32 encoder, consistent
with fp16 weights/intermediates and compiler arithmetic. This experiment does
not separately identify weight quantization, activation rounding, reduction
order, or compiler lowering, nor prove that an fp16 exact-erf graph would pass.

The tested binding exposes **no erf or exact GELU op**. Its activation enum
contains ReLU, tanh, leaky ReLU, sigmoid, ELU, linear, hard sigmoid, softplus and
softsign; elementwise unary operations include inverse, sqrt/rsqrt, abs,
threshold, log and exp. Inspection of the graph API and MIL operation mappings
found no additional erf lowering. No available composition with a demonstrated
fp32 error bound ≤1e-6 was established. This is a statement about the checked
API and available proof, not a mathematical impossibility claim about all
compositions. **No alternative approximation was substituted and no ANE graph
formula was changed.** The repeated ANE gate still fails, so full scoring and
new-arm metrics remain prohibited.

### Per-layer attribution on the two failing pairs

These are maximum absolute differences against hooked HF fp32 layer outputs,
**after each complete attention/MLP residual layer and before final norm**.
Layers are numbered 1–22. The maxima below include every channel and every
position of the padded tensors (64 positions for pair 3, 128 for pair 20).
The private evidence also retains separate active-token-only maxima and the
full hidden-state arrays; ANE channel-major storage was transposed to match HF
and CPU token-major storage before comparison. HF hook runs reproduced their
saved logits within 1e-6.

| Layer | Pair 3 ANE | Pair 3 CPU tanh | Pair 3 CPU erf | Pair 20 ANE | Pair 20 CPU tanh | Pair 20 CPU erf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.082054 | 0.014738 | 0.000029 | 0.096092 | 0.012903 | 0.000017 |
| 2 | 0.137875 | 0.069096 | 0.000034 | 0.164047 | 0.092384 | 0.000025 |
| 3 | 0.150352 | 0.080025 | 0.000032 | 0.190060 | 0.095562 | 0.000031 |
| 4 | 0.259487 | 0.088230 | 0.000076 | 0.174084 | 0.102562 | 0.000034 |
| 5 | 0.657913 | 0.182419 | 0.000137 | 0.849884 | 0.215790 | 0.000053 |
| 6 | 0.811279 | 0.223694 | 0.000168 | 0.975113 | 0.256958 | 0.000061 |
| 7 | 0.879044 | 0.256332 | 0.000168 | 0.963501 | 0.279663 | 0.000099 |
| 8 | 0.935806 | 0.263474 | 0.000168 | 0.962814 | 0.283813 | 0.000088 |
| 9 | 1.047882 | 0.294220 | 0.000198 | 1.050873 | 0.320206 | 0.000076 |
| 10 | 1.002899 | 0.326462 | 0.000122 | 3.025360 | 0.614639 | 0.000473 |
| 11 | 0.683304 | 0.234497 | 0.000130 | 2.800552 | 0.530869 | 0.000458 |
| 12 | 2.626526 | 0.546448 | 0.001831 | 2.522705 | 0.525879 | 0.001038 |
| 13 | 2.980774 | 0.554199 | 0.001831 | 2.233154 | 0.529358 | 0.000916 |
| 14 | 3.088684 | 0.561096 | 0.001831 | 2.137268 | 0.527374 | 0.000854 |
| 15 | 2.802673 | 0.572205 | 0.001831 | 1.715759 | 0.526031 | 0.000793 |
| 16 | 1.545898 | 0.360859 | 0.002136 | 4.955383 | 0.530792 | 0.000916 |
| 17 | 2.214216 | 0.465515 | 0.002228 | 5.006317 | 0.541004 | 0.000977 |
| 18 | 2.689034 | 0.575043 | 0.002319 | 5.186615 | 0.626965 | 0.000977 |
| 19 | 3.283520 | 0.982750 | 0.001999 | 5.783859 | 1.355728 | 0.001266 |
| 20 | 5.203156 | 0.978439 | 0.001434 | 5.686768 | 1.222523 | 0.001038 |
| 21 | 5.114960 | 1.043152 | 0.001251 | 5.456680 | 1.478600 | 0.001053 |
| 22 | 7.226776 | 1.633240 | 0.001404 | 5.803955 | 2.067627 | 0.001404 |

Error is already present after layer 1 and grows non-monotonically, with marked
increases at layer 12 and the final layers for pair 3, and layers 10 and 16 for
pair 20. These are unnormalized residual magnitudes, not score errors; final
normalization substantially changes the scale. Restricting the final-layer
maximum to active tokens gives ANE errors **6.170792 / 5.803955** for pairs
3/20, so the discrepancy is not merely padded output. The table identifies
where accumulated divergence is visible, not an isolated faulty operator.

### Embedding regression check

Although the shared ANE graph was **unchanged**, `modernbert_full` was rerun on
its existing `rows.jsonl` at 512 positions, one layer per executable, before and
after the diagnostic instrumentation. Its minimum cosine is **0.9991073 before
and 0.9991073 after**, above the unchanged **0.999** gate. Both runs are
deterministic; all returned embedding vectors match exactly across runs. The
original exact-erf CPU default was preserved. Both checks used the same row-set
hash, an unset `TMPDIR`, and an empty read-only campaign rig claim before
execution. Diagnostic ANE measurements also waited on that claim before each
pair. Layer-capture I/O is not included in the earlier latency claims.
