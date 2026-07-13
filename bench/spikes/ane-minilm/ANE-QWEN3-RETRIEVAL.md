# ANE Qwen3 retrieval-quality A/B: CosQA

## Verdict

**Verdict: drift is retrieval-visible on cosqa.** On the same 20,604-document
corpus and 500 queries, owned-Metal fp32 leads ANE fp16 by **0.00317349
NDCG@10**; four queries improve for fp32, one improves for ANE, and 495 tie.
The difference is small, but it is nonzero in this fixed A/B.

This is a numeric-path comparison, not a general Qwen3 model-quality claim.
The usual CosQA single-qrel limitation applies equally to both arms.

## Protocol

- **Host:** `[bench-host]` (Apple M1 Max, macOS 26.5.2), reached by
  SSH. Both quality runs used `nice -19` and never acquired
  `[bench-user-home]/bench.lock`; no M5 work was used.
- **Model:** `Qwen/Qwen3-Embedding-0.6B` snapshot
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; `model.safetensors` SHA-256
  `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`.
- **Corpus and queries:** public CosQA, deterministically prepared into
  20,604 corpus rows, 500 test queries, and 500 qrels. Documents have no
  prefix. Queries use the established Qwen retrieval prefix, with the raw
  query text appended directly after the colon:

  ```text
  Instruct: Given a web search query, retrieve relevant code snippets that answer the query
  Query:
  ```

- **Terminal-EOS policy:** ANE inputs were left-padded to the fixed seq512
  Core ML model and end in model-config EOS `151643`. The existing prep policy
  reserves the terminal position, so active sequences have at most 511 tokens.
  The Metal arm used `--max-length 511` to reproduce those active token IDs
  exactly before its own last-token pooling and L2 normalization.
- **Matching check:** all 20,604 corpus and 500 query active sequences were
  checked against the Metal tokenizer policy: 1,399,899 corpus tokens and
  13,955 query tokens matched. This avoids treating a tokenizer-boundary
  difference as a numeric-path result.
- **Long-text boundary:** in untruncated tokenization, 20,583 corpus rows are
  shorter than 512 tokens; 21 are longer (maximum 1,809 tokens) and are
  truncated identically in both arms. No query reaches the ceiling (prepared
  maximum 41 tokens).

### fp32 reference substitution

The scored reference is the owned `spike-unified-rt` Metal MPSGraph **fp32**
path, not CPU ORT. This is an intentional substitution: on the M1, the same
matching max-length-511 400-row check against the frozen ORT fp32 reference
reported mean cosine `0.9999999999930432` and mean top-10 overlap `1.000000`.
It therefore clears the standard `>= 0.9999` cosine and `>= 0.995` overlap
parity gates before acting as the fp32 arm.

The full fp32 run used the runtime's bounded bucketed batching to keep GPU
memory bounded. Its serving-only padding-efficiency assertion fired after it
had written all complete vectors (20.18% corpus padding and 56.74% query
padding versus that runner's 15% serving threshold). The completed row counts,
token counts, digests, and successful scorer loads were independently checked;
this is not a vector-fidelity failure. The ORT parity check above used exact
shapes and completed normally.

## Cross-arm wiring check

Five final corpus vectors were compared before scoring. All cosines are well
above the `0.999` miswiring stop threshold.

| ID | ANE fp16 cosine to owned-Metal fp32 |
|---|---:|
| `d1` | 0.9999750342 |
| `d10` | 0.9999775364 |
| `d14633` | 0.9999693040 |
| `d1927` | 0.9999698093 |
| `d5361` | 0.9999748862 |
| **mean / minimum** | **0.9999733140 / 0.9999693040** |

## Retrieval result

`bench/eval-coir/score.py` performed exact brute-force cosine retrieval and
scored the shared CosQA qrels at 10.

| Metric | owned-Metal fp32 | ANE fp16 | Delta (fp32 - ANE) |
|---|---:|---:|---:|
| NDCG@10 | 0.3492174711 | 0.3460439768 | +0.0031734943 |
| MRR@10 | 0.2543801587 | 0.2514134921 | +0.0029666667 |
| Recall@10 | 0.6040000000 | 0.6020000000 | +0.0020000000 |

Per-query wins/losses/ties are oriented to fp32: a win means that fp32's
per-query metric is strictly higher than ANE's; ties use an absolute tolerance
of `1e-12`.

| Per-query metric @10 | fp32 wins | ANE wins | ties |
|---|---:|---:|---:|
| NDCG | 4 | 1 | 495 |
| MRR | 4 | 1 | 495 |
| Recall | 1 | 0 | 499 |

The observed full-quality arm times were 232.431 s for the fp32 corpus,
4.292 s for the fp32 queries, 2,658.045 s for the ANE corpus, and 64.631 s
for the ANE queries. They are execution provenance only, not locked-machine
latency or power claims.

## Reproducibility digests

The repository contains digests rather than vector contents.

| Artifact | Rows | SHA-256 |
|---|---:|---|
| CosQA corpus JSONL | 20,604 | `a730b7ab09f86449a39780538220f9de0f2fa8ba0e8b4efce3e225e68b0fb098` |
| CosQA query JSONL | 500 | `086bd4f24bc6078c6aa92d36189a0febbedd967c2e40b1350b293a9d45956d6e` |
| CosQA qrels | 500 | `d3b9138da2302994d9664b1251e9a7f9995eb9ea52d4f216e6d5a56a6d28a757` |
| Qwen-prefixed query JSONL | 500 | `bd5b0dc62d8fe5d7963c5b127333e1ac1dd588181ad7f5b3fdaec0d5948e1b2c` |
| ANE corpus fixed-input JSONL | 20,604 | `6453f889e3046dbfbc901dffd989009fc6f3dd744c024bcacbca5bfd4247b6cd` |
| ANE query fixed-input JSONL | 500 | `1fd92e89365b1c31a359c26467781a28f1d71d3964104e67cb17765c42650e84` |
| owned-Metal fp32 corpus vectors | 20,604 | `efd7642ea8d0da9cf0f86a1caaeefb831ef06a5446165105f9eacf253c57908b` |
| owned-Metal fp32 query vectors | 500 | `d0469b18c9ad9fcbffd70c4fcec4eb18d13e9e9758238923a2c64145c61f3850` |
| ANE fp16 corpus vectors | 20,604 | `5f1fc88a311cfe5d68a41895284bcfba489ebd6f09c5693cc0cc958261ae198c` |
| ANE fp16 query vectors | 500 | `8f43627fdeee38cf4ed619916d202cc9ce7138a9ae0651af72d77e6c1d229047` |

For binary provenance, the owned Metal runner SHA-256 is
`3fae0455ccc650c1b597c7bc4b8c4dc8d08c224b45d471128fe360b9cddc9db3` and
the ANE runner SHA-256 is
`96642084e8ba6ead2bc638c3c32c0c94d5c83dae68c631820c8ef3d94e8e0b63`.

## Limits

- This is one 500-query dataset with one positive qrel per query. The five
  non-tied NDCG/MRR outcomes are enough to make the drift visible here, but
  not enough to establish a broad product-quality or statistical-significance
  claim.
- The 21 long corpus texts share the same terminal-EOS truncation policy, so
  truncation cannot explain the A/B delta, but a longer-context dataset could
  behave differently.
- The fp32 arm is parity-certified owned Metal, not a literal ORT corpus run;
  the frozen-ORT parity gate above is the evidence that makes that substitution
  sound.
- This result does not relax the Wave 2 fingerprint boundary: ANE fp16 and
  fp32 vectors remain separately versioned index spaces.
