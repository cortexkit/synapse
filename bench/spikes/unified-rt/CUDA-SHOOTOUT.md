# CUDA embedding shootout — RTX 4090, 2026-07-14

One RTX 4090, one 15,271-chunk corpus, and canonical model tokenizers were used to compare the owned CUDA runtime with llama.cpp, Hugging Face Text Embeddings Inference (TEI), and ONNX Runtime CUDA EP. Throughput cells are the slower of two clean full-corpus runs after a three-point batch sweep. All reported vectors passed the required mean-cosine `>=0.999` and top-10-overlap `>=0.99` gates against frozen ORT-fp32 references.

## Results

| Contender | Model | dtype | Real tok/s | GPU W avg / peak | J/Mtok | Peak VRAM | Cold load | Single-query p50 | Parity cosine / top-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OWNED | all-MiniLM-L6-v2 | f16 | **321,426** | 78.9 / 101.2 | 245 | 753 MiB | 2.66 s | 4.03 ms | 0.99999960 / 0.9990 |
| llama.cpp CUDA | all-MiniLM-L6-v2 | f16 GGUF | 20,844 | 52.7 / 56.5 | 2,529 | 509 MiB | 1.03 s | 5.61 ms | 0.99999674 / 0.9978 |
| TEI candle-cuda | all-MiniLM-L6-v2 | f16 | **776,438** | 110.7 / 135.9 | **143** | 827 MiB | 1.03 s | 3.16 ms | 0.99995383 / 0.9985 |
| ORT CUDA EP | all-MiniLM-L6-v2 | fp32 ONNX | 487,214 | 173.5 / 274.8 | 356 | 929 MiB | **0.55 s** | **1.30 ms** | 0.99999976 / 0.9995 |
| OWNED | gte-modernbert-base | f16 | **135,097** | 198.4 / 273.3 | 1,469 | 1,241 MiB | 4.09 s | 9.71 ms | 0.99999857 / 0.9975 |
| llama.cpp CUDA | gte-modernbert-base | f16 GGUF | 44,461 | 104.1 / 144.0 | 2,342 | 713 MiB | **1.03 s** | 4.79 ms | 0.99998800 / 0.9958 |
| TEI candle-cuda | gte-modernbert-base | f16 | **231,750** | 271.6 / 318.3 | **1,172** | 1,115 MiB | 2.04 s | 5.41 ms | 0.99999744 / 0.9980 |
| ORT CUDA EP | gte-modernbert-base | fp32 ONNX | 92,955 | 314.9 / 359.6 | 3,387 | 2,465 MiB | 1.36 s | **3.87 ms** | 0.99999981 / 0.9990 |
| OWNED | Qwen3-Embedding-0.6B | f16 | **63,330** | 291.9 / 389.6 | 4,608 | 2,303 MiB | 8.19 s | 12.99 ms | 0.99999785 / 0.9975 |
| llama.cpp CUDA | Qwen3-Embedding-0.6B | f16 GGUF | 9,205 | 120.1 / 130.6 | 13,050 | 2,303 MiB | **1.03 s** | 16.46 ms | 0.99997985 / 0.9948 |
| TEI candle-cuda | Qwen3-Embedding-0.6B | f16 | **87,456** | 348.5 / 394.5 | **3,985** | **1,979 MiB** | 2.04 s | **5.84 ms** | 0.99999848 / 0.9990 |
| ORT CUDA EP | Qwen3-Embedding-0.6B | — | **unsupported** | — | — | — | — | — | — |

Bold throughput and energy values mark the best comparable result for a model; OWNED is also bolded to make its position easy to scan. `J/Mtok = average board watts / real tok/s * 1,000,000`, so it includes idle board power during the timed window.

## Positioning verdict

TEI is the throughput and energy leader on all three models, beating OWNED by 2.42x on MiniLM, 1.72x on ModernBERT, and 1.38x on Qwen3. OWNED is second on ModernBERT and Qwen3 and third on MiniLM behind ORT, while beating llama.cpp by 15.4x, 3.04x, and 6.88x respectively with parity intact. The owned backend is therefore a credible broad f16 runtime but not the same-box CUDA serving leader; TEI is the performance choice, while ORT remains a compelling MiniLM latency/throughput baseline when a suitable ONNX graph exists.

## Method

- **Rig:** vast.ai contract 44740762, RTX 4090 24 GiB (SM 8.9), NVIDIA driver 570.133.20, CUDA 12.6 (`nvcc` build 12.6.34841621), Ubuntu 24.04, AMD EPYC 7443 with 48 visible threads and 503 GiB RAM. The driver passed the owned-kernel `>=570` gate before any build or run.
- **Corpus and denominator:** `bench/data/corpus-v2.jsonl`, 15,271 chunks. Canonical non-padded, sanitized tokenizer counts after 512-token truncation were 2,178,482 MiniLM tokens, 2,031,766 ModernBERT tokens, and 1,569,409 Qwen3 tokens. Qwen3 accounting enforces its canonical terminal EOS 151643; engine-reported token counts were never used.
- **Timing:** each selected configuration embedded the entire corpus twice from a fresh process; the lower tok/s repeat supplies the row and its power/VRAM/load measurements. Tuning probes used the full corpus for OWNED, TEI, and ORT; llama.cpp used a fixed 3,000-row prefix because its HTTP embedding path was much slower, followed by both required full-corpus repeats.
- **Telemetry:** a separate `nvidia-smi` sampler recorded board power, utilization, and used VRAM every 500 ms only during full-corpus inference. Every final window began at <=2% GPU utilization with no compute process, showed at most the contender's one compute process, and ended clean; no row was contaminated or rerun for foreign GPU work.
- **Latency:** 30 warmed sequential one-text requests with a different corpus text and nonce each iteration. TEI and llama use their HTTP APIs, OWNED uses its persistent framed-stdio protocol after standard-shape preparation, and ORT uses a persistent CUDA-EP session with the lane's tokenization/pooling policy.
- **Cold load:** OWNED/ORT use the lane's internal model/session load measurement. HTTP servers use process-spawn-to-healthy time; the health loop had 1 s resolution, so the `~1.03 s` server values are upper-resolution observations rather than sub-second precision claims.
- **Parity:** first 400 rows of the staged MiniLM 1,000-row set and the complete 400-row ModernBERT/Qwen3 sets. Cosine is candidate-vs-reference per row; top-10 overlap compares each candidate similarity ranking with the frozen ORT-fp32 ranking. All gates passed before full-corpus timing.

## Version pins and build status

| Component | Exact pin / build |
|---|---|
| OWNED | Synapse `edc9118cf04fe7f3bf7eb8fcfaf62dd65d1a016a`; `spike-unified-rt`, release, CUDA f16 |
| llama.cpp | current master/release `b9992`, commit `6eddde06a4f25d55d538b5d15628dcc2b6882147`; CUDA build, GNU 13.2 |
| TEI | release `v1.9.3`, commit `06670157fb6c1523482219bdb2d1660277d38088`; `text-embeddings-router 1.9.3` |
| TEI CUDA path | `candle-cuda,static-linking`; the resulting binary contains `candle-flash-attn` and served MiniLM, ModernBERT, and Qwen3 |
| ORT | `lane-ort-embed` with `ort` crate `2.0.0-rc.11`, dynamically loading `onnxruntime-gpu 1.23.2` (`libonnxruntime.so.1.23.2`) and CUDA EP |
| Toolchain | Rust 1.97.0; Python 3.12 client; tokenizers 0.21.4 |

The first TEI build command failed before compiling kernels because `cudarc` had neither a dynamic nor static linking feature selected. Rebuilding with `static-linking` succeeded, compiled `candle-flash-attn` for the Ada GPU, and was the binary used for every TEI result; there was no SM 8.9 flash-attention fallback.

## Configuration and tuning appendix

| Contender / model | Three-point sweep (tok/s) | Selected full-corpus configuration |
|---|---|---|
| OWNED / MiniLM | attention units 1M: 324,803; 4M: 322,750; 8M: 322,858 | 1M; CUDA graphs; bucket policy v1; f16; mean pool + L2 |
| OWNED / ModernBERT | 1M: 135,998; 4M: 134,279; 8M: 134,859 | 1M; CUDA graphs; bucket policy v1; f16; CLS + L2 |
| OWNED / Qwen3 | 1M: 63,616; 4M: 63,425; 8M: 63,494 | 1M; CUDA graphs; bucket policy v1; f16; last-token + L2 |
| TEI / MiniLM | max batch tokens 8,192: 802,771; 16,384: 796,099; 32,768: 818,705 | 32,768; dynamic batching; client batch 512; mean pool; f16 flash-attn |
| TEI / ModernBERT | 8,192: 227,460; 16,384: 236,036; 32,768: 216,599 | 16,384; dynamic batching; client batch 512; CLS; f16 flash-attn |
| TEI / Qwen3 | 8,192: 87,938; 16,384: 85,618; 32,768: 85,009 | 8,192; dynamic batching; client batch 512; last-token; f16 flash-attn |
| llama / MiniLM | `ub=b` 512: 22,735; 1,024: 21,829; 2,048: 21,922 | `-ngl 99 -c 512 -ub 512 -b 512 -fa on --cont-batching`; HTTP batch 64; mean pool |
| llama / ModernBERT | 512: 51,814; 1,024: 49,343; 2,048: 50,704 | same flags; HTTP batch 64; CLS pool |
| llama / Qwen3 | 512: 8,167; 1,024: 7,907; 2,048: 7,860 | `-ngl 99 -c 2048 -ub 512 -b 512 -fa on --cont-batching`; HTTP batch 32; last pool |
| ORT / MiniLM | attention units 1M: 501,194; 4M: 360,345; 8M: 258,087 | 1M; intra-op threads 4; fp32 ONNX; mean pool + L2 |
| ORT / ModernBERT | 1M: 93,113; 4M: 59,941; 8M: 56,869 | 1M; intra-op threads 4; fp32 ONNX; CLS + L2 |

OWNED's selected standard ladder was batch 8 at sequence lengths 64–320, batch 6 at 384, batch 4 at 448, and batch 3 at 512. llama-server warns and forces `n_batch = n_ubatch` for embedding mode, so the valid sweep used equal values; Qwen3 additionally needed `-c 2048` because its four automatic slots otherwise exhausted a `-c 512` KV cache. The two failed Qwen3 tuning attempts at `-c 512` hit a llama.cpp `GGML_ASSERT` after KV-cache exhaustion, not a CUDA driver/SASS failure, and were excluded before the clean sweep.

### Full-corpus repeats and host load

| Cell | Repeat 1 tok/s | Repeat 2 tok/s | Selected repeat | Host 1m load start -> end |
|---|---:|---:|---:|---:|
| OWNED MiniLM | 321,547 | 321,426 | 2 | 1.32 -> 1.50 |
| OWNED ModernBERT | 135,459 | 135,097 | 2 | 1.31 -> 1.22 |
| OWNED Qwen3 | 63,330 | 63,461 | 1 | 1.22 -> 1.33 |
| llama MiniLM | 20,844 | 21,023 | 1 | 0.99 -> 1.41 |
| llama ModernBERT | 44,461 | 44,719 | 1 | 0.65 -> 0.92 |
| llama Qwen3 | 9,205 | 9,206 | 1 | 4.65 -> 1.64 |
| TEI MiniLM | 776,438 | 831,010 | 1 | 1.07 -> 1.07 |
| TEI ModernBERT | 231,750 | 232,634 | 1 | 1.05 -> 1.05 |
| TEI Qwen3 | 87,467 | 87,456 | 2 | 1.56 -> 1.69 |
| ORT MiniLM | 487,214 | 492,248 | 1 | 2.37 -> 2.24 |
| ORT ModernBERT | 92,955 | 93,329 | 1 | 2.30 -> 6.93 |

## Coverage and comparability notes

- **TEI support is verified, not inferred:** v1.9.3 loaded and returned parity-passing vectors for `gte-modernbert-base` with CLS pooling and `Qwen3-Embedding-0.6B` with last-token pooling.
- **llama.cpp support is verified:** current master converted both MiniLM and ModernBERT from the staged Hugging Face snapshots and served all three models. Converter warnings only concerned GGUF special-token metadata; parity passed for every resulting model.
- **ORT Qwen3 is unsupported in this lane/campaign:** no Qwen3 ONNX graph was staged, and `lane-ort-embed` consumes a ready ONNX graph rather than exporting a Transformers checkpoint. Fabricating an export during the timed campaign would not be the requested staged-model baseline, so the cell is reported unsupported rather than substituting another model representation.
- **CPU contention:** the selected windows' one-minute loads are recorded above. The largest excursion was ORT ModernBERT (2.30 -> 6.93 on 48 visible threads); GPU utilization/power remained authoritative, and repeat throughput differed by only 0.4%.
- **Comparability:** the table does not label parity failures as throughput because there were none. ORT rows use staged fp32 graphs while the other contenders use f16, as explicitly shown in the dtype column.

## Artifact integrity

Every coordinate-file digest supplied in `RIG-CUDA-SHOOTOUT.txt` was recomputed on-box before use and matched. Important input and derived hashes are:

| Artifact | SHA-256 |
|---|---|
| `corpus-v2.jsonl` | `8f11c8e8a03b4979aa9c28d3b1597eda7f711a6d0e8cc280e7fe6df58a91dfa8` |
| MiniLM parity corpus / vectors | `b7c8424f5b6bc5df61d96146a03642671789c1d41cbe37e82864117330996a10` / `7589eea5148562f6141c864d3357bab5dceb6881055afcf93b80efbdcae7d24d` |
| ModernBERT parity corpus / vectors | `b4ff00f6d2d9f0652146b7438c2ecd421746bcead466cccf18ec79e45ff79aa8` / `d1fb6aaf48c36c8ed7b06b9c69e6244f01393e085d32f49b15194671f7a44000` |
| Qwen3 parity corpus / vectors | `5a9bfdc8c069657aa46cbb45bef91bc1a0ddc72602bfb96b189af31ba55f630c` / `cacee1f64d12704ea94cded9861f6aef903a018800b2e0a1ec67589c33c7cf46` |
| MiniLM ONNX / safetensors / derived GGUF | `bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5` / `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` / `3eb69b124070dd6396e28f2428a11893338f0d808b0290f39db6f85c0c7a3bcd` |
| ModernBERT safetensors / ONNX / derived GGUF | `3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba` / `947f31df7effaeec4edb57c50e4ed7e0f2034d9336063f92615b92e3e0d24d78` / `813903f20aea3cda083156df3af70a46480abf53a81dcd7ab56d4ae970cb7c7c` |
| Qwen3 safetensors / staged GGUF | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` / `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340` |
| Staged Synapse bundle | `13052737d2bae84c072fcb515ce9052e94de82d0c73afe5f5093faca71ef7254` |

Raw final, tuning, latency, and parity JSONs were copied to `bench/results/cuda-shootout-2026-07-14/`; that directory is intentionally gitignored. Reproduction helpers are `cuda_shootout_client.py`, `cuda_shootout_direct.sh`, `cuda_shootout_owned_latency.py`, and `cuda_shootout_ort_latency.py` in this directory.
