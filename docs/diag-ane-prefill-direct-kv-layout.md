# ANE-prefill direct K/V output layout

## Result

The sidecar now gives CoreML one custom-strided `MLMultiArray` output backing for
each K/V output. Each backing points into the worker-owned protocol-v2 mapping at
its final `[layer][key_or_value][head][cache_position][dimension]` location.
CoreML therefore writes worker-importable cache bytes during prediction; Swift no
longer performs the scalar 14.7 MB K/V re-stride after prediction.

The canonical W128 package accepted all 56 mapped K/V backings. The returned
objects, pointers, and strides matched all 56 supplied views, and comparison with
a normal prediction found zero active-value bit mismatches. The same checks were
also zero-mismatch for the canonical W256 and W512 packages. The existing v2 K/V
offset is 64-byte aligned rather than page aligned; CoreML accepted that real
alignment for all three package shapes, so no frame field or protocol bump was
needed.

## Transform removed

Before this change, each CoreML K/V output had logical shape
`[1, kv_heads, window, head_dimension]`. The sidecar read every element through
`MLMultiArray.strides` and stored it into a cache-sized destination plane. The
source position stride was based on `window`, while each destination head used a
`cache_tokens` stride. That scalar stride walk was correct for non-contiguous
CoreML output, but it dominated sidecar execution.

The mapped output view now has these element strides:

```text
[
  kv_heads * cache_tokens * head_dimension,
  cache_tokens * head_dimension,
  head_dimension,
  1,
]
```

The backing for each layer and key/value plane starts at that plane's final mmap
offset. CoreML writes only the fixed model window, leaving the cache-sized gap
between heads untouched. The mapping is cleared before prediction. When
`active_tokens < window`, the sidecar clears the fixed-window inactive tail after
prediction, before hashing or READY publication. Thus right-padding positions
cannot become decode state.

CoreML output-backing construction, prediction with a backing, returned-backing
identity validation, shape/stride validation, and inactive-tail validation all
fail closed as `kv_conversion_failure`. The retained helper for copy-based
adapters uses block copies for contiguous dimensions and a scalar walk only for
arbitrary strides; overflow and storage failures use the same token.

## Bit-faithfulness and failure proof

The Swift fixture materializes the same non-contiguous logical K/V values in two
ways:

1. the legacy active-only stride walk followed by expansion into the padded
   worker cache, and
2. simulated CoreML writes through the new mmap-backed views followed by inactive
   tail clearing.

The resulting complete imported cache byte strings and SHA-256 digests are
identical. Active-value bit mismatches and non-zero padding values are both zero.
Additional tests reject undersized and misaligned mappings, a returned output
that did not use the supplied backing, an invalid active-token count, and stride
arithmetic overflow. Every rejection asserts the exact
`kv_conversion_failure` code.

The real-package feasibility probe produced:

| Package | Returned object identity | Pointer match | Stride match | Active bit mismatches |
|---|---:|---:|---:|---:|
| W128 | 56 / 56 | 56 / 56 | 56 / 56 | 0 |
| W256 | 56 / 56 | 56 / 56 | 56 / 56 | 0 |
| W512 | 56 / 56 | 56 / 56 | 56 / 56 | 0 |

Verification commands:

```sh
cd workers/ane-prefill-sidecar
swift run -c release ane-prefill-sidecar-tests
swift build -c release
swift-format lint --strict --configuration <four-space-project-config> \
  Sources/AnePrefillSidecar/DirectLayout.swift \
  Sources/AnePrefillSidecarExecutable/main.swift \
  Sources/AnePrefillSidecarTests/main.swift
```

## Sidecar stage measurements

The baseline p50 values are the accepted W128 decomposition in
`docs/diag-ane-prefill-ttft.md`. The new values are from the quiet 20-sample
battery below. The baseline predates protocol-v2 mapped integrity publication,
so it has no separately comparable publication-hash row.

| Sidecar / worker stage | Baseline p50 (ms) | New f16-step p50 (ms) | New q8-step p50 (ms) |
|---|---:|---:|---:|
| Core ML prediction | 12.996 | 13.519 | 13.676 |
| K/V layout or backing work | **36.441** | **2.729** | **3.146** |
| Logits copy | 0.738 | 3.191 | 3.127 |
| Sidecar integrity / publication hash | not separate | 18.567 | 18.469 |
| Sidecar total | 50.299 | 38.252 | 38.544 |
| Worker `EXECUTE` boundary | 50.504 | 38.534 | 38.820 |
| Payload IPC / mapped validation | 123.081 socket payload | 18.668 | 18.627 |
| q8 engine-to-engine cache handoff | 0 | 0 | 20.724 |
| Metal upload | 6.159 | 5.194 | 12.236 |

The K/V stage dropped by **33.712 ms**, or **13.35x**, on f16-step. The new
sidecar `EXECUTE` path is approximately 38.5 ms p50. End-to-end time remains
higher because the sidecar publication hash and worker validation hash each walk
the complete 58.7 MB padded mapping; q8-step also pays its existing cache
handoff and larger upload.

## Quiet W128 TTFT battery

The run used the freshly built worktree sidecar and the owner-authorized
read-only canonical worker. This substitution proves interoperability with the
deployed worker; `git diff -- crates/` and the Swift shared-memory protocol files
were empty. A freshly built worktree worker was not used because that binary
stalled in `_dyld_start` before `--help` or the worker handshake, despite a valid
ad-hoc signature. No Rust or protocol source was changed.

The one-minute load average was below 6 at both boundaries:

- start: `{ 5.84 7.12 8.83 }`
- end: `{ 5.51 6.79 8.59 }`

Three artifact-warm requests per engine preceded the 20 sample-major,
ANE-then-GPU pairs for each decode configuration.

| Sample | f16 split (ms) | f16 GPU (ms) | q8 split (ms) | q8 GPU (ms) |
|---:|---:|---:|---:|---:|
| 0 | 65.852 | 340.614 | 99.083 | 403.374 |
| 1 | 63.701 | 313.834 | 11412.601 | 365.548 |
| 2 | 62.080 | 322.308 | 92.609 | 337.092 |
| 3 | 71.986 | 316.348 | 91.081 | 334.210 |
| 4 | 64.032 | 314.127 | 88.017 | 338.062 |
| 5 | 64.061 | 313.094 | 91.645 | 339.068 |
| 6 | 62.647 | 313.201 | 92.190 | 339.209 |
| 7 | 62.325 | 313.185 | 94.569 | 364.154 |
| 8 | 62.631 | 313.825 | 122.886 | 338.369 |
| 9 | 63.390 | 313.541 | 89.016 | 339.987 |
| 10 | 62.494 | 376.057 | 104.087 | 339.930 |
| 11 | 61.666 | 316.549 | 90.593 | 350.290 |
| 12 | 62.258 | 317.351 | 92.550 | 339.494 |
| 13 | 61.748 | 313.321 | 89.802 | 338.469 |
| 14 | 63.590 | 313.813 | 90.541 | 338.955 |
| 15 | 62.970 | 313.863 | 95.064 | 338.742 |
| 16 | 63.375 | 312.881 | 89.966 | 338.331 |
| 17 | 64.379 | 313.801 | 88.337 | 337.929 |
| 18 | 63.704 | 314.318 | 90.014 | 339.945 |
| 19 | 64.115 | 313.482 | 90.799 | 343.654 |

| Decode configuration | Split min / p50 / p95 / max (ms) | GPU min / p50 / p95 / max (ms) | GPU / split p50 |
|---|---:|---:|---:|
| f16-step | 61.666 / **63.383** / 65.852 / 71.986 | 312.881 / **313.830** / 340.614 / 376.057 | **4.951x** |
| q8-step | 88.017 / **91.363** / 122.886 / 11412.601 | 334.210 / **339.138** / 365.548 / 403.374 | **3.712x** |

The q8 sample-1 outlier spent 11.336 seconds inside the worker `EXECUTE`
boundary while the sidecar itself completed in 66.811 ms. It does not affect the
median, but it is retained rather than discarded.

The direct layout achieves the intended memcpy-class K/V stage and a roughly
38.5 ms sidecar execution boundary. The measured end-to-end f16 ratio against
the unchanged approximately 313 ms GPU baseline is **4.951x**, not the projected
7.8x and still just below the unchanged 5.0x floor. The remaining mapped-payload
hash, Metal upload, and (for q8) engine-to-engine cache handoff are outside the
removed Swift re-stride and remain in honest TTFT. No gate, contract, evidence
record, runtime enable, or configuration constant changed.
