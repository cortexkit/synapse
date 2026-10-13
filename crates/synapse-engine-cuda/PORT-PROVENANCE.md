# Owned-CUDA engine port provenance

This record accompanies the production `owned-cuda` engine. The spike tree is
read-only. Kernel arithmetic was copied from `bench/spikes/unified-rt` at
source revision `4d0ded67c30286fe2be37cc7413359ad745dd751`.

## Kernel source inventory

| production file | spike source | source revision | source SHA-256 | production SHA-256 | reviewed difference |
| --- | --- | --- | --- | --- | --- |
| `src/port/cuda_family_common.cuh` | `bench/spikes/unified-rt/src/cuda_family_common.cuh` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `937a01e329822277fa15eeac8498fcca6890b40665b531adb9c92f67b4eb53c5` | `937a01e329822277fa15eeac8498fcca6890b40665b531adb9c92f67b4eb53c5` | byte-identical; no arithmetic difference |
| `src/port/cuda_minilm.h` | `bench/spikes/unified-rt/src/cuda_minilm.h` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `21a0bcee2d21b17fea795807eab2070e848c416df1cb94c7ae7bb9adfa514de5` | `21a0bcee2d21b17fea795807eab2070e848c416df1cb94c7ae7bb9adfa514de5` | byte-identical; FFI declarations are retained |
| `src/port/cuda_minilm.cu` | `bench/spikes/unified-rt/src/cuda_minilm.cu` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `9848c050015d7f2ecd0ecc6aa628bb4e4d4280778e4a7eb6590475e937f30375` | `9e82e3b3b58588184a0140dcb264a32da4502e7e72b2f7d45e9cdf568222be13` | completed-layer events and callback seam; arithmetic unchanged |
| `src/port/cuda_modernbert.cu` | `bench/spikes/unified-rt/src/cuda_modernbert.cu` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `455e4419c0c6996b004153e1709e9d6ef001ca0d865e42bc7af2ad6809b350b9` | `cf5a5b91c026b30f7a5b65a2092a14a500c25f46ccbf2ea41b0f9e5e7e14710f` | completed-layer events and callback seam; arithmetic unchanged |
| `src/port/cuda_qwen3.cu` | `bench/spikes/unified-rt/src/cuda_qwen3.cu` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `a4c05d7cf119b39ffe52cdfa77f39859ddee8ac3e18139d8094219fbc8a1588b` | `893d55b1d61caeb3bd7846ecca747cd08a9aa6cb85972b39e7612bebbd9e8841` | completed-layer events and callback seam; arithmetic unchanged |

These rows retain the complete arithmetic port. Layer completion events and
their observer calls are instrumentation, not numerical kernel changes: no
kernel arithmetic line was retuned, reordered, or otherwise changed.

## Difference classification

The production-only files make only seam and distribution changes:

| file | classification | change |
| --- | --- | --- |
| `src/port/cuda_progress.cuh` | progress instrumentation | repository-local event/observer helper; SHA-256 `f58b6f5b21306c4d2deadb761e82d617d662e24989a93d56d563a27faeb46ef7`; captured external event-record nodes prove completion on each serialized replay, with no socket calls inside capture |
| `build.rs` | packaging | compiles the retained kernels as a production crate, emits compute-75 virtual PTX only, and links the CUDA runtime and cuBLAS libraries on Linux and Windows feature builds |
| `src/cuda.rs` | FFI, visibility | gives the existing C ABI typed Rust wrappers and keeps a default-disabled cross-platform stub; the wrappers do not alter kernel arguments or arithmetic |
| `src/model.rs` | wiring | loads production safetensors packages, prepares canonical token-id inputs, and maps each family model to its existing kernel ABI; host-side embedding lookup and normalization are seam preparation, not kernel changes |
| `src/lib.rs` | visibility, wiring | exposes `EmbedEngine`, family selection, resolved storage dtype, PTX identity, CUDA floor predicates, and supervised-load errors; it owns model handles and serializes access to each CUDA context |
| `dtype-resolution-v1.json` | wiring | records the immutable C1-C3 ABI evidence and selected manifest dtypes |

Every production difference is therefore visibility, packaging, FFI, wiring or
completed-work instrumentation; there is no arithmetic-class difference. The
kernel revision and numerical lane fingerprint inputs remain unchanged.

## Distribution identity

- Backend: `cuda-ptx`
- PTX virtual architecture: `compute_75` (no SASS or per-device fatbins)
- Minimum device compute capability: `7.5`
- Minimum CUDA driver API: `12040`
- Risk class: `abort_capable`
- C1/C2/C3 resolved storage dtype: `f16`
