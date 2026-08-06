# Owned-CUDA engine port provenance

This record accompanies the production `owned-cuda` engine. The spike tree is
read-only. Kernel arithmetic was copied from `bench/spikes/unified-rt` at
source revision `4d0ded67c30286fe2be37cc7413359ad745dd751`.

## Kernel source inventory

| production file | spike source | source revision | source SHA-256 | production SHA-256 | reviewed difference |
| --- | --- | --- | --- | --- | --- |
| `src/port/cuda_family_common.cuh` | `bench/spikes/unified-rt/src/cuda_family_common.cuh` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `937a01e329822277fa15eeac8498fcca6890b40665b531adb9c92f67b4eb53c5` | `937a01e329822277fa15eeac8498fcca6890b40665b531adb9c92f67b4eb53c5` | byte-identical; no arithmetic difference |
| `src/port/cuda_minilm.h` | `bench/spikes/unified-rt/src/cuda_minilm.h` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `21a0bcee2d21b17fea795807eab2070e848c416df1cb94c7ae7bb9adfa514de5` | `21a0bcee2d21b17fea795807eab2070e848c416df1cb94c7ae7bb9adfa514de5` | byte-identical; FFI declarations are retained |
| `src/port/cuda_minilm.cu` | `bench/spikes/unified-rt/src/cuda_minilm.cu` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `9848c050015d7f2ecd0ecc6aa628bb4e4d4280778e4a7eb6590475e937f30375` | `9848c050015d7f2ecd0ecc6aa628bb4e4d4280778e4a7eb6590475e937f30375` | byte-identical; no arithmetic difference |
| `src/port/cuda_modernbert.cu` | `bench/spikes/unified-rt/src/cuda_modernbert.cu` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `455e4419c0c6996b004153e1709e9d6ef001ca0d865e42bc7af2ad6809b350b9` | `455e4419c0c6996b004153e1709e9d6ef001ca0d865e42bc7af2ad6809b350b9` | byte-identical; no arithmetic difference |
| `src/port/cuda_qwen3.cu` | `bench/spikes/unified-rt/src/cuda_qwen3.cu` | `4d0ded67c30286fe2be37cc7413359ad745dd751` | `a4c05d7cf119b39ffe52cdfa77f39859ddee8ac3e18139d8094219fbc8a1588b` | `a4c05d7cf119b39ffe52cdfa77f39859ddee8ac3e18139d8094219fbc8a1588b` | byte-identical; no arithmetic difference |

The byte-identical rows are the complete arithmetic port. No kernel line was
retuned, reordered, or otherwise changed to alter numerical behavior.

## Difference classification

The production-only files make only seam and distribution changes:

| file | classification | change |
| --- | --- | --- |
| `build.rs` | packaging | compiles the retained kernels as a production crate, emits compute-75 virtual PTX only, and links the CUDA runtime and cuBLAS libraries on Linux and Windows feature builds |
| `src/cuda.rs` | FFI, visibility | gives the existing C ABI typed Rust wrappers and keeps a default-disabled cross-platform stub; the wrappers do not alter kernel arguments or arithmetic |
| `src/model.rs` | wiring | loads production safetensors packages, prepares canonical token-id inputs, and maps each family model to its existing kernel ABI; host-side embedding lookup and normalization are seam preparation, not kernel changes |
| `src/lib.rs` | visibility, wiring | exposes `EmbedEngine`, family selection, resolved storage dtype, PTX identity, CUDA floor predicates, and supervised-load errors; it owns model handles and serializes access to each CUDA context |
| `dtype-resolution-v1.json` | wiring | records the immutable C1-C3 ABI evidence and selected manifest dtypes |

Every production difference is therefore visibility, packaging, FFI, or wiring;
there is no arithmetic-class difference.

## Distribution identity

- Backend: `cuda-ptx`
- PTX virtual architecture: `compute_75` (no SASS or per-device fatbins)
- Minimum device compute capability: `7.5`
- Minimum CUDA driver API: `12040`
- Risk class: `abort_capable`
- C1/C2/C3 resolved storage dtype: `f16`
