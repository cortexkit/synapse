# Utility judge evaluation

The judge is blind to candidate labels and receives hydrated snippet bytes. Top-up calls are the headline utility cost; package score is secondary. F1 is the existing gold-overlap file F1, restricted to natural completions when a score report is available.

Calibration gate: **PASS** (gold mean top-up 0.40, empty mean 4.60, mismatched none 3/5).

Calibration prompt: iteration 1, SHA 2c27195db8c467edaa9f464bc92297d6b5478accc311f26b056ce8d76047af39; base blind two-phase protocol.
Calibration cost projection: $6.37 for 280 full-matrix packages (sample rows: 38).

## Calibration evidence
| control | rows | full / partial / none | top-up calls mean |
| --- | ---: | ---: | ---: |
| gold | 5 | 4 / 1 / 0 | 0.40 |
| empty | 5 | 0 / 1 / 4 | 4.60 |
| mismatched | 5 | 0 / 3 / 2 | 5.20 |

## Gold-anchor difficulty spread

12 of 40 gold packages needed repository top-ups after phase 1.

| top-up calls | gold jobs |
| ---: | ---: |
| 0 | 28 |
| 2 | 2 |
| 3 | 4 |
| 5 | 1 |
| 6 | 4 |
| 7 | 1 |

| job ID | request | phase-1 result | top-up calls | final result |
| --- | --- | --- | ---: | --- |
| 65567dba33519d5763f654c7bd977142039ef3360025ea47bc5f632ab30123a0 | Trace how a query processed through the ANE/CoreML spike path (coreml_spike.rs) differs from the standard CPU/Vulkan retrieval path — what shared interfaces or traits, if any, unify these backends in the core crate? | partial | 7 | partial |
| 4c5242d395e4a403cd198420bb5b3bad1527dd7fe43a3d3c76b87552c36bf30a | If storage.test.ts fails against clients/store/tests/golden/storage_vectors.json after a change to derivation.ts, which functions in derivation.ts and descriptor.ts are most likely responsible for the mismatch, and how would you locate the exact diverging byte offset? | partial | 6 | full |
| c9698dc46fd4c75ad5777b23014a84234bbaa77d51208e1ea033fc21995eeef9 | Which Rust workspace members (as declared in the top-level Cargo.toml) would need coordinated version bumps if the derivation logic mirrored in clients/store/src/derivation.ts were changed? | partial | 6 | partial |
| e52e6f3e8412a697a991bc6faebc02488a08ee50ee5f3860717fc1806dedb712 | Based on current import relationships between source modules, which files violate the boundaries described in the modular-code-enforcement rule, and what would need to change to bring them into compliance? | partial | 6 | partial |
| e67b5506e53355bf6eabb81d1a12a77ffb681aa0c99dd64aabef69fc6aa6a637 | What public functions or types does the core crate defined in Cargo.toml export from its crate root (lib.rs), and which of these are consumed by the TypeScript layer? | partial | 6 | partial |
| 85a1585b1aaa733b9901471d4926ae802f6d4191fe0f911352c2d7dc00d32ea5 | Trace the data flow from the Rust retrieval/embedding code in the main crate through to the Python scripts in bench/eval-coir — what intermediate file formats or serialization boundaries connect them? | partial | 5 | partial |
| 283bcb6fe5d96907a88829309f6672b196651f2a1ba58cc60a1155e8d4751aba | What is the entry point that registers the `publish` and `get-unpublished-changes` commands with the CLI dispatcher, and what module owns the command routing table? | partial | 3 | partial |
| 7889d0261c394decbcd3194318a6a8eea839b6f9b3ccca0ede3ea96c2bf33635 | What is the purpose of .cortexkit/magic-context.jsonc, and which part of the codebase loads or interprets its 'magic-context' entries? | partial | 3 | partial |
| 8218929b0d5299e445a9dc390b443f61dce8c77542873d4fde0829688c18fec9 | What frame-to-progress or duration calculation logic is duplicated between text-animations-typewriter.tsx and text-animations-word-highlight.tsx that could be extracted into a shared timing utility? | partial | 3 | partial |
| e40ecde09b35ad037ab65b87da3a9cd313602f7aff3832cb06dd255441f3d589 | Before restructuring the .cortexkit directory, which source files import or reference paths under .cortexkit/ (e.g. aft.jsonc, magic-context.jsonc) that would break if those files moved? | partial | 3 | partial |
| 254f0d586b454438e39f69480dbdeaf281d129cf112b0f4965e3285cb4a564e2 | What API or CLI command consumes .bg-shell/manifest.json, and what shape of object does it expect the manifest to expose? | not answerable | 2 | none |
| 9000662d93304f5f9601d2b6ba7f972631e10ceacc46ca3f1001e9dece0c4766 | In charts-bar-chart.tsx, what interpolation/easing function drives bar height animation, and could it return negative or NaN heights for frame values before the animation's declared start frame? | partial | 2 | full |

| system | phase-1 full / partial / none | final full / partial / none | top-up calls mean | top-up calls median | top-up tokens mean | score mean | F1 | skipped invalid | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| codegraph-explore | 1 / 10 / 29 | 1 / 12 / 27 | 4.33 | 4.00 | 35049 | 2.25 | 0.06 | 0 | 0 |
| deepseek-v4-flash-nothink | 18 / 2 / 0 | 17 / 3 / 0 | 0.25 | 0.00 | 8036 | 8.90 | 0.85 | 20 | 0 |
| deepseek-v4-flash-zeroshot | 16 / 3 / 0 | 18 / 1 / 0 | 0.95 | 0.00 | 8900 | 9.00 | 0.89 | 21 | 0 |
| gold-control | 28 / 11 / 1 | 28 / 11 / 1 | 1.30 | 0.00 | 13442 | 8.53 | 1.00 | 0 | 0 |
| qwen35-2b-sft-v1-fixed | 15 / 19 / 0 | 14 / 20 / 0 | 2.59 | 3.00 | 17049 | 7.35 | 0.60 | 6 | 0 |
| qwen35-4b-lora-v1 | 21 / 13 / 0 | 22 / 12 / 0 | 2.18 | 0.00 | 15279 | 8.38 | 0.64 | 6 | 0 |
| qwen35-9b-lora-v1 | 20 / 9 / 1 | 22 / 7 / 1 | 1.70 | 0.00 | 12422 | 8.33 | 0.69 | 10 | 0 |

## Ranking conclusion
Utility ranking (lower top-up is better): **deepseek-v4-flash-nothink < deepseek-v4-flash-zeroshot < qwen35-9b-lora-v1 < qwen35-4b-lora-v1 < qwen35-2b-sft-v1-fixed < codegraph-explore**. F1 ranking (higher is better): **deepseek-v4-flash-zeroshot > deepseek-v4-flash-nothink > qwen35-9b-lora-v1 > qwen35-4b-lora-v1 > qwen35-2b-sft-v1-fixed > codegraph-explore**. The rankings diverge.

## Two concrete divergence examples
- codegraph-explore vs deepseek-v4-flash-nothink on c18b34685a21fb678a019fd32a58b502c0d85d96e23b841a3051d0ec29cac21b (What public functions or types does clients/store/src/index.ts re-export from derivation.ts and descriptor.ts, and are there any internal-only symbols excluded from the public API?): codegraph-explore F1 0.29 with 4 top-up calls versus deepseek-v4-flash-nothink F1 1.00 with 0 calls.
- codegraph-explore vs deepseek-v4-flash-zeroshot on c18b34685a21fb678a019fd32a58b502c0d85d96e23b841a3051d0ec29cac21b (What public functions or types does clients/store/src/index.ts re-export from derivation.ts and descriptor.ts, and are there any internal-only symbols excluded from the public API?): codegraph-explore F1 0.29 with 4 top-up calls versus deepseek-v4-flash-zeroshot F1 1.00 with 0 calls.

Invalid or forced rows are recorded as `sufficiency=none` with zero judge calls and are reported separately as skipped; they are not evidence of a cheap sufficient package.
