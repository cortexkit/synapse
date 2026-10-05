# Real-weight Metal validation, 2026-10-05

These are the verbatim outputs of the committed parity evaluator, from the
ignored catalog hardware test on Mac17,6, macOS 27.0.1 (26A434), Xcode 27.0
(27A266a), cargo 1.99.0. All manifest-listed files were copied from their pinned
Hugging Face cache snapshots with symlinks dereferenced, then SHA-256 checked
before inference. No checkpoint digest mismatch occurred.

All four catalog profiles passed every evaluator gate. The 17 embedding cases
and 125 reranking cases per model include the 8192-token input; each reranker
includes both the 10- and 100-candidate ranking pools. These are engine math
checks, not release certification records or module-tokenization tests. The
JSON `fingerprint` field is explicitly labeled as a hardware check with the
manifest profile digest, not a module-generated lane fingerprint.

`preload-baseline.f32le` contains 125 raw GTE reranker logits, in committed
reference case order, generated from master commit
`d49576a7664d333e4d173adfa8758d850745d317`. Its engine directory tree is identical
to the implementation baseline `6c306520f63dffaa3307e75914a1166e100ee4cb`:
`d3a6c0a264d6365dcdf1827eda6d2942fada0ea1`.
The current preload path produced byte-identical outputs: 500 bytes, SHA-256
`34c95ae0c13e7891429e69355f363bd15f75bc0ee2f3d8927996ca7764d04c72`.
The baseline source was temporarily installed in this isolated worktree, built
and run with the same harness/weights/toolchain, then restored from the index.
No baseline source changes remain.

No parity defect was found, so no inference fix was needed after the hardware
runs. Selecting full Xcode instead of Command Line Tools and rebuilding the
engine generated the embedded Metal kernels, allowing all three decode kernel
tests to pass. The full owned package suite passed 58 unit tests, 9 integration
tests (12 ignored), and 2 doc tests.

## Repository gates

`cargo fmt --all --check` and the standalone hardware crate's formatting check
passed. `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo test
--workspace` passed: 1114 tests passed, 64 ignored, no failures. An initial
workspace attempt still used cached spike kernels built with Command Line
Tools; cleaning `spike-unified-rt` and rebuilding under full Xcode resolved
those three convolution-test failures without source changes.
