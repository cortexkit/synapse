# synapse-worker-ane

Supervised Synapse worker for fixed-bucket Core ML MiniLM artifacts on Apple
Neural Engine. The Rust binary is a tiny launcher; `build.rs` compiles the
self-contained Swift worker with `swiftc` so no Xcode project is required.

Runtime contract:

- `LOAD` expects an already-compiled `.mlmodelc` directory. Conversion from
  PyTorch / `.mlpackage` is an offline v1 tooling step; it is never performed by
  the worker.
- `EMBED_BATCH` receives pretokenized ids from the module, pads each row to the
  fixed Core ML bucket, rejects rows longer than that bucket, runs the Core ML
  model, then mask-mean-pools and L2-normalizes vectors.
- `PING` reports the Neural Engine placement share from the last
  `MLComputePlan` check. Certification uses the module-side placement threshold
  (default 0.9) as an additional gate.

Artifacts from the spike are expected under `~/bench-tools/ane-spike/models` or
via module config/env paths, for example
`all-MiniLM-L6-v2-seq256.mlmodelc`.
