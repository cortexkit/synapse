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
via module config/env paths. Production `model.load` accepts a package set:
put the seq128 `.mlmodelc` archive in `files.model` and seq256/seq512 archives
in ordered `files.extra` entries. Each archive is a zipped compiled bundle
(the worker verifies the SHA-256, unpacks it into a temporary directory, and
loads it with `CPU_AND_NE`). The worker dispatches each batch to the smallest
loaded bucket that fits its longest tokenized item. A direct `.mlmodelc`
directory remains valid for preload/runtime tests.
