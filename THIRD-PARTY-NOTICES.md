# Third-party notices

Synapse's own code is covered by [LICENSE](LICENSE). The notices below distinguish
adapted or vendored source from ordinary package dependencies. Dependency licenses
remain those declared by their distributions and lockfile; this file does not
relicense upstream work.

## Apple Neural Engine binding (`ane`)

- Upstream: https://github.com/mutable-state-inc/siliconswarm-at-ensue-plugin
- Revision: `ec54af9501d4bfd0cf3a4b162e59022dee2118cb`.
- Upstream source directory: `ane_kernel/crates/ane`.
- Repository files: `crates/synapse-worker-ane-direct/vendor/ane/src/**`,
  `build.rs` and `Cargo.toml` in that directory; the checksum and provenance files
  document the vendored inventory and local compatibility changes.
- License: **MIT**, declared by the upstream workspace and explicitly by the
  vendored crate. That pinned revision supplies no license text; no copyright
  notice or license text is fabricated here.

See [VENDORED.md](crates/synapse-worker-ane-direct/vendor/ane/VENDORED.md) for source
provenance, platform/dependency changes, diagnostics and checksum verification.
The copy excludes upstream examples, benchmarks and model weights.

## Laya sequence construction and decision head

- Upstream: https://github.com/NandhaKishorM/laya
- Revision: `9ac2115df190748bdfae6a7b1f6e57081f1b7527`.
- Attribution: **Convai Innovations**.
- License: **Apache License 2.0**; the complete upstream license is preserved at
  [bench/spikes/laya-owned/LICENSE-LAYA](bench/spikes/laya-owned/LICENSE-LAYA).
- Repository files: `bench/spikes/laya-owned/src/sequence.rs`,
  `bench/spikes/laya-owned/src/head.rs`, `bench/spikes/laya-owned/reference.py`,
  and `bench/spikes/laya-owned/battery.json`.

The Rust sequence builder and CPU head port `laya/common.py`. The reference battery
uses public preset questions and the Agent implementation. Local changes include
the Rust tokenizer interface, host matrix operations, owned encoder integration
and independent parity tooling. No upstream training code or model weights are
redistributed. The retained [local notice](bench/spikes/laya-owned/NOTICE.md)
provides the adaptation attribution.

## ModernBERT residual-stream rotation reference

- Reference: https://github.com/smpanaro/ModernBERT-AppleNeuralEngine
- Revision: `d0268940c0af4ffe2bca0f9eb3842fd05984215c`.
- Related repository implementation: `bench/parity/src/hadamard.rs`,
  `bench/parity/src/convert.rs` and the conversion experiment in
  `bench/spikes/ane-modernbert-full-context/`.
- Upstream license: **not specified by a LICENSE or COPYING file at that
  revision**. This notice asserts no upstream license grant or invented copyright
  attribution.

The reference describes residual-stream centering, normalization-scale folding and
an orthogonal Hadamard basis change. Synapse's deterministic rotation construction
and byte identity are specified in its own source and manifest; attribution to the
reference method must not be mistaken for inherited parity or permission to copy
unlicensed upstream source. Review upstream licensing before importing additional
source from that project.
