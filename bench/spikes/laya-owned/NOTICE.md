# Third-party attribution

This spike adapts the sequence construction, decision-head computation, and answer semantics from [Laya](https://github.com/NandhaKishorM/laya), revision `9ac2115df190748bdfae6a7b1f6e57081f1b7527`, by Convai Innovations, distributed under Apache License 2.0. The upstream license is preserved as `LICENSE-LAYA`.

The Rust sequence builder and CPU head are new Rust ports of `laya/common.py`; the reference battery uses Laya's public preset questions and Agent implementation. Changes include a Rust tokenizer interface, host Accelerate matrix operations, owned Metal encoder integration, and standalone parity/measurement tooling. No upstream training code or model weights are redistributed here.
