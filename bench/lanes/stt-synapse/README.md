# Synapse direct STT benchmark worker (never shipped)

See [the common runner guide](../stt-common/README.md#synapse-build-and-run-no-certification-capture)
for pinned preparation and exact M5/Windows build/run commands, and
[the worker protocol](../stt-common/protocol.md) for its JSON contract.

The lane calls the real engine APIs directly with observation disabled, after
the shared `synapse-speech` WAV frontend. It does not launch module workers,
change production profiles or invoke certification tests. Direct ANE is a
binary-only implementation, so this bench library compiles its existing source
modules unchanged; its library test harness is disabled to keep certification
intermediate-tensor and decoder-trace capture out of bench builds. The vendored
`ane` crate and default features are identical to the production worker. Graph
constants retain their production identifier order, preserving the program cache
identifiers for identical weights and shapes.

`stt-synapse-prepare` is an explicit offline package converter, not a timed model
download or an inference command. Measurement-only F32 selection exists solely
in this lane and the backend branches' explicit measurement APIs.

Portable dispatch test (no engines/models):

```sh
python3 -m unittest discover -s bench/lanes/stt-synapse -p test_matrix.py -v
```
