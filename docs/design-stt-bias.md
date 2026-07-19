# Context-Biased STT — design r1

Product goal (Ufuk, 2026-07-19): when the user says "memcpy" while looking at
code containing `memcpy`, the transcript reads exactly `memcpy`. The CK app
knows what the user is seeing (screen content, session history, project
vocabulary); Synapse turns that knowledge into transcription accuracy on the
terms that generic ASR reliably botches (memcpy → "mem copy", PAQ → "pak",
subc → "sub sea").

Division of responsibility (banked #8518): consumers DERIVE the bias context
and send it per request; Synapse APPLIES it per engine. Synapse never sees the
screen — it sees `bias_context`.

## Two stacks, one contract

Both arms take the same request shape and are scored by the same eval kit, so
the choice per hardware tier stays evidence-based (the D-005 pattern).

**Arm A — owned unified voice model (LFM2-Audio on unified-rt).**
The ASR decode runs through our own loop (pre-commit tap, token-exact vs
reference). Bias mechanisms, cheap to deep:
  A1. Prompt-prefix: vocabulary carried in the pre-audio text prompt (the
      backbone is a unified token space — this is native, no engine surface).
  A2. Trie logit boost at the tap: the json_constraint machinery repurposed
      from hard masking to soft boosting — trie over bias-term tokenizations
      (space-prefixed + bare), fixed logit delta on trie-continuing tokens.
      Tunable, engine-owned, and nobody else has this seam.
  A3. (later) Contextual shallow fusion: score interpolation with a tiny
      vocabulary LM — only if A2's ceiling proves insufficient.

**Arm B — cascade (reference: huggingface/speech-to-speech, Apache-2.0).**
VAD (Silero) → STT (Parakeet TDT / Whisper-class / Paraformer) → LLM → TTS,
OpenAI Realtime-compatible. Read finding (2026-07-19): the HF pipeline ships
NO context-biasing surface at all — no initial_prompt plumbing, no hotwords,
no bias hooks in any STT handler. Context bias is differentiation, not
catch-up. Bias mechanisms available per engine:
  B1. Whisper-class initial_prompt (helps spelling; known-unreliable for rare
      terms — measure, don't assume).
  B2. Hotword-native engines (Paraformer/Fun-ASR accept hotword lists) —
      strongest cascade-side mechanism where available.
  B3. Post-ASR rescore: microllm.oneshot correcting the transcript against the
      vocabulary ("pak" → "PAQ"). Engine-agnostic, works over ANY STT output,
      already a lane we serve. Also the fallback that composes with A1/A2.

## Wire surface (v1 sketch, poll-first per SUBC rules)

    stt.transcribe {
      audio: { format: "wav16k-mono" | "pcm16", data | file_ref },
      bias_context?: {
        terms: [ { text, weight? } ],   // exact surface forms; consumer-ranked
        prompt_text?: string,            // free-text domain context
      },
      language?: "auto" | tag,
      engine?: fingerprint,              // no silent substitution, as everywhere
    } -> { job_id }                      // job-tier; short clips may return inline

    Response envelope: transcript, per-segment timestamps, term_hits (which
    bias terms were applied/emitted — observability for the consumer's UX),
    fingerprint, content_sha256 of the audio.

Streaming (partial transcripts) rides the subc streamed-request lane (LAST-flag
chained frames, #8886) — design round with SUBC before any streaming ship;
poll-first chunked jobs are the v1 shape.

`bias_context` is per-REQUEST state, like input_type for embeddings: it never
enters the fingerprint (same model + same audio + different bias = different
transcript is the FEATURE, not a substitution violation — the envelope echoes
the bias terms so consumers can attribute differences).

## Evaluation (the part that keeps us honest)

Eval kit (bench/spikes/stt-bias/, mason in flight): ~120 CortexKit-vocabulary
terms in two classes (ASR-hostile technical terms; control words), ~3 carrier
sentences each, synthesized at 16 kHz via macOS say (bootstrap-grade; known
caveat that TTS itself mangles some code terms — those get excluded with
reasons, and a real-recording validation pass follows before any product
claim). Metrics:
  - term-exact accuracy on hostile terms (the product metric),
  - false-insertion rate on control sentences (the failure mode of
    over-aggressive biasing — a bias that hallucinates "memcpy" into ordinary
    speech is worse than none),
  - WER overall (regression guard),
  - case fidelity tracked separately (PAQ vs Paq vs pak).
Gate for the owned arm: unbiased path stays byte-identical to pinned reference
fixtures (bias must be strictly additive machinery).

## Sequencing

1. (in flight) Eval kit + owned arms A1/A2 measured — [task-id].
2. Cascade baseline: faster-whisper initial_prompt (B1) + microllm rescore
   (B3) on the same kit — comparison arm, separate mason.
3. Decision row: pick per-tier defaults from the table (the speed/energy knob
   logic applied to STT); wire-contract review with SUBC/MC; stt.transcribe
   lands behind probe certification like every other lane.
4. Later: hotword-native engine arm (B2) if the cascade stays in the picture;
   A3 shallow fusion if A2's ceiling disappoints; ANE quiet-tier STT encoder
   (FastConformer is encoder-shaped) once the CoreML fp16 lowering drift
   (#9036) is resolved.
