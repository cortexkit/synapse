# Parakeet transcription reference attribution

The Parakeet model, the FLEURS dataset and the LibriSpeech dataset are used under
the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The license permits sharing and adaptation with attribution. The original
authors do not endorse Synapse or these derived reference fixtures.

* **NVIDIA**, *Parakeet TDT 0.6B v3*:
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>, revision
  `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`.
* **Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur**,
  *LibriSpeech: an ASR corpus based on public domain audio books* (2015):
  <https://www.openslr.org/12/>. The audio originates in public-domain LibriVox
  readings. The reference uses the Hugging Face `openslr/librispeech_asr` mirror,
  revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`, test-clean and test-other.
* **Google and the FLEURS authors**, *FLEURS: Few-shot Learning Evaluation of
  Universal Representations of Speech* (2022):
  <https://huggingface.co/datasets/google/fleurs>, revision
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. The selected test languages are
  German, French, Spanish, Italian, Polish, Dutch, Russian and Greek.

`parakeet-corpus.json` records every selected utterance id, its original
ground-truth transcription, the immutable source URL and SHA-256, and the
SHA-256 of its audio member. Selection does not depend on recognition accuracy.

Changes made for this reference: decoding/scaling to mono float32, deterministic
resampling, deterministic silence/leading-silence and hop-boundary fixtures,
stereo PCM16/float32 resampling fixtures, concatenation and trimming for duration
ladder fixtures, derived log-mel/encoder tensors, and model-generated traces and
transcripts. The small self-check WAVs are PCM16 encodings of selected recordings
or deterministic word-boundary excerpts; their seals retain the source recipe.
The upstream transcripts are preserved, not replaced by the model's predictions.

The Python reference uses Hugging Face Transformers (Apache-2.0); its named
attention correction is a local implementation, not a modification of the
installed package. NeMo (Apache-2.0) is cited to explain masking and decoding
differences. Provenance and checksums are recorded in `upstream-sources.json`.
