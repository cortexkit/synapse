//! Chunked-prefill and batched-verification exactness battery (Qwen3 Metal step).
//!
//! Two gate families, both checkpoint-gated (`#[ignore]`, model snapshot env
//! var required) and machine-independent (they compare paths of the same
//! engine on the same GPU):
//!
//! 1. **Chunked prefill is bit-exact** (the quantum-bounding lever). The
//!    prefill path processes prompts in spans of at most the quantum token
//!    budget, one command buffer per span, so the scheduler can release the
//!    decode permit between spans. Chunking must not change the arithmetic:
//!    for per-token spans of 8/16/32 and for the 16-token batched spans the
//!    serving path runs, the KV cache after prefill must be byte-identical
//!    and the first-token argmax must match the uninterrupted
//!    single-command-buffer prefill. Same argument as chunked decode: each
//!    span's forward depends only on the KV state of earlier positions,
//!    which the spans write identically; host pacing only.
//!
//! 2. **Batched verification is byte-identical to sequential stepping** (the
//!    speed lever's exactness law). `verify_tokens_batch` runs up to 16
//!    positions through the mat-mat kernels with weights streamed once per
//!    layer. Batching parallelizes ACROSS positions; it never reorders the
//!    accumulation WITHIN one dot product (the mat-mat kernels are templated
//!    on the compile-time column count so the per-column accumulators stay
//!    register-resident in single-token order — the campaign gate that
//!    caught the runtime-column-count reordering lives in
//!    `bench/spikes/unified-rt/BATCHED-VERIFY.md`). These tests re-gate the
//!    production port: full f32 logits from the batched path must be
//!    bit-for-bit equal to sequential single-token `advance` logits at the
//!    same positions, for f16 and Q8_0, across K in {1,2,4,8,16} and prompt
//!    depths {1,5,33,128,469}. This is the arithmetic-identity fixture
//!    evidence that lets prefill use the batched path by default; if it ever
//!    fails, batched prefill is a different token function and must be
//!    treated as an arithmetic-identity rotation, not silently shipped.
//!
//! The batched-verify kernels (`.metal`), the Objective-C driver (`.m`), and
//! the FFI entry are byte-identical to the spike port; the gates mirror the
//! spike's campaign gates (`batched_verify_logits_are_byte_identical_to_sequential_*`,
//! `batched_verify_is_deterministic_*`,
//! `batched_verify_forced_rejection_preserves_continuation_*`).
//!
//! Env vars:
//! - `SYNAPSE_OWNED_DECODE_QWEN3_0_6B`: path to the Qwen3-Embedding-0.6B snapshot
//!
//! Run with:
//! ```text
//! DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
//! SYNAPSE_OWNED_DECODE_QWEN3_0_6B=<qwen3-snapshot> \
//! cargo test -p synapse-engine-owned --release --test owned_decode_prefill_chunking -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use synapse_engine_owned::owned_decode_engine::{
    DecodeKernel, MetalStepDecoder, MetalStepKvCache, Qwen3DecodeModel, WeightQuantization,
};
use synapse_engine_owned::Precision;

/// Decode context bucket for the gate decoders. 512 fits the deepest gate
/// fixture (469-token prompt + 16 draft positions).
const BUCKET: usize = 512;

fn model_path() -> PathBuf {
    PathBuf::from(
        std::env::var_os("SYNAPSE_OWNED_DECODE_QWEN3_0_6B")
            .expect("set SYNAPSE_OWNED_DECODE_QWEN3_0_6B to the Qwen3-Embedding-0.6B snapshot"),
    )
}

/// Build a Metal step decoder over the real model and hand both to `body`.
/// The model outlives the decoder borrow within the closure scope.
fn with_decoder<R>(
    weight_quant: WeightQuantization,
    body: impl FnOnce(&Qwen3DecodeModel, &mut MetalStepDecoder) -> R,
) -> R {
    let model = Qwen3DecodeModel::load_with_quant(&model_path(), Precision::F16, weight_quant)
        .expect("load Qwen3-0.6B");
    let mut decoder = MetalStepDecoder::new(&model, Precision::F16, BUCKET, weight_quant)
        .expect("construct Metal step decoder");
    body(&model, &mut decoder)
}

/// Host greedy argmax matching the sampler rule: highest logit, lowest id on
/// tie (replace only on strictly greater under total_cmp).
fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (id, value) in logits.iter().enumerate().skip(1) {
        if value.total_cmp(&logits[best]) == std::cmp::Ordering::Greater {
            best = id;
        }
    }
    best as u32
}

fn logits_bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|value| value.to_bits()).collect()
}

/// Deterministic synthetic prompt of a given length. The exact tokens do not
/// matter because the compared paths must produce identical results for ANY
/// input; varied lengths exercise short, medium, and deep-context attention
/// prefixes.
fn synthetic_prompt(length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| (1000 + index * 7919 % 5000) as u32)
        .collect()
}

/// Full KV cache bits (every layer, whole bucket) for byte-identical
/// comparison between prefill runs.
fn kv_cache_bits(decoder: &mut MetalStepDecoder, layers: usize) -> Vec<u16> {
    let mut bits = Vec::new();
    for layer in 0..layers {
        bits.extend(
            decoder
                .inspect_cache_bits(layer)
                .expect("inspect KV cache layer"),
        );
    }
    bits
}

fn assert_kv_identical(reference: &[u16], candidate: &[u16], what: &str) {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "{what}: KV cache length mismatch"
    );
    if let Some((index, (a, b))) = reference
        .iter()
        .zip(candidate)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!("{what}: KV cache diverges at element {index}: {a} != {b}");
    }
}

/// Chunked prefill must be bit-exact with the uninterrupted single-command-
/// buffer prefill: byte-identical KV state and the same first-token argmax,
/// for per-token spans of 8/16/32 and the 16-token batched spans.
fn chunked_prefill_bit_exact_for(weight_quant: WeightQuantization) {
    with_decoder(weight_quant, |model, decoder| {
        let layers = model.layers.len();
        for prompt_len in [1usize, 5, 33, 128] {
            let prompt = synthetic_prompt(prompt_len);

            // Reference: one command buffer over the whole prompt.
            let (ref_cache, reference_first) = decoder.prefill(&prompt).expect("reference prefill");
            assert_eq!(
                decoder.cache_position(&ref_cache),
                prompt_len,
                "reference prefill must advance to the prompt length"
            );
            let reference_kv = kv_cache_bits(decoder, layers);

            // Per-token path chunked at the candidate quantum budgets: each
            // span is one command buffer; host pacing only.
            for chunk_tokens in [8usize, 16, 32] {
                let mut cache = MetalStepKvCache { position: 0 };
                let mut argmaxes = Vec::with_capacity(prompt_len);
                for chunk in prompt.chunks(chunk_tokens) {
                    argmaxes.extend(
                        DecodeKernel::verify_tokens(decoder, &mut cache, chunk)
                            .expect("chunked prefill span"),
                    );
                }
                assert_eq!(
                    decoder.cache_position(&cache),
                    prompt_len,
                    "chunked prefill must advance to the prompt length ({weight_quant:?}, len={prompt_len}, chunk={chunk_tokens})"
                );
                let first = *argmaxes.last().expect("non-empty prompt");
                assert_eq!(
                    first, reference_first,
                    "chunked prefill first-token argmax diverged ({weight_quant:?}, len={prompt_len}, chunk={chunk_tokens})"
                );
                let chunked_kv = kv_cache_bits(decoder, layers);
                assert_kv_identical(
                    &reference_kv,
                    &chunked_kv,
                    &format!("per-token chunk {chunk_tokens}, len={prompt_len}, {weight_quant:?}"),
                );
            }

            // Batched path (mat-mat, weights streamed once per layer) at the
            // 16-token span size the quantum-bounded serving prefill runs.
            let mut cache = MetalStepKvCache { position: 0 };
            let mut argmaxes = Vec::with_capacity(prompt_len);
            for chunk in prompt.chunks(MetalStepDecoder::MAX_BATCH_VERIFY_TOKENS) {
                argmaxes.extend(
                    decoder
                        .verify_tokens_batch(&mut cache, chunk)
                        .expect("batched prefill span"),
                );
            }
            assert_eq!(
                decoder.cache_position(&cache),
                prompt_len,
                "batched chunked prefill must advance to the prompt length ({weight_quant:?}, len={prompt_len})"
            );
            let first = *argmaxes.last().expect("non-empty prompt");
            assert_eq!(
                first, reference_first,
                "batched chunked prefill first-token argmax diverged ({weight_quant:?}, len={prompt_len})"
            );
            let batched_kv = kv_cache_bits(decoder, layers);
            assert_kv_identical(
                &reference_kv,
                &batched_kv,
                &format!("batched chunk 16, len={prompt_len}, {weight_quant:?}"),
            );
        }
        println!("[prefill-chunking] {weight_quant:?}: chunked prefill bit-exact (per-token 8/16/32, batched 16)");
    });
}

#[test]
#[ignore]
fn chunked_prefill_is_bit_exact_f16() {
    chunked_prefill_bit_exact_for(WeightQuantization::None);
}

#[test]
#[ignore]
fn chunked_prefill_is_bit_exact_q8() {
    chunked_prefill_bit_exact_for(WeightQuantization::Q8_0);
}

/// Batched verification logits must be bit-for-bit equal to sequential
/// single-token `advance` logits at the same positions, for every K in
/// {1,2,4,8,16} and prompt depths {1,5,33,128,469}. Port of the spike's
/// campaign gate to the production engine.
fn byte_identical_gate_for(weight_quant: WeightQuantization) {
    with_decoder(weight_quant, |_model, decoder| {
        for prompt_len in [1usize, 5, 33, 128, 469] {
            let prompt = synthetic_prompt(prompt_len);
            let (cache, first) = decoder.prefill(&prompt).expect("prefill");
            let mut cache = cache;
            let base_position = decoder.cache_position(&cache);
            assert_eq!(base_position, prompt_len);

            // Sequential reference: the greedy draft (first generated token,
            // then argmax of each advance) and the logits that produced each
            // successor. seq_logits[i] is the logits after feeding draft[i].
            let mut draft = Vec::with_capacity(16);
            let mut seq_logits = Vec::with_capacity(16);
            let mut next = first;
            for _ in 0..16 {
                draft.push(next);
                let logits = decoder
                    .advance(&mut cache, next)
                    .expect("sequential advance");
                next = greedy_argmax(&logits);
                seq_logits.push(logits);
            }

            for k in [1usize, 2, 4, 8, 16] {
                let draft = &draft[..k];
                // Rewind to the prefix and run one batched forward over K tokens.
                decoder.rewind(&mut cache, base_position).expect("rewind");
                let batch_logits = decoder
                    .verify_tokens_batch_logits(&mut cache, draft)
                    .expect("batched verify logits");
                let vocab = seq_logits[0].len();
                assert_eq!(batch_logits.len(), k * vocab);
                for i in 0..k {
                    let batch_row = &batch_logits[i * vocab..(i + 1) * vocab];
                    assert_eq!(
                        logits_bits(batch_row),
                        logits_bits(&seq_logits[i]),
                        "batched logits diverge from sequential at prompt_len={prompt_len} k={k} position {i} ({weight_quant:?})"
                    );
                }
                // Argmax surface agrees too (this is what the session consumes).
                decoder.rewind(&mut cache, base_position).expect("rewind");
                let batch_argmaxes = decoder
                    .verify_tokens_batch(&mut cache, draft)
                    .expect("batched verify argmaxes");
                for i in 0..k {
                    assert_eq!(
                        batch_argmaxes[i],
                        greedy_argmax(&seq_logits[i]),
                        "batched argmax diverges at prompt_len={prompt_len} k={k} position {i} ({weight_quant:?})"
                    );
                }
            }
        }
        println!("[batched-verify] {weight_quant:?}: logits byte-identical to sequential for K in {{1,2,4,8,16}}");
    });
}

#[test]
#[ignore]
fn batched_verify_logits_are_byte_identical_to_sequential_f16() {
    byte_identical_gate_for(WeightQuantization::None);
}

#[test]
#[ignore]
fn batched_verify_logits_are_byte_identical_to_sequential_q8() {
    byte_identical_gate_for(WeightQuantization::Q8_0);
}

/// Two batched runs over the same draft must produce bit-identical logits.
fn determinism_gate_for(weight_quant: WeightQuantization) {
    with_decoder(weight_quant, |_model, decoder| {
        let prompt = synthetic_prompt(64);
        let (cache, first) = decoder.prefill(&prompt).expect("prefill");
        let mut cache = cache;
        let base_position = decoder.cache_position(&cache);
        let mut draft = Vec::with_capacity(8);
        let mut next = first;
        for _ in 0..8 {
            draft.push(next);
            let logits = decoder
                .advance(&mut cache, next)
                .expect("sequential advance");
            next = greedy_argmax(&logits);
        }

        decoder.rewind(&mut cache, base_position).expect("rewind");
        let first_run = decoder
            .verify_tokens_batch_logits(&mut cache, &draft)
            .expect("first batched run");
        decoder.rewind(&mut cache, base_position).expect("rewind");
        let second_run = decoder
            .verify_tokens_batch_logits(&mut cache, &draft)
            .expect("second batched run");
        assert_eq!(
            logits_bits(&first_run),
            logits_bits(&second_run),
            "batched verification is not deterministic ({weight_quant:?})"
        );
        println!("[batched-verify] {weight_quant:?}: deterministic");
    });
}

#[test]
#[ignore]
fn batched_verify_is_deterministic_f16() {
    determinism_gate_for(WeightQuantization::None);
}

#[test]
#[ignore]
fn batched_verify_is_deterministic_q8() {
    determinism_gate_for(WeightQuantization::Q8_0);
}

/// Forced-rejection rollback: verify a K-token draft whose token at index
/// `wrong` is corrupted, accept the correct prefix, rewind to it, and confirm
/// the greedy continuation is byte-exact with the target-only stream. Run for
/// every rejection position so each KV slot in the batch window is exercised
/// as the rollback boundary.
fn forced_rejection_gate_for(weight_quant: WeightQuantization) {
    with_decoder(weight_quant, |_model, decoder| {
        let prompt = synthetic_prompt(48);
        let (cache, first) = decoder.prefill(&prompt).expect("prefill");
        let mut cache = cache;
        let base_position = decoder.cache_position(&cache);
        // Target-only greedy reference, long enough to cover the continuation.
        let mut target = Vec::with_capacity(33);
        let mut next = first;
        for _ in 0..33 {
            target.push(next);
            let logits = decoder
                .advance(&mut cache, next)
                .expect("sequential advance");
            next = greedy_argmax(&logits);
        }
        let vocab = {
            decoder.rewind(&mut cache, base_position).expect("rewind");
            let probe = decoder.advance(&mut cache, target[0]).expect("probe");
            decoder.rewind(&mut cache, base_position).expect("rewind");
            probe.len()
        };

        for k in [4usize, 8] {
            for wrong in 0..k {
                // Corrupt one draft token; the prefix before it stays correct.
                let mut draft = target[..k].to_vec();
                draft[wrong] = (target[wrong] + 1) % vocab as u32;

                decoder.rewind(&mut cache, base_position).expect("rewind");
                decoder
                    .verify_tokens_batch(&mut cache, &draft)
                    .expect("batched verify");
                // Accept the `wrong` correct tokens, discard the rest.
                decoder
                    .rewind(&mut cache, base_position + wrong)
                    .expect("rewind to acceptance boundary");
                // Re-advance the correct token and follow greedy; every step
                // must match the target-only continuation.
                let mut next = target[wrong];
                for step in 0..8 {
                    let logits = decoder.advance(&mut cache, next).expect("continue");
                    let argmax = greedy_argmax(&logits);
                    assert_eq!(
                        argmax,
                        target[wrong + 1 + step],
                        "continuation diverged after rejection: k={k} wrong={wrong} step={step} ({weight_quant:?})"
                    );
                    next = argmax;
                }
            }
        }
        println!("[batched-verify] {weight_quant:?}: forced rejection preserves the continuation");
    });
}

#[test]
#[ignore]
fn batched_verify_forced_rejection_preserves_continuation_f16() {
    forced_rejection_gate_for(WeightQuantization::None);
}

#[test]
#[ignore]
fn batched_verify_forced_rejection_preserves_continuation_q8() {
    forced_rejection_gate_for(WeightQuantization::Q8_0);
}
