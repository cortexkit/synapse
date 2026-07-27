use std::time::Instant;

use anyhow::{ensure, Result};

use super::{DecodeKernel, DecodeRuntime, DecodeStageTimings, Model};
use crate::quant::WeightQuantization;
use crate::{vulkan_backend::Qwen3DecodeContext, Precision, VulkanGemm};

pub(crate) struct VulkanKvCache {
    position: usize,
}

pub(crate) struct VulkanDecoder<'a> {
    context: Qwen3DecodeContext<'a>,
    model: &'a Model,
    bucket: usize,
    timings: DecodeStageTimings,
}

impl<'a> VulkanDecoder<'a> {
    pub(crate) fn new(
        model: &'a Model,
        precision: Precision,
        gemm: VulkanGemm,
        pipeline_cache: Option<std::path::PathBuf>,
        bucket: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(precision, Precision::F16),
            "Qwen3 Vulkan decode requires --dtype f16"
        );
        let started = Instant::now();
        let context = Qwen3DecodeContext::new(gemm, pipeline_cache, model, bucket)?;
        Ok(Self {
            context,
            model,
            bucket,
            timings: DecodeStageTimings {
                graph_prepare_wall_s: started.elapsed().as_secs_f64(),
                ..DecodeStageTimings::default()
            },
        })
    }
}

impl DecodeKernel for VulkanDecoder<'_> {
    type Cache = VulkanKvCache;

    fn capacity(&self) -> usize {
        self.bucket
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
        let started = Instant::now();
        let logits = self.context.prefill(tokens)?;
        self.timings.execute_wall_s += started.elapsed().as_secs_f64();
        self.timings.prefill_calls += 1;
        Ok((
            VulkanKvCache {
                position: tokens.len(),
            },
            logits,
        ))
    }

    fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>> {
        ensure!(
            cache.position < self.bucket,
            "decode cache capacity exhausted"
        );
        let started = Instant::now();
        let logits = self.context.advance(token, cache.position)?;
        self.timings.execute_wall_s += started.elapsed().as_secs_f64();
        self.timings.step_calls += 1;
        cache.position += 1;
        Ok(logits)
    }

    fn cache_position(&self, cache: &Self::Cache) -> usize {
        cache.position
    }

    fn inspect_cache_layer(&self, _cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
        self.context.inspect_cache_layer(layer)
    }

    fn stage_timings(&self) -> DecodeStageTimings {
        self.timings
    }

    /// Batched verification: runs K draft tokens through one mat-mat forward so
    /// each layer's weights stream once instead of once per token. Returns the
    /// greedy argmax after each supplied token. By construction the per-position
    /// logits are bit-identical to K sequential single-token `advance` steps at
    /// the same positions; the batch only shares the weight read across
    /// positions, never reordering one dot's accumulation.
    fn verify_tokens(&mut self, cache: &mut Self::Cache, tokens: &[u32]) -> Result<Vec<u32>> {
        ensure!(
            !tokens.is_empty(),
            "verification requires at least one token"
        );
        ensure!(
            tokens.len() <= 16,
            "Vulkan batched verification supports at most 16 draft tokens, got {}",
            tokens.len()
        );
        let started = Instant::now();
        let logits = self.context.verify_batch_logits(tokens)?;
        self.timings.execute_wall_s += started.elapsed().as_secs_f64();
        self.timings.step_calls += tokens.len() as u64;
        cache.position += tokens.len();
        // Host greedy argmax matching the sampler rule: highest logit, lowest
        // id on tie (replace only on strictly greater under total_cmp).
        let vocab = self.model.config.vocab_size;
        let mut argmaxes = Vec::with_capacity(tokens.len());
        for k in 0..tokens.len() {
            let row = &logits[k * vocab..(k + 1) * vocab];
            let mut best = 0usize;
            for (id, value) in row.iter().enumerate().skip(1) {
                if value.total_cmp(&row[best]) == std::cmp::Ordering::Greater {
                    best = id;
                }
            }
            argmaxes.push(best as u32);
        }
        Ok(argmaxes)
    }

    /// Restores the logical cache length after speculative verification. KV
    /// data is addressed by [layer, head, position, dimension]; attention reads
    /// only positions <= cache.position, RoPE is recomputed from that position,
    /// and every activation scratch buffer is overwritten by the next command
    /// buffer. No auxiliary decode state advances with a batch, so changing
    /// this logical bound is sufficient.
    fn rewind(&mut self, cache: &mut Self::Cache, position: usize) -> Result<()> {
        ensure!(
            position <= cache.position,
            "cannot rewind Vulkan decode cache forward from {} to {position}",
            cache.position
        );
        cache.position = position;
        self.context.set_position(position);
        Ok(())
    }
}

impl DecodeRuntime for VulkanDecoder<'_> {
    fn lane(&self) -> &'static str {
        "owned-rt-vulkan-decode-plain"
    }

    fn kv_update_path(&self) -> &'static str {
        "vulkan-device-resident-f16-in-slot-kv-cache"
    }

    fn weight_feed_path(&self) -> &'static str {
        match self.model.weight_quantization {
            WeightQuantization::None => "vulkan-persistent-f16-serial-gemv",
            WeightQuantization::Q8_0 => "vulkan-persistent-q8_0-serial-gemv",
        }
    }

    fn optimization_level(&self) -> u8 {
        0
    }
}

#[cfg(test)]
mod tests {
    //! Real-model gates for the Vulkan batched (mat-mat) verification path.
    //! These require a Vulkan GPU and the Qwen3-0.6B weights, so they are
    //! `#[ignore]` and run explicitly:
    //!
    //! ```text
    //! SYNAPSE_UNIFIED_RT_QWEN3_0_6B=<snapshot dir> \
    //!   cargo test -p spike-unified-rt --release --features vulkan \
    //!     vulkan_batched_verify -- --ignored --nocapture
    //! ```
    //!
    //! The central invariant is machine-independent: batched verification must
    //! produce logits bit-identical to K sequential single-token steps, because
    //! batching only shares the per-layer weight read across the K positions and
    //! never reorders one dot product's accumulation. These tests prove that on
    //! whatever Vulkan GPU runs them. Performance is measured separately on the
    //! Ally (the Vulkan decode timing authority documented in VULKAN-DECODE.md).

    use super::{VulkanDecoder, VulkanKvCache};
    use crate::quant::WeightQuantization;
    use crate::qwen3_decode::DecodeKernel;
    use crate::{Precision, VulkanGemm};

    const BUCKET: usize = 1024;

    fn model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_QWEN3_0_6B")
                .expect("set SYNAPSE_UNIFIED_RT_QWEN3_0_6B to the Qwen3-0.6B snapshot directory"),
        )
    }

    /// Build a Vulkan decoder over the real model and hand it to `body`. The
    /// model outlives the decoder borrow within the closure scope.
    fn with_decoder<R>(
        weight_quant: WeightQuantization,
        body: impl FnOnce(&crate::qwen3::Model, &mut VulkanDecoder) -> R,
    ) -> R {
        let model =
            crate::qwen3::Model::load_with_quant(&model_path(), Precision::F16, weight_quant)
                .expect("load Qwen3-0.6B");
        let mut decoder =
            VulkanDecoder::new(&model, Precision::F16, VulkanGemm::Plain, None, BUCKET)
                .expect("construct Vulkan decoder");
        body(&model, &mut decoder)
    }

    /// Host greedy argmax matching the sampler rule: highest logit, lowest id
    /// on tie (replace only on strictly greater under total_cmp).
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

    /// Deterministic synthetic prompt of a given length. The exact tokens do
    /// not matter because the batched and sequential paths must produce
    /// identical logits for ANY input; varied lengths exercise short, medium,
    /// and deep-context attention prefixes.
    fn synthetic_prompt(length: usize) -> Vec<u32> {
        (0..length)
            .map(|index| (1000 + index * 7919 % 5000) as u32)
            .collect()
    }

    /// Greedy continuation tokens and the per-position logits that produced
    /// them, via sequential single-token advances starting from `cache`
    /// (already prefilled). Returns (tokens, logits) where logits[i] is the
    /// logits after feeding tokens[i]; the seed logits (before any feed) are
    /// returned too.
    fn sequential_greedy(
        decoder: &mut VulkanDecoder,
        cache: &mut VulkanKvCache,
        seed_logits: &[f32],
        count: usize,
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut tokens = Vec::with_capacity(count);
        let mut logits = Vec::with_capacity(count);
        let mut next = greedy_argmax(seed_logits);
        for _ in 0..count {
            tokens.push(next);
            let step_logits = decoder.advance(cache, next).expect("sequential advance");
            next = greedy_argmax(&step_logits);
            logits.push(step_logits);
        }
        (tokens, logits)
    }

    fn byte_identical_gate_for(weight_quant: WeightQuantization) {
        with_decoder(weight_quant, |_model, decoder| {
            for prompt_len in [1usize, 5, 33, 128, 469] {
                let prompt = synthetic_prompt(prompt_len);
                let (cache, seed_logits) = decoder.prefill(&prompt).expect("prefill");
                let mut cache = cache;
                let base_position = decoder.cache_position(&cache);
                assert_eq!(base_position, prompt_len);

                // Sequential reference: greedy draft tokens and their logits.
                let (draft, seq_logits) = sequential_greedy(decoder, &mut cache, &seed_logits, 16);

                for k in [1usize, 2, 4, 8, 16] {
                    let draft = &draft[..k];
                    // Rewind to the prefix and run one batched forward over K tokens.
                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    let batch_logits = decoder
                        .context
                        .verify_batch_logits(draft)
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
                        .verify_tokens(&mut cache, draft)
                        .expect("batched verify argmaxes");
                    assert_eq!(batch_argmaxes.len(), k);
                    for i in 0..k {
                        assert_eq!(
                            batch_argmaxes[i],
                            greedy_argmax(&seq_logits[i]),
                            "batched argmax diverges at prompt_len={prompt_len} k={k} position {i}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    #[ignore]
    fn vulkan_batched_verify_logits_are_byte_identical_to_sequential_f16() {
        byte_identical_gate_for(WeightQuantization::None);
    }

    #[test]
    #[ignore]
    fn vulkan_batched_verify_logits_are_byte_identical_to_sequential_q8() {
        byte_identical_gate_for(WeightQuantization::Q8_0);
    }

    fn determinism_gate_for(weight_quant: WeightQuantization) {
        with_decoder(weight_quant, |_model, decoder| {
            let prompt = synthetic_prompt(64);
            let (cache, seed_logits) = decoder.prefill(&prompt).expect("prefill");
            let mut cache = cache;
            let base_position = decoder.cache_position(&cache);
            let (draft, _) = sequential_greedy(decoder, &mut cache, &seed_logits, 8);

            decoder.rewind(&mut cache, base_position).expect("rewind");
            let first = decoder
                .context
                .verify_batch_logits(&draft)
                .expect("first batched run");
            decoder.rewind(&mut cache, base_position).expect("rewind");
            let second = decoder
                .context
                .verify_batch_logits(&draft)
                .expect("second batched run");
            assert_eq!(
                logits_bits(&first),
                logits_bits(&second),
                "batched verification is not deterministic ({weight_quant:?})"
            );
        });
    }

    #[test]
    #[ignore]
    fn vulkan_batched_verify_is_deterministic_f16() {
        determinism_gate_for(WeightQuantization::None);
    }

    #[test]
    #[ignore]
    fn vulkan_batched_verify_is_deterministic_q8() {
        determinism_gate_for(WeightQuantization::Q8_0);
    }

    /// Per-token batched-verify cost curve. Prints the median wall time per
    /// verify_batch(K) call and the per-token (wall/K) figure for K in
    /// {1,2,4,8,16}. The authoritative numbers are taken on the Ally (the
    /// Vulkan decode timing authority; see VULKAN-DECODE.md); on any other GPU
    /// this still works as a functional timing harness. Select weights with
    /// SYNAPSE_VULKAN_BATCHED_PROBE_QUANT=f16|q8 (default q8).
    fn timing_probe(weight_quant: WeightQuantization) {
        with_decoder(weight_quant, |_model, decoder| {
            let prompt = synthetic_prompt(64);
            let (cache, seed_logits) = decoder.prefill(&prompt).expect("prefill");
            let mut cache = cache;
            let base_position = decoder.cache_position(&cache);
            let (draft, _) = sequential_greedy(decoder, &mut cache, &seed_logits, 16);

            // Warmup so GPU clocks and pipeline caches are steady before timing.
            for _ in 0..5 {
                decoder.rewind(&mut cache, base_position).expect("rewind");
                decoder
                    .verify_tokens(&mut cache, &draft[..8])
                    .expect("warmup");
            }

            println!("VULKAN_BATCHED_PROBE quant={weight_quant:?} prompt_len={base_position}");
            // Single-token reference: sequential greedy `advance` (the unchanged
            // per-token decode path). Running it in this same harness and build
            // gives a direct baseline beside the batched numbers and confirms
            // the additive batched path does not perturb the existing per-token
            // path.
            {
                let steps = 64;
                let iterations = 8;
                let mut samples = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    let mut next = greedy_argmax(&seed_logits);
                    let started = std::time::Instant::now();
                    for _ in 0..steps {
                        let logits = decoder.advance(&mut cache, next).expect("advance");
                        next = greedy_argmax(&logits);
                    }
                    samples.push(started.elapsed().as_secs_f64() / steps as f64);
                }
                samples.sort_by(|a, b| a.total_cmp(b));
                let median = samples[iterations / 2];
                println!(
                    "SINGLE_TOKEN_REFERENCE per_token_ms={:.4} decode_tok_per_s={:.2}",
                    median * 1e3,
                    1.0 / median
                );
            }
            for k in [1usize, 2, 4, 8, 16] {
                let draft = &draft[..k];
                let iterations = 40;
                let mut samples = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    let started = std::time::Instant::now();
                    decoder
                        .verify_tokens(&mut cache, draft)
                        .expect("verify_batch");
                    samples.push(started.elapsed().as_secs_f64());
                }
                samples.sort_by(|a, b| a.total_cmp(b));
                let median = samples[iterations / 2];
                println!(
                    "VULKAN_BATCHED_PROBE k={k:>2} median_call_ms={:.4} per_token_ms={:.4} verify_tok_per_s={:.2}",
                    median * 1e3,
                    median * 1e3 / k as f64,
                    k as f64 / median
                );
            }
        });
    }

    #[test]
    #[ignore]
    fn vulkan_batched_verify_timing_probe() {
        let quant = match std::env::var("SYNAPSE_VULKAN_BATCHED_PROBE_QUANT")
            .ok()
            .as_deref()
        {
            Some("f16") => WeightQuantization::None,
            _ => WeightQuantization::Q8_0,
        };
        timing_probe(quant);
    }
}
