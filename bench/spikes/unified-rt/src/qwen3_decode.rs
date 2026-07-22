//! Instrumentable greedy decoding for Qwen3 on the owned Metal runtime.
//!
//! The decode controller is backend-agnostic so pause, tap, and splice semantics
//! can be proved without a model download. `MetalDecoder` supplies the resident
//! Qwen3 implementation and keeps its KV buffers alive across calls.

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::json_constraint::{DecodeConstraint, TokenMask};
use crate::{
    quant::Q8_0Tensor,
    qwen3::{Model, Weight},
    Tensor,
};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct TopLogit {
    pub(crate) token_id: u32,
    pub(crate) logit: f32,
}

#[derive(Debug)]
pub(crate) struct TokenTapEvent<'a> {
    pub(crate) step: usize,
    pub(crate) token_id: u32,
    pub(crate) top_logits: &'a [TopLogit],
}

pub(crate) trait TokenStreamTap {
    fn before_commit(&mut self, event: TokenTapEvent<'_>);
}

impl<F> TokenStreamTap for F
where
    F: FnMut(TokenTapEvent<'_>),
{
    fn before_commit(&mut self, event: TokenTapEvent<'_>) {
        self(event);
    }
}

#[allow(dead_code)]
pub(crate) trait DecodeKernel {
    type Cache;

    fn capacity(&self) -> usize;
    fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)>;
    fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>>;
    fn cache_position(&self, cache: &Self::Cache) -> usize;
    fn inspect_cache_layer(&self, cache: &Self::Cache, layer: usize) -> Result<Vec<f32>>;
    fn stage_timings(&self) -> DecodeStageTimings {
        DecodeStageTimings::default()
    }

    /// The GPU-chained decode span, or 1 when the backend has no chained path.
    /// A backend returning > 1 must make `advance_chain` produce exactly the
    /// same tokens as the same number of per-token `advance`/argmax steps.
    fn chain_span(&self) -> usize {
        1
    }

    /// Advance `steps` tokens in one fused submission, returning the argmax
    /// token id of every step. `seed` feeds the first step. The final element
    /// is the token committed at the caller's newest position; earlier elements
    /// are the intermediate tokens. Backends without a chained path (the
    /// default) must not be asked for this — `chain_span` returns 1 for them.
    fn advance_chain(&mut self, _cache: &mut Self::Cache, _seed: u32, _steps: usize) -> Result<Vec<u32>> {
        anyhow::bail!("this decode backend has no chained multi-token path")
    }
}

pub(crate) trait DecodeRuntime: DecodeKernel {
    fn lane(&self) -> &'static str;
    fn kv_update_path(&self) -> &'static str;
    fn weight_feed_path(&self) -> &'static str;
    fn optimization_level(&self) -> u8;
}

/// A paused session owns all logical state needed to resume generation exactly.
pub(crate) struct DecodeSession<'a, K: DecodeKernel> {
    kernel: &'a mut K,
    cache: K::Cache,
    sequence: Vec<u32>,
    generated: Vec<u32>,
    next_logits: Vec<f32>,
    sample_wall_s: f64,
    constraint_wall_s: f64,
}

#[allow(dead_code)]
impl<'a, K: DecodeKernel> DecodeSession<'a, K> {
    pub(crate) fn prefill(kernel: &'a mut K, prompt: &[u32]) -> Result<Self> {
        ensure!(
            !prompt.is_empty(),
            "decode prompt must contain at least one token"
        );
        let (cache, next_logits) = kernel.prefill(prompt)?;
        ensure!(
            kernel.cache_position(&cache) == prompt.len(),
            "prefill cache position does not match prompt length"
        );
        Ok(Self {
            kernel,
            cache,
            sequence: prompt.to_vec(),
            generated: Vec::new(),
            next_logits,
            sample_wall_s: 0.0,
            constraint_wall_s: 0.0,
        })
    }

    pub(crate) fn sequence(&self) -> &[u32] {
        &self.sequence
    }

    pub(crate) fn generated(&self) -> &[u32] {
        &self.generated
    }

    pub(crate) fn position(&self) -> usize {
        self.kernel.cache_position(&self.cache)
    }

    pub(crate) fn inspect_cache_layer(&self, layer: usize) -> Result<Vec<f32>> {
        self.kernel.inspect_cache_layer(&self.cache, layer)
    }

    pub(crate) fn sample_wall_s(&self) -> f64 {
        self.sample_wall_s
    }

    pub(crate) fn constraint_wall_s(&self) -> f64 {
        self.constraint_wall_s
    }

    pub(crate) fn stage_timings(&self) -> DecodeStageTimings {
        self.kernel.stage_timings()
    }

    /// The backend's GPU-chained decode span (1 when it has no chained path).
    pub(crate) fn chain_span(&self) -> usize {
        self.kernel.chain_span()
    }

    pub(crate) fn generate(
        &mut self,
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        top_k: usize,
        tap: &mut dyn TokenStreamTap,
    ) -> Result<Vec<u32>> {
        self.generate_inner(max_tokens, stop_tokens, top_k, None, tap)
    }

    pub(crate) fn generate_constrained(
        &mut self,
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        top_k: usize,
        constraint: &mut dyn DecodeConstraint,
        tap: &mut dyn TokenStreamTap,
    ) -> Result<Vec<u32>> {
        self.generate_inner(max_tokens, stop_tokens, top_k, Some(constraint), tap)
    }

    fn generate_inner(
        &mut self,
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        top_k: usize,
        mut constraint: Option<&mut dyn DecodeConstraint>,
        tap: &mut dyn TokenStreamTap,
    ) -> Result<Vec<u32>> {
        ensure!(top_k > 0, "decode tap top-k must be positive");
        let first_generated = self.generated.len();
        for _ in 0..max_tokens {
            ensure!(
                self.position() < self.kernel.capacity(),
                "decode cache capacity {} exhausted",
                self.kernel.capacity()
            );
            let sample_started = Instant::now();
            let top = if let Some(constraint) = constraint.as_deref_mut() {
                let constraint_started = Instant::now();
                let mask = constraint.allowed()?;
                let top = top_logits_masked(&self.next_logits, &mask, top_k);
                self.constraint_wall_s += constraint_started.elapsed().as_secs_f64();
                top
            } else {
                top_logits(&self.next_logits, top_k)
            };
            ensure!(
                !top.is_empty(),
                "decode constraint masked every model token"
            );
            let token = top[0].token_id;
            self.sample_wall_s += sample_started.elapsed().as_secs_f64();
            tap.before_commit(TokenTapEvent {
                step: self.generated.len(),
                token_id: token,
                top_logits: &top,
            });
            if let Some(constraint) = constraint.as_deref_mut() {
                let constraint_started = Instant::now();
                constraint.advance(token)?;
                self.constraint_wall_s += constraint_started.elapsed().as_secs_f64();
            }
            self.sequence.push(token);
            self.generated.push(token);
            self.next_logits = self.kernel.advance(&mut self.cache, token)?;
            if stop_tokens.contains(&token) {
                break;
            }
        }
        if let Some(constraint) = constraint {
            ensure!(
                constraint.is_complete(),
                "JSON constraint did not complete within {max_tokens} generated tokens: {}",
                constraint.describe()
            );
        }
        Ok(self.generated[first_generated..].to_vec())
    }

    /// GPU-chained greedy generation. Produces the same tokens as `generate`
    /// with the same `top_k=1` argmax, but decodes in spans of `chain_span()`:
    /// the host takes one argmax to seed a span, the backend runs the whole span
    /// device-side and returns its token ids, and the host commits them and
    /// checks stop tokens. Because the backend's chained argmax and embedding
    /// gather are byte-exact with the per-token path, the committed stream is
    /// identical to `generate`.
    ///
    /// This path intentionally emits no per-token top-logit tap and accepts no
    /// constraint: the requirement that the k=1 path stay fully instrumented
    /// (tap, pause, splice, JSON constraint) is met by `generate` at chain span
    /// 1, and callers needing any of those must use `generate`. Up to
    /// `chain_span - 1` tokens can
    /// be produced past a stop token within a span; they are truncated here so
    /// the returned stream stops exactly at the first stop token, matching
    /// `generate`.
    pub(crate) fn generate_chained(
        &mut self,
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        tap: &mut dyn TokenStreamTap,
    ) -> Result<Vec<u32>> {
        let span = self.kernel.chain_span();
        ensure!(span >= 1, "chain span must be at least 1");
        let first_generated = self.generated.len();
        let mut produced = 0usize;
        'outer: while produced < max_tokens {
            // Seed the span with the host argmax of the pending logits, exactly
            // as `generate` selects its next token.
            let sample_started = Instant::now();
            let top = top_logits(&self.next_logits, 1);
            ensure!(!top.is_empty(), "decode produced empty logits");
            let seed = top[0].token_id;
            self.sample_wall_s += sample_started.elapsed().as_secs_f64();

            // The span may not exceed the remaining budget or the cache.
            let remaining = max_tokens - produced;
            let capacity_left = self.kernel.capacity().saturating_sub(self.position());
            ensure!(capacity_left > 0, "decode cache capacity exhausted");
            // advance_chain(seed, steps) advances the cache `steps` positions and
            // returns the `steps` tokens that follow the seed. Together with the
            // host-selected seed that yields `steps + 1` committed tokens, so cap
            // the chain steps to keep the total within budget and capacity.
            let max_follow = remaining.saturating_sub(1);
            let steps = span.saturating_sub(1).min(max_follow).min(capacity_left - 1);

            // Commit the seed first (it occupies one cache position via the
            // chain's step 0), tapping it like `generate` does before commit.
            tap.before_commit(TokenTapEvent {
                step: self.generated.len(),
                token_id: seed,
                top_logits: &top,
            });
            self.sequence.push(seed);
            self.generated.push(seed);
            produced += 1;
            if stop_tokens.contains(&seed) {
                break;
            }
            if steps == 0 {
                // No room for a follow-on span; refresh logits per token so the
                // loop can select the next seed.
                self.next_logits = self.kernel.advance(&mut self.cache, seed)?;
                continue;
            }

            let followers = self.kernel.advance_chain(&mut self.cache, seed, steps)?;
            ensure!(
                followers.len() == steps,
                "chained decode returned the wrong number of tokens"
            );
            for (offset, &token) in followers.iter().enumerate() {
                tap.before_commit(TokenTapEvent {
                    step: self.generated.len(),
                    token_id: token,
                    top_logits: &[],
                });
                self.sequence.push(token);
                self.generated.push(token);
                produced += 1;
                if stop_tokens.contains(&token) {
                    break 'outer;
                }
                // The final follower's logits are needed to seed the next span.
                if offset + 1 == steps {
                    self.next_logits = self.kernel.advance(&mut self.cache, token)?;
                }
            }
        }
        Ok(self.generated[first_generated..].to_vec())
    }

    /// Commits externally supplied tokens and advances every layer's KV state.
    pub(crate) fn splice(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            self.position() + tokens.len() <= self.kernel.capacity(),
            "token splice exceeds decode cache capacity"
        );
        for &token in tokens {
            self.sequence.push(token);
            self.next_logits = self.kernel.advance(&mut self.cache, token)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct DecodeKernelTimings {
    /// GPU seconds attributed to the RMSNorm kernels.
    pub(crate) rmsnorm_s: f64,
    /// GPU seconds attributed to Q/K/V projection matvecs.
    pub(crate) qkv_matvec_s: f64,
    /// GPU seconds attributed to Q/K normalization and RoPE.
    pub(crate) qk_norm_rope_s: f64,
    /// GPU seconds attributed to attention score, softmax, and P/V work.
    pub(crate) attention_s: f64,
    /// GPU seconds attributed to the attention output projection.
    pub(crate) o_proj_s: f64,
    /// GPU seconds attributed to the fused residual RMSNorm.
    pub(crate) residual_rmsnorm_s: f64,
    /// GPU seconds attributed to the MLP down projection.
    pub(crate) down_proj_s: f64,
    /// GPU seconds attributed to the fused gate/up/SiLU product.
    pub(crate) gate_up_swiglu_s: f64,
    /// GPU seconds attributed to the vocabulary projection.
    pub(crate) lm_head_s: f64,
    /// Number of profiled command buffers contributing to these totals.
    pub(crate) samples: u64,
}

impl DecodeKernelTimings {
    pub(crate) fn delta(self, earlier: Self) -> Self {
        Self {
            rmsnorm_s: self.rmsnorm_s - earlier.rmsnorm_s,
            qkv_matvec_s: self.qkv_matvec_s - earlier.qkv_matvec_s,
            qk_norm_rope_s: self.qk_norm_rope_s - earlier.qk_norm_rope_s,
            attention_s: self.attention_s - earlier.attention_s,
            o_proj_s: self.o_proj_s - earlier.o_proj_s,
            residual_rmsnorm_s: self.residual_rmsnorm_s - earlier.residual_rmsnorm_s,
            down_proj_s: self.down_proj_s - earlier.down_proj_s,
            gate_up_swiglu_s: self.gate_up_swiglu_s - earlier.gate_up_swiglu_s,
            lm_head_s: self.lm_head_s - earlier.lm_head_s,
            samples: self.samples - earlier.samples,
        }
    }

    pub(crate) fn accumulate(&mut self, other: Self) {
        self.rmsnorm_s += other.rmsnorm_s;
        self.qkv_matvec_s += other.qkv_matvec_s;
        self.qk_norm_rope_s += other.qk_norm_rope_s;
        self.attention_s += other.attention_s;
        self.o_proj_s += other.o_proj_s;
        self.residual_rmsnorm_s += other.residual_rmsnorm_s;
        self.down_proj_s += other.down_proj_s;
        self.gate_up_swiglu_s += other.gate_up_swiglu_s;
        self.lm_head_s += other.lm_head_s;
        self.samples += other.samples;
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct DecodeStageTimings {
    pub(crate) graph_prepare_wall_s: f64,
    pub(crate) host_prepare_wall_s: f64,
    pub(crate) feed_wall_s: f64,
    pub(crate) execute_wall_s: f64,
    pub(crate) logits_readback_wall_s: f64,
    pub(crate) kv_update_wall_s: f64,
    pub(crate) kernel_gpu: DecodeKernelTimings,
    pub(crate) prefill_calls: u64,
    pub(crate) step_calls: u64,
}

impl DecodeStageTimings {
    pub(crate) fn delta(self, earlier: Self) -> Self {
        Self {
            graph_prepare_wall_s: self.graph_prepare_wall_s - earlier.graph_prepare_wall_s,
            host_prepare_wall_s: self.host_prepare_wall_s - earlier.host_prepare_wall_s,
            feed_wall_s: self.feed_wall_s - earlier.feed_wall_s,
            execute_wall_s: self.execute_wall_s - earlier.execute_wall_s,
            logits_readback_wall_s: self.logits_readback_wall_s - earlier.logits_readback_wall_s,
            kv_update_wall_s: self.kv_update_wall_s - earlier.kv_update_wall_s,
            kernel_gpu: self.kernel_gpu.delta(earlier.kernel_gpu),
            prefill_calls: self.prefill_calls - earlier.prefill_calls,
            step_calls: self.step_calls - earlier.step_calls,
        }
    }

    pub(crate) fn accumulate(&mut self, other: Self) {
        self.graph_prepare_wall_s += other.graph_prepare_wall_s;
        self.host_prepare_wall_s += other.host_prepare_wall_s;
        self.feed_wall_s += other.feed_wall_s;
        self.execute_wall_s += other.execute_wall_s;
        self.logits_readback_wall_s += other.logits_readback_wall_s;
        self.kv_update_wall_s += other.kv_update_wall_s;
        self.kernel_gpu.accumulate(other.kernel_gpu);
        self.prefill_calls += other.prefill_calls;
        self.step_calls += other.step_calls;
    }
}

pub(crate) fn top_logits(logits: &[f32], top_k: usize) -> Vec<TopLogit> {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(top_k > 0, "top-k must be positive");
    let mut top = Vec::<TopLogit>::with_capacity(top_k.min(logits.len()));
    for (token_id, &logit) in logits.iter().enumerate() {
        let candidate = TopLogit {
            token_id: token_id as u32,
            logit,
        };
        if top.len() == top_k && !logit_precedes(&candidate, &top[top_k - 1]) {
            continue;
        }
        let insertion = top
            .iter()
            .position(|current| logit_precedes(&candidate, current))
            .unwrap_or(top.len());
        if insertion < top_k {
            top.insert(insertion, candidate);
            if top.len() > top_k {
                top.pop();
            }
        }
    }
    top
}

pub(crate) fn top_logits_masked(logits: &[f32], mask: &TokenMask, top_k: usize) -> Vec<TopLogit> {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(top_k > 0, "top-k must be positive");
    let mut top = Vec::<TopLogit>::with_capacity(top_k.min(mask.len()));
    for token_id in mask.token_ids() {
        let Some(&logit) = logits.get(token_id as usize) else {
            continue;
        };
        insert_top_logit(&mut top, TopLogit { token_id, logit }, top_k);
    }
    top
}

fn insert_top_logit(top: &mut Vec<TopLogit>, candidate: TopLogit, top_k: usize) {
    if top.len() == top_k && !logit_precedes(&candidate, &top[top_k - 1]) {
        return;
    }
    let insertion = top
        .iter()
        .position(|current| logit_precedes(&candidate, current))
        .unwrap_or(top.len());
    if insertion < top_k {
        top.insert(insertion, candidate);
        if top.len() > top_k {
            top.pop();
        }
    }
}

fn logit_precedes(candidate: &TopLogit, current: &TopLogit) -> bool {
    candidate.logit.total_cmp(&current.logit).is_gt()
        || (candidate.logit.total_cmp(&current.logit).is_eq()
            && candidate.token_id < current.token_id)
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub(crate) struct WeightRegionKey {
    pub(crate) layer: Option<usize>,
    pub(crate) tensor_name: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct WeightRegion {
    /// Address of active host storage, valid until model drop; quantized regions span whole GGUF blocks.
    pub(crate) buffer_handle: usize,
    pub(crate) byte_len: usize,
    pub(crate) checksum: u64,
}

impl Model {
    pub(crate) fn weight_regions(&self) -> BTreeMap<WeightRegionKey, WeightRegion> {
        let mut regions = BTreeMap::new();
        insert_region(&mut regions, None, "embed_tokens.weight", &self.embeddings);
        insert_region(&mut regions, None, "norm.weight", &self.final_norm.weight);
        if let Some(lm_head) = &self.lm_head {
            insert_weight_region(&mut regions, None, "lm_head.weight", lm_head);
        } else if let Some(lm_head) = &self.tied_lm_head_q8_0 {
            insert_q8_region(&mut regions, None, "lm_head.weight", lm_head);
        }
        for (index, layer) in self.layers.iter().enumerate() {
            insert_region(
                &mut regions,
                Some(index),
                "input_layernorm.weight",
                &layer.input_norm.weight,
            );
            insert_region(
                &mut regions,
                Some(index),
                "post_attention_layernorm.weight",
                &layer.post_attention_norm.weight,
            );
            insert_weight_region(
                &mut regions,
                Some(index),
                "self_attn.q_proj.weight",
                &layer.q_proj,
            );
            insert_region(
                &mut regions,
                Some(index),
                "self_attn.q_norm.weight",
                &layer.q_norm.weight,
            );
            insert_weight_region(
                &mut regions,
                Some(index),
                "self_attn.k_proj.weight",
                &layer.k_proj,
            );
            insert_region(
                &mut regions,
                Some(index),
                "self_attn.k_norm.weight",
                &layer.k_norm.weight,
            );
            for (name, weight) in [
                ("self_attn.v_proj.weight", &layer.v_proj),
                ("self_attn.o_proj.weight", &layer.o_proj),
                ("mlp.gate_proj.weight", &layer.gate_proj),
                ("mlp.up_proj.weight", &layer.up_proj),
                ("mlp.down_proj.weight", &layer.down_proj),
            ] {
                insert_weight_region(&mut regions, Some(index), name, weight);
            }
        }
        regions
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn weight_region_bytes(&self, key: &WeightRegionKey) -> Option<Vec<u8>> {
        match (key.layer, key.tensor_name) {
            (None, "embed_tokens.weight") => Some(active_tensor_bytes(&self.embeddings)),
            (None, "norm.weight") => Some(active_tensor_bytes(&self.final_norm.weight)),
            (None, "lm_head.weight") => {
                self.lm_head.as_ref().map(active_weight_bytes).or_else(|| {
                    self.tied_lm_head_q8_0
                        .as_ref()
                        .map(|weight| weight.as_bytes().to_vec())
                })
            }
            (Some(index), name) => self.layers.get(index).and_then(|layer| match name {
                "input_layernorm.weight" => Some(active_tensor_bytes(&layer.input_norm.weight)),
                "post_attention_layernorm.weight" => {
                    Some(active_tensor_bytes(&layer.post_attention_norm.weight))
                }
                "self_attn.q_proj.weight" => Some(active_weight_bytes(&layer.q_proj)),
                "self_attn.q_norm.weight" => Some(active_tensor_bytes(&layer.q_norm.weight)),
                "self_attn.k_proj.weight" => Some(active_weight_bytes(&layer.k_proj)),
                "self_attn.k_norm.weight" => Some(active_tensor_bytes(&layer.k_norm.weight)),
                "self_attn.v_proj.weight" => Some(active_weight_bytes(&layer.v_proj)),
                "self_attn.o_proj.weight" => Some(active_weight_bytes(&layer.o_proj)),
                "mlp.gate_proj.weight" => Some(active_weight_bytes(&layer.gate_proj)),
                "mlp.up_proj.weight" => Some(active_weight_bytes(&layer.up_proj)),
                "mlp.down_proj.weight" => Some(active_weight_bytes(&layer.down_proj)),
                _ => None,
            }),
            _ => None,
        }
    }
}

fn insert_weight_region(
    regions: &mut BTreeMap<WeightRegionKey, WeightRegion>,
    layer: Option<usize>,
    tensor_name: &'static str,
    weight: &Weight,
) {
    if let Some(quantized) = &weight.q8_0 {
        insert_q8_region(regions, layer, tensor_name, quantized);
    } else {
        insert_region(regions, layer, tensor_name, &weight.tensor);
    }
}

fn insert_q8_region(
    regions: &mut BTreeMap<WeightRegionKey, WeightRegion>,
    layer: Option<usize>,
    tensor_name: &'static str,
    quantized: &Q8_0Tensor,
) {
    let checksum = quantized
        .as_bytes()
        .iter()
        .copied()
        .fold(1469598103934665603u64, checksum_byte);
    regions.insert(
        WeightRegionKey { layer, tensor_name },
        WeightRegion {
            buffer_handle: quantized.as_bytes().as_ptr() as usize,
            byte_len: quantized.as_bytes().len(),
            checksum,
        },
    );
}

#[cfg_attr(not(test), allow(dead_code))]
fn active_weight_bytes(weight: &Weight) -> Vec<u8> {
    weight.q8_0.as_ref().map_or_else(
        || active_tensor_bytes(&weight.tensor),
        |quantized| quantized.as_bytes().to_vec(),
    )
}

fn insert_region(
    regions: &mut BTreeMap<WeightRegionKey, WeightRegion>,
    layer: Option<usize>,
    tensor_name: &'static str,
    tensor: &Tensor,
) {
    let (buffer_handle, byte_len, checksum) = tensor.metal_f16_bits.as_ref().map_or_else(
        || {
            let checksum = tensor
                .data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .fold(1469598103934665603u64, checksum_byte);
            (
                tensor.data.as_ptr() as usize,
                std::mem::size_of_val(tensor.data.as_slice()),
                checksum,
            )
        },
        |bits| {
            let checksum = bits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .fold(1469598103934665603u64, checksum_byte);
            (
                bits.as_ptr() as usize,
                std::mem::size_of_val(bits.as_slice()),
                checksum,
            )
        },
    );
    regions.insert(
        WeightRegionKey { layer, tensor_name },
        WeightRegion {
            buffer_handle,
            byte_len,
            checksum,
        },
    );
}

fn checksum_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
}

#[cfg_attr(not(test), allow(dead_code))]
fn active_tensor_bytes(tensor: &Tensor) -> Vec<u8> {
    tensor.metal_f16_bits.as_ref().map_or_else(
        || {
            tensor
                .data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        },
        |bits| bits.iter().flat_map(|value| value.to_le_bytes()).collect(),
    )
}

#[cfg(target_os = "macos")]
mod metal {
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::path::Path;
    use std::ptr::NonNull;
    use std::time::Instant;

    use anyhow::{bail, ensure, Result};

    use super::{DecodeKernel, DecodeKernelTimings, DecodeStageTimings, Model};
    use crate::{encode_f16_bits, Execution, MetalExecutionConfig, Precision};

    #[repr(C)]
    struct LayerParams {
        input_norm: *const c_void,
        post_attention_norm: *const c_void,
        q_weight: *const c_void,
        q_norm: *const c_void,
        k_weight: *const c_void,
        k_norm: *const c_void,
        v_weight: *const c_void,
        o_weight: *const c_void,
        gate_weight: *const c_void,
        up_weight: *const c_void,
        down_weight: *const c_void,
    }

    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct NativeDecodeStageTimings {
        graph_prepare_wall_s: f64,
        feed_wall_s: f64,
        execute_wall_s: f64,
        logits_readback_wall_s: f64,
        kv_update_wall_s: f64,
        prefill_calls: u64,
        step_calls: u64,
    }

    pub(crate) struct MetalKvCache {
        pub(crate) position: usize,
    }

    pub(crate) struct MetalDecoder<'a> {
        raw: NonNull<c_void>,
        model: &'a Model,
        bucket: usize,
        prefill_package: Option<CString>,
        step_package: Option<CString>,
        host_prepare_wall_s: f64,
        legacy_cpu_readback: bool,
        optimization_level_one: bool,
    }

    impl<'a> MetalDecoder<'a> {
        pub(crate) fn new(
            model: &'a Model,
            precision: Precision,
            execution: &MetalExecutionConfig,
            bucket: usize,
        ) -> Result<Self> {
            ensure!(
                matches!(precision, Precision::F16),
                "Qwen3 decode currently requires f16"
            );
            ensure!(
                matches!(execution.execution, Execution::Explicit),
                "Qwen3 decode requires explicit graph compilation"
            );
            ensure!(
                [512, 1024, 2048].contains(&bucket),
                "decode cache bucket must be 512, 1024, or 2048"
            );
            let raw = NonNull::new(unsafe {
                synapse_qwen3_decode_context_new(
                    bucket as u64,
                    model.layers.len() as u64,
                    model.config.num_key_value_heads as u64,
                    model.config.head_dim as u64,
                )
            })
            .ok_or_else(last_error)?;
            let prefill_package =
                package_cstring(execution.decode_package_path("prefill", bucket).as_deref())?;
            let step_package =
                package_cstring(execution.decode_package_path("step", bucket).as_deref())?;
            let decoder = Self {
                raw,
                model,
                bucket,
                prefill_package,
                step_package,
                host_prepare_wall_s: 0.0,
                legacy_cpu_readback: std::env::var_os("SYNAPSE_QWEN3_DECODE_LEGACY_READBACK")
                    .is_some_and(|value| value == std::ffi::OsStr::new("1")),
                optimization_level_one: std::env::var_os("SYNAPSE_QWEN3_DECODE_OPT_LEVEL")
                    .is_none_or(|value| value != std::ffi::OsStr::new("0")),
            };
            let status = unsafe {
                synapse_qwen3_decode_prepare(
                    decoder.raw.as_ptr(),
                    model.config.hidden_size as u64,
                    model.config.num_attention_heads as u64,
                    model.config.num_key_value_heads as u64,
                    model.config.head_dim as u64,
                    model.config.intermediate_size as u64,
                    model.layers.len() as u64,
                    model.config.vocab_size as u64,
                    model.config.rms_norm_eps,
                    decoder
                        .prefill_package
                        .as_ref()
                        .map_or(std::ptr::null(), |path| path.as_ptr()),
                    decoder
                        .step_package
                        .as_ref()
                        .map_or(std::ptr::null(), |path| path.as_ptr()),
                )
            };
            if status != 0 {
                bail!(
                    "Qwen3 Metal decode preparation failed with status {status}: {}",
                    last_error()
                );
            }
            Ok(decoder)
        }

        fn layer_params(&self) -> Result<Vec<LayerParams>> {
            self.model
                .layers
                .iter()
                .map(|layer| {
                    Ok(LayerParams {
                        input_norm: layer.input_norm.weight.metal_f16_bits()?.as_ptr().cast(),
                        post_attention_norm: layer
                            .post_attention_norm
                            .weight
                            .metal_f16_bits()?
                            .as_ptr()
                            .cast(),
                        q_weight: layer.q_proj.tensor.metal_f16_bits()?.as_ptr().cast(),
                        q_norm: layer.q_norm.weight.metal_f16_bits()?.as_ptr().cast(),
                        k_weight: layer.k_proj.tensor.metal_f16_bits()?.as_ptr().cast(),
                        k_norm: layer.k_norm.weight.metal_f16_bits()?.as_ptr().cast(),
                        v_weight: layer.v_proj.tensor.metal_f16_bits()?.as_ptr().cast(),
                        o_weight: layer.o_proj.tensor.metal_f16_bits()?.as_ptr().cast(),
                        gate_weight: layer.gate_proj.tensor.metal_f16_bits()?.as_ptr().cast(),
                        up_weight: layer.up_proj.tensor.metal_f16_bits()?.as_ptr().cast(),
                        down_weight: layer.down_proj.tensor.metal_f16_bits()?.as_ptr().cast(),
                    })
                })
                .collect()
        }

        fn rope(&self, positions: std::ops::Range<usize>) -> (Vec<u16>, Vec<u16>) {
            let head_dim = self.model.config.head_dim;
            let mut cosine = Vec::with_capacity(positions.len() * head_dim);
            let mut sine = Vec::with_capacity(positions.len() * head_dim);
            for position in positions {
                for index in 0..head_dim {
                    let rotary_index = index % (head_dim / 2);
                    let frequency = 1.0
                        / self
                            .model
                            .config
                            .rope_theta
                            .powf((2 * rotary_index) as f32 / self.model.config.head_dim as f32);
                    let (sin, cos) = (position as f32 * frequency).sin_cos();
                    cosine.push(cos);
                    sine.push(sin);
                }
            }
            (encode_f16_bits(&cosine), encode_f16_bits(&sine))
        }

        fn embedding(&self, token: u32) -> Result<&[f32]> {
            let token = token as usize;
            ensure!(
                token < self.model.config.vocab_size,
                "token id {token} outside Qwen3 vocab"
            );
            let hidden = self.model.config.hidden_size;
            Ok(&self.model.embeddings.data[token * hidden..(token + 1) * hidden])
        }

        fn common_call_args(&self) -> Result<(Vec<LayerParams>, *const c_void, *const c_void)> {
            let layers = self.layer_params()?;
            let final_norm = self
                .model
                .final_norm
                .weight
                .metal_f16_bits()?
                .as_ptr()
                .cast();
            let lm_head = self.model.lm_head()?.metal_f16_bits()?.as_ptr().cast();
            Ok((layers, final_norm, lm_head))
        }

        pub(crate) fn kv_update_path(&self) -> &'static str {
            if self.legacy_cpu_readback {
                "cpu-readback-control"
            } else {
                "device-resident-blit"
            }
        }

        pub(crate) fn weight_feed_path(&self) -> &'static str {
            "f16-static-feeds-with-f16-attention-matmuls"
        }

        pub(crate) fn optimization_level(&self) -> u8 {
            u8::from(self.optimization_level_one)
        }

        pub(crate) fn stage_timings(&self) -> DecodeStageTimings {
            let mut native = NativeDecodeStageTimings::default();
            unsafe { synapse_qwen3_decode_stage_timings(self.raw.as_ptr(), &mut native) };
            DecodeStageTimings {
                graph_prepare_wall_s: native.graph_prepare_wall_s,
                host_prepare_wall_s: self.host_prepare_wall_s,
                feed_wall_s: native.feed_wall_s,
                execute_wall_s: native.execute_wall_s,
                logits_readback_wall_s: native.logits_readback_wall_s,
                kv_update_wall_s: native.kv_update_wall_s,
                kernel_gpu: DecodeKernelTimings::default(),
                prefill_calls: native.prefill_calls,
                step_calls: native.step_calls,
            }
        }

        pub(crate) fn inspect_cache_bits(&self, layer: usize) -> Result<Vec<u16>> {
            ensure!(
                layer < self.model.layers.len(),
                "KV cache layer {layer} out of range"
            );
            let elements = 2
                * self.model.config.num_key_value_heads
                * self.bucket
                * self.model.config.head_dim;
            let mut bits = vec![0u16; elements];
            let status = unsafe {
                synapse_qwen3_decode_cache_copy(
                    self.raw.as_ptr(),
                    layer as u64,
                    bits.as_mut_ptr(),
                    elements as u64,
                )
            };
            if status != 0 {
                bail!(
                    "Qwen3 Metal cache inspection failed: {status}: {}",
                    last_error()
                );
            }
            Ok(bits)
        }
    }

    impl DecodeKernel for MetalDecoder<'_> {
        type Cache = MetalKvCache;

        fn capacity(&self) -> usize {
            self.bucket
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
            let host_prepare_started = Instant::now();
            ensure!(!tokens.is_empty(), "decode prompt must not be empty");
            ensure!(
                tokens.len() <= self.bucket,
                "decode prompt exceeds cache bucket"
            );
            let hidden = self.model.config.hidden_size;
            let mut input = vec![0.0f32; self.bucket * hidden];
            for (position, &token) in tokens.iter().enumerate() {
                input[position * hidden..(position + 1) * hidden]
                    .copy_from_slice(self.embedding(token)?);
            }
            let input = encode_f16_bits(&input);
            let mut mask = vec![-10_000.0f32; self.bucket * self.bucket];
            for query in 0..self.bucket {
                for key in 0..tokens.len().min(query + 1) {
                    mask[query * self.bucket + key] = 0.0;
                }
            }
            let mut selector = vec![0.0f32; self.bucket];
            selector[tokens.len() - 1] = 1.0;
            let selector = encode_f16_bits(&selector);
            let (rope_cos, rope_sin) = self.rope(0..self.bucket);
            let (layers, final_norm, lm_head) = self.common_call_args()?;
            let mut logits = vec![0.0f32; self.model.config.vocab_size];
            self.host_prepare_wall_s += host_prepare_started.elapsed().as_secs_f64();
            let status = unsafe {
                synapse_qwen3_decode_prefill(
                    self.raw.as_ptr(),
                    self.model.config.hidden_size as u64,
                    self.model.config.num_attention_heads as u64,
                    self.model.config.num_key_value_heads as u64,
                    self.model.config.head_dim as u64,
                    self.model.config.intermediate_size as u64,
                    self.model.layers.len() as u64,
                    self.model.config.vocab_size as u64,
                    self.model.config.rms_norm_eps,
                    self.prefill_package
                        .as_ref()
                        .map_or(std::ptr::null(), |path| path.as_ptr()),
                    input.as_ptr().cast(),
                    mask.as_ptr(),
                    rope_cos.as_ptr().cast(),
                    rope_sin.as_ptr().cast(),
                    selector.as_ptr().cast(),
                    layers.as_ptr(),
                    final_norm,
                    lm_head,
                    logits.as_mut_ptr(),
                )
            };
            if status != 0 {
                bail!(
                    "Qwen3 Metal prefill failed with status {status}: {}",
                    last_error()
                );
            }
            Ok((
                MetalKvCache {
                    position: tokens.len(),
                },
                logits,
            ))
        }

        fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>> {
            let host_prepare_started = Instant::now();
            ensure!(
                cache.position < self.bucket,
                "decode cache capacity exhausted"
            );
            let input = encode_f16_bits(self.embedding(token)?);
            let keys = self.bucket + 1;
            let mut mask = vec![-10_000.0f32; keys];
            mask[..cache.position].fill(0.0);
            mask[self.bucket] = 0.0;
            let (rope_cos, rope_sin) = self.rope(cache.position..cache.position + 1);
            let (layers, final_norm, lm_head) = self.common_call_args()?;
            let mut logits = vec![0.0f32; self.model.config.vocab_size];
            self.host_prepare_wall_s += host_prepare_started.elapsed().as_secs_f64();
            let status = unsafe {
                synapse_qwen3_decode_step(
                    self.raw.as_ptr(),
                    cache.position as u64,
                    self.model.config.hidden_size as u64,
                    self.model.config.num_attention_heads as u64,
                    self.model.config.num_key_value_heads as u64,
                    self.model.config.head_dim as u64,
                    self.model.config.intermediate_size as u64,
                    self.model.layers.len() as u64,
                    self.model.config.vocab_size as u64,
                    self.model.config.rms_norm_eps,
                    self.step_package
                        .as_ref()
                        .map_or(std::ptr::null(), |path| path.as_ptr()),
                    input.as_ptr().cast(),
                    mask.as_ptr(),
                    rope_cos.as_ptr().cast(),
                    rope_sin.as_ptr().cast(),
                    layers.as_ptr(),
                    final_norm,
                    lm_head,
                    logits.as_mut_ptr(),
                )
            };
            if status != 0 {
                bail!(
                    "Qwen3 Metal decode step failed with status {status}: {}",
                    last_error()
                );
            }
            cache.position += 1;
            Ok(logits)
        }

        fn cache_position(&self, cache: &Self::Cache) -> usize {
            cache.position
        }

        fn stage_timings(&self) -> DecodeStageTimings {
            MetalDecoder::stage_timings(self)
        }

        fn inspect_cache_layer(&self, _cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
            Ok(self
                .inspect_cache_bits(layer)?
                .into_iter()
                .map(half::f16::from_bits)
                .map(f32::from)
                .collect())
        }
    }

    impl super::DecodeRuntime for MetalDecoder<'_> {
        fn lane(&self) -> &'static str {
            "owned-rt-metal"
        }

        fn kv_update_path(&self) -> &'static str {
            MetalDecoder::kv_update_path(self)
        }

        fn weight_feed_path(&self) -> &'static str {
            MetalDecoder::weight_feed_path(self)
        }

        fn optimization_level(&self) -> u8 {
            MetalDecoder::optimization_level(self)
        }
    }

    impl Drop for MetalDecoder<'_> {
        fn drop(&mut self) {
            unsafe { synapse_qwen3_decode_context_free(self.raw.as_ptr()) }
        }
    }

    fn package_cstring(path: Option<&Path>) -> Result<Option<CString>> {
        path.map(|path| CString::new(path.to_string_lossy().as_bytes()).map_err(Into::into))
            .transpose()
    }

    fn last_error() -> anyhow::Error {
        unsafe {
            let raw = synapse_qwen3_decode_last_error();
            if raw.is_null() {
                anyhow::anyhow!("unknown Qwen3 decode MPSGraph error")
            } else {
                anyhow::anyhow!(CStr::from_ptr(raw).to_string_lossy().into_owned())
            }
        }
    }

    unsafe extern "C" {
        fn synapse_qwen3_decode_context_new(
            bucket: u64,
            layer_count: u64,
            kv_heads: u64,
            head_dim: u64,
        ) -> *mut c_void;
        fn synapse_qwen3_decode_context_free(context: *mut c_void);
        fn synapse_qwen3_decode_prepare(
            context: *mut c_void,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            vocab: u64,
            epsilon: f32,
            prefill_package_path: *const c_char,
            step_package_path: *const c_char,
        ) -> i32;
        fn synapse_qwen3_decode_prefill(
            context: *mut c_void,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            vocab: u64,
            epsilon: f32,
            package_path: *const c_char,
            input: *const c_void,
            mask: *const f32,
            rope_cos: *const c_void,
            rope_sin: *const c_void,
            selector: *const c_void,
            layers: *const LayerParams,
            final_norm: *const c_void,
            lm_head: *const c_void,
            logits: *mut f32,
        ) -> i32;
        fn synapse_qwen3_decode_step(
            context: *mut c_void,
            position: u64,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            vocab: u64,
            epsilon: f32,
            package_path: *const c_char,
            input: *const c_void,
            mask: *const f32,
            rope_cos: *const c_void,
            rope_sin: *const c_void,
            layers: *const LayerParams,
            final_norm: *const c_void,
            lm_head: *const c_void,
            logits: *mut f32,
        ) -> i32;
        fn synapse_qwen3_decode_stage_timings(
            context: *mut c_void,
            timings: *mut NativeDecodeStageTimings,
        );
        fn synapse_qwen3_decode_cache_copy(
            context: *mut c_void,
            layer: u64,
            output: *mut u16,
            elements: u64,
        ) -> i32;
        fn synapse_qwen3_decode_last_error() -> *const c_char;
    }
}

#[cfg(target_os = "macos")]
#[path = "qwen3_decode_metal_step.rs"]
mod metal_step;
#[cfg(target_os = "macos")]
pub(crate) use metal_step::MetalStepDecoder;

#[cfg(all(target_os = "linux", feature = "cuda"))]
mod cuda {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use super::{DecodeKernel, DecodeRuntime, Model};
    use crate::quant::{CudaWeight, WeightQuantization};

    #[repr(C)]
    struct LayerParams {
        input_norm: *const f32,
        post_attention_norm: *const f32,
        q_weight: CudaWeight,
        q_norm: *const f32,
        k_weight: CudaWeight,
        k_norm: *const f32,
        v_weight: CudaWeight,
        o_weight: CudaWeight,
        gate_weight: CudaWeight,
        up_weight: CudaWeight,
        down_weight: CudaWeight,
    }

    pub(crate) struct CudaKvCache {
        position: usize,
    }

    pub(crate) struct CudaDecoder<'a> {
        raw: NonNull<c_void>,
        model: &'a Model,
        bucket: usize,
    }

    impl<'a> CudaDecoder<'a> {
        pub(crate) fn new(model: &'a Model, bucket: usize) -> Result<Self> {
            ensure!(
                bucket > 0,
                "Qwen3 CUDA decode cache bucket must be positive"
            );
            let raw = NonNull::new(unsafe { synapse_cuda_qwen3_decode_context_new(bucket as u64) })
                .ok_or_else(last_error)?;
            let decoder = Self { raw, model, bucket };
            let layers = decoder.layer_params();
            let head = model.lm_head()?;
            let lm_head = CudaWeight::new(&head.data, model.lm_head_q8_0());
            let status = unsafe {
                synapse_cuda_qwen3_decode_prepare(
                    raw.as_ptr(),
                    model.config.hidden_size as u64,
                    model.config.num_attention_heads as u64,
                    model.config.num_key_value_heads as u64,
                    model.config.head_dim as u64,
                    model.config.intermediate_size as u64,
                    model.layers.len() as u64,
                    model.config.vocab_size as u64,
                    model.config.rms_norm_eps,
                    model.config.rope_theta,
                    layers.as_ptr(),
                    model.final_norm.weight.data.as_ptr(),
                    &lm_head,
                )
            };
            if status != 0 {
                bail!("Qwen3 CUDA decode preparation failed: {}", last_error());
            }
            Ok(decoder)
        }

        fn layer_params(&self) -> Vec<LayerParams> {
            self.model
                .layers
                .iter()
                .map(|layer| LayerParams {
                    input_norm: layer.input_norm.weight.data.as_ptr(),
                    post_attention_norm: layer.post_attention_norm.weight.data.as_ptr(),
                    q_weight: CudaWeight::new(
                        &layer.q_proj.tensor.data,
                        layer.q_proj.q8_0.as_ref(),
                    ),
                    q_norm: layer.q_norm.weight.data.as_ptr(),
                    k_weight: CudaWeight::new(
                        &layer.k_proj.tensor.data,
                        layer.k_proj.q8_0.as_ref(),
                    ),
                    k_norm: layer.k_norm.weight.data.as_ptr(),
                    v_weight: CudaWeight::new(
                        &layer.v_proj.tensor.data,
                        layer.v_proj.q8_0.as_ref(),
                    ),
                    o_weight: CudaWeight::new(
                        &layer.o_proj.tensor.data,
                        layer.o_proj.q8_0.as_ref(),
                    ),
                    gate_weight: CudaWeight::new(
                        &layer.gate_proj.tensor.data,
                        layer.gate_proj.q8_0.as_ref(),
                    ),
                    up_weight: CudaWeight::new(
                        &layer.up_proj.tensor.data,
                        layer.up_proj.q8_0.as_ref(),
                    ),
                    down_weight: CudaWeight::new(
                        &layer.down_proj.tensor.data,
                        layer.down_proj.q8_0.as_ref(),
                    ),
                })
                .collect()
        }

        fn embedding(&self, token: u32) -> Result<&[f32]> {
            let token = token as usize;
            ensure!(
                token < self.model.config.vocab_size,
                "token id outside Qwen3 vocab"
            );
            let hidden = self.model.config.hidden_size;
            Ok(&self.model.embeddings.data[token * hidden..(token + 1) * hidden])
        }
    }

    impl DecodeKernel for CudaDecoder<'_> {
        type Cache = CudaKvCache;

        fn capacity(&self) -> usize {
            self.bucket
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
            ensure!(!tokens.is_empty(), "decode prompt must not be empty");
            ensure!(
                tokens.len() <= self.bucket,
                "decode prompt exceeds cache bucket"
            );
            let hidden = self.model.config.hidden_size;
            let mut embeddings = Vec::with_capacity(tokens.len() * hidden);
            for &token in tokens {
                embeddings.extend_from_slice(self.embedding(token)?);
            }
            let mut logits = vec![0.0; self.model.config.vocab_size];
            let status = unsafe {
                synapse_cuda_qwen3_decode_prefill(
                    self.raw.as_ptr(),
                    tokens.len() as u64,
                    embeddings.as_ptr(),
                    logits.as_mut_ptr(),
                )
            };
            if status != 0 {
                bail!("Qwen3 CUDA prefill failed: {}", last_error());
            }
            Ok((
                CudaKvCache {
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
            let mut logits = vec![0.0; self.model.config.vocab_size];
            let status = unsafe {
                synapse_cuda_qwen3_decode_step(
                    self.raw.as_ptr(),
                    cache.position as u64,
                    self.embedding(token)?.as_ptr(),
                    logits.as_mut_ptr(),
                )
            };
            if status != 0 {
                bail!("Qwen3 CUDA decode step failed: {}", last_error());
            }
            cache.position += 1;
            Ok(logits)
        }

        fn cache_position(&self, cache: &Self::Cache) -> usize {
            cache.position
        }

        fn inspect_cache_layer(&self, _cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
            ensure!(
                layer < self.model.layers.len(),
                "KV cache layer out of range"
            );
            let elements = 2
                * self.bucket
                * self.model.config.num_key_value_heads
                * self.model.config.head_dim;
            let mut values = vec![0.0; elements];
            let status = unsafe {
                synapse_cuda_qwen3_decode_cache_copy(
                    self.raw.as_ptr(),
                    layer as u64,
                    values.as_mut_ptr(),
                    elements as u64,
                )
            };
            if status != 0 {
                bail!("Qwen3 CUDA cache inspection failed: {}", last_error());
            }
            Ok(values)
        }
    }

    impl DecodeRuntime for CudaDecoder<'_> {
        fn lane(&self) -> &'static str {
            "owned-rt-cuda"
        }

        fn kv_update_path(&self) -> &'static str {
            "cuda-resident-fp32-kv-cache"
        }

        fn weight_feed_path(&self) -> &'static str {
            match self.model.weight_quantization {
                WeightQuantization::None => "cuda-persistent-fp32-matvec",
                WeightQuantization::Q8_0 => "cuda-persistent-q8_0-fused-dequant-matvec",
            }
        }

        fn optimization_level(&self) -> u8 {
            1
        }
    }

    impl Drop for CudaDecoder<'_> {
        fn drop(&mut self) {
            unsafe { synapse_cuda_qwen3_decode_context_free(self.raw.as_ptr()) }
        }
    }

    fn last_error() -> anyhow::Error {
        unsafe {
            let raw = synapse_cuda_last_error();
            if raw.is_null() {
                anyhow::anyhow!("unknown Qwen3 CUDA decode error")
            } else {
                anyhow::anyhow!(CStr::from_ptr(raw).to_string_lossy().into_owned())
            }
        }
    }

    unsafe extern "C" {
        fn synapse_cuda_qwen3_decode_context_new(capacity: u64) -> *mut c_void;
        fn synapse_cuda_qwen3_decode_context_free(context: *mut c_void);
        fn synapse_cuda_qwen3_decode_prepare(
            context: *mut c_void,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            vocab: u64,
            epsilon: f32,
            rope_theta: f32,
            layers: *const LayerParams,
            final_norm: *const f32,
            lm_head: *const CudaWeight,
        ) -> i32;
        fn synapse_cuda_qwen3_decode_prefill(
            context: *mut c_void,
            sequence: u64,
            embeddings: *const f32,
            logits: *mut f32,
        ) -> i32;
        fn synapse_cuda_qwen3_decode_step(
            context: *mut c_void,
            position: u64,
            embedding: *const f32,
            logits: *mut f32,
        ) -> i32;
        fn synapse_cuda_qwen3_decode_cache_copy(
            context: *mut c_void,
            layer: u64,
            output: *mut f32,
            elements: u64,
        ) -> i32;
        fn synapse_cuda_last_error() -> *const c_char;
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(crate) use cuda::CudaDecoder;
#[cfg(target_os = "macos")]
pub(crate) use metal::MetalDecoder;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::json_constraint::{JsonConstraint, TokenVocabulary};

    #[derive(Default)]
    struct MockKernel {
        capacity: usize,
    }

    impl MockKernel {
        fn logits(tokens: &[u32]) -> Vec<f32> {
            let next = tokens.iter().fold(7u32, |state, token| {
                state.wrapping_mul(31).wrapping_add(*token)
            }) % 11;
            (0..11)
                .map(|token| if token == next { 10.0 } else { -(token as f32) })
                .collect()
        }
    }

    impl DecodeKernel for MockKernel {
        type Cache = Vec<u32>;

        fn capacity(&self) -> usize {
            self.capacity
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
            Ok((tokens.to_vec(), Self::logits(tokens)))
        }

        fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>> {
            cache.push(token);
            Ok(Self::logits(cache))
        }

        fn cache_position(&self, cache: &Self::Cache) -> usize {
            cache.len()
        }

        fn inspect_cache_layer(&self, cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
            ensure!(layer == 0, "mock has one cache layer");
            Ok(cache.iter().map(|token| *token as f32).collect())
        }
    }

    fn no_stops() -> HashSet<u32> {
        HashSet::new()
    }

    /// A MockKernel that also exposes a chained multi-token path, modelling the
    /// real Metal backend's contract: `advance_chain` runs `steps` deterministic
    /// advances and returns each step's greedy argmax (highest logit, lowest id
    /// on ties). It lets the chained==per-token invariant be proved with no GPU.
    struct ChainedMockKernel {
        inner: MockKernel,
        span: usize,
    }

    impl DecodeKernel for ChainedMockKernel {
        type Cache = Vec<u32>;

        fn capacity(&self) -> usize {
            self.inner.capacity()
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
            self.inner.prefill(tokens)
        }

        fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>> {
            self.inner.advance(cache, token)
        }

        fn cache_position(&self, cache: &Self::Cache) -> usize {
            self.inner.cache_position(cache)
        }

        fn inspect_cache_layer(&self, cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
            self.inner.inspect_cache_layer(cache, layer)
        }

        fn chain_span(&self) -> usize {
            self.span
        }

        fn advance_chain(
            &mut self,
            cache: &mut Self::Cache,
            seed: u32,
            steps: usize,
        ) -> Result<Vec<u32>> {
            // Mirror the device chain: feed the seed, argmax its logits into the
            // first follower, feed that, and so on. The last follower's logits
            // are not consumed here (the session refreshes them per span).
            let mut followers = Vec::with_capacity(steps);
            let mut current = seed;
            for _ in 0..steps {
                let logits = self.inner.advance(cache, current)?;
                let token = top_logits(&logits, 1)[0].token_id;
                followers.push(token);
                current = token;
            }
            Ok(followers)
        }
    }

    #[test]
    fn chained_generation_matches_per_token_generation() {
        // The chained path must produce the exact token stream the fully
        // instrumented per-token path produces; only the submission shape
        // differs. Proven here across several spans and a stop token, so a
        // regression that changes tokens under chaining fails in CI without a
        // GPU. This is the machine-independent half of the chained-decode
        // exactness check; the Metal backend adds the same check on real logits.
        for span in [2usize, 3, 5, 8] {
            let mut per_token_kernel = MockKernel { capacity: 128 };
            let mut per_token = DecodeSession::prefill(&mut per_token_kernel, &[3, 1, 4]).unwrap();
            let expected = per_token
                .generate(40, &no_stops(), 1, &mut |_: TokenTapEvent<'_>| {})
                .unwrap();

            let mut chained_kernel = ChainedMockKernel {
                inner: MockKernel { capacity: 128 },
                span,
            };
            let mut chained = DecodeSession::prefill(&mut chained_kernel, &[3, 1, 4]).unwrap();
            let actual = chained
                .generate_chained(40, &no_stops(), &mut |_: TokenTapEvent<'_>| {})
                .unwrap();
            assert_eq!(actual, expected, "chained span {span} diverged from per-token");
        }
    }

    #[test]
    fn chained_generation_truncates_at_stop_token_like_per_token() {
        // A stop token reached mid-span must truncate the returned stream at the
        // stop, exactly as per-token generation does, even though the device may
        // produce up to span-1 tokens past it within the fused submission.
        let stops = HashSet::from([0u32]);
        let mut per_token_kernel = MockKernel { capacity: 128 };
        let mut per_token = DecodeSession::prefill(&mut per_token_kernel, &[2, 2]).unwrap();
        let expected = per_token
            .generate(40, &stops, 1, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();

        let mut chained_kernel = ChainedMockKernel {
            inner: MockKernel { capacity: 128 },
            span: 8,
        };
        let mut chained = DecodeSession::prefill(&mut chained_kernel, &[2, 2]).unwrap();
        let actual = chained
            .generate_chained(40, &stops, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();
        assert_eq!(actual, expected);
        assert!(
            expected.last().is_none_or(|last| stops.contains(last) || expected.len() == 40),
            "test fixture should exercise a stop-token truncation"
        );
    }

    struct ConstraintKernel;

    impl DecodeKernel for ConstraintKernel {
        type Cache = usize;

        fn capacity(&self) -> usize {
            8
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
            Ok((tokens.len(), vec![100.0, 10.0, 1.0]))
        }

        fn advance(&mut self, cache: &mut Self::Cache, _token: u32) -> Result<Vec<f32>> {
            *cache += 1;
            Ok(vec![100.0, 10.0, 1.0])
        }

        fn cache_position(&self, cache: &Self::Cache) -> usize {
            *cache
        }

        fn inspect_cache_layer(&self, _cache: &Self::Cache, _layer: usize) -> Result<Vec<f32>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn constraint_masks_invalid_logits_before_the_token_tap() {
        let vocabulary = Arc::new(TokenVocabulary::from_pieces(vec![
            Some(b"prose".to_vec()),
            Some(b"{}".to_vec()),
            None,
        ]));
        let stops = HashSet::from([2]);
        let mut constraint = JsonConstraint::new(vocabulary, None, &stops);
        let mut kernel = ConstraintKernel;
        let mut session = DecodeSession::prefill(&mut kernel, &[0]).unwrap();
        let mut tapped = Vec::new();
        let generated = session
            .generate_constrained(
                4,
                &stops,
                3,
                &mut constraint,
                &mut |event: TokenTapEvent<'_>| tapped.push(event.token_id),
            )
            .unwrap();
        assert_eq!(generated, vec![1, 2]);
        assert_eq!(tapped, generated);
    }

    #[test]
    fn token_stream_tap_observes_before_commit_without_changing_tokens() {
        let mut plain_kernel = MockKernel { capacity: 64 };
        let mut plain = DecodeSession::prefill(&mut plain_kernel, &[1, 2, 3]).unwrap();
        let expected = plain
            .generate(12, &no_stops(), 5, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();

        let mut tapped_kernel = MockKernel { capacity: 64 };
        let mut tapped = DecodeSession::prefill(&mut tapped_kernel, &[1, 2, 3]).unwrap();
        let mut events = Vec::new();
        let actual = tapped
            .generate(12, &no_stops(), 5, &mut |event: TokenTapEvent<'_>| {
                events.push((event.step, event.token_id, event.top_logits.to_vec()));
            })
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(events.len(), 12);
        assert!(events
            .iter()
            .all(|(_, token, top)| top[0].token_id == *token));
    }

    #[test]
    fn paused_state_resumes_to_uninterrupted_tokens() {
        let mut full_kernel = MockKernel { capacity: 64 };
        let mut full = DecodeSession::prefill(&mut full_kernel, &[4, 5]).unwrap();
        let expected = full
            .generate(16, &no_stops(), 3, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();

        let mut paused_kernel = MockKernel { capacity: 64 };
        let mut paused = DecodeSession::prefill(&mut paused_kernel, &[4, 5]).unwrap();
        paused
            .generate(6, &no_stops(), 3, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();
        assert_eq!(paused.position(), 8);
        assert_eq!(paused.inspect_cache_layer(0).unwrap().len(), 8);
        paused
            .generate(10, &no_stops(), 3, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();
        assert_eq!(paused.generated(), expected);
    }

    #[test]
    fn splice_matches_prefilling_the_concatenated_sequence() {
        let mut spliced_kernel = MockKernel { capacity: 64 };
        let mut spliced = DecodeSession::prefill(&mut spliced_kernel, &[2, 7]).unwrap();
        spliced
            .generate(3, &no_stops(), 2, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();
        spliced.splice(&[9, 8, 7]).unwrap();
        let concatenated = spliced.sequence().to_vec();
        let actual = spliced
            .generate(10, &no_stops(), 2, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();

        let mut fresh_kernel = MockKernel { capacity: 64 };
        let mut fresh = DecodeSession::prefill(&mut fresh_kernel, &concatenated).unwrap();
        let expected = fresh
            .generate(10, &no_stops(), 2, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            spliced.inspect_cache_layer(0).unwrap(),
            fresh.inspect_cache_layer(0).unwrap()
        );
    }

    #[test]
    fn greedy_argmax_uses_lowest_token_id_for_exact_ties() {
        let top = top_logits(&[1.0, 3.0, 3.0, 2.0], 3);
        assert_eq!(
            top.iter().map(|entry| entry.token_id).collect::<Vec<_>>(),
            [1, 2, 3]
        );
    }

    fn tiny_qwen_snapshot() -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "synapse-qwen3-weight-regions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            r#"{
                "model_type":"qwen3",
                "architectures":["Qwen3ForCausalLM"],
                "hidden_size":32,
                "intermediate_size":32,
                "num_attention_heads":2,
                "num_hidden_layers":1,
                "num_key_value_heads":1,
                "head_dim":16,
                "rms_norm_eps":0.000001,
                "rope_theta":1000000.0,
                "vocab_size":8,
                "tie_word_embeddings":true,
                "eos_token_id":7
            }"#,
        )
        .unwrap();
        let shapes = [
            ("embed_tokens.weight", vec![8, 32]),
            ("layers.0.input_layernorm.weight", vec![32]),
            ("layers.0.post_attention_layernorm.weight", vec![32]),
            ("layers.0.self_attn.q_proj.weight", vec![32, 32]),
            ("layers.0.self_attn.q_norm.weight", vec![16]),
            ("layers.0.self_attn.k_proj.weight", vec![16, 32]),
            ("layers.0.self_attn.k_norm.weight", vec![16]),
            ("layers.0.self_attn.v_proj.weight", vec![16, 32]),
            ("layers.0.self_attn.o_proj.weight", vec![32, 32]),
            ("layers.0.mlp.gate_proj.weight", vec![32, 32]),
            ("layers.0.mlp.up_proj.weight", vec![32, 32]),
            ("layers.0.mlp.down_proj.weight", vec![32, 32]),
            ("norm.weight", vec![32]),
        ];
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (tensor_index, (name, shape)) in shapes.into_iter().enumerate() {
            let start = payload.len();
            let elements = shape.iter().product::<usize>();
            for element in 0..elements {
                let value = (tensor_index * 100 + element + 1) as f32 / 100.0;
                payload.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let mut encoded_header = serde_json::to_vec(&header).unwrap();
        while encoded_header.len() % 8 != 0 {
            encoded_header.push(b' ');
        }
        let mut safetensors = (encoded_header.len() as u64).to_le_bytes().to_vec();
        safetensors.extend_from_slice(&encoded_header);
        safetensors.extend_from_slice(&payload);
        std::fs::write(root.join("model.safetensors"), safetensors).unwrap();
        root
    }

    #[test]
    fn addressable_weight_regions_are_byte_identical_across_loads() {
        let path = tiny_qwen_snapshot();
        let first = Model::load(&path, crate::Precision::F16).unwrap();
        let second = Model::load(&path, crate::Precision::F16).unwrap();
        let first_regions = first.weight_regions();
        let second_regions = second.weight_regions();
        assert_eq!(
            first_regions.keys().collect::<Vec<_>>(),
            second_regions.keys().collect::<Vec<_>>()
        );
        for (key, first_region) in &first_regions {
            let second_region = &second_regions[key];
            assert_ne!(first_region.buffer_handle, 0);
            assert_ne!(second_region.buffer_handle, 0);
            assert_eq!(first_region.byte_len, second_region.byte_len);
            assert_eq!(first_region.checksum, second_region.checksum);
            assert_eq!(
                first.weight_region_bytes(key),
                second.weight_region_bytes(key)
            );
        }
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn quantized_regions_address_complete_q8_0_blocks() {
        let path = tiny_qwen_snapshot();
        let baseline = Model::load(&path, crate::Precision::F32).unwrap();
        let model = Model::load_with_quant(
            &path,
            crate::Precision::F32,
            crate::quant::WeightQuantization::Q8_0,
        )
        .unwrap();
        assert_eq!(
            baseline.layers[0].q_proj.tensor.data,
            model.layers[0].q_proj.tensor.data
        );
        let key = WeightRegionKey {
            layer: Some(0),
            tensor_name: "self_attn.q_proj.weight",
        };
        let region = &model.weight_regions()[&key];
        assert_eq!(region.byte_len, 32 * 34);
        assert_eq!(region.byte_len % 34, 0);
        assert_eq!(
            model.weight_region_bytes(&key).unwrap().len(),
            region.byte_len
        );
        assert_eq!(model.quantized_weight_sha256().unwrap().len(), 64);
        let lm_head = WeightRegionKey {
            layer: None,
            tensor_name: "lm_head.weight",
        };
        assert_eq!(model.weight_regions()[&lm_head].byte_len, 8 * 34);
        std::fs::remove_dir_all(path).unwrap();
    }
}
