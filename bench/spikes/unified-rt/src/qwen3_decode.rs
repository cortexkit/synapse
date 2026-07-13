//! Instrumentable greedy decoding for Qwen3 on the owned Metal runtime.
//!
//! The decode controller is backend-agnostic so pause, tap, and splice semantics
//! can be proved without a model download. `MetalDecoder` supplies the resident
//! Qwen3 implementation and keeps its KV buffers alive across calls.

use std::collections::{BTreeMap, HashSet};

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::{qwen3::Model, Tensor};

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
}

/// A paused session owns all logical state needed to resume generation exactly.
pub(crate) struct DecodeSession<'a, K: DecodeKernel> {
    kernel: &'a mut K,
    cache: K::Cache,
    sequence: Vec<u32>,
    generated: Vec<u32>,
    next_logits: Vec<f32>,
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

    pub(crate) fn generate(
        &mut self,
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        top_k: usize,
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
            let top = top_logits(&self.next_logits, top_k);
            let token = top[0].token_id;
            tap.before_commit(TokenTapEvent {
                step: self.generated.len(),
                token_id: token,
                top_logits: &top,
            });
            self.sequence.push(token);
            self.generated.push(token);
            self.next_logits = self.kernel.advance(&mut self.cache, token)?;
            if stop_tokens.contains(&token) {
                break;
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

pub(crate) fn top_logits(logits: &[f32], top_k: usize) -> Vec<TopLogit> {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(top_k > 0, "top-k must be positive");
    let mut top = Vec::<TopLogit>::with_capacity(top_k.min(logits.len()));
    for (token_id, &logit) in logits.iter().enumerate() {
        let candidate = TopLogit {
            token_id: token_id as u32,
            logit,
        };
        let insertion = top
            .iter()
            .position(|current| {
                candidate.logit.total_cmp(&current.logit).is_gt()
                    || (candidate.logit.total_cmp(&current.logit).is_eq()
                        && candidate.token_id < current.token_id)
            })
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub(crate) struct WeightRegionKey {
    pub(crate) layer: Option<usize>,
    pub(crate) tensor_name: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct WeightRegion {
    /// Stable for the lifetime of one loaded model and used by the Metal buffer cache.
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
            insert_region(&mut regions, None, "lm_head.weight", &lm_head.tensor);
        }
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, tensor) in [
                ("input_layernorm.weight", &layer.input_norm.weight),
                (
                    "post_attention_layernorm.weight",
                    &layer.post_attention_norm.weight,
                ),
                ("self_attn.q_proj.weight", &layer.q_proj.tensor),
                ("self_attn.q_norm.weight", &layer.q_norm.weight),
                ("self_attn.k_proj.weight", &layer.k_proj.tensor),
                ("self_attn.k_norm.weight", &layer.k_norm.weight),
                ("self_attn.v_proj.weight", &layer.v_proj.tensor),
                ("self_attn.o_proj.weight", &layer.o_proj.tensor),
                ("mlp.gate_proj.weight", &layer.gate_proj.tensor),
                ("mlp.up_proj.weight", &layer.up_proj.tensor),
                ("mlp.down_proj.weight", &layer.down_proj.tensor),
            ] {
                insert_region(&mut regions, Some(index), name, tensor);
            }
        }
        regions
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn weight_region_bytes(&self, key: &WeightRegionKey) -> Option<Vec<u8>> {
        let tensor = match (key.layer, key.tensor_name) {
            (None, "embed_tokens.weight") => Some(&self.embeddings),
            (None, "norm.weight") => Some(&self.final_norm.weight),
            (None, "lm_head.weight") => self.lm_head.as_ref().map(|head| &head.tensor),
            (Some(index), name) => self.layers.get(index).and_then(|layer| match name {
                "input_layernorm.weight" => Some(&layer.input_norm.weight),
                "post_attention_layernorm.weight" => Some(&layer.post_attention_norm.weight),
                "self_attn.q_proj.weight" => Some(&layer.q_proj.tensor),
                "self_attn.q_norm.weight" => Some(&layer.q_norm.weight),
                "self_attn.k_proj.weight" => Some(&layer.k_proj.tensor),
                "self_attn.k_norm.weight" => Some(&layer.k_norm.weight),
                "self_attn.v_proj.weight" => Some(&layer.v_proj.tensor),
                "self_attn.o_proj.weight" => Some(&layer.o_proj.tensor),
                "mlp.gate_proj.weight" => Some(&layer.gate_proj.tensor),
                "mlp.up_proj.weight" => Some(&layer.up_proj.tensor),
                "mlp.down_proj.weight" => Some(&layer.down_proj.tensor),
                _ => None,
            }),
            _ => None,
        }?;
        Some(active_tensor_bytes(tensor))
    }
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

    use anyhow::{bail, ensure, Result};

    use super::{DecodeKernel, Model};
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

    pub(crate) struct MetalKvCache {
        position: usize,
    }

    pub(crate) struct MetalDecoder<'a> {
        raw: NonNull<c_void>,
        model: &'a Model,
        bucket: usize,
        prefill_package: Option<CString>,
        step_package: Option<CString>,
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
                "Qwen3 decode requires explicit O0 graph compilation"
            );
            ensure!(
                [512, 1024, 2048].contains(&bucket),
                "decode cache bucket must be 512, 1024, or 2048"
            );
            let raw = unsafe {
                synapse_qwen3_decode_context_new(
                    bucket as u64,
                    model.layers.len() as u64,
                    model.config.num_key_value_heads as u64,
                    model.config.head_dim as u64,
                )
            };
            let prefill_package =
                package_cstring(execution.decode_package_path("prefill", bucket).as_deref())?;
            let step_package =
                package_cstring(execution.decode_package_path("step", bucket).as_deref())?;
            Ok(Self {
                raw: NonNull::new(raw).ok_or_else(last_error)?,
                model,
                bucket,
                prefill_package,
                step_package,
            })
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
    }

    impl DecodeKernel for MetalDecoder<'_> {
        type Cache = MetalKvCache;

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

        fn inspect_cache_layer(&self, _cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
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
                    "Qwen3 Metal cache inspection failed with status {status}: {}",
                    last_error()
                );
            }
            Ok(bits
                .into_iter()
                .map(half::f16::from_bits)
                .map(f32::from)
                .collect())
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
pub(crate) use metal::MetalDecoder;

#[cfg(test)]
mod tests {
    use super::*;

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
                "hidden_size":4,
                "intermediate_size":6,
                "num_attention_heads":2,
                "num_hidden_layers":1,
                "num_key_value_heads":1,
                "head_dim":2,
                "rms_norm_eps":0.000001,
                "rope_theta":1000000.0,
                "vocab_size":8,
                "tie_word_embeddings":true,
                "eos_token_id":7
            }"#,
        )
        .unwrap();
        let shapes = [
            ("embed_tokens.weight", vec![8, 4]),
            ("layers.0.input_layernorm.weight", vec![4]),
            ("layers.0.post_attention_layernorm.weight", vec![4]),
            ("layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("layers.0.self_attn.q_norm.weight", vec![2]),
            ("layers.0.self_attn.k_proj.weight", vec![2, 4]),
            ("layers.0.self_attn.k_norm.weight", vec![2]),
            ("layers.0.self_attn.v_proj.weight", vec![2, 4]),
            ("layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("layers.0.mlp.gate_proj.weight", vec![6, 4]),
            ("layers.0.mlp.up_proj.weight", vec![6, 4]),
            ("layers.0.mlp.down_proj.weight", vec![4, 6]),
            ("norm.weight", vec![4]),
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
}
