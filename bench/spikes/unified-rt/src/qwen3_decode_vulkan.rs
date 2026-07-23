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
