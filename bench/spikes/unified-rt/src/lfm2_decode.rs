//! Cached LFM2 decoding over the shared instrumentable decode controller.

use std::collections::HashSet;

use anyhow::{ensure, Context, Result};

use crate::json_constraint::DecodeConstraint;
use crate::lfm2::{DecodeCache, LayerCache, Model};
use crate::qwen3_decode::{top_logits, top_logits_masked, DecodeKernel};
use crate::KernelProvider;

pub(crate) struct Decoder<'model, 'provider> {
    model: &'model Model,
    provider: &'provider mut dyn KernelProvider,
    capacity: usize,
}

impl<'model, 'provider> Decoder<'model, 'provider> {
    pub(crate) fn new(
        model: &'model Model,
        provider: &'provider mut dyn KernelProvider,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(capacity > 0, "LFM2 decode cache capacity must be positive");
        Ok(Self {
            model,
            provider,
            capacity,
        })
    }

    pub(crate) fn full_hidden(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.model.forward_hidden(self.provider, tokens)
    }

    pub(crate) fn full_reprefill_tokens(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
    ) -> Result<Vec<u32>> {
        self.full_reprefill_tokens_inner(prompt, max_tokens, stop_tokens, None)
    }

    pub(crate) fn full_reprefill_tokens_constrained(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        constraint: &mut dyn DecodeConstraint,
    ) -> Result<Vec<u32>> {
        self.full_reprefill_tokens_inner(prompt, max_tokens, stop_tokens, Some(constraint))
    }

    fn full_reprefill_tokens_inner(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        mut constraint: Option<&mut dyn DecodeConstraint>,
    ) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "decode prompt must not be empty");
        ensure!(
            prompt.len() + max_tokens <= self.capacity,
            "full-reprefill decode exceeds cache capacity"
        );
        let mut sequence = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let logits = self.model.forward_logits(self.provider, &sequence)?;
            let token = if let Some(constraint) = constraint.as_deref_mut() {
                let mask = constraint.allowed()?;
                top_logits_masked(&logits, &mask, 1)[0].token_id
            } else {
                top_logits(&logits, 1)[0].token_id
            };
            if let Some(constraint) = constraint.as_deref_mut() {
                constraint.advance(token)?;
            }
            sequence.push(token);
            generated.push(token);
            if stop_tokens.contains(&token) {
                break;
            }
        }
        if let Some(constraint) = constraint {
            ensure!(
                constraint.is_complete(),
                "full-reprefill JSON constraint did not complete: {}",
                constraint.describe()
            );
        }
        Ok(generated)
    }

    #[allow(dead_code)]
    pub(crate) fn weight_count(&self) -> usize {
        self.model.weight_count()
    }

    pub(crate) fn prefill_embeddings(
        &mut self,
        embeddings: &[Vec<f32>],
    ) -> Result<(DecodeCache, Vec<f32>)> {
        ensure!(
            !embeddings.is_empty(),
            "decode prefill embeddings must not be empty"
        );
        ensure!(
            embeddings.len() <= self.capacity,
            "decode prefill embeddings exceed cache capacity"
        );
        if let Some(prefilled) =
            self.model
                .prefill_embeddings(self.provider, embeddings, self.capacity)?
        {
            return Ok(prefilled);
        }
        let mut cache = self.model.empty_decode_cache(self.capacity);
        let mut logits = None;
        for embedding in embeddings {
            let (_, next_logits) =
                self.model
                    .decode_embedding(self.provider, &mut cache, embedding)?;
            logits = Some(next_logits);
        }
        Ok((
            cache,
            logits.context("decode prefill embeddings must not be empty")?,
        ))
    }

    pub(crate) fn advance_token(
        &mut self,
        cache: &mut DecodeCache,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.model
            .decode_token(self.provider, cache, token)
            .map(|(_, logits)| logits)
    }
}

impl DecodeKernel for Decoder<'_, '_> {
    type Cache = DecodeCache;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
        ensure!(!tokens.is_empty(), "decode prompt must not be empty");
        ensure!(
            tokens.len() <= self.capacity,
            "decode prompt exceeds cache capacity"
        );
        let embeddings = tokens
            .iter()
            .map(|&token| self.model.token_embedding(token).map(<[f32]>::to_vec))
            .collect::<Result<Vec<_>>>()?;
        if let Some(prefilled) =
            self.model
                .prefill_embeddings(self.provider, &embeddings, self.capacity)?
        {
            return Ok(prefilled);
        }
        let mut cache = self.model.empty_decode_cache(self.capacity);
        let mut logits = None;
        for &token in tokens {
            let (_, next_logits) = self.model.decode_token(self.provider, &mut cache, token)?;
            logits = Some(next_logits);
        }
        Ok((
            cache,
            logits.context("decode prompt must contain at least one token")?,
        ))
    }

    fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>> {
        self.model
            .decode_token(self.provider, cache, token)
            .map(|(_, logits)| logits)
    }

    fn cache_position(&self, cache: &Self::Cache) -> usize {
        cache.position
    }

    fn inspect_cache_layer(&self, cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
        let layer = cache
            .layers
            .get(layer)
            .with_context(|| format!("LFM2 cache layer {layer} out of range"))?;
        Ok(match layer {
            LayerCache::Conv { state } => state.clone(),
            LayerCache::Attention { keys, values } => {
                let mut combined = Vec::with_capacity(keys.len() + values.len());
                combined.extend_from_slice(keys);
                combined.extend_from_slice(values);
                combined
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm2::tiny_test_model;
    use crate::qwen3_decode::{DecodeSession, TokenTapEvent};
    use crate::{BLayout, KernelProvider};

    struct TestProvider;

    impl KernelProvider for TestProvider {
        fn name(&self) -> &'static str {
            "lfm2-test"
        }

        fn matmul(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            a: &[f32],
            b: &[f32],
            b_layout: BLayout,
            c: &mut [f32],
        ) -> Result<()> {
            for row in 0..m {
                for column in 0..n {
                    c[row * n + column] = (0..k)
                        .map(|inner| {
                            let rhs = match b_layout {
                                BLayout::RowMajorKn => b[inner * n + column],
                                BLayout::RowMajorNkTransposed => b[column * k + inner],
                            };
                            a[row * k + inner] * rhs
                        })
                        .sum();
                }
            }
            Ok(())
        }
    }

    #[test]
    fn cached_decode_matches_full_reprefill_for_hybrid_layers() {
        let model = tiny_test_model();
        let prompt = [1, 3, 2];
        let mut provider = TestProvider;
        let mut decoder = Decoder::new(&model, &mut provider, 32).unwrap();
        let mut session = DecodeSession::prefill(&mut decoder, &prompt).unwrap();
        let cached = session
            .generate(8, &HashSet::new(), 3, &mut |_: TokenTapEvent<'_>| {})
            .unwrap();
        drop(session);
        let reprefilled = decoder
            .full_reprefill_tokens(&prompt, 8, &HashSet::new())
            .unwrap();
        assert_eq!(cached, reprefilled);
    }

    #[test]
    fn incremental_hidden_states_match_full_prefill() {
        let model = tiny_test_model();
        let tokens = [1, 4, 3, 2];
        let mut full_provider = TestProvider;
        let full = model.forward_hidden(&mut full_provider, &tokens).unwrap();

        let mut incremental_provider = TestProvider;
        let mut cache = model.empty_decode_cache(16);
        let mut incremental = Vec::new();
        for token in tokens {
            let (hidden, _) = model
                .decode_token(&mut incremental_provider, &mut cache, token)
                .unwrap();
            incremental.push(hidden);
        }
        for (full, incremental) in full.iter().zip(&incremental) {
            for (full, incremental) in full.iter().zip(incremental) {
                assert!((full - incremental).abs() < 2e-5);
            }
        }
        assert_eq!(cache.position, tokens.len());
        assert_eq!(cache.layers.len(), 2);
    }
}
