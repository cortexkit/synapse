//! Shared decode infrastructure ported from the spike decode controller.
//!
//! The `DecodeKernel` trait and `top_logits` selector are backend-agnostic so
//! the production owned-decode engines (Qwen3 Metal step, LFM2 hybrid step) and
//! any future CPU reference share one greedy-selection contract. The spike
//! tree's tap/pause/splice hooks are deliberately NOT ported: they are dormant
//! spike-only capabilities with no production wire contract, and the spec's
//! non-goals confirm they are not part of this epic.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TopLogit {
    pub token_id: u32,
    pub logit: f32,
}

/// Backend-agnostic greedy decode contract. The production Metal step engines
/// implement this trait; the module-owned decode controller drives them through
/// it so the greedy-selection and stop-token logic is single-sourced.
pub trait DecodeKernel {
    type Cache;

    fn capacity(&self) -> usize;

    /// Device-resident causal prefill of `tokens`. Returns the advanced cache
    /// and the greedy argmax token id after the final prompt token (the first
    /// generated token). The step engines compute this argmax on device and
    /// expose no host-visible logits vector for it, so the contract returns
    /// the token id itself rather than a logits vector a caller could mistake
    /// for real logits. Callers that need per-token logits after prefill use
    /// `advance`; callers that only need greedy tokens use `advance_chain`.
    fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, u32)>;

    /// Feed `token` and return the full f32 logits for the token after it.
    /// Backends whose single-token path is implemented correctly return real
    /// logits; a backend without a correct single-token logits path must not
    /// return placeholder values here (see the LFM2 engine's host embedding
    /// gather for the production solution).
    fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>>;
    fn cache_position(&self, cache: &Self::Cache) -> usize;
    fn inspect_cache_layer(&self, cache: &Self::Cache, layer: usize) -> Result<Vec<f32>>;

    /// Runs the draft tokens through the verifier and returns the greedy argmax
    /// after each token. The first proposal is compared with the session's
    /// pending logits; element `i - 1` verifies proposal `i`. The default
    /// implementation loops over `advance`; engines with a device-resident
    /// verify path override it.
    fn verify_tokens(&mut self, cache: &mut Self::Cache, tokens: &[u32]) -> Result<Vec<u32>> {
        ensure!(
            !tokens.is_empty(),
            "verification requires at least one token"
        );
        let mut argmaxes = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let logits = self.advance(cache, token)?;
            let next = top_logits(&logits, 1)
                .first()
                .context("verifier produced empty logits")?
                .token_id;
            argmaxes.push(next);
        }
        Ok(argmaxes)
    }

    /// Restores the logical cache length after speculative verification. A
    /// backend must guarantee that reads at later positions are excluded after
    /// this call; overwritten in-slot data may remain physically allocated.
    fn rewind(&mut self, _cache: &mut Self::Cache, _position: usize) -> Result<()> {
        anyhow::bail!("this decode backend cannot rewind a speculative verification")
    }

    /// The GPU-chained decode span, or 1 when the backend has no chained path.
    /// A backend returning > 1 must make `advance_chain` produce exactly the
    /// same tokens as the same number of per-token `advance`/argmax steps.
    /// Production baseline is K=1 (chain_span=1); chaining is opt-in.
    fn chain_span(&self) -> usize {
        1
    }

    /// Advance `steps` tokens in one fused submission, returning the argmax
    /// token id of every step. `seed` feeds the first step. Backends without a
    /// chained path (the default) must not be asked for this.
    fn advance_chain(
        &mut self,
        _cache: &mut Self::Cache,
        _seed: u32,
        _steps: usize,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("this decode backend has no chained multi-token path")
    }
}

pub trait DecodeRuntime: DecodeKernel {
    fn lane(&self) -> &'static str;
    fn kv_update_path(&self) -> &'static str;
    fn weight_feed_path(&self) -> &'static str;
    fn optimization_level(&self) -> u8;
}

/// Greedy top-1 selection: highest logit wins, lowest token id breaks ties.
/// This is the exact selector the spike engines and their pinned fixtures use;
/// porting it byte-for-byte preserves the token-exactness contract.
pub fn top_logits(logits: &[f32], top_k: usize) -> Vec<TopLogit> {
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

fn logit_precedes(candidate: &TopLogit, current: &TopLogit) -> bool {
    candidate.logit.total_cmp(&current.logit).is_gt()
        || (candidate.logit.total_cmp(&current.logit).is_eq()
            && candidate.token_id < current.token_id)
}
