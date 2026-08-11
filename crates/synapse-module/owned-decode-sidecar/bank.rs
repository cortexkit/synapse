//! Target-tokenized sidecar hint-bank construction and suffix lookup.

use std::fmt::Display;

use synapse_core::{SanitizedTokenizer, SidecarHintBank};
use thiserror::Error;

use crate::client::PreparedSidecarResult;

/// The bounded suffix window fixed for sidecar pickup.
pub const MAX_SUFFIX_MATCH_TOKENS: usize = 7;
/// The maximum number of target tokens proposed by one bank lookup.
pub const MAX_HINT_PROPOSAL_TOKENS: usize = 16;

/// Deterministic work bounds for target-tokenized bank construction.
///
/// These bounds limit which rendered views may be admitted. They do not alter
/// rendering, tokenization, ordering, or the content digest; completion timing is
/// supplied separately as observation metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SidecarWorkBounds {
    pub max_views: usize,
    pub max_rendered_bytes_per_view: usize,
    pub max_tokens_per_view: usize,
}

impl Default for SidecarWorkBounds {
    fn default() -> Self {
        Self {
            max_views: 1,
            max_rendered_bytes_per_view: 64 * 1024,
            max_tokens_per_view: 4 * 1024,
        }
    }
}

/// The minimal target tokenizer seam needed by sidecar banking.
pub trait TargetTokenizer {
    type Error: Display;

    /// Tokenize one complete runtime-rendered target-lane view without inserting
    /// special tokens or view separators.
    fn tokenize_target_view(&self, view: &str) -> Result<Vec<u32>, Self::Error>;
}

impl TargetTokenizer for SanitizedTokenizer {
    type Error = String;

    fn tokenize_target_view(&self, view: &str) -> Result<Vec<u32>, Self::Error> {
        self.tokenizer()
            .encode(view, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }
}

/// Build a request-scoped hint bank from separately rendered views.
///
/// `built_at` is an external timing observation. For identical rendered views,
/// schema identity, and render-policy digest, changing it does not change the
/// content digest, provided the bounds admit the views.
pub fn build_hint_bank<T: TargetTokenizer>(
    tokenizer: &T,
    schema_identity: impl Into<String>,
    render_policy_digest: impl Into<String>,
    rendered_views: &[String],
    bounds: SidecarWorkBounds,
    built_at: u64,
) -> Result<SidecarHintBank, HintBankError> {
    validate_bounds(bounds)?;
    if rendered_views.is_empty() {
        return Err(HintBankError::NoViews);
    }
    if rendered_views.len() > bounds.max_views {
        return Err(HintBankError::ViewLimitExceeded {
            actual: rendered_views.len(),
            limit: bounds.max_views,
        });
    }

    let mut views = Vec::with_capacity(rendered_views.len());
    for (view_index, view) in rendered_views.iter().enumerate() {
        if view.len() > bounds.max_rendered_bytes_per_view {
            return Err(HintBankError::RenderedByteLimitExceeded {
                view_index,
                actual: view.len(),
                limit: bounds.max_rendered_bytes_per_view,
            });
        }
        let ids =
            tokenizer
                .tokenize_target_view(view)
                .map_err(|error| HintBankError::Tokenization {
                    view_index,
                    message: error.to_string(),
                })?;
        if ids.is_empty() {
            return Err(HintBankError::EmptyTokenizedView { view_index });
        }
        if ids.len() > bounds.max_tokens_per_view {
            return Err(HintBankError::TokenLimitExceeded {
                view_index,
                actual: ids.len(),
                limit: bounds.max_tokens_per_view,
            });
        }
        views.push(ids);
    }

    Ok(SidecarHintBank {
        views,
        schema_identity: schema_identity.into(),
        render_policy_digest: render_policy_digest.into(),
        built_at,
    })
}

/// Build the default one-view bank directly from a normalized result.
pub fn build_default_hint_bank<T: TargetTokenizer>(
    tokenizer: &T,
    schema_identity: impl Into<String>,
    render_policy_digest: impl Into<String>,
    prepared: &PreparedSidecarResult,
    bounds: SidecarWorkBounds,
    built_at: u64,
) -> Result<SidecarHintBank, HintBankError> {
    build_hint_bank(
        tokenizer,
        schema_identity,
        render_policy_digest,
        std::slice::from_ref(&prepared.rendered_view),
        bounds,
        built_at,
    )
}

/// A continuation selected from one view of a bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintContinuation {
    pub view_index: usize,
    /// Offset of the first proposed token in its view.
    pub bank_offset: usize,
    pub matched_suffix_len: usize,
    pub tokens: Vec<u32>,
}

/// Find the deterministic next continuation for the committed generated prefix.
///
/// Try suffix lengths from at most seven down to one. Equal matches retain the
/// first candidate, ordered by lowest view index then lowest bank offset. Each
/// view is searched independently, so matches never cross view boundaries or
/// insert separator tokens.
#[must_use]
pub fn find_hint_continuation(
    bank: &SidecarHintBank,
    committed: &[u32],
    context_remaining: usize,
    output_remaining: usize,
    proposal_limit: usize,
) -> Option<HintContinuation> {
    let limit = proposal_limit
        .min(MAX_HINT_PROPOSAL_TOKENS)
        .min(context_remaining)
        .min(output_remaining);
    if limit == 0 {
        return None;
    }

    if committed.is_empty() {
        return bank
            .views
            .iter()
            .enumerate()
            .find_map(|(view_index, view)| {
                let tokens = view.iter().take(limit).copied().collect::<Vec<_>>();
                (!tokens.is_empty()).then_some(HintContinuation {
                    view_index,
                    bank_offset: 0,
                    matched_suffix_len: 0,
                    tokens,
                })
            });
    }

    let maximum_suffix = committed.len().min(MAX_SUFFIX_MATCH_TOKENS);
    let mut best: Option<HintContinuation> = None;
    for (view_index, view) in bank.views.iter().enumerate() {
        for bank_offset in 1..view.len() {
            let preceding = &view[..bank_offset];
            let suffix_len = maximum_suffix.min(preceding.len());
            let matched_suffix_len = (1..=suffix_len).rev().find(|&len| {
                preceding[preceding.len() - len..] == committed[committed.len() - len..]
            });
            let Some(matched_suffix_len) = matched_suffix_len else {
                continue;
            };
            let tokens = view[bank_offset..]
                .iter()
                .take(limit)
                .copied()
                .collect::<Vec<_>>();
            if tokens.is_empty() {
                continue;
            }
            let candidate = HintContinuation {
                view_index,
                bank_offset,
                matched_suffix_len,
                tokens,
            };
            if best
                .as_ref()
                .is_none_or(|current| candidate.matched_suffix_len > current.matched_suffix_len)
            {
                best = Some(candidate);
            }
        }
    }
    best
}

/// Hint-bank construction failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HintBankError {
    #[error("sidecar bank needs at least one rendered view")]
    NoViews,
    #[error("sidecar work bounds must all be greater than zero")]
    ZeroWorkBound,
    #[error("sidecar produced {actual} views, exceeding the {limit} view limit")]
    ViewLimitExceeded { actual: usize, limit: usize },
    #[error("rendered view {view_index} has {actual} bytes, exceeding the {limit} byte limit")]
    RenderedByteLimitExceeded {
        view_index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("target tokenization failed for view {view_index}: {message}")]
    Tokenization { view_index: usize, message: String },
    #[error("target tokenization produced no tokens for view {view_index}")]
    EmptyTokenizedView { view_index: usize },
    #[error("target tokenization produced {actual} tokens for view {view_index}, exceeding the {limit} token limit")]
    TokenLimitExceeded {
        view_index: usize,
        actual: usize,
        limit: usize,
    },
}

fn validate_bounds(bounds: SidecarWorkBounds) -> Result<(), HintBankError> {
    if bounds.max_views == 0
        || bounds.max_rendered_bytes_per_view == 0
        || bounds.max_tokens_per_view == 0
    {
        return Err(HintBankError::ZeroWorkBound);
    }
    Ok(())
}
