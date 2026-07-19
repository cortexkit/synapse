//! Soft, per-request vocabulary bias for greedy ASR decoding.
//!
//! The trie is intentionally advisory: candidates receive a finite logit bonus
//! instead of a mask, so a supplied vocabulary cannot make ordinary speech
//! impossible to emit.

use std::collections::{BTreeSet, HashMap};

use anyhow::{ensure, Result};
use tokenizers::Tokenizer;

#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<u32, usize>,
}

/// Adds a fixed bonus to vocabulary-token continuations during greedy decoding.
///
/// A root path is always eligible so a term can begin at the next generated
/// token. Paths matching suffixes of recently committed output remain eligible
/// for `window` tokens, which lets multi-token terms finish without imposing a
/// hard constraint on the rest of the transcript.
#[derive(Debug)]
pub(crate) struct SoftTrieBias {
    nodes: Vec<TrieNode>,
    committed: Vec<u32>,
    window: usize,
    delta: f32,
}

impl SoftTrieBias {
    pub(crate) fn new(
        tokenizer: &Tokenizer,
        terms: &[String],
        delta: f32,
        window: usize,
    ) -> Result<Self> {
        ensure!(
            delta.is_finite() && delta >= 0.0,
            "ASR trie delta must be finite and non-negative"
        );
        ensure!(window > 0, "ASR trie window must be positive");

        let sequences = term_token_sequences(tokenizer, terms)?;
        ensure!(
            !sequences.is_empty(),
            "ASR trie bias needs at least one tokenizable vocabulary term"
        );
        Ok(Self::from_sequences(sequences, delta, window))
    }

    fn from_sequences(sequences: Vec<Vec<u32>>, delta: f32, window: usize) -> Self {
        let mut nodes = vec![TrieNode::default()];
        for sequence in sequences {
            let mut node = 0;
            for token in sequence {
                let next = if let Some(&next) = nodes[node].children.get(&token) {
                    next
                } else {
                    let next = nodes.len();
                    nodes.push(TrieNode::default());
                    nodes[node].children.insert(token, next);
                    next
                };
                node = next;
            }
        }
        Self {
            nodes,
            committed: Vec::new(),
            window,
            delta,
        }
    }

    /// Applies this step's vocabulary bonus before the token is committed.
    pub(crate) fn apply(&self, logits: &mut [f32]) {
        if self.delta == 0.0 {
            return;
        }
        for token in self.candidate_tokens() {
            if let Some(logit) = logits.get_mut(token as usize) {
                *logit += self.delta;
            }
        }
    }

    /// Records the selected token so matching continuations are eligible next step.
    pub(crate) fn commit(&mut self, token: u32) {
        self.committed.push(token);
    }

    fn candidate_tokens(&self) -> BTreeSet<u32> {
        // The root represents a term beginning at the next token. Every suffix
        // beginning in the configured recent window represents an active term
        // that may need one more token to complete.
        let mut active = BTreeSet::from([0usize]);
        let start = self.committed.len().saturating_sub(self.window);
        for suffix_start in start..self.committed.len() {
            let mut node = 0;
            let mut matches_prefix = true;
            for &token in &self.committed[suffix_start..] {
                let Some(&next) = self.nodes[node].children.get(&token) else {
                    matches_prefix = false;
                    break;
                };
                node = next;
            }
            if matches_prefix {
                active.insert(node);
            }
        }
        active
            .into_iter()
            .flat_map(|node| self.nodes[node].children.keys().copied())
            .collect()
    }
}

/// Returns every useful tokenizer spelling for the supplied terms.
///
/// A transcript term may begin after normal text or at the beginning of a
/// sequence, so each casing is encoded both bare and with a leading space.
fn term_token_sequences(tokenizer: &Tokenizer, terms: &[String]) -> Result<Vec<Vec<u32>>> {
    let mut forms = BTreeSet::new();
    for term in terms {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        forms.insert(term.to_owned());
        forms.insert(term.to_lowercase());
        forms.insert(term.to_uppercase());
        forms.insert(capitalize(&term.to_lowercase()));
    }

    let mut sequences = BTreeSet::new();
    for form in forms {
        for text in [form.clone(), format!(" {form}")] {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|error| anyhow::anyhow!("tokenize ASR bias term {form:?}: {error}"))?;
            if !encoding.get_ids().is_empty() {
                sequences.insert(encoding.get_ids().to_vec());
            }
        }
    }
    Ok(sequences.into_iter().collect())
}

fn capitalize(value: &str) -> String {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return String::new();
    };
    first.to_uppercase().chain(characters).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trie_boosts_starts_and_recent_continuations_only() {
        let mut bias = SoftTrieBias::from_sequences(vec![vec![1, 2], vec![1, 3], vec![5]], 4.0, 2);
        assert_eq!(bias.candidate_tokens(), BTreeSet::from([1, 5]));

        bias.commit(1);
        assert_eq!(bias.candidate_tokens(), BTreeSet::from([1, 2, 3, 5]));

        bias.commit(9);
        assert_eq!(bias.candidate_tokens(), BTreeSet::from([1, 5]));
    }

    #[test]
    fn trie_bonus_is_additive_and_leaves_other_logits_untouched() {
        let bias = SoftTrieBias::from_sequences(vec![vec![1]], 2.0, 4);
        let mut logits = vec![0.0, 3.0, -1.0];
        bias.apply(&mut logits);
        assert_eq!(logits, vec![0.0, 5.0, -1.0]);
    }

    #[test]
    fn capitalize_handles_empty_and_unicode_inputs() {
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("mpsgraph"), "Mpsgraph");
        assert_eq!(capitalize("éclair"), "Éclair");
    }
}
