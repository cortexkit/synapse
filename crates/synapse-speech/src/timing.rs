use unicode_casefold::UnicodeCaseFold;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result, Word};

pub const WORD_NORMALIZER_REVISION: &str = "synapse-wer-norm-v1";

/// Unicode compatibility composition (NFKC), full case folding, removal of
/// Unicode punctuation, then whitespace splitting.
/// Normalization is for matching only; returned transcripts keep their spelling.
pub fn normalize_words(text: &str) -> Vec<String> {
    text.nfkc()
        .case_fold()
        .filter(|c| {
            !matches!(
                get_general_category(*c),
                GeneralCategory::ConnectorPunctuation
                    | GeneralCategory::DashPunctuation
                    | GeneralCategory::ClosePunctuation
                    | GeneralCategory::FinalPunctuation
                    | GeneralCategory::InitialPunctuation
                    | GeneralCategory::OpenPunctuation
                    | GeneralCategory::OtherPunctuation
            )
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Align whitespace-split Qwen3-ASR words to Parakeet's timed words. Matches and
/// substitutions retain the companion's times; unaligned runs share the gap.
/// Invalid anchors are refused, never adjusted. Nonempty text with no companion
/// words is an internal-invariant error: serving treats zero Parakeet words as
/// `no_speech` and must never run Qwen3-ASR in that case. Empty text returns no words.
pub fn align_words(text: &str, parakeet: &[Word], duration_ms: f64) -> Result<Vec<Word>> {
    if !duration_ms.is_finite() || duration_ms < 0.0 {
        return Err(Error::InvalidWordTiming("invalid duration"));
    }
    let mut previous_start = 0.0;
    for word in parakeet {
        if !word.start_ms.is_finite()
            || !word.end_ms.is_finite()
            || word.start_ms < previous_start
            || word.start_ms > word.end_ms
            || word.end_ms > duration_ms
        {
            return Err(Error::InvalidWordTiming("invalid anchor"));
        }
        previous_start = word.start_ms;
    }
    let mut words: Vec<_> = text
        .split_whitespace()
        .map(|word| Word {
            word: word.to_owned(),
            start_ms: 0.0,
            end_ms: 0.0,
        })
        .collect();
    if words.is_empty() {
        return Ok(words);
    }
    if parakeet.is_empty() {
        return Err(Error::InvalidWordTiming("missing companion words"));
    }
    let qwen: Vec<_> = words
        .iter()
        .enumerate()
        .filter_map(|(i, word)| {
            let normalized = normalize_words(&word.word).join(" ");
            (!normalized.is_empty()).then_some((i, normalized))
        })
        .collect();
    let companion: Vec<_> = parakeet
        .iter()
        .map(|word| normalize_words(&word.word).join(" "))
        .collect();
    // Suffix costs let traceback move forward. Diagonal ties prefer a match,
    // then a substitution; skipping Qwen before Parakeet preserves the earliest
    // companion position when neither diagonal operation is optimal.
    let cols = companion.len() + 1;
    let rows = qwen.len() + 1;
    let mut costs = vec![0usize; rows * cols];
    for i in (0..rows).rev() {
        for j in (0..cols).rev() {
            costs[i * cols + j] = if i == qwen.len() {
                companion.len() - j
            } else if j == companion.len() {
                qwen.len() - i
            } else {
                (costs[(i + 1) * cols + j + 1] + usize::from(qwen[i].1 != companion[j]))
                    .min(1 + costs[(i + 1) * cols + j])
                    .min(1 + costs[i * cols + j + 1])
            };
        }
    }
    let mut anchors = vec![false; words.len()];
    let (mut i, mut j) = (0, 0);
    while i < qwen.len() && j < companion.len() {
        let cost = costs[i * cols + j];
        if cost == costs[(i + 1) * cols + j + 1] + usize::from(qwen[i].1 != companion[j]) {
            let word = &mut words[qwen[i].0];
            word.start_ms = parakeet[j].start_ms;
            word.end_ms = parakeet[j].end_ms;
            anchors[qwen[i].0] = true;
            i += 1;
            j += 1;
        } else if cost == 1 + costs[(i + 1) * cols + j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    if !anchors.iter().any(|&anchor| anchor) {
        let a = parakeet.first().unwrap().start_ms;
        let b = parakeet.last().unwrap().end_ms;
        share_gap(&mut words, a, b);
        return Ok(words);
    }
    let mut start = 0;
    while start < words.len() {
        if anchors[start] {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < words.len() && !anchors[end] {
            end += 1;
        }
        let (a, b) = match (start.checked_sub(1), words.get(end)) {
            (Some(p), Some(n)) => (
                words[p].start_ms.max(words[p].end_ms.min(n.start_ms)),
                n.start_ms,
            ),
            (None, Some(n)) => (0.0, n.start_ms),
            (Some(p), None) => (words[p].end_ms, duration_ms),
            (None, None) => unreachable!("at least one anchor"),
        };
        share_gap(&mut words[start..end], a, b);
        start = end;
    }
    Ok(words)
}

fn share_gap(words: &mut [Word], a: f64, b: f64) {
    let count = words.len() as f64;
    for (i, word) in words.iter_mut().enumerate() {
        word.start_ms = a + (b - a) * (i as f64 / count);
        word.end_ms = a + (b - a) * ((i + 1) as f64 / count);
    }
}
