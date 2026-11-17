//! JSON-in/counts-out bridge to the canonical parity WER normalizer.
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, io};
use synapse_parity::speech::{normalize_words, word_edits};

#[derive(Deserialize)]
struct Row {
    id: String,
    truth: String,
    baseline: String,
    actual: String,
}

fn score(rows: Vec<Row>) -> Result<BTreeMap<String, serde_json::Value>, &'static str> {
    let mut counts = BTreeMap::new();
    for row in rows {
        let truth = normalize_words(&row.truth);
        let count = json!({
            "words": truth.len(),
            "baseline_edits": word_edits(&truth, &normalize_words(&row.baseline)),
            "actual_edits": word_edits(&truth, &normalize_words(&row.actual)),
        });
        if counts.insert(row.id, count).is_some() {
            return Err("duplicate WER fixture");
        }
    }
    Ok(counts)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows: Vec<Row> = serde_json::from_reader(io::stdin().lock())?;
    println!("{}", serde_json::to_string(&score(rows)?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> Row {
        Row {
            id: "fixture".into(),
            truth: "Straße Ａ, café!".into(),
            baseline: "STRASSE a café".into(),
            actual: "strasse a wrong extra".into(),
        }
    }

    #[test]
    fn canonical_normalization_and_raw_counts() {
        assert_eq!(
            score(vec![row()]).unwrap()["fixture"],
            json!({"words": 3, "baseline_edits": 0, "actual_edits": 2})
        );
    }

    #[test]
    fn duplicate_fixture_is_not_silently_overwritten() {
        assert_eq!(
            score(vec![row(), row()]).unwrap_err(),
            "duplicate WER fixture"
        );
    }
}
