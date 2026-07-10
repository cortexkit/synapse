use thiserror::Error;

use super::openai_compat::EmbeddingResponse;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("provider protocol violation: {reason}")]
pub(super) struct ProviderProtocolViolation {
    pub reason: String,
}

impl ProviderProtocolViolation {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

pub(super) fn validate_embedding_response(
    response: &EmbeddingResponse,
    expected_items: usize,
    expected_dimensions: usize,
) -> Result<Vec<Vec<f64>>, ProviderProtocolViolation> {
    if response.data.len() != expected_items {
        return Err(ProviderProtocolViolation::new(format!(
            "item count mismatch: expected {expected_items}, received {}",
            response.data.len()
        )));
    }

    let mut vectors_by_index = vec![None; expected_items];
    for item in &response.data {
        if item.index >= expected_items {
            return Err(ProviderProtocolViolation::new(format!(
                "invalid index permutation: index {} is outside expected range 0..{expected_items}",
                item.index
            )));
        }
        if vectors_by_index[item.index].is_some() {
            return Err(ProviderProtocolViolation::new(format!(
                "invalid index permutation: index {} appears more than once",
                item.index
            )));
        }
        if item.embedding.len() != expected_dimensions {
            return Err(ProviderProtocolViolation::new(format!(
                "dimension mismatch at index {}: expected {expected_dimensions}, received {}",
                item.index,
                item.embedding.len()
            )));
        }
        if let Some((coordinate, value)) = item
            .embedding
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ProviderProtocolViolation::new(format!(
                "non-finite embedding value at index {}, coordinate {coordinate}: {value}",
                item.index
            )));
        }

        vectors_by_index[item.index] = Some(item.embedding.clone());
    }

    vectors_by_index
        .into_iter()
        .enumerate()
        .map(|(index, vector)| {
            vector.ok_or_else(|| {
                ProviderProtocolViolation::new(format!(
                    "invalid index permutation: index {index} is missing"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::openai_compat::{parse_embedding_response, EmbeddingData, EmbeddingUsage};

    fn response(data: Vec<EmbeddingData>) -> EmbeddingResponse {
        EmbeddingResponse {
            data,
            model: "model".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
        }
    }

    fn item(index: usize, embedding: Vec<f64>) -> EmbeddingData {
        EmbeddingData { index, embedding }
    }

    #[test]
    fn accepts_and_reorders_an_exact_index_permutation() {
        let response = response(vec![item(1, vec![3.0, 4.0]), item(0, vec![1.0, 2.0])]);
        let vectors = validate_embedding_response(&response, 2, 2).unwrap();
        assert_eq!(vectors, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn accepts_an_empty_response_for_an_empty_request() {
        let vectors = validate_embedding_response(&response(vec![]), 0, 8).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn rejects_short_and_long_item_counts() {
        let short =
            validate_embedding_response(&response(vec![item(0, vec![1.0])]), 2, 1).unwrap_err();
        assert_eq!(short.reason, "item count mismatch: expected 2, received 1");

        let long = validate_embedding_response(
            &response(vec![item(0, vec![1.0]), item(1, vec![2.0])]),
            1,
            1,
        )
        .unwrap_err();
        assert_eq!(long.reason, "item count mismatch: expected 1, received 2");
    }

    #[test]
    fn rejects_duplicate_missing_and_out_of_range_indexes() {
        let cases = [
            (
                vec![item(0, vec![1.0]), item(0, vec![2.0])],
                "index 0 appears more than once",
            ),
            (
                vec![item(0, vec![1.0]), item(2, vec![2.0])],
                "index 2 is outside expected range 0..2",
            ),
            (
                vec![item(1, vec![1.0]), item(1, vec![2.0])],
                "index 1 appears more than once",
            ),
        ];

        for (data, expected_reason) in cases {
            let error = validate_embedding_response(&response(data), 2, 1).unwrap_err();
            assert_eq!(
                error.reason,
                format!("invalid index permutation: {expected_reason}")
            );
        }
    }

    #[test]
    fn rejects_ragged_short_and_long_dimensions() {
        for data in [
            vec![item(0, vec![1.0, 2.0]), item(1, vec![3.0])],
            vec![item(0, vec![1.0, 2.0]), item(1, vec![3.0, 4.0, 5.0])],
        ] {
            let error = validate_embedding_response(&response(data), 2, 2).unwrap_err();
            assert!(error.reason.starts_with("dimension mismatch at index 1"));
        }
    }

    #[test]
    fn rejects_each_non_finite_float_value() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = validate_embedding_response(&response(vec![item(0, vec![value])]), 1, 1)
                .unwrap_err();
            assert!(error
                .reason
                .starts_with("non-finite embedding value at index 0"));
        }
    }

    #[test]
    fn json_number_smuggling_cannot_bypass_finite_validation() {
        assert!(parse_embedding_response(
            br#"{"data":[{"index":0,"embedding":[NaN]}],"model":"m","usage":{"prompt_tokens":1,"total_tokens":1}}"#
        )
        .is_err());

        let overflow = parse_embedding_response(
            br#"{"data":[{"index":0,"embedding":[1e999]}],"model":"m","usage":{"prompt_tokens":1,"total_tokens":1}}"#,
        );
        match overflow {
            Err(_) => {}
            Ok(response) => {
                let error = validate_embedding_response(&response, 1, 1).unwrap_err();
                assert!(error.reason.starts_with("non-finite embedding value"));
            }
        }
    }
}
