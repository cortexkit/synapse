use serde::{Deserialize, Serialize};

pub(super) const EMBEDDINGS_PATH: &str = "/embeddings";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(super) struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(super) struct EmbeddingData {
    pub index: usize,
    pub embedding: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(super) struct EmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}

pub(super) fn parse_embedding_response(
    body: &[u8],
) -> Result<EmbeddingResponse, serde_json::Error> {
    serde_json::from_slice(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_serializes_openai_shape_and_omits_absent_dimensions() {
        let request = EmbeddingRequest {
            model: "text-embedding".to_string(),
            input: vec!["alpha".to_string(), "beta".to_string()],
            dimensions: None,
        };

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({"model": "text-embedding", "input": ["alpha", "beta"]})
        );
    }

    #[test]
    fn request_serializes_dimensions_when_present() {
        let request = EmbeddingRequest {
            model: "text-embedding".to_string(),
            input: vec!["alpha".to_string()],
            dimensions: Some(384),
        };

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({"model": "text-embedding", "input": ["alpha"], "dimensions": 384})
        );
    }

    #[test]
    fn response_parser_tolerates_provider_extension_fields_at_every_level() {
        let body = br#"{
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.25, -0.5],
                "provider_extension": true
            }],
            "model": "provider/model",
            "usage": {
                "prompt_tokens": 3,
                "total_tokens": 3,
                "cached_tokens": 1
            },
            "provider_request_id": "req-123"
        }"#;

        let response = parse_embedding_response(body).unwrap();
        assert_eq!(response.model, "provider/model");
        assert_eq!(response.data[0].embedding, vec![0.25, -0.5]);
        assert_eq!(response.usage.total_tokens, 3);
    }

    #[test]
    fn response_parser_requires_the_owned_openai_shape() {
        let error = parse_embedding_response(br#"{"data": [], "model": "m"}"#).unwrap_err();
        assert!(error.to_string().contains("usage"));
    }
}
