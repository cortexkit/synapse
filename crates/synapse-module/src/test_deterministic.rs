//! Deterministic inference double, compiled only for module tests.
use synapse_core::{
    EmbedEngine, EngineError, EngineIdentity, LoadedModel, RerankEngine, RerankRequest,
    RerankScores, RuntimeConfig, TokenBatch, TokenIds, ValidatedArtifact, Vector, Vectors,
};

pub const NAME: &str = "test-deterministic";
/// Vector width when no override is set.
pub const DEFAULT_DIMS: usize = 384;

/// Vector width for this process. Tests that need a production-sized reply
/// (Qwen3-Embedding emits 1,024 values per row) set
/// SYNAPSE_TEST_DETERMINISTIC_DIMS; any value below 2 falls back to the
/// default, because the seed component and at least one token bin are needed.
pub fn dims() -> usize {
    std::env::var("SYNAPSE_TEST_DETERMINISTIC_DIMS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&dims| dims >= 2)
        .unwrap_or(DEFAULT_DIMS)
}

/// SYNAPSE_TEST_DETERMINISTIC_DENSE=1 fills every component (see `vector`).
fn dense() -> bool {
    std::env::var("SYNAPSE_TEST_DETERMINISTIC_DENSE").as_deref() == Ok("1")
}

#[derive(Default)]
pub struct TestDeterministic;

pub fn vector(ids: &[u32]) -> Vector {
    vector_with(ids, dims(), dense())
}

fn vector_with(ids: &[u32], dims: usize, dense: bool) -> Vector {
    let mut vector = vec![0.0_f32; dims];
    // A seed component gives even an empty token sequence a unit-norm result.
    vector[0] = 1.0;
    for &id in ids {
        let bin = (id.wrapping_mul(2654435761) % (dims as u32 - 1)) as usize + 1;
        vector[bin] += 1.0;
    }
    if dense {
        // Production vectors have no zero components, so a reply of mostly
        // zeros would understate the JSON reply size and its encode/decode
        // cost. Dense mode gives every empty bin a tiny value derived from the
        // tokens. At most 1e-6 per bin leaves the direction, and so the
        // probe's reference vectors, unchanged to within 1e-6 cosine, while
        // every component prints with full precision like a real embedding.
        let mut state = ids.iter().fold(0x9e37_79b9_u32, |state, &id| {
            state.rotate_left(5) ^ id.wrapping_mul(0x85eb_ca6b)
        }) | 1; // xorshift never leaves the all-zero state
        for value in vector.iter_mut().filter(|value| **value == 0.0) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *value = (state as f32 / u32::MAX as f32 - 0.5) * 2.0e-6;
        }
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    for v in &mut vector {
        *v /= norm;
    }
    vector
}

impl EmbedEngine for TestDeterministic {
    fn identity(&self) -> EngineIdentity {
        EngineIdentity {
            engine: NAME.into(),
            version: "token-bag-v1".into(),
            build_flags: Default::default(),
        }
    }
    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        _: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError> {
        Ok(LoadedModel {
            model_id: format!("{NAME}:{}", artifact.digest),
        })
    }
    fn embed_batch(&self, _: &LoadedModel, batch: TokenBatch) -> Result<Vectors, EngineError> {
        // A real engine takes measurable time per batch, and the admission tests
        // assert on requests that are queued or still running (waiter counts,
        // concurrent bursts). An instant engine would finish before they could
        // observe anything, so each batch sleeps briefly; a test can lengthen
        // the sleep with SYNAPSE_TEST_DETERMINISTIC_DELAY_MS.
        let delay = std::env::var("SYNAPSE_TEST_DETERMINISTIC_DELAY_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);
        std::thread::sleep(std::time::Duration::from_millis(delay));
        Ok(batch.items.iter().map(|ids| vector(ids)).collect())
    }
    fn embed_one(&self, _: &LoadedModel, ids: TokenIds) -> Result<Vector, EngineError> {
        Ok(vector(&ids))
    }
    fn unload(&mut self, _: &LoadedModel) {}
}

impl RerankEngine for TestDeterministic {
    fn identity(&self) -> EngineIdentity {
        EmbedEngine::identity(self)
    }
    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError> {
        EmbedEngine::load(self, artifact, cfg)
    }
    fn rerank(&self, _: &LoadedModel, request: RerankRequest) -> Result<RerankScores, EngineError> {
        let query = vector(&request.query);
        Ok(RerankScores {
            scores: request
                .candidates
                .iter()
                .map(|ids| query.iter().zip(vector(ids)).map(|(q, d)| q * d).sum())
                .collect(),
        })
    }
    fn unload(&mut self, _: &LoadedModel) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_ids_determine_unit_vectors_and_batch_order() {
        let engine = TestDeterministic;
        let model = LoadedModel {
            model_id: "test".into(),
        };
        let vectors = engine
            .embed_batch(
                &model,
                TokenBatch {
                    items: vec![vec![1], vec![2], vec![]],
                },
            )
            .unwrap();
        assert_ne!(vectors[0], vectors[1]);
        assert_eq!(vectors[0], engine.embed_one(&model, vec![1]).unwrap());
        for v in vectors {
            assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        }
        let scores = engine
            .rerank(
                &model,
                RerankRequest {
                    query: vec![1],
                    candidates: vec![vec![1], vec![2]],
                },
            )
            .unwrap()
            .scores;
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn dense_mode_fills_every_component_without_moving_the_direction() {
        let ids = [1, 2, 3, 2];
        let sparse = vector_with(&ids, 1_024, false);
        let dense = vector_with(&ids, 1_024, true);
        assert_eq!(dense.len(), 1_024);
        assert!(sparse.iter().filter(|&&v| v == 0.0).count() > 1_000);
        assert!(
            dense.iter().all(|&v| v != 0.0 && v.is_finite()),
            "dense mode left a zero component"
        );
        assert!((dense.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        let cosine = sparse.iter().zip(&dense).map(|(a, b)| a * b).sum::<f32>();
        assert!(
            cosine > 1.0 - 1e-6,
            "dense mode moved the direction: {cosine}"
        );
        assert_eq!(
            dense,
            vector_with(&ids, 1_024, true),
            "dense mode is deterministic"
        );
    }
}
