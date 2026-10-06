//! Deterministic inference double, compiled only for module tests.
use synapse_core::{
    EmbedEngine, EngineError, EngineIdentity, LoadedModel, RerankEngine, RerankRequest,
    RerankScores, RuntimeConfig, TokenBatch, TokenIds, ValidatedArtifact, Vector, Vectors,
};

pub const NAME: &str = "test-deterministic";
pub const DIMS: usize = 384;

#[derive(Default)]
pub struct TestDeterministic;

pub fn vector(ids: &[u32]) -> Vector {
    let mut vector = vec![0.0_f32; DIMS];
    // A seed component gives even an empty token sequence a unit-norm result.
    vector[0] = 1.0;
    for &id in ids {
        let bin = (id.wrapping_mul(2654435761) % (DIMS as u32 - 1)) as usize + 1;
        vector[bin] += 1.0;
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
}
