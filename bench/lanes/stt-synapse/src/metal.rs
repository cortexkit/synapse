use super::*;
use synapse_core::{RuntimeConfig, TranscribeEngine, ValidatedArtifact};
use synapse_engine_owned::transcribe::OwnedMetalTranscribeEngine;

struct Metal {
    engine: OwnedMetalTranscribeEngine,
    model: synapse_core::LoadedModel,
}

pub fn load(path: &Path) -> Result<Box<dyn Engine>> {
    let mut engine = OwnedMetalTranscribeEngine::new();
    let model = engine
        .load(
            &ValidatedArtifact {
                format: "safetensors-package".into(),
                digest: "benchmark".into(),
            },
            &RuntimeConfig {
                values: [
                    ("model_path".into(), path.to_string_lossy().into_owned()),
                    ("max_encoder_frames".into(), "7500".into()),
                ]
                .into(),
            },
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(Box::new(Metal { engine, model }))
}

impl Engine for Metal {
    fn tokens(&mut self, features: Features) -> Result<Vec<u32>> {
        token_ids(
            self.engine
                .transcribe(&self.model, request(features))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?,
        )
    }
}
