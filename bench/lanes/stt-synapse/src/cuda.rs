use super::*;
use synapse_core::{RuntimeConfig, TranscribeEngine, ValidatedArtifact};
use synapse_engine_cuda::transcribe::OwnedCudaTranscribeEngine;

struct Cuda {
    engine: OwnedCudaTranscribeEngine,
    model: synapse_core::LoadedModel,
}

pub fn load(path: &Path, f32: bool) -> Result<Box<dyn Engine>> {
    let path = package(path);
    let digest = format!(
        "sha256:{}",
        stt_common::artifact::artifacts(&path)?[0].sha256
    );
    let artifact = ValidatedArtifact {
        format: "safetensors-package".into(),
        digest,
    };
    let cfg = RuntimeConfig {
        values: [
            ("profile".into(), CUDA_PROFILE.into()),
            ("model_path".into(), path.to_string_lossy().into_owned()),
        ]
        .into(),
    };
    let mut engine = OwnedCudaTranscribeEngine::default();
    let model = if f32 {
        engine
            .load_f32_measurement(&artifact, &cfg)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
    } else {
        engine
            .load(&artifact, &cfg)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
    };
    Ok(Box::new(Cuda { engine, model }))
}

impl Engine for Cuda {
    fn tokens(&mut self, features: Features) -> Result<Vec<u32>> {
        token_ids(
            self.engine
                .transcribe(&self.model, request(features))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?,
        )
    }
}
