use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineIdentity {
    pub engine: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub build_flags: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineRiskClass {
    AbortSafe,
    AbortCapable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineErrorStage {
    Load,
    Inference,
    Unload,
    WorkerCrash,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineError {
    pub stage: EngineErrorStage,
    pub risk_class: EngineRiskClass,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    pub safe_to_retry_same_request: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedArtifact {
    pub digest: String,
    pub format: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub values: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadedModel {
    pub model_id: String,
}

pub type TokenIds = Vec<u32>;
pub type Vector = Vec<f32>;
pub type Vectors = Vec<Vector>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBatch {
    #[serde(default)]
    pub items: Vec<TokenIds>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankRequest {
    pub query: TokenIds,
    #[serde(default)]
    pub candidates: Vec<TokenIds>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RerankScores {
    #[serde(default)]
    pub scores: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: TokenIds,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateOutput {
    pub text: String,
    pub finish_reason: String,
    pub n_prompt: usize,
    pub n_gen: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generated_token_ids: Vec<u32>,
}

/// Normalized log-mel features in row-major order (128 bins per frame), before
/// engine rung padding. Front-end padding may follow `valid_mel_frames`.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscribeRequest {
    pub mel: Vec<f32>,
    pub mel_frames: usize,
    pub valid_mel_frames: usize,
    pub observe: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscribeOutput {
    pub valid_encoder_frames: usize,
    pub triples: Vec<crate::TranscriptRecord>,
    pub observation: Option<TranscribeObservation>,
}

/// Certification-only copies of engine results; observation must not change decoding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscribeObservation {
    pub encoder_output: Vec<f32>,
    pub decode_trace: Vec<crate::DecodeTraceRecord>,
}

pub trait EmbedEngine {
    fn identity(&self) -> EngineIdentity;
    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError>;
    fn embed_batch(&self, model: &LoadedModel, batch: TokenBatch) -> Result<Vectors, EngineError>;
    fn embed_one(&self, model: &LoadedModel, ids: TokenIds) -> Result<Vector, EngineError>;
    fn unload(&mut self, model: &LoadedModel);
}

pub trait RerankEngine {
    fn identity(&self) -> EngineIdentity;
    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError>;
    fn rerank(
        &self,
        model: &LoadedModel,
        request: RerankRequest,
    ) -> Result<RerankScores, EngineError>;
    fn unload(&mut self, model: &LoadedModel);
}

pub trait GenerateEngine {
    fn identity(&self) -> EngineIdentity;
    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError>;
    fn generate(
        &self,
        model: &LoadedModel,
        request: GenerateRequest,
    ) -> Result<GenerateOutput, EngineError>;
    fn unload(&mut self, model: &LoadedModel);
}

pub trait TranscribeEngine {
    fn identity(&self) -> EngineIdentity;
    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError>;
    fn transcribe(
        &self,
        model: &LoadedModel,
        request: TranscribeRequest,
    ) -> Result<TranscribeOutput, EngineError>;
    fn unload(&mut self, model: &LoadedModel);
}
