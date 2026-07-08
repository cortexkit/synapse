#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use half::f16;
use ndarray::{Array2, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use sha2::{Digest, Sha256};
use synapse_core::{
    EmbedEngine, EngineError, EngineErrorStage, EngineIdentity, EngineRiskClass, LoadedModel,
    RuntimeConfig, TokenBatch, TokenIds, ValidatedArtifact, Vector, Vectors, WorkerPooling,
};

const ENGINE_VERSION: &str = "ort-2.0.0-rc.11";

pub struct OrtEmbedEngine {
    models: HashMap<String, OrtLoadedModel>,
    next_model: u64,
}

struct OrtLoadedModel {
    session: Mutex<Session>,
    input_names: Vec<(String, Vec<i64>)>,
    pooling: WorkerPooling,
    normalize: bool,
}

impl Default for OrtEmbedEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OrtEmbedEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_model: 0,
        }
    }

    #[must_use]
    pub fn default_intra_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .div_ceil(2)
            .max(1)
    }

    fn error(stage: EngineErrorStage, message: impl Into<String>) -> EngineError {
        EngineError {
            stage,
            risk_class: EngineRiskClass::AbortSafe,
            message: message.into(),
            retry_after_ms: None,
            safe_to_retry_same_request: matches!(stage, EngineErrorStage::Inference),
        }
    }

    fn model_path(cfg: &RuntimeConfig) -> Result<PathBuf, EngineError> {
        cfg.values
            .get("model_path")
            .or_else(|| cfg.values.get("artifact_path"))
            .map(PathBuf::from)
            .ok_or_else(|| {
                Self::error(
                    EngineErrorStage::Load,
                    "ORT load requires runtime_config model_path or artifact_path",
                )
            })
    }

    fn pooling(cfg: &RuntimeConfig) -> Result<WorkerPooling, EngineError> {
        match cfg.values.get("pooling").map(String::as_str) {
            Some(value) => WorkerPooling::parse(value).ok_or_else(|| {
                Self::error(
                    EngineErrorStage::Load,
                    format!("unsupported ORT pooling mode '{value}'"),
                )
            }),
            None => Ok(WorkerPooling::Mean),
        }
    }

    fn normalize(cfg: &RuntimeConfig) -> bool {
        cfg.values
            .get("normalize")
            .map(|value| value != "false" && value != "0")
            .unwrap_or(true)
    }

    fn intra_threads(cfg: &RuntimeConfig) -> Result<usize, EngineError> {
        match cfg.values.get("intra_threads") {
            Some(value) => value.parse::<usize>().map_err(|error| {
                Self::error(
                    EngineErrorStage::Load,
                    format!("invalid intra_threads '{value}': {error}"),
                )
            }),
            None => Ok(Self::default_intra_threads()),
        }
    }

    fn verify_digest(path: &Path, digest: &str) -> Result<(), EngineError> {
        if digest.trim().is_empty() {
            return Ok(());
        }
        let expected = digest.strip_prefix("sha256:").unwrap_or(digest);
        let actual = sha256_hex(path).map_err(|error| {
            Self::error(
                EngineErrorStage::Load,
                format!("failed to hash {}: {error}", path.display()),
            )
        })?;
        if actual == expected {
            Ok(())
        } else {
            Err(Self::error(
                EngineErrorStage::Load,
                format!(
                    "artifact digest mismatch for {}: expected {expected}, got {actual}",
                    path.display()
                ),
            ))
        }
    }

    fn input_names(session: &Session) -> Vec<(String, Vec<i64>)> {
        session
            .inputs()
            .iter()
            .map(|input| {
                let dims = match input.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } => shape.to_vec(),
                    _ => Vec::new(),
                };
                (input.name().to_string(), dims)
            })
            .collect()
    }

    pub fn load_from_path(
        &mut self,
        path: impl AsRef<Path>,
        pooling: WorkerPooling,
    ) -> Result<LoadedModel, EngineError> {
        let mut cfg = RuntimeConfig::default();
        cfg.values.insert(
            "model_path".to_string(),
            path.as_ref().to_string_lossy().to_string(),
        );
        cfg.values
            .insert("pooling".to_string(), pooling.as_str().to_string());
        self.load(
            &ValidatedArtifact {
                digest: String::new(),
                format: "onnx".to_string(),
            },
            &cfg,
        )
    }

    fn run_batch(
        &self,
        loaded: &OrtLoadedModel,
        batch: TokenBatch,
    ) -> Result<Vectors, EngineError> {
        if batch.items.is_empty() {
            return Ok(Vec::new());
        }

        let batch_len = batch.items.len();
        let max_len = batch.items.iter().map(Vec::len).max().unwrap_or(0).max(1);
        if let Some(index) = batch.items.iter().position(Vec::is_empty) {
            return Err(Self::error(
                EngineErrorStage::Inference,
                format!("token batch item {index} is empty"),
            ));
        }

        let mut ids = vec![0_i64; batch_len * max_len];
        let mut mask = vec![0_i64; batch_len * max_len];
        for (row, token_ids) in batch.items.iter().enumerate() {
            for (col, token_id) in token_ids.iter().copied().enumerate() {
                ids[row * max_len + col] = i64::from(token_id);
                mask[row * max_len + col] = 1;
            }
        }

        let ids_array =
            Array2::<i64>::from_shape_vec((batch_len, max_len), ids).map_err(|error| {
                Self::error(
                    EngineErrorStage::Inference,
                    format!("failed to build input_ids tensor: {error}"),
                )
            })?;
        let mask_array = Array2::<i64>::from_shape_vec((batch_len, max_len), mask.clone())
            .map_err(|error| {
                Self::error(
                    EngineErrorStage::Inference,
                    format!("failed to build attention_mask tensor: {error}"),
                )
            })?;

        let mut inputs: Vec<(&str, ort::value::DynValue)> = Vec::new();
        for (name, dims) in &loaded.input_names {
            match name.as_str() {
                "input_ids" => inputs.push((
                    name.as_str(),
                    ort::value::Tensor::from_array(ids_array.clone())
                        .map_err(|error| {
                            Self::error(EngineErrorStage::Inference, error.to_string())
                        })?
                        .into_dyn(),
                )),
                "attention_mask" => inputs.push((
                    name.as_str(),
                    ort::value::Tensor::from_array(mask_array.clone())
                        .map_err(|error| {
                            Self::error(EngineErrorStage::Inference, error.to_string())
                        })?
                        .into_dyn(),
                )),
                "token_type_ids" => {
                    let token_types = Array2::<i64>::zeros((batch_len, max_len));
                    inputs.push((
                        name.as_str(),
                        ort::value::Tensor::from_array(token_types)
                            .map_err(|error| {
                                Self::error(EngineErrorStage::Inference, error.to_string())
                            })?
                            .into_dyn(),
                    ));
                }
                "position_ids" => {
                    let mut positions = Array2::<i64>::zeros((batch_len, max_len));
                    for row in 0..batch_len {
                        for col in 0..max_len {
                            positions[[row, col]] = col as i64;
                        }
                    }
                    inputs.push((
                        name.as_str(),
                        ort::value::Tensor::from_array(positions)
                            .map_err(|error| {
                                Self::error(EngineErrorStage::Inference, error.to_string())
                            })?
                            .into_dyn(),
                    ));
                }
                other if other.starts_with("past_key_values.") => {
                    if dims.len() != 4 {
                        return Err(Self::error(
                            EngineErrorStage::Inference,
                            format!("unexpected past KV input shape for {other}: {dims:?}"),
                        ));
                    }
                    let kv_heads = dims[1].max(1) as usize;
                    let head_dim = dims[3].max(1) as usize;
                    let empty = Array4::<f32>::zeros((batch_len, kv_heads, 0, head_dim));
                    inputs.push((
                        name.as_str(),
                        ort::value::Tensor::from_array(empty)
                            .map_err(|error| {
                                Self::error(EngineErrorStage::Inference, error.to_string())
                            })?
                            .into_dyn(),
                    ));
                }
                other => {
                    return Err(Self::error(
                        EngineErrorStage::Inference,
                        format!("unexpected ORT model input '{other}'"),
                    ));
                }
            }
        }

        let mut session = loaded.session.lock().map_err(|_| {
            Self::error(
                EngineErrorStage::Inference,
                "ORT session mutex was poisoned during inference",
            )
        })?;
        let outputs = session.run(inputs).map_err(|error| {
            Self::error(EngineErrorStage::Inference, format!("ORT run: {error}"))
        })?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map(|(shape, data)| (shape.to_vec(), data.to_vec()))
            .or_else(|_| -> Result<_, ort::Error> {
                let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
                Ok((
                    shape.to_vec(),
                    data.iter().map(|value| value.to_f32()).collect(),
                ))
            })
            .map_err(|error| {
                Self::error(
                    EngineErrorStage::Inference,
                    format!("extract ORT output tensor: {error}"),
                )
            })?;

        pool_outputs(
            &shape,
            &data,
            batch_len,
            max_len,
            &mask,
            loaded.pooling,
            loaded.normalize,
        )
    }
}

impl EmbedEngine for OrtEmbedEngine {
    fn identity(&self) -> EngineIdentity {
        let mut build_flags = BTreeMap::new();
        build_flags.insert("risk_class".to_string(), "abort_safe".to_string());
        build_flags.insert("execution_provider".to_string(), "cpu".to_string());
        build_flags.insert(
            "thread_policy".to_string(),
            "ceil(available_parallelism/2)".to_string(),
        );
        EngineIdentity {
            engine: "ort".to_string(),
            version: ENGINE_VERSION.to_string(),
            build_flags,
        }
    }

    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError> {
        if artifact.format != "onnx" {
            return Err(Self::error(
                EngineErrorStage::Load,
                format!(
                    "ORT engine only loads onnx artifacts, got {}",
                    artifact.format
                ),
            ));
        }
        let path = Self::model_path(cfg)?;
        Self::verify_digest(&path, &artifact.digest)?;
        let pooling = Self::pooling(cfg)?;
        let normalize = Self::normalize(cfg);
        let intra_threads = Self::intra_threads(cfg)?;

        let session = Session::builder()
            .map_err(|error| Self::error(EngineErrorStage::Load, format!("ORT builder: {error}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| {
                Self::error(
                    EngineErrorStage::Load,
                    format!("ORT optimization level: {error}"),
                )
            })?
            .with_intra_threads(intra_threads)
            .map_err(|error| {
                Self::error(
                    EngineErrorStage::Load,
                    format!("ORT intra threads: {error}"),
                )
            })?
            .commit_from_file(&path)
            .map_err(|error| {
                Self::error(
                    EngineErrorStage::Load,
                    format!("load ORT model {}: {error}", path.display()),
                )
            })?;
        let input_names = Self::input_names(&session);
        let model_id = format!("ort:{}", self.next_model);
        self.next_model += 1;
        self.models.insert(
            model_id.clone(),
            OrtLoadedModel {
                session: Mutex::new(session),
                input_names,
                pooling,
                normalize,
            },
        );
        Ok(LoadedModel { model_id })
    }

    fn embed_batch(&self, model: &LoadedModel, batch: TokenBatch) -> Result<Vectors, EngineError> {
        let loaded = self.models.get(&model.model_id).ok_or_else(|| {
            Self::error(
                EngineErrorStage::Inference,
                format!("unknown ORT model ref '{}'", model.model_id),
            )
        })?;
        self.run_batch(loaded, batch)
    }

    fn embed_one(&self, model: &LoadedModel, ids: TokenIds) -> Result<Vector, EngineError> {
        let mut vectors = self.embed_batch(model, TokenBatch { items: vec![ids] })?;
        vectors.pop().ok_or_else(|| {
            Self::error(
                EngineErrorStage::Inference,
                "ORT returned no vector for single-item batch",
            )
        })
    }

    fn unload(&mut self, model: &LoadedModel) {
        self.models.remove(&model.model_id);
    }
}

fn pool_outputs(
    shape: &[i64],
    data: &[f32],
    expected_batch: usize,
    expected_seq: usize,
    mask: &[i64],
    pooling: WorkerPooling,
    normalize: bool,
) -> Result<Vectors, EngineError> {
    if shape.len() != 3 {
        return Err(OrtEmbedEngine::error(
            EngineErrorStage::Inference,
            format!("expected ORT output [batch, seq, hidden], got {shape:?}"),
        ));
    }
    let (batch, seq, hidden) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    if batch != expected_batch || seq != expected_seq {
        return Err(OrtEmbedEngine::error(
            EngineErrorStage::Inference,
            format!(
                "ORT output shape mismatch: got batch={batch}, seq={seq}; expected batch={expected_batch}, seq={expected_seq}"
            ),
        ));
    }
    if data.len() != batch * seq * hidden {
        return Err(OrtEmbedEngine::error(
            EngineErrorStage::Inference,
            format!(
                "ORT output data length mismatch: got {}, expected {}",
                data.len(),
                batch * seq * hidden
            ),
        ));
    }

    let mut result = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut vector = vec![0.0_f32; hidden];
        match pooling {
            WorkerPooling::Mean => {
                let mut count = 0.0_f32;
                for col in 0..seq {
                    if mask[row * expected_seq + col] == 1 {
                        count += 1.0;
                        for dim in 0..hidden {
                            vector[dim] += data[(row * seq + col) * hidden + dim];
                        }
                    }
                }
                let denom = count.max(1.0);
                for value in &mut vector {
                    *value /= denom;
                }
            }
            WorkerPooling::Cls => {
                vector.copy_from_slice(&data[row * seq * hidden..row * seq * hidden + hidden]);
            }
            WorkerPooling::Last => {
                let last = (0..seq)
                    .rev()
                    .find(|&col| mask[row * expected_seq + col] == 1)
                    .unwrap_or(0);
                vector.copy_from_slice(
                    &data[(row * seq + last) * hidden..(row * seq + last + 1) * hidden],
                );
            }
        }
        if normalize {
            l2_normalize(&mut vector);
        }
        result.push(vector);
    }
    Ok(result)
}

pub fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt() + 1e-12;
    for value in vector {
        *value /= norm;
    }
}

fn sha256_hex(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_unit_length_vectors() {
        let mut vector = vec![3.0, 4.0];
        l2_normalize(&mut vector);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn pools_mean_cls_and_last() {
        let shape = [2, 3, 2];
        let data = vec![1.0, 0.0, 3.0, 0.0, 100.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0];
        let mask = vec![1, 1, 0, 1, 1, 1];
        let mean = pool_outputs(&shape, &data, 2, 3, &mask, WorkerPooling::Mean, false).unwrap();
        assert_eq!(mean, vec![vec![2.0, 0.0], vec![0.0, 4.0]]);
        let cls = pool_outputs(&shape, &data, 2, 3, &mask, WorkerPooling::Cls, false).unwrap();
        assert_eq!(cls, vec![vec![1.0, 0.0], vec![0.0, 2.0]]);
        let last = pool_outputs(&shape, &data, 2, 3, &mask, WorkerPooling::Last, false).unwrap();
        assert_eq!(last, vec![vec![3.0, 0.0], vec![0.0, 6.0]]);
    }
}
