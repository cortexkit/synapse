use crate::{modernbert, qwen};
use ane::{Executable, Graph, NSQualityOfService, TensorData};
use anyhow::{ensure, Context, Result};
use half::f16;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use synapse_core::{AneExecutable, AnePlacementInventory};

pub const REVISION: &str = "ane-direct-graph-v1";
const PRODUCTION_SHAPE_LIMIT: usize = 4;
pub const LADDER: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
pub fn rung(tokens: usize) -> Result<usize> {
    ensure!(tokens > 0, "invalid_request");
    LADDER
        .into_iter()
        .find(|&n| n >= tokens)
        .context("context_limit_exceeded")
}

#[derive(Clone)]
pub struct Profile {
    pub id: String,
    pub model: Value,
    pub numeric: Value,
}
impl Profile {
    pub fn select(id: &str, operation: &str) -> Result<Self> {
        let manifest: Value =
            serde_json::from_slice(include_bytes!(concat!(env!("OUT_DIR"), "/models.json")))?;
        let numeric = manifest["profiles"][id].clone();
        ensure!(numeric["lane"] == "ane-direct-worker", "model_unsupported");
        let model =
            manifest["models"][numeric["model"].as_str().context("model_unsupported")?].clone();
        ensure!(model["operation"] == operation, "operation_mismatch");
        ensure!(numeric["compute_dtype"] == "f16", "model_unsupported");
        Ok(Self {
            id: id.into(),
            model,
            numeric,
        })
    }
    pub fn params(&self) -> &Value {
        &self.model["architecture"]["params"]
    }
    pub fn n(&self, key: &str) -> usize {
        self.params()[key].as_u64().expect("manifest integer") as usize
    }
    pub fn f(&self, key: &str) -> f32 {
        self.params()[key].as_f64().expect("manifest float") as f32
    }
    pub fn modern(&self) -> bool {
        self.model["architecture"]["family"] == "modernbert"
    }
    pub fn operation(&self) -> &str {
        self.model["operation"].as_str().unwrap()
    }
    pub fn prefix(&self) -> &str {
        self.model["tensor_prefix"].as_str().unwrap()
    }
    pub fn inventory(&self, shape: usize, os: &str) -> Result<AnePlacementInventory> {
        let stages: Vec<String> = serde_json::from_value(self.numeric["cpu_stages"].clone())?;
        let executables = (0..self.n("num_hidden_layers"))
            .map(|layer| AneExecutable {
                id: self.key(shape, layer, os),
                layers: vec![layer as u32],
            })
            .collect();
        let inventory = AnePlacementInventory {
            executables,
            cpu_stages: stages,
        };
        validate_inventory(&inventory, self.n("num_hidden_layers"))?;
        Ok(inventory)
    }
    pub fn key(&self, shape: usize, layer: usize, os: &str) -> String {
        let identity = serde_json::json!({"model_digest":self.numeric["converted_package_digest"],
            "operation": self.operation(), "numeric_profile":self.numeric,
            "rotation":self.numeric["rotation"], "shape":shape, "layer":layer,
            "graph_revision":REVISION, "worker_version":env!("CARGO_PKG_VERSION"), "os_build":os});
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&identity).unwrap())
        )
    }
}
pub fn validate_inventory(inventory: &AnePlacementInventory, layers: usize) -> Result<()> {
    const CPU: [&str; 8] = [
        "token_embedding",
        "mask_position",
        "rotation_in",
        "rotation_out",
        "final_norm",
        "pooling",
        "gte_classifier_head",
        "qwen_yes_no_readout",
    ];
    ensure!(
        inventory
            .cpu_stages
            .iter()
            .all(|stage| CPU.contains(&stage.as_str())),
        "invalid_placement"
    );
    let mut coverage = vec![0usize; layers];
    for executable in &inventory.executables {
        ensure!(!executable.layers.is_empty(), "invalid_placement");
        for &layer in &executable.layers {
            let count = coverage
                .get_mut(layer as usize)
                .context("invalid_placement")?;
            *count += 1;
        }
    }
    ensure!(coverage.into_iter().all(|n| n == 1), "invalid_placement");
    Ok(())
}

pub struct Model {
    pub profile: Profile,
    pub tensors: BTreeMap<String, Vec<f32>>,
    pub resident: BTreeMap<usize, Resident>,
}
pub struct Resident {
    pub inventory: AnePlacementInventory,
    executables: Vec<Executable>,
    a: TensorData,
    b: TensorData,
    residual: TensorData,
    mask: TensorData,
}
impl Drop for Model {
    fn drop(&mut self) {
        // Model unload, replacement, and connection close release all cached programs.
        ane::autoreleasepool(|_| self.resident.clear());
    }
}

impl Model {
    pub fn load(profile: Profile, path: &Path, digest: &str) -> Result<Self> {
        ensure!(
            profile.numeric["converted_package_digest"] == digest,
            "package_digest_mismatch"
        );
        let path = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.into()
        };
        let bytes = std::fs::read(path)?;
        ensure!(
            format!("sha256:{:x}", Sha256::digest(&bytes)) == digest,
            "package_digest_mismatch"
        );
        let (metadata, st) = SafeTensors::read_metadata(&bytes)?;
        let _ = metadata;
        ensure!(
            st.metadata()
                .as_ref()
                .is_some_and(|m| m.get("profile") == Some(&profile.id)),
            "package_digest_mismatch"
        );
        let st = SafeTensors::deserialize(&bytes)?;
        if let Some(heads) = profile.model["head"]["tensors"].as_object() {
            for head in heads.values() {
                let key = head["key"].as_str().context("head_tensor_missing")?;
                let tensor = st
                    .tensor(key)
                    .map_err(|_| anyhow::anyhow!("head_tensor_missing"))?;
                ensure!(
                    serde_json::to_value(tensor.shape())? == head["shape"],
                    "head_tensor_missing"
                );
            }
        }
        if let Some(keys) = profile.model["head"]["forbidden_tensor_keys"].as_array() {
            ensure!(
                keys.iter().all(|k| st.tensor(k.as_str().unwrap()).is_err()),
                "head_tensor_missing"
            );
        }
        let fp32: Vec<String> = serde_json::from_value(profile.numeric["fp32_tensors"].clone())?;
        let mut tensors = BTreeMap::new();
        for (name, tensor) in st.tensors() {
            let expected = if fp32.contains(&name) {
                Dtype::F32
            } else {
                Dtype::F16
            };
            ensure!(tensor.dtype() == expected, "package_digest_mismatch");
            let values = match expected {
                Dtype::F32 => tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
                Dtype::F16 => tensor
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                    .collect(),
                _ => unreachable!(),
            };
            tensors.insert(name, values);
        }
        Ok(Self {
            profile,
            tensors,
            resident: BTreeMap::new(),
        })
    }
    pub fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| format!("tensor_missing:{name}"))
    }
    pub fn admit(&mut self, shape: usize, os: &str) -> Result<AnePlacementInventory> {
        let profile = crate::profile::LaneProfile::new();
        let cached = self.resident.contains_key(&shape);
        // Compile temporaries retain ANE programs until the autorelease pool drains.
        let result = ane::autoreleasepool(|_| {
            self.admit_with_compiler(shape, os, |graph, _| {
                graph.compile(NSQualityOfService::UserInteractive)
            })
        });
        profile.finish(serde_json::json!({"kind":"admit", "shape":shape, "cached":cached, "ok":result.is_ok()}));
        result
    }
    fn admit_with_compiler(
        &mut self,
        shape: usize,
        os: &str,
        compile: impl FnMut(&Graph, usize) -> std::result::Result<ane::Executable, ane::Error>,
    ) -> Result<AnePlacementInventory> {
        self.admit_with_limit(shape, os, PRODUCTION_SHAPE_LIMIT, compile)
    }
    fn admit_with_limit(
        &mut self,
        shape: usize,
        os: &str,
        limit: usize,
        mut compile: impl FnMut(&Graph, usize) -> std::result::Result<ane::Executable, ane::Error>,
    ) -> Result<AnePlacementInventory> {
        ensure!(LADDER.contains(&shape), "invalid_shape");
        if let Some(resident) = self.resident.get(&shape) {
            return Ok(resident.inventory.clone());
        }
        ensure!(self.resident.len() < limit, "ane_residency_limit");
        let inventory = self.profile.inventory(shape, os)?;
        let hidden = self.profile.n("hidden_size");
        let mut executables = Vec::new();
        for layer in 0..self.profile.n("num_hidden_layers") {
            let mut graph = Graph::new();
            let input = graph.placeholder(modernbert::shape(shape, hidden));
            let mask = graph.placeholder(modernbert::shape(shape, 1));
            if self.profile.modern() {
                let residual = if layer == 0 {
                    graph.placeholder(modernbert::shape(shape, hidden))
                } else {
                    input
                };
                let config: modernbert::Config =
                    serde_json::from_value(self.profile.params().clone())?;
                let base = format!("{}layers.{layer}", self.profile.prefix());
                let linear = |name: &str| -> Result<modernbert::Linear> {
                    Ok(modernbert::Linear {
                        weight: self.tensor(&format!("{base}.{name}.weight"))?.to_vec(),
                    })
                };
                let weights = modernbert::LayerWeights {
                    qkv: linear("attn.Wqkv")?,
                    attention_output: linear("attn.Wo")?,
                    attention_norm: (layer > 0).then(|| vec![1.0; hidden]),
                    mlp_input: linear("mlp.Wi")?,
                    mlp_output: linear("mlp.Wo")?,
                    mlp_norm: vec![1.0; hidden],
                };
                let _ = modernbert::layer_graph(
                    &mut graph, input, residual, mask, &weights, &config, layer, shape,
                );
            } else {
                qwen::layer_graph(&mut graph, input, mask, self, layer, shape)?;
            }
            #[cfg(test)]
            let diagnostic = if std::env::var_os("ANE_DIAGNOSTICS").is_some() {
                Some(diagnostic_payload(&graph, shape, layer))
            } else {
                None
            };
            #[cfg(test)]
            let layer_started = std::time::Instant::now();
            let result = compile(&graph, layer);
            #[cfg(test)]
            if let Some(payload) = diagnostic {
                diagnostic_artifact(payload, result.is_ok());
                println!("LAYER_LOAD shape={shape} layer={layer} prior_loaded={} elapsed_ms={:.3} outcome={}", executables.len(), layer_started.elapsed().as_secs_f64()*1000.0, result.as_ref().map(|_| "LOADED".to_owned()).unwrap_or_else(|e| e.to_string()));
            }
            // The vector stays local until every layer loads. Returning an
            // error drops all previous executables before the worker replies.
            executables.push(result.map_err(|error| admission_error(error, layer))?);
        }
        self.resident.insert(
            shape,
            Resident {
                inventory: inventory.clone(),
                executables,
                a: TensorData::new(modernbert::shape(shape, hidden)),
                b: TensorData::new(modernbert::shape(shape, hidden)),
                residual: TensorData::new(modernbert::shape(shape, hidden)),
                mask: TensorData::new(modernbert::shape(shape, 1)),
            },
        );
        Ok(inventory)
    }
    pub fn evict(&mut self, shape: usize) {
        let profile = crate::profile::LaneProfile::new();
        ane::autoreleasepool(|_| drop(self.resident.remove(&shape)));
        profile.finish(serde_json::json!({"kind":"evict", "shape":shape}));
    }

    pub fn run(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        ane::autoreleasepool(|_| self.run_stages(tokens, false))
    }
    fn run_stages(&self, tokens: &[u32], traced: bool) -> Result<Vec<f32>> {
        self.run_stages_at_rung(tokens, traced, rung(tokens.len())?)
    }
    pub(crate) fn run_stages_at_rung(
        &self,
        tokens: &[u32],
        traced: bool,
        shape: usize,
    ) -> Result<Vec<f32>> {
        let mut stages = StageClock::new(traced);
        let mut profile = crate::profile::LaneProfile::new();
        let resident = self.resident.get(&shape).context("shape_not_admitted")?;
        let hidden = self.profile.n("hidden_size");
        let pad = self.profile.n("pad_token_id") as u32;
        ensure!(!tokens.is_empty(), "invalid_request");
        let embedding_name = if self.profile.modern() {
            "embeddings.tok_embeddings.weight"
        } else {
            "embed_tokens.weight"
        };
        let embeddings = self.tensor(&format!("{}{embedding_name}", self.profile.prefix()))?;
        let mut input = vec![0.0; hidden * shape];
        let padded = padded_ids(tokens, pad, shape);
        for (position, token) in padded.into_iter().enumerate() {
            let token = token as usize;
            let embedding = embeddings
                .get(token * hidden..(token + 1) * hidden)
                .context("invalid_request")?;
            let mut row = embedding.to_vec();
            if self.profile.modern() {
                rms_cpu(&mut row, None, self.profile.f("norm_eps"));
            }
            for channel in 0..hidden {
                input[channel * shape + position] = row[channel];
            }
        }
        stages.emit("host_prologue");
        profile.phase("embedding_gather");
        resident.a.copy_from_f32(&input);
        stages.emit("initial_input_copy_fp32_to_fp16");
        profile.phase("input_pack");
        if self.profile.modern() {
            let matrix = self.tensor("rotation_in.weight")?;
            // The converted residual projection is a dense fp32 matrix, not
            // just the Hadamard basis. Batch its sequence columns on the CPU.
            let mut residual = matmul_channels(matrix, &input, hidden, shape, true)?;
            let direction = rotation_center_direction();
            for position in 0..shape {
                let mean = (0..hidden)
                    .map(|c| residual[c * shape + position] * direction[c])
                    .sum::<f32>()
                    / hidden as f32;
                for c in 0..hidden {
                    residual[c * shape + position] -= mean * direction[c];
                }
            }
            stages.emit("rotation_in_fp32");
            resident.residual.copy_from_f32(&residual);
            stages.emit("residual_copy_fp32_to_fp16");
        }
        resident
            .mask
            .copy_from_f32(&padding_mask(tokens, pad, shape));
        stages.emit("mask_copy_fp32_to_fp16");
        profile.phase("mask_pack");
        for (layer, executable) in resident.executables.iter().enumerate() {
            let (src, dst) = if layer % 2 == 0 {
                (&resident.a, &resident.b)
            } else {
                (&resident.b, &resident.a)
            };
            let inputs: &[&TensorData] = if layer == 0 && self.profile.modern() {
                &[src, &resident.mask, &resident.residual]
            } else {
                &[src, &resident.mask]
            };
            if profile.enabled() && std::env::var_os("SYNAPSE_ANE_PROFILE_HW").is_some() {
                let started = std::time::Instant::now();
                let hw_ns = executable.run_cached_with_stats(inputs, &[dst])?;
                profile.hardware_layer(layer, started.elapsed(), hw_ns);
                stages.restart();
            } else if traced || profile.enabled() {
                let (prepare, evaluate, created) =
                    executable.run_cached_profiled(inputs, &[dst])?;
                if traced {
                    println!("FORWARD_LAYER layer={layer} request_prepare_ms={:.6} sync_submit_and_wait_ms={:.6} request_created={created} interlayer_host_copy_bytes=0 iosurface_allocations=0", prepare.as_secs_f64()*1000.0, evaluate.as_secs_f64()*1000.0);
                }
                if profile.enabled() {
                    profile.layer(layer, prepare, evaluate, created);
                }
                stages.restart();
            } else {
                executable.run_cached(inputs, &[dst])?;
            }
        }
        let surface = if resident.executables.len() % 2 == 0 {
            &resident.a
        } else {
            &resident.b
        };
        let raw = surface.read_f32();
        stages.emit("final_readback_fp16_to_fp32");
        profile.phase("output_readback");
        // CLS pooling and last-token readout consume only one row. The
        // ModernBERT classifier needs every unpadded row for mean pooling.
        let positions: Vec<usize> = if self.profile.modern() && self.profile.operation() == "rerank"
        {
            (0..tokens.len()).collect()
        } else if self.profile.modern() {
            vec![0]
        } else {
            vec![tokens.len() - 1]
        };
        let columns = positions.len();
        let mut normalized = vec![0.0; hidden * columns];
        let norm_weight = if self.profile.modern() {
            None
        } else {
            Some(self.tensor(&format!("{}norm.weight", self.profile.prefix()))?)
        };
        for (column, position) in positions.into_iter().enumerate() {
            let mut row: Vec<_> = (0..hidden).map(|c| raw[c * shape + position]).collect();
            rms_cpu(&mut row, norm_weight, self.profile.f("norm_eps"));
            for c in 0..hidden {
                normalized[c * columns + column] = row[c];
            }
        }
        let projected = if self.profile.modern() {
            matmul_channels(
                self.tensor("rotation_out.weight")?,
                &normalized,
                hidden,
                columns,
                false,
            )?
        } else {
            normalized
        };
        let mut rows: Vec<Vec<f32>> = (0..columns)
            .map(|position| {
                (0..hidden)
                    .map(|c| projected[c * columns + position])
                    .collect()
            })
            .collect();
        stages.emit("rotation_out_and_final_norm_fp32");
        if self.profile.operation() == "embed" {
            let mut vector = if self.profile.modern() {
                rows.remove(0)
            } else {
                rows.pop().unwrap()
            };
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for value in &mut vector {
                *value /= norm;
            }
            stages.emit("cpu_head");
            profile.phase("cpu_tail");
            profile.finish(serde_json::json!({"kind":"row", "tokens":tokens.len(), "shape":shape}));
            return Ok(vector);
        }
        let score = if self.profile.modern() {
            let mut mean = vec![0.0; hidden];
            for row in &rows {
                for c in 0..hidden {
                    mean[c] += row[c] / rows.len() as f32;
                }
            }
            let mut dense = linear_cpu(&mean, self.tensor("head.dense.weight")?, hidden)?;
            for v in &mut dense {
                *v = 0.5 * *v * (1.0 + libm::erff(*v / std::f32::consts::SQRT_2));
            }
            let mean = dense.iter().sum::<f32>() / hidden as f32;
            for v in &mut dense {
                *v -= mean;
            }
            rms_cpu(
                &mut dense,
                Some(self.tensor("head.norm.weight")?),
                self.profile.f("norm_eps"),
            );
            let logit = linear_cpu(&dense, self.tensor("classifier.weight")?, 1)?[0]
                + self.tensor("classifier.bias")?[0];
            1.0 / (1.0 + (-logit).exp())
        } else {
            let row = rows.pop().unwrap();
            let readout = &self.profile.model["grammar"]["readout"];
            let yes = readout["yes"]["id"]
                .as_u64()
                .context("head_tensor_missing")? as usize;
            let no = readout["no"]["id"]
                .as_u64()
                .context("head_tensor_missing")? as usize;
            let dot = |token: usize| -> f32 {
                row.iter()
                    .zip(&embeddings[token * hidden..(token + 1) * hidden])
                    .map(|(a, b)| a * b)
                    .sum()
            };
            1.0 / (1.0 + (dot(no) - dot(yes)).exp())
        };
        stages.emit("cpu_head");
        profile.phase("cpu_tail");
        profile.finish(serde_json::json!({"kind":"row", "tokens":tokens.len(), "shape":shape}));
        Ok(vec![score])
    }
}
fn padded_ids(tokens: &[u32], pad: u32, shape: usize) -> Vec<u32> {
    let mut ids = tokens.to_vec();
    ids.resize(shape, pad);
    ids
}
pub fn padding_mask(tokens: &[u32], pad: u32, shape: usize) -> Vec<f32> {
    let _ = pad;
    (0..shape)
        .map(|i| if i < tokens.len() { 0.0 } else { -10_000.0 })
        .collect()
}
pub fn linear_cpu(input: &[f32], matrix: &[f32], rows: usize) -> Result<Vec<f32>> {
    ensure!(matrix.len() == rows * input.len(), "tensor_shape_mismatch");
    Ok(matrix
        .chunks_exact(input.len())
        .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
        .collect())
}
pub fn rms_cpu(row: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let inv = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + eps)
        .sqrt()
        .recip();
    for (c, v) in row.iter_mut().enumerate() {
        *v *= inv * weight.map_or(1.0, |w| w[c]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placement_requires_exact_layer_cover_and_closed_cpu_set() {
        let p = Profile::select("gte-modernbert-base.ane-direct-worker", "embed").unwrap();
        let mut inventory = p.inventory(128, "test-os").unwrap();
        inventory.executables.pop();
        assert!(validate_inventory(&inventory, 22).is_err());
        let mut inventory = p.inventory(128, "test-os").unwrap();
        inventory.cpu_stages.push("attention".into());
        assert!(validate_inventory(&inventory, 22).is_err());
        let mut inventory = p.inventory(128, "test-os").unwrap();
        inventory.executables[0].layers.push(1);
        assert!(validate_inventory(&inventory, 22).is_err());
    }
    #[test]
    fn compiled_identity_separates_rotation_dtype_operation_shape_and_os() {
        let p = Profile::select("gte-modernbert-base.ane-direct-worker", "embed").unwrap();
        let original = p.key(128, 0, "os-a");
        let mut changed = p.clone();
        changed.numeric["rotation"] = Value::String("none".into());
        assert_ne!(original, changed.key(128, 0, "os-a"));
        changed = p.clone();
        changed.numeric["compute_dtype"] = Value::String("f32".into());
        assert_ne!(original, changed.key(128, 0, "os-a"));
        assert_ne!(original, p.key(256, 0, "os-a"));
        assert_ne!(original, p.key(128, 1, "os-a"));
        assert_ne!(original, p.key(128, 0, "os-b"));
    }
    #[test]
    fn smallest_rung_and_right_padding_mask() {
        for (length, expected) in [(1, 128), (128, 128), (129, 256), (257, 512), (8192, 8192)] {
            assert_eq!(rung(length).unwrap(), expected);
        }
        assert!(rung(8193).is_err());
        assert!(rung(0).is_err());
        let mask = padding_mask(&[10, 11, 0, 12], 0, 128);
        assert_eq!(&mask[..5], &[0.0, 0.0, 0.0, 0.0, -10_000.0]);
        assert!(mask[4..].iter().all(|&v| v == -10_000.0));
    }
    #[test]
    fn load_does_not_compile_and_nonresident_inference_refuses() {
        let profile = Profile::select("gte-modernbert-base.ane-direct-worker", "embed").unwrap();
        let model = Model {
            profile,
            tensors: BTreeMap::new(),
            resident: BTreeMap::new(),
        };
        assert_eq!(
            model.run(&[50281, 50282]).unwrap_err().to_string(),
            "shape_not_admitted"
        );
    }
    #[test]
    #[ignore = "requires four converted packages in ANE_TEST_PACKAGES and private ANE hardware"]
    fn real_weight_short_parity_all_four_profiles() {
        crate::worker::test_private_api().unwrap();
        let root = std::path::PathBuf::from(
            std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"),
        );
        let mut failures = Vec::new();
        for slug in [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ] {
            let operation = if slug.contains("reranker") {
                "rerank"
            } else {
                "embed"
            };
            let p = Profile::select(&format!("{slug}.ane-direct-worker"), operation).unwrap();
            let digest = p.numeric["converted_package_digest"]
                .as_str()
                .unwrap()
                .to_string();
            let mut model =
                Model::load(p, &root.join(format!("{slug}.safetensors")), &digest).unwrap();
            assert!(model.resident.is_empty());
            let fixture: Value = serde_json::from_slice(&std::fs::read(format!("../../bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json")).unwrap()).unwrap();
            let case = &fixture["cases"][0];
            let tokens: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            let shape = rung(tokens.len()).unwrap();
            let start = std::time::Instant::now();
            let inventory = model.admit(shape, "hardware-test").unwrap();
            assert_eq!(inventory, model.admit(shape, "hardware-test").unwrap());
            let warm = std::time::Instant::now();
            let output = model.run(&tokens).unwrap();
            let warm_ms = warm.elapsed().as_secs_f64() * 1000.0;
            if operation == "embed" {
                let expected: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
                let cosine = output
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>()
                    / (output.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                        * expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
                    .sqrt();
                println!(
                    "{slug}: shape={shape} cosine={cosine:.9} warm_ms={warm_ms:.3} total_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
                if cosine < 0.999 {
                    failures.push(format!("{slug} cosine={cosine}"));
                }
            } else {
                let expected = case["output"].as_f64().unwrap();
                let abs = (output[0] as f64 - expected).abs();
                println!("{slug}: shape={shape} score={} expected={expected} abs={abs:.9} warm_ms={warm_ms:.3} total_ms={:.3}", output[0], start.elapsed().as_secs_f64()*1000.0);
                if abs > 0.01 {
                    failures.push(format!("{slug} abs={abs}"));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));
    }
}

fn rotation_center_direction() -> &'static Vec<f32> {
    static DIRECTION: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    DIRECTION.get_or_init(|| {
        // This vector is the manifest's generated Hadamard matrix transposed
        // and applied to an all-ones vector. Subtracting its projection removes
        // the original channel mean without undoing the residual rotation.
        let mut h = [[0i8; 12]; 12];
        for (i, row) in h.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let x = (j as i64 - i as i64).rem_euclid(11);
                let chi = if x == 0 {
                    0
                } else if (1..11).any(|v| v * v % 11 == x) {
                    1
                } else {
                    -1
                };
                let skew = match (i, j) {
                    (0, 0) => 0,
                    (0, _) => 1,
                    (_, 0) => -1,
                    _ => chi,
                };
                *cell = skew + i8::from(i == j);
            }
        }
        let mut matrix = Vec::new();
        let mut direction = vec![0.0f64; 768];
        let scale = 1.0 / 768f64.sqrt();
        for r in 0..768 {
            let mut hash = Sha256::new();
            hash.update(b"modernbert-hadamard-768");
            hash.update(0u64.to_le_bytes());
            hash.update(((r / 256) as u64).to_le_bytes());
            let digest = hash.finalize();
            let bit = r % 256;
            let sign = if (digest[bit / 8] >> (bit % 8)) & 1 == 1 {
                -1
            } else {
                1
            };
            for c in 0..768 {
                let sylvester = if ((r % 64) & (c % 64)).count_ones() % 2 == 0 {
                    1
                } else {
                    -1
                };
                let value = (sign * h[r / 64][c / 64] * sylvester) as f64 * scale;
                matrix.extend_from_slice(&value.to_le_bytes());
                direction[c] += value;
            }
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(matrix)),
            "5cd34f0d01ce33615f987bd62be88b8e595f5083a3c94b42291c053d5cf87d53"
        );
        direction.into_iter().map(|v| v as f32).collect()
    })
}

#[cfg(test)]
mod ladder_hardware {
    use super::*;
    #[test]
    #[ignore = "requires pinned converted packages and private ANE hardware"]
    fn real_weight_admission_all_ladder_shapes() {
        crate::worker::test_private_api().unwrap();
        let root = std::path::PathBuf::from(
            std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"),
        );
        let mut requests = 0;
        for slug in [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ] {
            let operation = if slug.contains("reranker") {
                "rerank"
            } else {
                "embed"
            };
            let profile = Profile::select(&format!("{slug}.ane-direct-worker"), operation).unwrap();
            let digest = profile.numeric["converted_package_digest"]
                .as_str()
                .unwrap()
                .to_string();
            let mut model =
                Model::load(profile, &root.join(format!("{slug}.safetensors")), &digest).unwrap();
            for shape in LADDER {
                model.resident.clear();
                let started = std::time::Instant::now();
                let result = model.admit(shape, "hardware-test");
                println!(
                    "{slug}: ADMIT shape={shape} compile_ms={:.3} outcome={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    result
                        .as_ref()
                        .map(|_| "ADMITTED".into())
                        .unwrap_or_else(|e| format!("{e:#}"))
                );
                result.unwrap();
                requests += 1;
            }
        }
        assert_eq!(requests, 28);
    }
}

#[cfg(test)]
mod fresh_process_hardware {
    use super::*;
    fn model(slug: &str) -> Model {
        crate::worker::test_private_api().unwrap();
        let root = std::path::PathBuf::from(
            std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"),
        );
        let operation = if slug.contains("reranker") {
            "rerank"
        } else {
            "embed"
        };
        let profile = Profile::select(&format!("{slug}.ane-direct-worker"), operation).unwrap();
        let digest = profile.numeric["converted_package_digest"]
            .as_str()
            .unwrap()
            .to_owned();
        Model::load(profile, &root.join(format!("{slug}.safetensors")), &digest).unwrap()
    }
    #[test]
    fn production_shape_limit_refuses_fifth_shape() {
        let profile = Profile::select("gte-modernbert-base.ane-direct-worker", "embed").unwrap();
        let mut model = Model {
            profile,
            tensors: BTreeMap::new(),
            resident: BTreeMap::new(),
        };
        // Empty executables isolate the fifth-shape rejection, which must precede graph construction.
        for shape in [128, 256, 512, 1024] {
            model.resident.insert(
                shape,
                Resident {
                    inventory: model
                        .profile
                        .inventory(shape, "capacity-unit-test")
                        .unwrap(),
                    executables: Vec::new(),
                    a: TensorData::new(modernbert::shape(1, 1)),
                    b: TensorData::new(modernbert::shape(1, 1)),
                    residual: TensorData::new(modernbert::shape(1, 1)),
                    mask: TensorData::new(modernbert::shape(1, 1)),
                },
            );
        }
        assert_eq!(
            model
                .admit(2048, "capacity-unit-test")
                .unwrap_err()
                .to_string(),
            "ane_residency_limit"
        );
        assert_eq!(model.resident.len(), 4);
    }

    #[test]
    #[ignore = "Sequential capacity experiment with pinned weights; run alone in a fresh process"]
    fn fresh_process_residency_capacity() {
        let mix = std::env::var("ANE_CAPACITY_MIX").expect("ANE_CAPACITY_MIX");
        let slugs: &[&str] = match mix.as_str() {
            "gte" | "gte-large" => &["gte-modernbert-base"],
            "qwen" => &["qwen3-embedding-0.6b"],
            "alternating" => &["gte-modernbert-base", "qwen3-embedding-0.6b"],
            "all-four" => &[
                "gte-modernbert-base",
                "gte-reranker-modernbert-base",
                "qwen3-embedding-0.6b",
                "qwen3-reranker-0.6b",
            ],
            _ => panic!("unknown capacity mix"),
        };
        let mut models: Vec<_> = slugs.iter().map(|slug| model(slug)).collect();
        let mut resident = Vec::new();
        let mut resident_mil = 0usize;
        let mut resident_weights = 0usize;
        let mut submitted_mil = 0usize;
        let mut submitted_weights = 0usize;
        let mut resident_executables = 0usize;
        let mut admissions = Vec::new();
        let mut exhausted = false;
        let mut layer_events = Vec::new();
        let ladder: &[usize] = if mix == "gte-large" {
            &[4096, 8192, 128, 256, 512, 1024, 2048]
        } else {
            &[128, 256, 512, 1024, 2048]
        };
        'ladder: for &shape in ladder {
            for (index, model) in models.iter_mut().enumerate() {
                let mut mil_bytes = 0usize;
                let mut weight_bytes = 0usize;
                let mut loaded = 0usize;
                let started = std::time::Instant::now();
                // The ignored capacity probe bypasses the four-shape cap to measure hardware exhaustion.
                let result =
                    model.admit_with_limit(shape, "capacity-development", 64, |graph, _| {
                        let (mil, weights) = graph.source_payload();
                        mil_bytes += mil.len();
                        weight_bytes += weights.len();
                        let result = graph.compile(NSQualityOfService::UserInteractive);
                        layer_events.push(serde_json::json!({"unix_ns": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos().to_string(), "loaded": result.is_ok()}));
                        if result.is_ok() {
                            loaded += 1;
                        }
                        result
                    });
                submitted_mil += mil_bytes;
                submitted_weights += weight_bytes;
                if result.is_ok() {
                    resident.push(serde_json::json!({"model": slugs[index], "length": shape, "executables": loaded, "mil_bytes": mil_bytes, "weight_bytes": weight_bytes}));
                    resident_executables += loaded;
                    resident_mil += mil_bytes;
                    resident_weights += weight_bytes;
                }
                let error = result.err().map(|e| format!("{e:#}"));
                admissions.push(serde_json::json!({"model": slugs[index], "length": shape, "elapsed_ms": started.elapsed().as_secs_f64()*1000.0, "loaded_before_rollback": loaded, "mil_bytes_submitted": mil_bytes, "weight_bytes_submitted": weight_bytes, "error": error}));
                if let Some(error) = error {
                    assert!(
                        error.starts_with("ane_resources_exhausted:"),
                        "unexpected admission failure: {error}"
                    );
                    exhausted = true;
                    break 'ladder;
                }
            }
        }
        let report = serde_json::json!({
            "classification": "development", "mix": mix, "exhausted": exhausted,
            "resident_shapes": resident, "resident_executables": resident_executables,
            "resident_mil_bytes": resident_mil, "resident_weight_bytes": resident_weights,
            "submitted_mil_bytes": submitted_mil, "submitted_weight_bytes": submitted_weights,
            "admissions": admissions, "layer_events": layer_events,
            "accounting": "Compiler source payload, not opaque executable memory. Failed admission's partial executables are rolled back; submitted totals include its failing layer."
        });
        std::fs::write(
            std::env::var_os("ANE_CAPACITY_OUT").expect("ANE_CAPACITY_OUT"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("CAPACITY_RESULT {report}");
        if let Some(release) = std::env::var_os("ANE_CAPACITY_RELEASE") {
            // Hold successful shapes until the paired process reports, preserving simultaneous residency.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
            while !std::path::Path::new(&release).exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "paired-process release deadline"
                );
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        drop(models);
        cleanup_diagnostic_artifacts();
    }

    fn write_experiment(report: serde_json::Value) {
        std::fs::write(
            std::env::var_os("ANE_CAPACITY_OUT").expect("ANE_CAPACITY_OUT"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "Fresh-process unload/reclaim probe with real weights"]
    fn fresh_process_reclaim() {
        let mut model = model("gte-modernbert-base");
        let boundary = std::env::var_os("ANE_RECLAIM_OWNER_EXITED");
        let mut before_exit_error = None;
        if let Some(ref marker) = boundary {
            before_exit_error = model
                .admit_with_limit(128, "reclaim-development", 64, |graph, _| {
                    graph.compile(NSQualityOfService::UserInteractive)
                })
                .err()
                .map(|e| format!("{e:#}"));
            assert!(
                before_exit_error
                    .as_ref()
                    .is_some_and(|e| e.starts_with("ane_resources_exhausted:")),
                "owner must fill hardware before exit control"
            );
            std::fs::write(
                std::env::var_os("ANE_RECLAIM_READY").expect("ANE_RECLAIM_READY"),
                b"ready",
            )
            .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
            while !std::path::Path::new(marker).exists() {
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        } else {
            for shape in [128, 256, 512, 1024, 2048] {
                model
                    .admit_with_limit(shape, "reclaim-development", 64, |graph, _| {
                        graph.compile(NSQualityOfService::UserInteractive)
                    })
                    .unwrap();
            }
            assert_eq!(
                model
                    .resident
                    .values()
                    .map(|s| s.executables.len())
                    .sum::<usize>(),
                110
            );
            drop(model.resident.remove(&128));
        }
        let released = std::time::Instant::now();
        let target = if boundary.is_some() { 128 } else { 4096 };
        let mut attempts = Vec::new();
        for delay_ms in [0u64, 50, 200, 1000, 5000] {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            let actual_start_ms = released.elapsed().as_secs_f64() * 1000.0;
            let mut loaded = 0;
            let result = model.admit_with_limit(target, "reclaim-development", 64, |graph, _| {
                let result = graph.compile(NSQualityOfService::UserInteractive);
                if result.is_ok() {
                    loaded += 1;
                }
                result
            });
            let error = result.err().map(|e| format!("{e:#}"));
            let success = error.is_none();
            attempts.push(serde_json::json!({"delay_after_previous_attempt_ms": delay_ms, "actual_start_after_release_ms": actual_start_ms, "finished_after_release_ms": released.elapsed().as_secs_f64()*1000.0, "loaded_before_rollback": loaded, "error": error}));
            if success {
                break;
            }
        }
        write_experiment(
            serde_json::json!({"classification":"development", "experiment":if boundary.is_some(){"owner-exit-reclaim"}else{"in-process-reclaim"}, "target_length":target, "before_exit_error":before_exit_error, "attempts":attempts, "resident_executables":model.resident.values().map(|s|s.executables.len()).sum::<usize>(), "timing":"Actual times include graph construction and compilation; configured intervals are delays between attempts, not a hardware reclamation clock."}),
        );
        drop(model);
    }

    #[test]
    #[ignore = "Scoped command pools with live retained executables and single-shape eviction"]
    fn fresh_process_scoped_pools_reclaim() {
        let slug = std::env::var("ANE_TEST_MODEL").unwrap_or_else(|_| "gte-modernbert-base".into());
        let method =
            std::env::var("ANE_RECLAIM_PATH").unwrap_or_else(|_| "scoped-single-evict".into());
        let qwen = slug == "qwen3-embedding-0.6b";
        let lengths = if qwen {
            vec![128, 256, 512, 1024]
        } else {
            vec![128, 256, 512, 1024, 2048]
        };
        let replacement = if qwen { 2048 } else { 4096 };
        let mut model = model(&slug);
        for shape in lengths {
            let mut compile = || {
                model.admit_with_limit(shape, "scoped-pools-development", 64, |graph, _| {
                    graph.compile(NSQualityOfService::UserInteractive)
                })
            };
            if method == "scoped-compile-no-pool" {
                compile()
            } else {
                ane::diagnostics::with_autorelease_pool(compile)
            }
            .unwrap();
        }
        let initial_executables = model
            .resident
            .values()
            .map(|resident| resident.executables.len())
            .sum::<usize>();
        assert_eq!(initial_executables, if qwen { 112 } else { 110 });
        let tokens = vec![1u32; 200];
        let before = ane::diagnostics::with_autorelease_pool(|| model.run(&tokens)).unwrap();
        assert!(!before.is_empty());
        assert!(before.iter().all(|value| value.is_finite()));
        let started = std::time::Instant::now();
        if method == "scoped-single-evict-no-pool" {
            drop(model.resident.remove(&128));
        } else {
            ane::diagnostics::with_autorelease_pool(|| drop(model.resident.remove(&128)));
        }
        let offset_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut loaded = 0;
        let mut compile = || {
            model.admit_with_limit(replacement, "scoped-pools-development", 64, |graph, _| {
                let result = graph.compile(NSQualityOfService::UserInteractive);
                if result.is_ok() {
                    loaded += 1;
                }
                result
            })
        };
        let result = if method == "scoped-compile-no-pool" {
            compile()
        } else {
            ane::diagnostics::with_autorelease_pool(compile)
        };
        let error = result.err().map(|error| format!("{error:#}"));
        let after = ane::diagnostics::with_autorelease_pool(|| model.run(&tokens)).unwrap();
        assert_eq!(before,after,"retained sibling must remain executable and byte-identical across scoped-pool releases");
        write_experiment(
            serde_json::json!({"classification":"development","experiment":"in-process-reclaim-paths","method":method,"model":slug,"replacement_shape":replacement,"initial_executables":initial_executables,"observations":["retained sibling runs with finite byte-identical output before and after eviction/new compilation"],"attempts":[{"actual_start_after_reclaim_ms":offset_ms,"finished_after_reclaim_ms":started.elapsed().as_secs_f64()*1000.0,"loaded":loaded,"error":error}],"remaining_fully_resident_executables":model.resident.values().map(|resident|resident.executables.len()).sum::<usize>()}),
        );
    }

    #[test]
    #[ignore = "Bounded autorelease-pool reclamation control with every shape object released"]
    fn fresh_process_autorelease_reclaim() {
        let mut model = model("gte-modernbert-base");
        let mut started = std::time::Instant::now();
        ane::diagnostics::with_autorelease_pool(|| {
            for shape in [128, 256, 512, 1024, 2048] {
                model
                    .admit_with_limit(shape, "autorelease-reclaim-development", 64, |graph, _| {
                        graph.compile(NSQualityOfService::UserInteractive)
                    })
                    .unwrap();
            }
            started = std::time::Instant::now();
            model.resident.clear();
        });
        let mut attempts = Vec::new();
        for delay_ms in [0, 5000] {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            let offset_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut loaded = 0;
            let result = ane::diagnostics::with_autorelease_pool(|| {
                model.admit_with_limit(4096, "autorelease-reclaim-development", 64, |graph, _| {
                    let result = graph.compile(NSQualityOfService::UserInteractive);
                    if result.is_ok() {
                        loaded += 1;
                    }
                    result
                })
            });
            let error = result.err().map(|error| format!("{error:#}"));
            let success = error.is_none();
            attempts.push(serde_json::json!({"actual_start_after_reclaim_ms":offset_ms,"finished_after_reclaim_ms":started.elapsed().as_secs_f64()*1000.0,"delay_after_previous_attempt_ms":delay_ms,"loaded":loaded,"error":error}));
            if success {
                break;
            }
        }
        write_experiment(
            serde_json::json!({"classification":"development","experiment":"in-process-reclaim-paths","method":"autorelease-all","initial_executables":110,"observations":["all shapes dropped and enclosing Objective-C autorelease pool drained before replacement"],"attempts":attempts,"remaining_fully_resident_executables":model.resident.values().map(|resident|resident.executables.len()).sum::<usize>()}),
        );
    }

    #[test]
    #[ignore = "Bounded in-process reclaim alternatives with runtime-enumerated selectors"]
    fn fresh_process_reclaim_paths() {
        let method = std::env::var("ANE_RECLAIM_PATH").unwrap();
        let mut model = model("gte-modernbert-base");
        for shape in [128, 256, 512, 1024, 2048] {
            model
                .admit_with_limit(shape, "reclaim-path-development", 64, |graph, _| {
                    graph.compile(NSQualityOfService::UserInteractive)
                })
                .unwrap();
        }
        let started = std::time::Instant::now();
        let observations = if method == "release-all" {
            model.resident.clear();
            vec!["all resident executable/model objects and IOSurfaces dropped".to_owned()]
        } else {
            let resident = model.resident.remove(&128).unwrap();
            let probe = match method.as_str() {
                "model-purge" => ane::diagnostics::ReclaimProbe::ModelPurge,
                "client-purge" => ane::diagnostics::ReclaimProbe::ClientPurge,
                "fresh-client" => ane::diagnostics::ReclaimProbe::FreshClientUnload,
                "fresh-allocated-client" => {
                    ane::diagnostics::ReclaimProbe::FreshAllocatedClientUnload
                }
                "drop-client-references" => {
                    ane::diagnostics::ReclaimProbe::DropSharedClientReferences
                }
                _ => panic!("unknown reclaim path"),
            };
            let observations = ane::diagnostics::reclaim(resident.executables, probe);
            drop((
                resident.a,
                resident.b,
                resident.residual,
                resident.mask,
                resident.inventory,
            ));
            observations
        };
        let mut attempts = Vec::new();
        for delay_ms in [0, 5000] {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            let offset_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut loaded = 0;
            let result =
                model.admit_with_limit(4096, "reclaim-path-development", 64, |graph, _| {
                    let result = graph.compile(NSQualityOfService::UserInteractive);
                    if result.is_ok() {
                        loaded += 1;
                    }
                    result
                });
            let error = result.err().map(|e| format!("{e:#}"));
            let success = error.is_none();
            attempts.push(serde_json::json!({"delay_after_previous_attempt_ms":delay_ms,"actual_start_after_reclaim_ms":offset_ms,"finished_after_reclaim_ms":started.elapsed().as_secs_f64()*1000.0,"loaded":loaded,"error":error}));
            if success {
                break;
            }
        }
        write_experiment(
            serde_json::json!({"classification":"development","experiment":"in-process-reclaim-paths","method":method,"initial_executables":110,"observations":observations,"attempts":attempts,"remaining_fully_resident_executables":model.resident.values().map(|resident|resident.executables.len()).sum::<usize>()}),
        );
        drop(model);
    }

    #[test]
    #[ignore = "Fresh-process concurrent full-shape and first-layer compiler controls"]
    fn fresh_process_concurrent_compiles() {
        let count: usize = std::env::var("ANE_COMPILE_COUNT").unwrap().parse().unwrap();
        let full = std::env::var_os("ANE_COMPILE_FULL_SHAPE").is_some();
        assert!(if full {
            [2, 4].contains(&count)
        } else {
            [2, 4, 8, 16, 32].contains(&count)
        });
        let model = model("gte-modernbert-base");
        let mut jobs = Vec::new();
        let lengths = [128, 256, 512, 1024];
        let shapes = if full {
            lengths[..count].to_vec()
        } else {
            (1..=count).map(|index| index * 128).collect::<Vec<_>>()
        };
        for shape in shapes {
            // Build graphs before the barrier to isolate concurrent compile/load, not graph construction.
            let graphs = (0..if full { 22 } else { 1 })
                .map(|layer| gte_layer_graph(&model, shape, layer))
                .collect::<Vec<_>>();
            jobs.push((shape, graphs));
        }
        drop(model);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(count));
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for (shape, graphs) in jobs {
            let barrier = barrier.clone();
            let events = events.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let mut executables = Vec::new(); let mut error = None;
                for graph in graphs {
                    let result = graph.compile(NSQualityOfService::UserInteractive);
                    events.lock().unwrap().push(serde_json::json!({"shape":shape,"loaded":result.is_ok(),"unix_ns":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos().to_string()}));
                    match result { Ok(executable) => executables.push(executable), Err(e) => { error=Some(e.to_string()); break; } }
                }
                (shape, executables, error)
            }));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        write_experiment(
            serde_json::json!({"classification":"development", "experiment":"concurrent-compiles", "full_shapes":full, "count":count, "peak_transient_counter":"Private binding exposes no hardware instance counter; events count successful load callbacks retained until all threads finish.", "loaded_total":results.iter().map(|(_,e,_)|e.len()).sum::<usize>(), "results":results.iter().map(|(shape,e,error)|serde_json::json!({"shape":shape,"loaded":e.len(),"error":error})).collect::<Vec<_>>(), "layer_events":*events.lock().unwrap()}),
        );
        drop(results);
    }

    fn gte_layer_graph(model: &Model, shape: usize, layer: usize) -> Graph {
        let hidden = model.profile.n("hidden_size");
        let mut graph = Graph::new();
        let input = graph.placeholder(modernbert::shape(shape, hidden));
        let mask = graph.placeholder(modernbert::shape(shape, 1));
        let residual = if layer == 0 {
            graph.placeholder(modernbert::shape(shape, hidden))
        } else {
            input
        };
        let config: modernbert::Config =
            serde_json::from_value(model.profile.params().clone()).unwrap();
        let base = format!("{}layers.{layer}", model.profile.prefix());
        let linear = |name: &str| modernbert::Linear {
            weight: model
                .tensor(&format!("{base}.{name}.weight"))
                .unwrap()
                .to_vec(),
        };
        let weights = modernbert::LayerWeights {
            qkv: linear("attn.Wqkv"),
            attention_output: linear("attn.Wo"),
            attention_norm: (layer > 0).then(|| vec![1.0; hidden]),
            mlp_input: linear("mlp.Wi"),
            mlp_output: linear("mlp.Wo"),
            mlp_norm: vec![1.0; hidden],
        };
        let _ = modernbert::layer_graph(
            &mut graph, input, residual, mask, &weights, &config, layer, shape,
        );
        graph
    }

    #[test]
    #[ignore = "Nth-load rollback uses real ANE executables and pinned weights"]
    fn nth_load_exhaustion_leaves_shape_fully_absent() {
        let mut model = model("gte-modernbert-base");
        let mut attempts = 0;
        let failure = model
            .admit_with_compiler(128, "rollback-test", |graph, layer| {
                attempts += 1;
                if layer == 3 {
                    Err(ane::Error::Load(
                        "no ANE resources (transient; retry)".into(),
                    ))
                } else {
                    graph.compile(NSQualityOfService::UserInteractive)
                }
            })
            .unwrap_err();
        assert!(failure.to_string().starts_with("ane_resources_exhausted:"));
        assert_eq!(attempts, 4);
        assert!(
            model.resident.is_empty(),
            "partially loaded shape became resident"
        );
        assert_eq!(
            model.run(&[1]).unwrap_err().to_string(),
            "shape_not_admitted"
        );
        let inventory = model.admit(128, "rollback-test").unwrap();
        assert_eq!(inventory.executables.len(), 22);
        assert_eq!(model.resident.len(), 1);
        drop(model);
        cleanup_diagnostic_artifacts();
    }
    fn resident_surface_bytes(model: &Model) {
        for (&shape, resident) in &model.resident {
            let sizes = [
                resident.a.surface().allocationSize(),
                resident.b.surface().allocationSize(),
                resident.residual.surface().allocationSize(),
                resident.mask.surface().allocationSize(),
            ];
            println!("RESIDENT_IO shape={shape} iosurface_bytes={sizes:?} total_bytes={} fp32_scratch_bytes={}", sizes.iter().sum::<isize>(), (3 * model.profile.n("hidden_size") + 1) * shape * 4);
        }
    }
    #[test]
    #[ignore = "fresh-process experiment with real weights; ANE_TEST_SHAPE required"]
    fn fresh_process_gte_single_shape_admission() {
        let mut model = model("gte-modernbert-base");
        assert!(model.resident.is_empty());
        let shape: usize = std::env::var("ANE_TEST_SHAPE").unwrap().parse().unwrap();
        let started = std::time::Instant::now();
        let result = model.admit(shape, "fresh-process-release");
        println!(
            "FRESH_SHAPE shape={shape} prior_resident=[] elapsed_ms={:.3} resident={:?} outcome={}",
            started.elapsed().as_secs_f64() * 1000.0,
            model.resident.keys().collect::<Vec<_>>(),
            result
                .as_ref()
                .map(|_| "ADMITTED".to_owned())
                .unwrap_or_else(|e| format!("{e:#}"))
        );
        resident_surface_bytes(&model);
        let success = result.is_ok();
        drop(result);
        drop(model);
        cleanup_diagnostic_artifacts();
        assert!(success, "shape admission failed; see per-layer diagnostics");
    }
    #[test]
    #[ignore = "quiet-window Qwen capacity measurement with manifest-pinned packages"]
    fn fresh_process_qwen_single_shape_admission() {
        let mut load = [0.0; 3];
        // By default, run this capacity measurement only when the one-minute
        // system load average is below 16, and refuse before creating Neural
        // Engine programs so a busy-host result is not reported as quiet-host
        // evidence. SYNAPSE_BENCH_ALLOW_LOAD=1 runs it anyway for a rough
        // answer; the load is printed either way.
        assert_eq!(unsafe { libc::getloadavg(load.as_mut_ptr(), 3) }, 3);
        println!("QWEN_SHAPE load_before={load:?}");
        let allow_load = std::env::var_os("SYNAPSE_BENCH_ALLOW_LOAD").is_some_and(|v| v == "1");
        assert!(
            allow_load || load[0] < 16.0,
            "quiet-window load gate refused"
        );
        let slug = std::env::var("ANE_TEST_MODEL").expect("ANE_TEST_MODEL");
        assert!(matches!(
            slug.as_str(),
            "qwen3-embedding-0.6b" | "qwen3-reranker-0.6b"
        ));
        let mut model = model(&slug);
        let shape: usize = std::env::var("ANE_TEST_SHAPE")
            .expect("ANE_TEST_SHAPE")
            .parse()
            .unwrap();
        let started = std::time::Instant::now();
        let result = model.admit(shape, "fresh-process-qwen-capacity");
        println!("QWEN_SHAPE model={slug} shape={shape} prior_resident=[] compile_load_ms={:.3} resident={:?} outcome={}", started.elapsed().as_secs_f64() * 1000.0, model.resident.keys().collect::<Vec<_>>(), result.as_ref().map(|_| "ADMITTED".to_owned()).unwrap_or_else(|error| format!("{error:#}")));
        let success = result.is_ok();
        if let Ok(inventory) = &result {
            assert_eq!(inventory.executables.len(), 28);
        }
        drop(result);
        drop(model);
        cleanup_diagnostic_artifacts();
        assert!(
            success,
            "Qwen shape admission failed; do not lower the catalog ceiling"
        );
    }
    #[test]
    #[ignore = "fresh-process single-layer diagnostic with real weights"]
    fn fresh_process_gte_single_layer_load() {
        let model = model("gte-modernbert-base");
        let shape: usize = std::env::var("ANE_TEST_SHAPE").unwrap().parse().unwrap();
        let layer: usize = std::env::var("ANE_TEST_LAYER").unwrap().parse().unwrap();
        let hidden = model.profile.n("hidden_size");
        let mut graph = Graph::new();
        let input = graph.placeholder(modernbert::shape(shape, hidden));
        let mask = graph.placeholder(modernbert::shape(shape, 1));
        let residual = if layer == 0 {
            graph.placeholder(modernbert::shape(shape, hidden))
        } else {
            input
        };
        let config: modernbert::Config =
            serde_json::from_value(model.profile.params().clone()).unwrap();
        let base = format!("{}layers.{layer}", model.profile.prefix());
        let linear = |name: &str| modernbert::Linear {
            weight: model
                .tensor(&format!("{base}.{name}.weight"))
                .unwrap()
                .to_vec(),
        };
        let weights = modernbert::LayerWeights {
            qkv: linear("attn.Wqkv"),
            attention_output: linear("attn.Wo"),
            attention_norm: (layer > 0).then(|| vec![1.0; hidden]),
            mlp_input: linear("mlp.Wi"),
            mlp_output: linear("mlp.Wo"),
            mlp_norm: vec![1.0; hidden],
        };
        let _ = modernbert::layer_graph(
            &mut graph, input, residual, mask, &weights, &config, layer, shape,
        );
        let payload = diagnostic_payload(&graph, shape, layer);
        let started = std::time::Instant::now();
        let result = graph.compile(NSQualityOfService::UserInteractive);
        println!(
            "FRESH_LAYER shape={shape} layer={layer} elapsed_ms={:.3} outcome={}",
            started.elapsed().as_secs_f64() * 1000.0,
            result
                .as_ref()
                .map(|_| "LOADED".to_owned())
                .unwrap_or_else(|e| e.to_string())
        );
        diagnostic_artifact(payload, result.is_ok());
        let success = result.is_ok();
        drop(result);
        cleanup_diagnostic_artifacts();
        assert!(success, "single layer failed; see diagnostics");
    }
    #[test]
    #[ignore = "real-weight 512-token latency; ANE_TEST_MODEL required"]
    fn real_weight_512_warm_latency() {
        let slug = std::env::var("ANE_TEST_MODEL").unwrap();
        let mut model = model(&slug);
        let fixture: Value = serde_json::from_slice(
            &std::fs::read(format!(
                "../../bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| {
                case["input_ids"]
                    .as_array()
                    .is_some_and(|ids| ids.len() == 512)
            })
            .expect("512-token fixture");
        let tokens: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
        let started = std::time::Instant::now();
        model.admit(512, "latency-test").unwrap();
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = std::time::Instant::now();
        let output = model.run(&tokens).unwrap();
        let first_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut times = Vec::new();
        for _ in 0..5 {
            let started = std::time::Instant::now();
            let repeat = model.run(&tokens).unwrap();
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(output, repeat, "nondeterministic warm result");
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let metric = if model.profile.operation() == "embed" {
            let expected: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
            let dot = output
                .iter()
                .zip(&expected)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            let norm = (output.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                * expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
            .sqrt();
            format!("cosine={:.9}", dot / norm)
        } else {
            format!(
                "score={} expected={} abs={:.9}",
                output[0],
                case["output"],
                (output[0] as f64 - case["output"].as_f64().unwrap()).abs()
            )
        };
        println!("LATENCY model={slug} tokens=512 debug_assertions={} compile_ms={compile_ms:.3} first_ms={first_ms:.3} warm_ms={times:?} median_ms={:.3} {metric}", cfg!(debug_assertions), sorted[2]);
        if std::env::var_os("ANE_TRACE_FORWARD").is_some() {
            model.run_stages(&tokens, true).unwrap();
        }
        drop(model);
        cleanup_diagnostic_artifacts();
    }
}

#[cfg(test)]
fn diagnostic_paths() -> &'static std::sync::Mutex<Vec<std::path::PathBuf>> {
    static PATHS: std::sync::OnceLock<std::sync::Mutex<Vec<std::path::PathBuf>>> =
        std::sync::OnceLock::new();
    PATHS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}
#[cfg(test)]
fn diagnostic_payload(graph: &Graph, shape: usize, layer: usize) -> (String, String, usize, usize) {
    let (mil, weights) = graph.source_payload();
    let prefix = format!("{:X}_", Sha256::digest(mil.as_bytes()));
    println!("LAYER_PAYLOAD shape={shape} layer={layer} attention={} mil_bytes={} weight_bytes={} total_bytes={}", if layer.is_multiple_of(3) { "global" } else { "local" }, mil.len(), weights.len(), mil.len()+weights.len());
    (
        prefix,
        format!("{:x}", Sha256::digest(&weights)),
        mil.len(),
        weights.len(),
    )
}
#[cfg(test)]
fn diagnostic_artifact(
    (prefix, weight_digest, mil_bytes, weight_bytes): (String, String, usize, usize),
    loaded: bool,
) {
    let paths: Vec<_> = std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.unwrap();
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix))
                .then(|| entry.path())
                .filter(|path| {
                    std::fs::read(path.join("weights/weight.bin"))
                        .is_ok_and(|bytes| format!("{:x}", Sha256::digest(bytes)) == weight_digest)
                })
        })
        .collect();
    // An injected or pre-emission failure has no artifact to inspect.
    if paths.is_empty() && !loaded {
        return;
    }
    assert_eq!(
        paths.len(),
        1,
        "expected one artifact for this exact MIL/weights payload"
    );
    let path = paths.into_iter().next().unwrap();
    assert_eq!(
        std::fs::metadata(path.join("model.mil")).unwrap().len() as usize,
        mil_bytes
    );
    assert_eq!(
        std::fs::metadata(path.join("weights/weight.bin"))
            .unwrap()
            .len() as usize,
        weight_bytes
    );
    diagnostic_paths().lock().unwrap().push(path);
}
#[cfg(test)]
fn cleanup_diagnostic_artifacts() {
    for path in diagnostic_paths().lock().unwrap().drain(..) {
        match std::fs::remove_dir_all(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("remove diagnostic model artifact: {error}"),
        }
    }
}

struct StageClock {
    enabled: bool,
    last: std::time::Instant,
}
impl StageClock {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last: std::time::Instant::now(),
        }
    }
    fn emit(&mut self, stage: &str) {
        if self.enabled {
            println!(
                "FORWARD_STAGE name={stage} elapsed_ms={:.6}",
                self.last.elapsed().as_secs_f64() * 1000.0
            );
            self.restart();
        }
    }
    fn restart(&mut self) {
        self.last = std::time::Instant::now();
    }
}

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn matmul_channels(
    matrix: &[f32],
    input: &[f32],
    hidden: usize,
    columns: usize,
    transpose: bool,
) -> Result<Vec<f32>> {
    ensure!(
        matrix.len()
            == hidden
                .checked_mul(hidden)
                .context("invalid_rotation_shape")?
            && input.len()
                == hidden
                    .checked_mul(columns)
                    .context("invalid_rotation_shape")?,
        "invalid_rotation_shape"
    );
    let h = i32::try_from(hidden)?;
    let n = i32::try_from(columns)?;
    ensure!(h > 0 && n > 0, "invalid_rotation_shape");
    let mut output = vec![0.0f32; input.len()];
    // Row-major matrices are hidden-by-hidden and hidden-by-columns. The
    // lengths/leading dimensions above ensure BLAS cannot overrun either slice.
    // SGEMM accumulates fp32 on the CPU; no transformer operation moves off ANE.
    unsafe {
        cblas_sgemm(
            101,
            if transpose { 112 } else { 111 },
            111,
            h,
            n,
            h,
            1.0,
            matrix.as_ptr(),
            h,
            input.as_ptr(),
            n,
            0.0,
            output.as_mut_ptr(),
            n,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod cpu_rotation_tests {
    use super::*;
    #[test]
    fn batched_fp32_projection_matches_independent_scalar_rows() {
        let matrix = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 9.0];
        let input = [0.25, -0.5, 2.0, 1.0, 0.125, 4.0];
        for transposed in [false, true] {
            let actual = matmul_channels(&matrix, &input, 3, 2, transposed).unwrap();
            for c in 0..3 {
                for position in 0..2 {
                    let mut expected = 0.0f32;
                    for i in 0..3 {
                        expected += input[i * 2 + position]
                            * matrix[if transposed { i * 3 + c } else { c * 3 + i }];
                    }
                    assert!((actual[c * 2 + position] - expected).abs() < 1e-5);
                }
            }
        }
        assert!(matmul_channels(&matrix[..8], &input, 3, 2, false).is_err());
    }
}

fn admission_error(error: ane::Error, layer: usize) -> anyhow::Error {
    if matches!(&error, ane::Error::Load(message) if message.contains("no ANE resources")) {
        anyhow::anyhow!("ane_resources_exhausted:compile layer {layer}: {error}")
    } else {
        anyhow::Error::new(error).context(format!("compile layer {layer}"))
    }
}

#[cfg(test)]
mod admission_error_tests {
    use super::*;
    #[test]
    fn only_resource_load_failure_gets_exhaustion_code() {
        let transient = admission_error(
            ane::Error::Load("no ANE resources (transient; retry)".into()),
            3,
        );
        assert!(transient
            .to_string()
            .starts_with("ane_resources_exhausted:"));
        for other in [
            ane::Error::Load("another load error".into()),
            ane::Error::Compile("no ANE resources".into()),
            ane::Error::Evaluate("no ANE resources".into()),
        ] {
            assert_eq!(admission_error(other, 3).to_string(), "compile layer 3");
        }
    }
}

#[cfg(test)]
mod padding_golden_tests {
    use super::*;
    #[test]
    fn direct_ane_padding_matches_independent_committed_golden() {
        let fixture: Value =
            serde_json::from_slice(include_bytes!("../tests/fixtures/direct-ane-padding.json"))
                .unwrap();
        assert_eq!(fixture["lane"], "ane-direct-worker");
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 12);
        for case in cases {
            let slug = case["model"].as_str().unwrap();
            let operation = if slug.contains("reranker") {
                "rerank"
            } else {
                "embed"
            };
            let profile = Profile::select(&format!("{slug}.ane-direct-worker"), operation).unwrap();
            let ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            let expected_ids: Vec<u32> =
                serde_json::from_value(case["padded_ids"].clone()).unwrap();
            let expected_mask: Vec<f32> =
                serde_json::from_value(case["additive_mask"].clone()).unwrap();
            let shape = rung(ids.len()).unwrap();
            let pad = profile.n("pad_token_id") as u32;
            assert_eq!(pad as u64, case["pad_id"].as_u64().unwrap());
            assert_eq!(shape as u64, case["shape"].as_u64().unwrap());
            assert_eq!(
                padded_ids(&ids, pad, shape),
                expected_ids,
                "{slug} padded ids"
            );
            assert_eq!(
                padding_mask(&ids, pad, shape),
                expected_mask,
                "{slug} derived mask"
            );
        }
    }
}
