use super::*;
use synapse_worker_vulkan::{runtime, transcribe};

pub fn load(path: &Path, f32: bool) -> Result<Box<dyn Engine>> {
    let manifest = if f32 {
        vulkan_f32_manifest()?
    } else {
        manifest()?
    };
    let id = if f32 {
        VULKAN_F32_PROFILE
    } else {
        VULKAN_PROFILE
    };
    let profile = &manifest.profiles[id];
    let model = &manifest.models[&profile.model];
    let bytes = std::fs::read(package(path))?;
    if !f32 {
        ensure!(
            profile
                .converted_package_digest
                .as_ref()
                .is_some_and(
                    |pin| pin.trim_start_matches("sha256:") == stt_common::input::sha256(&bytes)
                ),
            "Vulkan F16 package differs from pin"
        );
    }
    synapse_parity::safetensors::verify_package_structure(&bytes, id)?;
    let (raw, data) = synapse_parity::safetensors::split_file(&bytes)?;
    let header = synapse_parity::safetensors::parse_header_json(raw)?;
    let precision = if f32 {
        transcribe::WeightPrecision::F32
    } else {
        transcribe::WeightPrecision::F16
    };
    let expected = synapse_parity::arch::expected_tensors(model)?;
    ensure!(
        expected.len() == header.tensors.len(),
        "unexpected Vulkan tensor count"
    );
    for (name, shape) in expected {
        use synapse_parity::safetensors::StDType;
        let tensor = header
            .tensors
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        let dtype = if name.ends_with("num_batches_tracked") {
            StDType::I64
        } else if f32 {
            StDType::F32
        } else {
            StDType::F16
        };
        ensure!(
            tensor.shape == shape && tensor.dtype == dtype && tensor.data_offsets.1 <= data.len(),
            "invalid tensor {name}"
        );
    }
    let required =
        transcribe::floors_for_precision(model, transcribe::maximum(profile)?, precision)?;
    let prepared = runtime::prepare(required)?;
    let engine =
        runtime::transcribe::Engine::load(model, profile, prepared, required, &header, data)?;
    Ok(Box::new(Vulkan(engine)))
}

struct Vulkan(runtime::transcribe::Engine);

impl Engine for Vulkan {
    fn tokens(&mut self, features: Features) -> Result<Vec<u32>> {
        let output = self.0.transcribe(
            &features.values,
            features.mel_frames,
            features.valid_mel_frames,
            false,
            &mut || Ok(()),
        )?;
        ensure!(
            output.encoder.is_empty() && output.trace.is_empty(),
            "benchmark engine returned observations"
        );
        output
            .triples
            .iter()
            .map(|t| Ok(u32::try_from(t.token_id)?))
            .collect()
    }
}
