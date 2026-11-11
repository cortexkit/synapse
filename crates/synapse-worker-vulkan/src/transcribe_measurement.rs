//! Explicit, unpinned storage-precision measurements for the native test only.
use super::*;

pub(super) const F32_PROFILE: &str = "parakeet-tdt-0.6b-v3.owned-vulkan.measurement-f32";
pub(super) const F32_REVISION: &str = "measurement-only-vulkan-transcribe-f32-v1";

pub(super) fn f32_enabled() -> bool {
    match std::env::var("SYNAPSE_VULKAN_SPEECH_PRECISION").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("f16") => false,
        Ok("f32") => {
            #[cfg(not(feature = "transcribe-f32-measurement"))]
            {
                panic!("f32 requires --features transcribe-f32-measurement");
            }
            #[cfg(feature = "transcribe-f32-measurement")]
            {
                true
            }
        }
        other => panic!("unknown measurement precision: {other:?}"),
    }
}

fn profile_copy() -> synapse_parity::manifest::Manifest {
    use synapse_parity::manifest::{AllMarker, DType, Fp32Tensors};
    let mut manifest = crate::manifest();
    let mut profile = manifest.profiles["parakeet-tdt-0.6b-v3.owned-vulkan"].clone();
    profile.fp32_tensors = Fp32Tensors::All(AllMarker::All);
    profile.storage_dtype = DType::F32;
    profile.compute_dtype = DType::F32;
    profile.converted_package_digest = None;
    let maximum = crate::transcribe::maximum(&profile).unwrap();
    let required = crate::transcribe::floors_for_precision(
        &manifest.models[&profile.model],
        maximum,
        crate::transcribe::WeightPrecision::F32,
    )
    .unwrap();
    profile.vulkan_min_device_local_bytes = Some(required.min_device_local_bytes);
    profile.vulkan_min_storage_buffer_range = Some(required.min_storage_buffer_range);
    manifest.profiles.insert(F32_PROFILE.into(), profile);
    manifest
}

#[cfg(feature = "transcribe-f32-measurement")]
pub(super) fn load_f32() -> Result<(Loaded, Value)> {
    let manifest = profile_copy();
    let profile = &manifest.profiles[F32_PROFILE];
    let model = &manifest.models[&profile.model];
    let conversion = Instant::now();
    println!("converting pinned checkpoint for unpinned f32 measurement");
    let bytes = synapse_parity::convert::convert_profile_file(
        &manifest,
        F32_PROFILE,
        &env_path("SYNAPSE_VULKAN_SPEECH_CHECKPOINT"),
    )?;
    let conversion_s = conversion.elapsed().as_secs_f64();
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let load = Instant::now();
    synapse_parity::safetensors::verify_package_structure(&bytes, F32_PROFILE)?;
    let (raw, data) = synapse_parity::safetensors::split_file(&bytes)?;
    let header = parse_header_json(raw)?;
    let maximum = crate::transcribe::maximum(profile)?;
    let required = crate::transcribe::floors_for_precision(
        model,
        maximum,
        crate::transcribe::WeightPrecision::F32,
    )?;
    #[cfg(all(target_os = "macos", feature = "moltenvk-diagnostic"))]
    let prepared =
        crate::runtime::prepare_moltenvk_parity(&env_path("SYNAPSE_MOLTENVK_LOADER"), required)?;
    #[cfg(not(all(target_os = "macos", feature = "moltenvk-diagnostic")))]
    let prepared = crate::runtime::prepare(required)?;
    let engine = crate::runtime::transcribe::Engine::load(
        model, profile, prepared, required, &header, data,
    )?;
    let metadata = json!({"measurement_only":true,"weight_precision":"f32","profile":F32_PROFILE,
        "measurement_kernel_revision":F32_REVISION,"package_sha256":digest,"package_bytes":bytes.len(),
        "checkpoint_sha256":model.files["model.safetensors"],"conversion_s":conversion_s,"load_s":load.elapsed().as_secs_f64(),
        "measurement_profile_digest":manifest.profile_digest(F32_PROFILE)?,"production_load_binding_checked":false});
    Ok((
        Loaded {
            model_ref: digest,
            operation: Operation::Transcribe,
            dims: 0,
            engine: LoadedEngine::Speech(Box::new(engine)),
        },
        metadata,
    ))
}

#[derive(Default, serde::Serialize)]
pub(super) struct GroupTiming {
    pub cases: usize,
    pub feature_preparation_s: f64,
    pub observe_inference_s: f64,
    pub unobserved_inference_s: f64,
}

pub(super) fn group(id: &str, duration_ms: u64, refused: bool) -> &'static str {
    if refused {
        "expected_refusals"
    } else if id.starts_with("ladder-") {
        "ladders"
    } else if duration_ms <= 30_000 {
        "short_clips_le_30s"
    } else {
        "long_clips_gt_30s"
    }
}

pub(super) fn heap_usage(loaded: &Loaded) -> Option<u64> {
    #[cfg(feature = "transcribe-f32-measurement")]
    if let LoadedEngine::Speech(engine) = &loaded.engine {
        return engine.debug_device_heap_usage();
    }
    let _ = loaded;
    None
}

#[test]
fn transcribe_f32_measurement_copy_preserves_production_manifest_and_pins() {
    let original = crate::manifest();
    let copied = profile_copy();
    let id = "parakeet-tdt-0.6b-v3.owned-vulkan";
    assert_eq!(original.profiles[id], copied.profiles[id]);
    assert_eq!(
        original.profile_digest(id).unwrap(),
        copied.profile_digest(id).unwrap()
    );
    assert_eq!(original.profiles.len() + 1, copied.profiles.len());
    let measurement = &copied.profiles[F32_PROFILE];
    assert!(matches!(
        measurement.fp32_tensors,
        synapse_parity::manifest::Fp32Tensors::All(_)
    ));
    assert!(measurement.converted_package_digest.is_none());
    assert_eq!(
        measurement.vulkan_min_device_local_bytes,
        Some(3_276_015_104)
    );
}

#[test]
fn transcribe_measurement_grouping_separates_refusals_ladders_and_short_clips() {
    assert_eq!(group("ladder-8193", 655_440, true), "expected_refusals");
    assert_eq!(group("ladder-1025", 81_930, false), "ladders");
    assert_eq!(group("silence-30s", 30_000, false), "short_clips_le_30s");
    assert_eq!(group("librispeech", 31_000, false), "long_clips_gt_30s");
}
