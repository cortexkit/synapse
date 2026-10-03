//! The numeric-profile rule, written once so `models.json` can be checked
//! against it rather than trusted.
//!
//! - Storage dtype: f16 on every lane, except the Metal gte-reranker profile,
//!   which stores and computes in f32. Compute dtype follows storage.
//! - fp32-kept tensors: none on CUDA and Vulkan; every tensor on the f32 Metal
//!   profile and none on the f16 Metal profiles; on direct ANE exactly the
//!   tensors read only by the profile's CPU stages (every transformer layer
//!   runs on the ANE in fp16).
//! - `rerank_tolerance_class` is `fp32` only when every transformer layer
//!   stores and computes in fp32, which leaves the Metal gte-reranker profile
//!   as the only `fp32` profile.
//! - Direct ANE: ModernBERT profiles use the committed Hadamard rotation and
//!   the tanh GELU; Qwen3 profiles pin rotation `none` and have no GELU.

use std::collections::BTreeSet;

use crate::arch::{embedding_key, final_norm_key};
use crate::manifest::{
    AllMarker, DType, Family, Fp32Tensors, Lane, Model, Operation, Profile, ToleranceClass,
};
use crate::{perr, Result};

/// Name of the rotation every direct-ANE ModernBERT profile uses.
pub const MODERNBERT_ROTATION: &str = "modernbert-hadamard-768-v2";
/// Package tensor names of the two runtime rotation projections.
pub const ROTATION_IN_KEY: &str = "rotation_in.weight";
pub const ROTATION_OUT_KEY: &str = "rotation_out.weight";

/// The closed set of stages the direct-ANE lane may run on the CPU.
pub const ANE_CPU_STAGES: [&str; 8] = [
    "token_embedding",
    "mask_position",
    "rotation_in",
    "rotation_out",
    "final_norm",
    "pooling",
    "gte_classifier_head",
    "qwen_yes_no_readout",
];

pub fn storage_dtype(model_slug: &str, lane: Lane) -> DType {
    if lane == Lane::OwnedMetal && model_slug == "gte-reranker-modernbert-base" {
        DType::F32
    } else {
        DType::F16
    }
}

/// CPU stages a direct-ANE profile uses: embedding lookup, mask and position
/// setup, the fp32 final norm and pooling always; the two rotation
/// projections on ModernBERT; and the operation's head.
pub fn ane_cpu_stages(model: &Model) -> Vec<String> {
    let mut stages = vec!["token_embedding", "mask_position"];
    if model.architecture.family == Family::Modernbert {
        stages.extend(["rotation_in", "rotation_out"]);
    }
    stages.extend(["final_norm", "pooling"]);
    match (model.architecture.family, model.operation) {
        (Family::Modernbert, Operation::Rerank) => stages.push("gte_classifier_head"),
        (Family::Qwen3, Operation::Rerank) => stages.push("qwen_yes_no_readout"),
        (_, Operation::Embed) => {}
    }
    stages.into_iter().map(str::to_string).collect()
}

/// Package tensors each CPU stage reads.
pub fn stage_tensors(model: &Model, stage: &str) -> Result<Vec<String>> {
    Ok(match stage {
        "token_embedding" => vec![embedding_key(model)],
        "mask_position" | "pooling" => vec![],
        "rotation_in" => vec![ROTATION_IN_KEY.to_string()],
        "rotation_out" => vec![ROTATION_OUT_KEY.to_string()],
        // On rotated ModernBERT the final norm is parameter-free: its scale is
        // folded into the unrotation projection, so the stage reads nothing.
        "final_norm" => match model.architecture.family {
            Family::Modernbert => vec![],
            Family::Qwen3 => vec![final_norm_key(model)],
        },
        "gte_classifier_head" => {
            let head = model
                .head
                .as_ref()
                .ok_or_else(|| perr!("gte_classifier_head on a model without a head"))?;
            head.tensors
                .values()
                .map(|tensor| tensor.key.clone())
                .collect()
        }
        "qwen_yes_no_readout" => vec![model
            .grammar
            .readout
            .weight
            .clone()
            .ok_or_else(|| perr!("qwen_yes_no_readout on a model without a readout weight"))?],
        other => return Err(perr!("`{other}` is not an allowed direct-ANE CPU stage")),
    })
}

pub fn fp32_tensors(model_slug: &str, model: &Model, lane: Lane) -> Result<Fp32Tensors> {
    Ok(match lane {
        Lane::OwnedCuda | Lane::OwnedVulkan => Fp32Tensors::List(vec![]),
        Lane::OwnedMetal => match storage_dtype(model_slug, lane) {
            DType::F32 => Fp32Tensors::All(AllMarker::All),
            DType::F16 => Fp32Tensors::List(vec![]),
        },
        Lane::AneDirect => {
            let mut set = BTreeSet::new();
            for stage in ane_cpu_stages(model) {
                set.extend(stage_tensors(model, &stage)?);
            }
            Fp32Tensors::List(set.into_iter().collect())
        }
    })
}

pub fn tolerance_class(storage: DType, compute: DType, fp32: &Fp32Tensors) -> ToleranceClass {
    if storage == DType::F32 && compute == DType::F32 && *fp32 == Fp32Tensors::All(AllMarker::All) {
        ToleranceClass::Fp32
    } else {
        ToleranceClass::Fp16
    }
}

/// The profile the rule produces for a model on a lane, minus the values that
/// are measured or computed elsewhere (package digest, Vulkan floors).
pub fn derived_profile(model_slug: &str, model: &Model, lane: Lane) -> Result<Profile> {
    let storage = storage_dtype(model_slug, lane);
    let fp32 = fp32_tensors(model_slug, model, lane)?;
    let modernbert = model.architecture.family == Family::Modernbert;
    Ok(Profile {
        model: model_slug.to_string(),
        lane,
        storage_dtype: storage,
        compute_dtype: storage,
        rerank_tolerance_class: tolerance_class(storage, storage, &fp32),
        fp32_tensors: fp32,
        conversion_rule: if lane.is_worker() { "v1" } else { "none" }.to_string(),
        converted_package_digest: None,
        cpu_stages: (lane == Lane::AneDirect).then(|| ane_cpu_stages(model)),
        rotation: (lane == Lane::AneDirect).then(|| {
            if modernbert {
                MODERNBERT_ROTATION
            } else {
                "none"
            }
            .to_string()
        }),
        gelu_lowering: (lane == Lane::AneDirect)
            .then(|| if modernbert { "tanh" } else { "none" }.to_string()),
        vulkan_sub_batch_max_tokens: None,
        vulkan_min_storage_buffer_range: None,
        vulkan_min_device_local_bytes: None,
        cuda_min_driver_api: if lane == Lane::OwnedCuda {
            crate::manifest::Manifest::from_slice(include_bytes!("../models.json"))?
                .profiles
                .get(&format!("{model_slug}.owned-cuda"))
                .and_then(|p| p.cuda_min_driver_api)
        } else {
            None
        },
        cuda_min_compute_major: if lane == Lane::OwnedCuda {
            crate::manifest::Manifest::from_slice(include_bytes!("../models.json"))?
                .profiles
                .get(&format!("{model_slug}.owned-cuda"))
                .and_then(|p| p.cuda_min_compute_major)
        } else {
            None
        },
        cuda_min_compute_minor: if lane == Lane::OwnedCuda {
            crate::manifest::Manifest::from_slice(include_bytes!("../models.json"))?
                .profiles
                .get(&format!("{model_slug}.owned-cuda"))
                .and_then(|p| p.cuda_min_compute_minor)
        } else {
            None
        },
    })
}

/// Compare a committed profile with the rule. The converted-package digest
/// and the three Vulkan keys are not derived by the rule, so they are copied
/// from the committed profile here; `validate` checks the Vulkan floors
/// against `vulkan::vulkan_floors`, and `parity-manifest reconvert` checks the
/// package digest against a fresh conversion.
pub fn check_profile(
    profile_id: &str,
    model_slug: &str,
    model: &Model,
    profile: &Profile,
) -> Result<()> {
    let mut expected = derived_profile(model_slug, model, profile.lane)?;
    expected.converted_package_digest = profile.converted_package_digest.clone();
    expected.vulkan_sub_batch_max_tokens = profile.vulkan_sub_batch_max_tokens;
    expected.vulkan_min_storage_buffer_range = profile.vulkan_min_storage_buffer_range;
    expected.vulkan_min_device_local_bytes = profile.vulkan_min_device_local_bytes;
    if &expected != profile {
        return Err(perr!(
            "profile `{profile_id}` disagrees with the numeric-profile rule:\n  committed: {}\n  rule:      {}",
            serde_json::to_string(profile).expect("profile serializes"),
            serde_json::to_string(&expected).expect("profile serializes"),
        ));
    }
    Ok(())
}
