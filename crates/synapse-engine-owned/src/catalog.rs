//! Catalog-installed Metal models use the embedded manifest's architecture parameters,
//! not config.json beside the weights, so editing that file cannot change inference.
use crate::{ModelFamily, OwnedDType};
use serde_json::Value;

// Only the Metal load path calls these, and that path is compiled on macOS alone;
// the crate itself still builds on Linux and Windows for its CPU-side code.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn model(profile: &str) -> Result<(ModelFamily, OwnedDType, Value), String> {
    let manifest: Value =
        serde_json::from_slice(include_bytes!("../../../bench/parity/models.json"))
            .map_err(|e| e.to_string())?;
    let entry = manifest["profiles"]
        .get(profile)
        .ok_or("model_unsupported")?;
    if !profile.ends_with(".owned-metal") {
        return Err("model_unsupported".into());
    }
    let slug = profile
        .strip_suffix(".owned-metal")
        .ok_or("model_unsupported")?;
    let model = manifest["models"]
        .get(slug)
        .ok_or("model_unsupported")?
        .clone();
    let family = match model["architecture"]["family"].as_str() {
        Some("modernbert") => ModelFamily::GteModernBert,
        Some("qwen3") => ModelFamily::Qwen3,
        _ => return Err("model_unsupported".into()),
    };
    let dtype = OwnedDType::parse(
        entry["storage_dtype"]
            .as_str()
            .ok_or("missing storage dtype")?,
    )
    .map_err(|e| e.to_string())?;
    Ok((family, dtype, model))
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn config(model: &Value) -> Value {
    let mut config = model["architecture"]["params"].clone();
    config["model_type"] = model["architecture"]["family"].clone();
    config["hidden_activation"] = config["activation"].clone();
    config["rms_norm_eps"] = config["norm_eps"].clone();
    if model["operation"] == "rerank" && model["architecture"]["family"] == "modernbert" {
        config["classifier_pooling"] = Value::from("mean");
        config["classifier_activation"] = config["activation"].clone();
        config["classifier_bias"] = Value::from(false);
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metal_revision_matches_graph_and_bucket() {
        // The shared constant names the base graph revision, which every family and
        // dtype uses except the f16 ModernBERT graph.
        assert_eq!(
            synapse_core::METAL_KERNEL_REVISION,
            format!(
                "owned-metal-graph-{}-bucket-{}",
                crate::GRAPH_REVISION,
                crate::BUCKET_POLICY_VERSION
            )
        );
        assert_eq!(
            crate::metal_kernel_revision(ModelFamily::Qwen3, OwnedDType::F16),
            synapse_core::METAL_KERNEL_REVISION
        );
        assert_eq!(
            crate::metal_kernel_revision(ModelFamily::GteModernBert, OwnedDType::F32),
            synapse_core::METAL_KERNEL_REVISION
        );
        assert_eq!(
            crate::metal_kernel_revision(ModelFamily::GteModernBert, OwnedDType::F16),
            "owned-metal-graph-5-bucket-2"
        );
    }
    #[test]
    fn metal_catalog_precision_is_operation_specific() {
        for (slug, dtype) in [
            ("gte-modernbert-base", OwnedDType::F16),
            ("gte-reranker-modernbert-base", OwnedDType::F32),
            ("qwen3-embedding-0.6b", OwnedDType::F16),
            ("qwen3-reranker-0.6b", OwnedDType::F16),
        ] {
            assert_eq!(model(&format!("{slug}.owned-metal")).unwrap().1, dtype);
        }
    }
}
