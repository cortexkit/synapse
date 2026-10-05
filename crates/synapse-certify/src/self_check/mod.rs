//! Compact load-time references derived from the sealed parity corpus.
use crate::{refuse, Result};
use serde_json::Value;
use synapse_parity::{canonical::sha256_hex, evaluator::FixtureSet, manifest::Manifest};

// These cases exercise short input and both sides of the 128-token shape boundary
// without running the costly 8192 case on load. Rerank also checks a three-item
// pool so the evaluator's existing ranking gate has real observations to grade.
pub const BASE_CASES: [&str; 4] = ["short-0", "boundary-127", "boundary-128", "boundary-129"];
pub const POOL_CASES: [&str; 3] = ["pool10-000", "pool10-001", "pool10-002"];
const INDEX: &[u8] = include_bytes!("index.json");
const SOURCE_INDEX: &[u8] = include_bytes!("../../../../bench/parity/fixtures/index.json");
const MANIFEST: &[u8] = include_bytes!("../../../../bench/parity/models.json");

pub struct References {
    pub manifest: Manifest,
    pub fixtures: FixtureSet,
    pub cases: Vec<Value>,
    pub digest: String,
}
fn bytes(model: &str) -> Result<&'static [u8]> {
    match model {
        "gte-modernbert-base" => Ok(include_bytes!("gte-modernbert-base.json")),
        "gte-reranker-modernbert-base" => Ok(include_bytes!("gte-reranker-modernbert-base.json")),
        "qwen3-embedding-0.6b" => Ok(include_bytes!("qwen3-embedding-0.6b.json")),
        "qwen3-reranker-0.6b" => Ok(include_bytes!("qwen3-reranker-0.6b.json")),
        _ => Err(refuse(format!("self-check references missing for {model}"))),
    }
}
pub fn load(profile: &str) -> Result<References> {
    let manifest = Manifest::from_slice(MANIFEST).map_err(|error| refuse(error.to_string()))?;
    let model = &manifest
        .profiles
        .get(profile)
        .ok_or_else(|| refuse(format!("self-check profile missing: {profile}")))?
        .model;
    load_bytes(profile, &manifest, model, bytes(model)?)
}
fn load_bytes(profile: &str, manifest: &Manifest, model: &str, bytes: &[u8]) -> Result<References> {
    let at = |error| {
        refuse(format!(
            "self-check references invalid for {profile}: {error}"
        ))
    };
    let index: Value = serde_json::from_slice(INDEX).map_err(at)?;
    let source: Value = serde_json::from_slice(SOURCE_INDEX).map_err(at)?;
    let seal = &index[model];
    let fixture_id = synapse_parity::evaluator::fixture_set_id(manifest, model);
    if seal["fixture_set_id"] != fixture_id
        || seal["source_sha256"] != source[&fixture_id]["sha256"]
        || source[&fixture_id]["model"] != model
    {
        return Err(refuse(format!(
            "self-check source seal mismatch for {profile}"
        )));
    }
    let expected = seal["sha256"]
        .as_str()
        .ok_or_else(|| refuse(format!("self-check subset seal missing for {profile}")))?;
    let fixtures = FixtureSet::from_bytes(manifest, model, bytes, expected).map_err(|error| {
        refuse(format!(
            "self-check references invalid for {profile}: {error}"
        ))
    })?;
    let document: Value = serde_json::from_slice(bytes).map_err(at)?;
    let cases = document["cases"]
        .as_array()
        .ok_or_else(|| refuse(format!("self-check cases missing for {profile}")))?
        .clone();
    let expected_cases = BASE_CASES
        .into_iter()
        .chain(
            (manifest
                .model(model)
                .map_err(|error| refuse(error.to_string()))?
                .operation
                == synapse_parity::manifest::Operation::Rerank)
                .then_some(POOL_CASES)
                .into_iter()
                .flatten(),
        )
        .collect::<Vec<_>>();
    if cases
        .iter()
        .map(|case| case["id"].as_str().unwrap_or(""))
        .collect::<Vec<_>>()
        != expected_cases
    {
        return Err(refuse(format!(
            "self-check fixed subset mismatch for {profile}"
        )));
    }
    Ok(References {
        manifest: manifest.clone(),
        fixtures,
        cases,
        digest: sha256_hex(bytes),
    })
}

/// Returns the committed load-time subset's digest, not the full corpus digest.
/// Including it in the stored check key prevents reusing a pass after the subset changes.
pub fn seal_digest(profile: &str) -> Result<String> {
    let manifest = Manifest::from_slice(MANIFEST).map_err(|error| refuse(error.to_string()))?;
    let model = &manifest
        .profiles
        .get(profile)
        .ok_or_else(|| refuse(format!("self-check profile missing: {profile}")))?
        .model;
    let index: Value = serde_json::from_slice(INDEX).map_err(|error| refuse(error.to_string()))?;
    index[model]["sha256"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| refuse(format!("self-check subset seal missing: {profile}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_sixteen_profiles_have_sealed_model_specific_subsets() {
        let manifest = Manifest::from_slice(MANIFEST).unwrap();
        assert_eq!(manifest.profiles.len(), 16);
        for profile in manifest.profiles.keys() {
            assert!(load(profile).is_ok(), "{profile}");
        }
    }
    /// The load-time seal only proves each subset matches its own recorded
    /// digest. This ties the subset to the sealed parity corpus it claims to
    /// come from: every metadata field and every case must be byte-for-byte
    /// what the source fixture holds, so a hand-edited subset (with its digest
    /// re-recorded) can't pass as a parity reference.
    #[test]
    fn subsets_are_exact_copies_of_the_sealed_source_cases() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
        let source_index: Value = serde_json::from_slice(SOURCE_INDEX).unwrap();
        let index: Value = serde_json::from_slice(INDEX).unwrap();
        for model in [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ] {
            let fixture_id = index[model]["fixture_set_id"].as_str().unwrap();
            let entry = &source_index[fixture_id];
            let source_bytes = std::fs::read(root.join(entry["path"].as_str().unwrap())).unwrap();
            assert_eq!(
                sha256_hex(&source_bytes),
                entry["sha256"].as_str().unwrap(),
                "{model}: source fixture digest"
            );
            let source: Value = serde_json::from_slice(&source_bytes).unwrap();
            let subset: Value = serde_json::from_slice(bytes(model).unwrap()).unwrap();
            for (key, value) in subset.as_object().unwrap() {
                if key != "cases" {
                    assert_eq!(value, &source[key], "{model}: field {key}");
                }
            }
            let source_cases = source["cases"].as_array().unwrap();
            for case in subset["cases"].as_array().unwrap() {
                let id = case["id"].as_str().unwrap();
                let original = source_cases
                    .iter()
                    .find(|candidate| candidate["id"] == id)
                    .unwrap_or_else(|| panic!("{model}: case {id} is not in the source corpus"));
                assert_eq!(case, original, "{model}: case {id}");
            }
        }
    }

    #[test]
    fn tampered_subset_digest_refuses_and_names_profile() {
        let profile = "qwen3-reranker-0.6b.owned-metal";
        let manifest = Manifest::from_slice(MANIFEST).unwrap();
        let mut tampered = bytes("qwen3-reranker-0.6b").unwrap().to_vec();
        tampered.push(b' ');
        let error = load_bytes(profile, &manifest, "qwen3-reranker-0.6b", &tampered)
            .err()
            .unwrap();
        assert!(error.to_string().contains("fixture_digest_mismatch"));
        assert!(error.to_string().contains(profile));
    }
}
