#![cfg(target_os = "macos")]

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs;
    use std::io::Read;
    use std::path::PathBuf;
    use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
    use synapse_engine_owned::{ModelFamily, OwnedDType, OwnedMetalEmbedEngine};
    use synapse_parity::evaluator::{evaluate, FixtureSet, ObservedCase, Output};
    use synapse_parity::manifest::{DType, Family, Manifest, Operation};

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }
    fn manifest() -> Manifest {
        Manifest::from_slice(&fs::read(synapse_parity::parity_dir().join("models.json")).unwrap())
            .unwrap()
    }
    fn verify_assets(manifest: &Manifest, slug: &str) -> PathBuf {
        let path = root().join("assets").join(slug);
        for (name, expected) in &manifest.models[slug].files {
            let file = path.join(name);
            assert!(
                !fs::symlink_metadata(&file)
                    .unwrap()
                    .file_type()
                    .is_symlink(),
                "asset must be a dereferenced copy: {}",
                file.display()
            );
            let mut reader = fs::File::open(&file).unwrap();
            let mut hash = Sha256::new();
            let mut buffer = vec![0; 8 * 1024 * 1024];
            loop {
                let n = reader.read(&mut buffer).unwrap();
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                *expected,
                "checkpoint digest mismatch: {slug}/{name}"
            );
        }
        path
    }
    fn load(
        manifest: &Manifest,
        slug: &str,
        catalog: bool,
    ) -> (OwnedMetalEmbedEngine, synapse_core::LoadedModel) {
        let path = verify_assets(manifest, slug);
        let profile_id = format!("{slug}.owned-metal");
        let profile = &manifest.profiles[&profile_id];
        let model = &manifest.models[slug];
        let family = match model.architecture.family {
            Family::Modernbert => ModelFamily::GteModernBert,
            Family::Qwen3 => ModelFamily::Qwen3,
        };
        let dtype = match profile.storage_dtype {
            DType::F16 => OwnedDType::F16,
            DType::F32 => OwnedDType::F32,
        };
        let mut config = RuntimeConfig::default();
        for (key, value) in [
            (
                "model_path",
                path.join("model.safetensors").display().to_string(),
            ),
            (
                "package_cache_root",
                root()
                    .join("packages")
                    .join(if catalog { "catalog" } else { "preload" })
                    .display()
                    .to_string(),
            ),
            ("max_tokens", "8192".into()),
            ("attention_units", "67108864".into()),
            ("execution", "explicit".into()),
        ] {
            config.values.insert(key.into(), value);
        }
        if catalog {
            config.values.insert("profile".into(), profile_id);
            config.values.insert(
                "operation".into(),
                match model.operation {
                    Operation::Embed => "embed",
                    Operation::Rerank => "rerank",
                }
                .into(),
            );
        }
        let mut engine = OwnedMetalEmbedEngine::new(family, dtype);
        let loaded = engine
            .load(
                &ValidatedArtifact {
                    digest: model.checkpoint_digest.clone(),
                    format: "safetensors-package".into(),
                },
                &config,
            )
            .unwrap();
        (engine, loaded)
    }
    fn fixtures(manifest: &Manifest, slug: &str) -> FixtureSet {
        FixtureSet::load(&synapse_parity::parity_dir(), manifest, slug).unwrap()
    }
    fn write_result(name: &str, bytes: &[u8]) {
        fs::create_dir_all(root().join("results")).unwrap();
        fs::write(root().join("results").join(name), bytes).unwrap();
    }

    #[test]
    #[ignore = "requires full Xcode, Metal GPU, and manifest-verified local snapshots"]
    fn catalog_real_weights_pass_committed_evaluator() {
        let manifest = manifest();
        // Check all four snapshots before any weight tensor is loaded.
        for slug in manifest.models.keys() {
            verify_assets(&manifest, slug);
        }
        let selected = std::env::var("METAL_PARITY_MODEL").ok();
        let mut failures = Vec::new();
        for (slug, model) in &manifest.models {
            if selected.as_ref().is_some_and(|s| s != slug) {
                continue;
            }
            let profile_id = format!("{slug}.owned-metal");
            let reference = fixtures(&manifest, slug);
            let (mut engine, loaded) = load(&manifest, slug, true);
            let mut observed = BTreeMap::new();
            for chunk in reference.cases().chunks(8) {
                eprintln!(
                    "METAL_PARITY {slug} cases {}..{}",
                    chunk.first().unwrap().id,
                    chunk.last().unwrap().id
                );
                let sequences = chunk
                    .iter()
                    .map(|c| c.input_ids.clone())
                    .collect::<Vec<_>>();
                let outputs = match model.operation {
                    Operation::Embed => engine
                        .embed_batch(&loaded, TokenBatch { items: sequences })
                        .unwrap()
                        .into_iter()
                        .map(|v| Output::Embedding(v.into_iter().map(f64::from).collect()))
                        .collect::<Vec<_>>(),
                    Operation::Rerank => engine
                        .rerank_pairs(&loaded, sequences)
                        .unwrap()
                        .scores
                        .into_iter()
                        .map(|s| Output::Score(f64::from(s)))
                        .collect(),
                };
                assert_eq!(outputs.len(), chunk.len());
                let readout = model
                    .grammar
                    .readout
                    .yes
                    .as_ref()
                    .zip(model.grammar.readout.no.as_ref())
                    .map(|(y, n)| (y.id, n.id));
                for (case, output) in chunk.iter().zip(outputs) {
                    observed.insert(
                        case.id.clone(),
                        ObservedCase {
                            output,
                            input_ids: case.input_ids.clone(),
                            readout,
                        },
                    );
                }
            }
            let result = evaluate(
                &manifest,
                &profile_id,
                &format!("hardware-check:{}", manifest.digests.profiles[&profile_id]),
                &reference,
                &observed,
            )
            .unwrap();
            let json = serde_json::to_string_pretty(&result).unwrap();
            println!("{json}");
            write_result(&format!("{slug}.evaluation.json"), json.as_bytes());
            if !result.failures.is_empty() {
                failures.push((slug.clone(), result.failures));
            }
            engine.unload(&loaded);
        }
        assert!(
            failures.is_empty(),
            "catalog hardware parity failed: {failures:?}"
        );
    }

    #[test]
    #[ignore = "requires verified GTE reranker weights; writes or compares raw preload logits"]
    fn preload_gte_raw_logits_match_baseline() {
        let manifest = manifest();
        let slug = "gte-reranker-modernbert-base";
        let reference = fixtures(&manifest, slug);
        let (mut engine, loaded) = load(&manifest, slug, false);
        let mut bytes = Vec::new();
        for chunk in reference.cases().chunks(8) {
            let pairs = chunk.iter().map(|c| c.input_ids.clone()).collect();
            for score in engine.rerank_pairs(&loaded, pairs).unwrap().scores {
                bytes.extend_from_slice(&score.to_le_bytes());
            }
        }
        engine.unload(&loaded);
        let label = std::env::var("METAL_PRELOAD_LABEL").unwrap_or_else(|_| "current".into());
        write_result(&format!("preload-{label}.f32le"), &bytes);
        if label == "current" {
            let baseline = fs::read(root().join("evidence/preload-baseline.f32le"))
                .expect("committed master preload baseline");
            assert_eq!(bytes, baseline, "preload logits changed");
            println!(
                "PRELOAD_BYTE_IDENTITY: {} pairs, {} bytes, sha256={:x}",
                reference.cases().len(),
                bytes.len(),
                Sha256::digest(&bytes)
            );
        }
    }
}
