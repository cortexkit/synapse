use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::engine::EngineIdentity;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NumericProfileId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Fingerprint(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStrategy {
    Mean,
    Cls,
    LastToken,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationMode {
    None,
    L2,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericDType {
    F16,
    F32,
    Bf16,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionSetting {
    Disabled,
    Enabled,
    Auto,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadPolicyClass {
    Quiet,
    Balanced,
    Performance,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertifiedShapeEnvelope {
    pub max_context_tokens: u32,
    pub max_batch_tokens: u32,
    pub max_micro_batch_tokens: u32,
    pub max_sequences: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumericProfile {
    pub model_digest: String,
    pub quant: String,
    pub engine: EngineIdentity,
    pub sanitized_tokenizer_digest: String,
    pub pooling: PoolingStrategy,
    pub normalization: NormalizationMode,
    pub dtype: NumericDType,
    pub flash_attention: FlashAttentionSetting,
    pub certified_shape: CertifiedShapeEnvelope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_template: Option<String>,
    pub thread_policy: ThreadPolicyClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_grammar: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converted_package_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_profile_digest: Option<String>,
}

impl NumericProfile {
    pub fn stable_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("numeric profile should always serialize")
    }

    pub fn numeric_profile_id(&self) -> NumericProfileId {
        NumericProfileId(sha256_hex(&self.stable_bytes()))
    }

    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint(sha256_hex(
            &serde_json::to_vec(&serde_json::json!([
                self.model_digest,
                self.quant,
                self.numeric_profile_id().0,
            ]))
            .expect("fingerprint payload should serialize"),
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AliasRow {
    #[serde(rename = "fingerprint_a", alias = "left", alias = "left_fingerprint")]
    pub fingerprint_a: Fingerprint,
    #[serde(rename = "fingerprint_b", alias = "right", alias = "right_fingerprint")]
    pub fingerprint_b: Fingerprint,
    #[serde(rename = "valid_from_ms", alias = "valid_from_epoch")]
    pub valid_from_ms: u64,
    #[serde(
        default,
        rename = "valid_to_ms",
        alias = "valid_to_epoch_exclusive",
        skip_serializing_if = "Option::is_none"
    )]
    pub valid_to_ms: Option<u64>,
    #[serde(default = "empty_evidence")]
    pub evidence: Value,
}

impl AliasRow {
    pub fn with_evidence(
        fingerprint_a: Fingerprint,
        fingerprint_b: Fingerprint,
        valid_from_ms: u64,
        valid_to_ms: Option<u64>,
        evidence: Value,
    ) -> Self {
        let (fingerprint_a, fingerprint_b) = canonical_pair(fingerprint_a, fingerprint_b);
        Self {
            fingerprint_a,
            fingerprint_b,
            valid_from_ms,
            valid_to_ms,
            evidence,
        }
    }

    pub fn is_active_at(&self, at_ms: u64) -> bool {
        self.valid_from_ms <= at_ms && self.valid_to_ms.map(|until| at_ms < until).unwrap_or(true)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AliasTable {
    pub table_epoch: u64,
    #[serde(default)]
    pub rows: Vec<AliasRow>,
}

impl AliasTable {
    pub fn equivalent_fingerprints_at(
        &self,
        fingerprint: &Fingerprint,
        at_epoch: u64,
    ) -> BTreeSet<Fingerprint> {
        self.rows
            .iter()
            .filter(|row| row.is_active_at(at_epoch))
            .filter_map(|row| {
                if &row.fingerprint_a == fingerprint {
                    Some(row.fingerprint_b.clone())
                } else if &row.fingerprint_b == fingerprint {
                    Some(row.fingerprint_a.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn check_index(
        &self,
        index_fingerprint: &Fingerprint,
        provenance_set: &BTreeSet<Fingerprint>,
    ) -> AliasCheckVerdict {
        for row in &self.rows {
            if provenance_set.contains(&row.fingerprint_a)
                && provenance_set.contains(&row.fingerprint_b)
                && row.valid_to_ms.is_some()
            {
                return AliasCheckVerdict::MigrationRequired {
                    retracted_pair: RetractedAliasPair {
                        fingerprint_a: row.fingerprint_a.clone(),
                        fingerprint_b: row.fingerprint_b.clone(),
                    },
                    rebuild_target: index_fingerprint.clone(),
                };
            }
        }
        AliasCheckVerdict::Valid
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetractedAliasPair {
    #[serde(rename = "fingerprint_a", alias = "left")]
    pub fingerprint_a: Fingerprint,
    #[serde(rename = "fingerprint_b", alias = "right")]
    pub fingerprint_b: Fingerprint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum AliasCheckVerdict {
    Valid,
    MigrationRequired {
        retracted_pair: RetractedAliasPair,
        rebuild_target: Fingerprint,
    },
}

fn empty_evidence() -> Value {
    Value::Object(Default::default())
}

fn canonical_pair(left: Fingerprint, right: Fingerprint) -> (Fingerprint, Fingerprint) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

/// Formats the digest of an already canonicalized manifest grammar entry.
pub fn input_grammar_identity(canonical_entry: &[u8]) -> String {
    format!("synapse-input-grammar-v1:{}", sha256_hex(canonical_entry))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::engine::EngineIdentity;

    fn sample_profile() -> NumericProfile {
        let mut build_flags = BTreeMap::new();
        build_flags.insert("backend".to_string(), "metal".to_string());
        build_flags.insert("simd".to_string(), "neon".to_string());
        NumericProfile {
            model_digest: "sha256:model".to_string(),
            quant: "f16".to_string(),
            engine: EngineIdentity {
                engine: "llama.cpp".to_string(),
                version: "1.2.3".to_string(),
                build_flags,
            },
            sanitized_tokenizer_digest: "sha256:tok".to_string(),
            pooling: PoolingStrategy::Mean,
            normalization: NormalizationMode::L2,
            dtype: NumericDType::F32,
            flash_attention: FlashAttentionSetting::Enabled,
            certified_shape: CertifiedShapeEnvelope {
                max_context_tokens: 8192,
                max_batch_tokens: 2048,
                max_micro_batch_tokens: 512,
                max_sequences: 16,
            },
            prompt_template: Some("query: {{text}}".to_string()),
            prefix_template: Some("passage: ".to_string()),
            thread_policy: ThreadPolicyClass::Balanced,
            operation: None,
            input_grammar: None,
            kernel_revision: None,
            rotation: None,
            converted_package_digest: None,
            manifest_profile_digest: None,
        }
    }

    #[test]
    fn numeric_profile_id_is_stable_for_identical_inputs() {
        let profile = sample_profile();
        let first = profile.numeric_profile_id();
        let second = sample_profile().numeric_profile_id();
        assert_eq!(first, second);
        assert_eq!(
            first.0,
            "9969360fa4e031b5043b254fb6f9b8a230774077a9b2e3996f46a66266814273"
        );
    }

    fn legacy_metal_profile(rerank: bool) -> NumericProfile {
        let mut profile = sample_profile();
        profile.model_digest = if rerank {
            "sha256:87b3ec4d78a348ed34fb846c931d89f6fa0471e3370fbc4ada54b17c1db8867b"
        } else {
            "sha256:9a63560f30b6bcc9a71239c507575125f91175cc52e5b2cb24f4339f06e43bad"
        }
        .to_string();
        profile.quant = if rerank { "f32" } else { "f16" }.to_string();
        profile.engine = EngineIdentity {
            engine: "owned-metal".to_string(),
            version: "owned-metal-v1".to_string(),
            build_flags: BTreeMap::from([
                ("backend".to_string(), "metal-mpsgraph".to_string()),
                ("bucket_policy".to_string(), "v2".to_string()),
                ("dtype".to_string(), profile.quant.clone()),
                ("family".to_string(), "gte-modernbert".to_string()),
                ("graph_revision".to_string(), "4".to_string()),
                ("risk_class".to_string(), "abort_safe".to_string()),
            ]),
        };
        profile.sanitized_tokenizer_digest =
            "sha256:f64652c0d4292921662f8a34068ed38c9db4a8e78daae667eaff88a30494ef8b".to_string();
        profile.dtype = if rerank {
            NumericDType::F32
        } else {
            NumericDType::F16
        };
        profile.flash_attention = FlashAttentionSetting::Disabled;
        profile.certified_shape = CertifiedShapeEnvelope {
            max_context_tokens: 8192,
            max_batch_tokens: 8192,
            max_micro_batch_tokens: 3072,
            max_sequences: 64,
        };
        profile.prompt_template =
            rerank.then(|| "synapse-rerank-bos-query-sep-doc-eos-v1".to_string());
        profile.prefix_template = None;
        profile
    }

    #[test]
    fn legacy_production_metal_fingerprints_remain_pinned() {
        // Captured profiles from bench/parity/preload pin the fingerprints consumers
        // use to identify stored vectors; adding absent fields must not rotate them.
        assert_eq!(
            legacy_metal_profile(false).fingerprint().0,
            "24cc5271f42dbc2d154f963e2d67d1adef322cbdfbb541b3eddfaa8ae8859dfb"
        );
        assert_eq!(
            legacy_metal_profile(true).fingerprint().0,
            "2fa5f24c0208f30c6db4bf18eb66bc0b2c46f765882cb744fcc58cd080b2b92d"
        );
    }

    #[test]
    fn legacy_coreml_bytes_and_hash_derivation_are_unchanged() {
        // Artifact digests here are sentinels, not a capture of the deployed Core ML package.
        // The ANE preload builder hashes these JSON bytes. Preserve their field order
        // and omit absent catalog fields so existing Core ML vectors keep their identity.
        let legacy = r#"{"model_digest":"sha256:model","quant":"fp16","engine":{"engine":"ane-coreml-worker","version":"protocol-v1","build_flags":{"placement_gate":"neural-engine","risk_class":"abort_capable","transport":"unix-socket-worker"}},"sanitized_tokenizer_digest":"sha256:tok","pooling":"mean","normalization":"l2","dtype":"f16","flash_attention":"disabled","certified_shape":{"max_context_tokens":512,"max_batch_tokens":8192,"max_micro_batch_tokens":3072,"max_sequences":64},"thread_policy":"balanced"}"#;
        let profile: NumericProfile = serde_json::from_str(legacy).unwrap();
        assert_eq!(profile.stable_bytes(), legacy.as_bytes());
        let legacy_id = sha256_hex(legacy.as_bytes());
        assert_eq!(profile.numeric_profile_id().0, legacy_id);
        let legacy_fingerprint = sha256_hex(
            &serde_json::to_vec(&serde_json::json!(["sha256:model", "fp16", legacy_id])).unwrap(),
        );
        assert_eq!(profile.fingerprint().0, legacy_fingerprint);
    }

    #[test]
    fn every_fingerprint_input_changes_identity_independently() {
        let original = sample_profile();
        let fields: &[(&str, fn(&mut NumericProfile))] = &[
            ("model_digest", |p| p.model_digest.push('x')),
            ("quant", |p| p.quant.push('x')),
            ("lane", |p| p.engine.engine.push('x')),
            ("engine_version", |p| p.engine.version.push('x')),
            ("engine_flags", |p| {
                p.engine.build_flags.insert("new".into(), "flag".into());
            }),
            ("tokenizer", |p| p.sanitized_tokenizer_digest.push('x')),
            ("pooling", |p| p.pooling = PoolingStrategy::Cls),
            ("normalization", |p| {
                p.normalization = NormalizationMode::None
            }),
            ("dtype", |p| p.dtype = NumericDType::F16),
            ("flash_attention", |p| {
                p.flash_attention = FlashAttentionSetting::Disabled
            }),
            ("context", |p| p.certified_shape.max_context_tokens += 1),
            ("batch", |p| p.certified_shape.max_batch_tokens += 1),
            ("micro_batch", |p| {
                p.certified_shape.max_micro_batch_tokens += 1
            }),
            ("sequences", |p| p.certified_shape.max_sequences += 1),
            ("prompt", |p| p.prompt_template = None),
            ("prefix", |p| p.prefix_template = None),
            ("thread_policy", |p| {
                p.thread_policy = ThreadPolicyClass::Quiet
            }),
            ("operation", |p| p.operation = Some("embed".into())),
            ("input_grammar", |p| {
                p.input_grammar = Some(input_grammar_identity(b"{}"))
            }),
            ("kernel_revision", |p| {
                p.kernel_revision = Some("revision".into())
            }),
            ("rotation", |p| p.rotation = Some("none".into())),
            ("converted_package_digest", |p| {
                p.converted_package_digest = Some("package".into())
            }),
            ("manifest_profile_digest", |p| {
                p.manifest_profile_digest = Some("manifest".into())
            }),
        ];
        for (name, mutate) in fields {
            let mut changed = original.clone();
            mutate(&mut changed);
            assert_ne!(changed.fingerprint(), original.fingerprint(), "{name}");
        }
    }

    #[test]
    fn grammar_identity_hashes_canonical_entry_and_gelu_changes_profile_digest() {
        assert_eq!(input_grammar_identity(b"{}"),
            "synapse-input-grammar-v1:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a");
        let mut profile = sample_profile();
        profile.manifest_profile_digest = Some(sha256_hex(br#"{"gelu_lowering":"tanh"}"#));
        let before = profile.fingerprint();
        profile.manifest_profile_digest = Some(sha256_hex(br#"{"gelu_lowering":"exact"}"#));
        assert_ne!(before, profile.fingerprint());
    }

    #[test]
    fn coreml_and_direct_ane_have_distinct_fingerprints() {
        let mut profile = sample_profile();
        profile.engine.engine = crate::worker_engine_names::ANE_WORKER_ENGINE.into();
        let coreml = profile.fingerprint();
        profile.engine.engine = crate::worker_engine_names::ANE_DIRECT_WORKER_ENGINE.into();
        assert_ne!(coreml, profile.fingerprint());
    }

    #[test]
    fn alias_validity_queries_detect_mid_flight_retractions() {
        let a = Fingerprint("fp-a".to_string());
        let b = Fingerprint("fp-b".to_string());
        let active = AliasTable {
            table_epoch: 4,
            rows: vec![AliasRow::with_evidence(
                a.clone(),
                b.clone(),
                1,
                None,
                empty_evidence(),
            )],
        };
        assert_eq!(
            active.equivalent_fingerprints_at(&a, 4),
            BTreeSet::from([b.clone()])
        );
        assert_eq!(
            active.check_index(&a, &BTreeSet::from([a.clone(), b.clone()])),
            AliasCheckVerdict::Valid
        );

        let retracted = AliasTable {
            table_epoch: 5,
            rows: vec![AliasRow::with_evidence(
                a.clone(),
                b.clone(),
                1,
                Some(5),
                empty_evidence(),
            )],
        };
        assert_eq!(
            retracted.check_index(&a, &BTreeSet::from([a.clone(), b.clone()])),
            AliasCheckVerdict::MigrationRequired {
                retracted_pair: RetractedAliasPair {
                    fingerprint_a: a.clone(),
                    fingerprint_b: b.clone(),
                },
                rebuild_target: a.clone(),
            }
        );
        assert_eq!(
            retracted.check_index(&a, &BTreeSet::from([a.clone()])),
            AliasCheckVerdict::Valid
        );
    }
}
