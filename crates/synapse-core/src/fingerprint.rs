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
