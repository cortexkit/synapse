use std::collections::BTreeSet;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasRow {
    pub left: Fingerprint,
    pub right: Fingerprint,
    pub valid_from_epoch: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to_epoch_exclusive: Option<u64>,
}

impl AliasRow {
    pub fn new(
        left: Fingerprint,
        right: Fingerprint,
        valid_from_epoch: u64,
        valid_to_epoch_exclusive: Option<u64>,
    ) -> Self {
        let (left, right) = canonical_pair(left, right);
        Self {
            left,
            right,
            valid_from_epoch,
            valid_to_epoch_exclusive,
        }
    }

    pub fn is_active_at(&self, table_epoch: u64) -> bool {
        self.valid_from_epoch <= table_epoch
            && self
                .valid_to_epoch_exclusive
                .map(|until| table_epoch < until)
                .unwrap_or(true)
    }

    pub fn was_retracted_by(&self, table_epoch: u64) -> bool {
        self.valid_to_epoch_exclusive
            .map(|until| until <= table_epoch)
            .unwrap_or(false)
    }

    pub fn spans(&self, a: &Fingerprint, b: &Fingerprint) -> bool {
        let (a, b) = canonical_pair(a.clone(), b.clone());
        self.left == a && self.right == b
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
                if &row.left == fingerprint {
                    Some(row.right.clone())
                } else if &row.right == fingerprint {
                    Some(row.left.clone())
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
            if provenance_set.contains(&row.left)
                && provenance_set.contains(&row.right)
                && row.was_retracted_by(self.table_epoch)
            {
                return AliasCheckVerdict::MigrationRequired {
                    retracted_pair: RetractedAliasPair {
                        left: row.left.clone(),
                        right: row.right.clone(),
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
    pub left: Fingerprint,
    pub right: Fingerprint,
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

#[derive(Debug, Error)]
pub enum AliasStoreError {
    #[error("sqlite alias store: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

pub trait AliasTableStore {
    fn migrate(&self) -> Result<(), AliasStoreError>;
    fn load_alias_table(&self) -> Result<AliasTable, AliasStoreError>;
    fn store_alias_row(&self, row: &AliasRow) -> Result<(), AliasStoreError>;
    fn set_table_epoch(&self, table_epoch: u64) -> Result<(), AliasStoreError>;
}

pub struct SqliteAliasTableStore<'conn> {
    conn: &'conn Connection,
}

impl<'conn> SqliteAliasTableStore<'conn> {
    pub fn new(conn: &'conn Connection) -> Self {
        Self { conn }
    }
}

impl AliasTableStore for SqliteAliasTableStore<'_> {
    fn migrate(&self) -> Result<(), AliasStoreError> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS alias_table_meta (\
                 id INTEGER PRIMARY KEY CHECK (id = 0),\
                 table_epoch INTEGER NOT NULL\
             );\
             INSERT OR IGNORE INTO alias_table_meta (id, table_epoch) VALUES (0, 0);\
             CREATE TABLE IF NOT EXISTS alias_rows (\
                 left_fingerprint TEXT NOT NULL,\
                 right_fingerprint TEXT NOT NULL,\
                 valid_from_epoch INTEGER NOT NULL,\
                 valid_to_epoch_exclusive INTEGER,\
                 PRIMARY KEY (left_fingerprint, right_fingerprint, valid_from_epoch)\
             );",
        )?;
        Ok(())
    }

    fn load_alias_table(&self) -> Result<AliasTable, AliasStoreError> {
        let table_epoch = self
            .conn
            .query_row(
                "SELECT table_epoch FROM alias_table_meta WHERE id = 0",
                [],
                |row| row.get::<_, i64>(0),
            )? as u64;

        let mut stmt = self.conn.prepare(
            "SELECT left_fingerprint, right_fingerprint, valid_from_epoch, valid_to_epoch_exclusive \
             FROM alias_rows \
             ORDER BY left_fingerprint, right_fingerprint, valid_from_epoch",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(AliasRow {
                    left: Fingerprint(row.get(0)?),
                    right: Fingerprint(row.get(1)?),
                    valid_from_epoch: row.get::<_, i64>(2)? as u64,
                    valid_to_epoch_exclusive: row
                        .get::<_, Option<i64>>(3)?
                        .map(|value| value as u64),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AliasTable { table_epoch, rows })
    }

    fn store_alias_row(&self, row: &AliasRow) -> Result<(), AliasStoreError> {
        let canonical = AliasRow::new(
            row.left.clone(),
            row.right.clone(),
            row.valid_from_epoch,
            row.valid_to_epoch_exclusive,
        );
        self.conn.execute(
            "INSERT INTO alias_rows (left_fingerprint, right_fingerprint, valid_from_epoch, valid_to_epoch_exclusive) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(left_fingerprint, right_fingerprint, valid_from_epoch) DO UPDATE SET \
             valid_to_epoch_exclusive = excluded.valid_to_epoch_exclusive",
            params![
                canonical.left.0,
                canonical.right.0,
                canonical.valid_from_epoch as i64,
                canonical
                    .valid_to_epoch_exclusive
                    .map(|value| value as i64),
            ],
        )?;
        Ok(())
    }

    fn set_table_epoch(&self, table_epoch: u64) -> Result<(), AliasStoreError> {
        self.conn.execute(
            "UPDATE alias_table_meta SET table_epoch = ?1 WHERE id = 0",
            params![table_epoch as i64],
        )?;
        Ok(())
    }
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
            rows: vec![AliasRow::new(a.clone(), b.clone(), 1, Some(5))],
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
            rows: active.rows.clone(),
        };
        assert_eq!(
            retracted.check_index(&a, &BTreeSet::from([a.clone(), b.clone()])),
            AliasCheckVerdict::MigrationRequired {
                retracted_pair: RetractedAliasPair {
                    left: a.clone(),
                    right: b.clone(),
                },
                rebuild_target: a.clone(),
            }
        );
        assert_eq!(
            retracted.check_index(&a, &BTreeSet::from([a.clone()])),
            AliasCheckVerdict::Valid
        );
    }

    #[test]
    fn sqlite_alias_store_round_trips_rows() {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        let store = SqliteAliasTableStore::new(&conn);
        store.migrate().expect("migrate alias schema");
        store.set_table_epoch(7).expect("set table epoch");
        store
            .store_alias_row(&AliasRow::new(
                Fingerprint("fp-a".to_string()),
                Fingerprint("fp-b".to_string()),
                3,
                Some(9),
            ))
            .expect("store alias row");

        let loaded = store.load_alias_table().expect("load alias table");
        assert_eq!(loaded.table_epoch, 7);
        assert_eq!(loaded.rows.len(), 1);
        assert_eq!(loaded.rows[0].left.0, "fp-a");
        assert_eq!(loaded.rows[0].right.0, "fp-b");
        assert_eq!(loaded.rows[0].valid_from_epoch, 3);
        assert_eq!(loaded.rows[0].valid_to_epoch_exclusive, Some(9));
    }
}
