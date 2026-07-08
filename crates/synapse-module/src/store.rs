use cortexkit_store::{open_sqlite, Migration, SqliteStore, StoreError};
use cortexkit_store_types::StorageDescriptor;
use rusqlite::params;
use serde::Serialize;
use synapse_core::{AliasRow, Fingerprint};
use thiserror::Error;

const NAMESPACE: &str = "synapse_module";
const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    statements: "CREATE TABLE module_meta (\
                     id INTEGER PRIMARY KEY CHECK (id = 0),\
                     module_generation INTEGER NOT NULL DEFAULT 0,\
                     table_epoch INTEGER NOT NULL DEFAULT 0\
                 );\
                 INSERT OR IGNORE INTO module_meta (id, module_generation, table_epoch) VALUES (0, 0, 0);\
                 CREATE TABLE jobs (\
                     job_id TEXT PRIMARY KEY,\
                     request_key TEXT NOT NULL UNIQUE,\
                     module_generation INTEGER NOT NULL,\
                     state TEXT NOT NULL,\
                     created_ms INTEGER NOT NULL\
                 );\
                 CREATE TABLE alias_rows (\
                     left_fingerprint TEXT NOT NULL,\
                     right_fingerprint TEXT NOT NULL,\
                     valid_from_epoch INTEGER NOT NULL,\
                     valid_to_epoch_exclusive INTEGER,\
                     PRIMARY KEY (left_fingerprint, right_fingerprint, valid_from_epoch)\
                 );\
                 CREATE TABLE cert_rows (\
                     fingerprint TEXT NOT NULL,\
                     machine_profile_hash TEXT NOT NULL,\
                     certified_shape_json TEXT NOT NULL,\
                     created_ms INTEGER NOT NULL,\
                     PRIMARY KEY (fingerprint, machine_profile_hash)\
                 );",
}];

#[derive(Debug, Error)]
pub enum SynapseStoreError {
    #[error("synapse store: {0}")]
    Store(#[from] StoreError),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelCatalogEntry {
    pub model_id: String,
    pub state: String,
    pub fingerprints: Vec<Fingerprint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CatalogSnapshot {
    pub table_epoch: u64,
    pub models: Vec<ModelCatalogEntry>,
    pub alias_rows: Vec<AliasRow>,
}

pub struct SynapseStore {
    store: SqliteStore,
}

impl SynapseStore {
    pub fn open(descriptor: &StorageDescriptor) -> Result<Self, SynapseStoreError> {
        let store = open_sqlite(descriptor)?;
        store.migrate(NAMESPACE, MIGRATIONS)?;
        Ok(Self { store })
    }

    pub fn next_module_generation(&self) -> Result<u64, SynapseStoreError> {
        let next = self.store.with_conn_fenced(|tx| {
            let current: i64 = tx.query_row(
                "SELECT module_generation FROM module_meta WHERE id = 0",
                [],
                |row| row.get(0),
            )?;
            let next = current + 1;
            tx.execute(
                "UPDATE module_meta SET module_generation = ?1 WHERE id = 0",
                params![next],
            )?;
            Ok(next as u64)
        })?;
        Ok(next)
    }

    pub fn catalog_snapshot(&self) -> Result<CatalogSnapshot, SynapseStoreError> {
        let snapshot = self.store.with_conn(|conn| {
            let table_epoch: i64 = conn.query_row(
                "SELECT table_epoch FROM module_meta WHERE id = 0",
                [],
                |row| row.get(0),
            )?;
            let mut stmt = conn.prepare(
                "SELECT left_fingerprint, right_fingerprint, valid_from_epoch, valid_to_epoch_exclusive \
                 FROM alias_rows \
                 ORDER BY left_fingerprint, right_fingerprint, valid_from_epoch",
            )?;
            let alias_rows = stmt
                .query_map([], |row| {
                    Ok(AliasRow::new(
                        Fingerprint(row.get(0)?),
                        Fingerprint(row.get(1)?),
                        row.get::<_, i64>(2)? as u64,
                        row.get::<_, Option<i64>>(3)?.map(|value| value as u64),
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CatalogSnapshot {
                table_epoch: table_epoch as u64,
                models: Vec::new(),
                alias_rows,
            })
        })?;
        Ok(snapshot)
    }
}
