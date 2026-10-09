use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read},
};

use rusqlite::{params, OptionalExtension, Transaction};
use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_core::cache::{ModelCache, ModelCacheError, ModelCacheReadGuard};

use crate::{
    catalog::{self, CatalogEntry, CatalogFile},
    catalog_self_check_id, ModuleError, SynapseStore, SynapseStoreError,
};

// Older releases included roles and backend membership in the manifest digest.
// These fixed keys identify their persisted rows. Recalculating from the live
// catalog would lose those rows as soon as its backend membership changes.
pub(crate) const LEGACY_MANIFEST_DIGESTS: [(&str, &str); 4] = [
    (
        "gte-modernbert-base",
        "9efe05f1170baa48eb1bc6b44c84ff1ea2fbb2c0ae0778e7a9e3a0f493f35c19",
    ),
    (
        "gte-reranker-modernbert-base",
        "f0f4c8f6497d270dd11216be43606e2de0b35935f062c59d842ad220a311720b",
    ),
    (
        "qwen3-embedding-0.6b",
        "ae666f103da54abdb6745b803fe49888aad0352fba2c708c449760fec043de30",
    ),
    (
        "qwen3-reranker-0.6b",
        "b832e032523d2e5b5d94dd5b18dc7af2085ca3d6948e6262fdc35a940c937b39",
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MigrationOutcome {
    Rekeyed,
    Replaced,
    Miss,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ManifestDigestMigration {
    pub catalog_id: String,
    pub backend: String,
    pub outcome: MigrationOutcome,
}

pub(crate) fn migrate_compiled_catalog_digests(
    store: &SynapseStore,
    cache: &ModelCache,
) -> Result<Vec<ManifestDigestMigration>, ModuleError> {
    let entries = &catalog::compiled_catalog()
        .map_err(|error| ModuleError::Config(error.to_string()))?
        .models;
    let results = migrate_legacy_manifest_digests(store, cache, entries, &LEGACY_MANIFEST_DIGESTS)?;
    for result in &results {
        tracing::info!(
            catalog_id = result.catalog_id,
            backend = result.backend,
            outcome = ?result.outcome,
            "catalog manifest digest migration"
        );
    }
    Ok(results)
}

pub(crate) fn migrate_legacy_manifest_digests(
    store: &SynapseStore,
    cache: &ModelCache,
    entries: &[CatalogEntry],
    pre_digests: &[(&str, &str)],
) -> Result<Vec<ManifestDigestMigration>, SynapseStoreError> {
    // The fenced callback is one SQLite transaction, including verification.
    // Keep shared blob leases until commit so cache GC cannot remove verified bytes.
    let mut readers = Vec::new();
    Ok(store.store.with_conn_fenced(|tx| {
        let mut results = Vec::new();
        for entry in entries {
            let Some((_, old_digest)) = pre_digests.iter().find(|(id, _)| *id == entry.id) else {
                continue;
            };
            let new_digest = entry.manifest_digest();
            if *old_digest == new_digest {
                continue;
            }
            let mut groups = tx.prepare(
                "SELECT backend FROM catalog_installs WHERE catalog_id=?1 AND manifest_digest=?2
                 UNION SELECT backend FROM catalog_install_members WHERE catalog_id=?1 AND manifest_digest=?2
                 UNION SELECT backend FROM catalog_file_verifications WHERE catalog_id=?1 AND manifest_digest=?2
                 UNION SELECT backend FROM catalog_self_checks WHERE catalog_id=?1 AND manifest_digest=?2
                 ORDER BY backend",
            )?;
            let backends = groups
                .query_map(params![entry.id, old_digest], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            drop(groups);
            for backend in backends {
                let outcome = if !verify_group(tx, cache, entry, old_digest, &backend, &mut readers)? {
                    MigrationOutcome::Miss
                } else if has_install(tx, &entry.id, &new_digest, &backend)? {
                    delete_group(tx, &entry.id, old_digest, &backend)?;
                    MigrationOutcome::Replaced
                } else {
                    rekey_group(tx, &entry.id, old_digest, &new_digest, &backend)?;
                    MigrationOutcome::Rekeyed
                };
                results.push(ManifestDigestMigration {
                    catalog_id: entry.id.clone(),
                    backend,
                    outcome,
                });
            }
        }
        Ok(results)
    })?)
}

fn has_install(
    tx: &Transaction<'_>,
    catalog_id: &str,
    digest: &str,
    backend: &str,
) -> rusqlite::Result<bool> {
    Ok(tx
        .query_row(
            "SELECT 1 FROM catalog_installs WHERE catalog_id=?1 AND manifest_digest=?2 AND backend=?3",
            params![catalog_id, digest, backend],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn verify_group(
    tx: &Transaction<'_>,
    cache: &ModelCache,
    entry: &CatalogEntry,
    old_digest: &str,
    backend: &str,
    readers: &mut Vec<ModelCacheReadGuard>,
) -> rusqlite::Result<bool> {
    if entry.backend(backend).is_none() || !has_install(tx, &entry.id, old_digest, backend)? {
        return Ok(false);
    }
    let files = entry
        .files
        .iter()
        .filter(|file| file.backends.iter().any(|name| name == backend))
        .collect::<Vec<_>>();
    let mut statement = tx.prepare(
        "SELECT path, digest FROM catalog_install_members
         WHERE catalog_id=?1 AND manifest_digest=?2 AND backend=?3",
    )?;
    let members = statement
        .query_map(params![entry.id, old_digest, backend], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<BTreeMap<_, _>>>()?;
    if files.is_empty()
        || members.len() != files.len()
        || files
            .iter()
            .any(|file| members.get(&file.path) != Some(&file.sha256))
    {
        return Ok(false);
    }
    for file in files {
        let reader = match cache.acquire_read(&file.sha256) {
            Ok(reader) => reader,
            Err(ModelCacheError::NotFound(_)) => return Ok(false),
            Err(ModelCacheError::Io { source, .. }) if source.kind() == io::ErrorKind::NotFound => {
                return Ok(false);
            }
            Err(error) => return Err(sql_error(error)),
        };
        if !verify_file(reader.blob_path(), file)? {
            return Ok(false);
        }
        readers.push(reader);
    }
    Ok(true)
}

fn sql_error(error: impl std::error::Error + Send + Sync + 'static) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(error))
}

fn verify_file(path: &std::path::Path, expected: &CatalogFile) -> rusqlite::Result<bool> {
    let verify = || -> io::Result<bool> {
        let mut file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache blob is not a file",
            ));
        }
        if metadata.len() != expected.size_bytes {
            return Ok(false);
        }
        let mut hash = Sha256::new();
        let mut buffer = [0; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        Ok(hex::encode(hash.finalize()) == expected.sha256)
    };
    match verify() {
        Ok(matches) => Ok(matches),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(sql_error(io::Error::new(
            error.kind(),
            format!("verifying cached catalog file {}: {error}", path.display()),
        ))),
    }
}

fn delete_group(
    tx: &Transaction<'_>,
    catalog_id: &str,
    old_digest: &str,
    backend: &str,
) -> rusqlite::Result<()> {
    for table in [
        "catalog_installs",
        "catalog_install_members",
        "catalog_file_verifications",
        "catalog_self_checks",
    ] {
        tx.execute(
            &format!(
                "DELETE FROM {table} WHERE catalog_id=?1 AND manifest_digest=?2 AND backend=?3"
            ),
            params![catalog_id, old_digest, backend],
        )?;
    }
    Ok(())
}

fn rekey_group(
    tx: &Transaction<'_>,
    catalog_id: &str,
    old_digest: &str,
    new_digest: &str,
    backend: &str,
) -> rusqlite::Result<()> {
    for table in [
        "catalog_installs",
        "catalog_install_members",
        "catalog_file_verifications",
    ] {
        // Member UPDATE does not fire the DELETE cleanup trigger; stamps must
        // be moved explicitly to retain their unchanged-file verification.
        tx.execute(
            &format!("UPDATE {table} SET manifest_digest=?4 WHERE catalog_id=?1 AND manifest_digest=?2 AND backend=?3"),
            params![catalog_id, old_digest, backend, new_digest],
        )?;
    }
    let mut statement = tx.prepare(
        "SELECT check_id, fingerprint, engine_identity_json, os_build, fixture_revision
         FROM catalog_self_checks WHERE catalog_id=?1 AND manifest_digest=?2 AND backend=?3",
    )?;
    let checks = statement
        .query_map(params![catalog_id, old_digest, backend], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    for (old_id, fingerprint, identity, os_build, fixture_revision) in checks {
        let identity: Value = serde_json::from_str(&identity).map_err(sql_error)?;
        let new_id = catalog_self_check_id(
            catalog_id,
            new_digest,
            backend,
            &fingerprint,
            &identity,
            &os_build,
            &fixture_revision,
        )
        .map_err(|error| sql_error(io::Error::new(io::ErrorKind::InvalidData, error)))?;
        let exists = tx
            .query_row(
                "SELECT 1 FROM catalog_self_checks WHERE check_id=?1",
                [&new_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if exists {
            tx.execute(
                "DELETE FROM catalog_self_checks WHERE check_id=?1",
                [&old_id],
            )?;
        } else {
            tx.execute(
                "UPDATE catalog_self_checks SET check_id=?2, manifest_digest=?3 WHERE check_id=?1",
                params![old_id, new_id, new_digest],
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "catalog_migration_tests.rs"]
mod tests;
