use std::{collections::BTreeMap, fs, path::PathBuf};

use rusqlite::types::Value as SqlValue;
use serde_json::json;

use super::*;
use crate::{sha256_hex, store::DownloadJobRequest};

const TABLES: [&str; 4] = [
    "catalog_installs",
    "catalog_install_members",
    "catalog_file_verifications",
    "catalog_self_checks",
];
const HISTORY: [&str; 3] = ["jobs", "download_key_bindings", "download_acquisitions"];
type Rows = Vec<Vec<SqlValue>>;

struct TempRoot(PathBuf);
impl Drop for TempRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

struct Fixture {
    store: SynapseStore,
    cache: ModelCache,
    entries: Vec<CatalogEntry>,
    old_digests: Vec<String>,
    _root: TempRoot,
}

impl Fixture {
    fn new(label: &str, count: usize) -> Self {
        let (root, descriptor) = crate::tests::test_storage_descriptor(label);
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = ModelCache::new(root.join("cache"));
        fs::create_dir_all(cache.root().join("blobs")).unwrap();
        let mut entries = Vec::new();
        let mut old_digests = Vec::new();
        for index in 0..count {
            let mut entry = catalog::compiled_catalog()
                .unwrap()
                .entry("qwen3-embedding-0.6b")
                .unwrap()
                .clone();
            entry.id = format!("fixture-{index}");
            entry.files.clear();
            for (path, role, backends) in [
                ("ane-model.bin", "model", vec!["ane"]),
                ("metal-model.bin", "model", vec!["metal"]),
                ("tokenizer.json", "tokenizer", vec!["ane", "metal"]),
            ] {
                let bytes = format!("{} {path} fixture bytes", entry.id);
                let sha256 = sha256_hex(bytes.as_bytes());
                fs::write(cache.blob_path(&sha256), bytes.as_bytes()).unwrap();
                entry.files.push(CatalogFile {
                    path: path.into(),
                    role: role.into(),
                    sha256,
                    size_bytes: bytes.len() as u64,
                    backends: backends.into_iter().map(str::to_string).collect(),
                });
            }
            let old = sha256_hex(
                catalog::jcs(&json!({"upstream": entry.upstream, "files": entry.files}))
                    .unwrap()
                    .as_bytes(),
            );
            assert_ne!(old, entry.manifest_digest());
            for backend in ["ane", "metal"] {
                seed_group(&store, &entry, &old, backend);
            }
            entries.push(entry);
            old_digests.push(old);
        }
        seed_history(&store);
        Self {
            store,
            cache,
            entries,
            old_digests,
            _root: TempRoot(root),
        }
    }

    fn run(&self) -> Result<Vec<ManifestDigestMigration>, SynapseStoreError> {
        let keys = self
            .entries
            .iter()
            .zip(&self.old_digests)
            .map(|(entry, digest)| (entry.id.as_str(), digest.as_str()))
            .collect::<Vec<_>>();
        migrate_legacy_manifest_digests(&self.store, &self.cache, &self.entries, &keys)
    }

    fn group(&self, index: usize, digest: &str, backend: &str) -> BTreeMap<String, Rows> {
        TABLES
            .into_iter()
            .map(|table| {
                let rows = self
                    .store
                    .store
                    .with_conn(|conn| {
                        let mut stmt = conn.prepare(&format!(
                            "SELECT * FROM {table} WHERE catalog_id=?1 AND manifest_digest=?2 AND backend=?3 ORDER BY rowid"
                        ))?;
                        let columns = stmt.column_count();
                        let rows = stmt.query_map(params![self.entries[index].id, digest, backend], |row| {
                            (0..columns).map(|column| row.get(column)).collect()
                        })?;
                        rows.collect()
                    })
                    .unwrap();
                (table.into(), rows)
            })
            .collect()
    }
}

fn seed_history(store: &SynapseStore) {
    let params = json!({"catalog_id": "fixture-0", "manifest_digest": "historical-manifest"});
    let admission = store
        .admit_download_job(&DownloadJobRequest {
            request_key: "historical-download-key",
            request_digest: "historical-download-digest",
            module_generation: 1,
            params_json: &params,
            now_ms: 1,
            result_retention_ttl_ms: 100_000,
            entry_complete: false,
        })
        .unwrap();
    store
        .store
        .with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO download_acquisitions VALUES (?1, 'historical-blob', 1)",
                [&admission.record().job_id],
            )?;
            Ok(())
        })
        .unwrap();
}

fn seed_group(store: &SynapseStore, entry: &CatalogEntry, digest: &str, backend: &str) {
    store.store.with_conn_fenced(|tx| {
        tx.execute("INSERT INTO catalog_installs VALUES (?1,?2,?3)", params![entry.id,digest,backend])?;
        for file in entry.files.iter().filter(|file| file.backends.iter().any(|name| name == backend)) {
            tx.execute("INSERT INTO catalog_install_members VALUES (?1,?2,?3,?4,?5)",
                params![entry.id,digest,backend,file.path,file.sha256])?;
            tx.execute("INSERT INTO catalog_file_verifications VALUES (?1,?2,?3,?4,?5,?6,'inode','size','mtime','ctime')",
                params![entry.id,digest,backend,file.path,file.sha256,format!("device-{backend}")])?;
        }
        insert_check(tx, entry, digest, backend, "passed", 7)?;
        Ok(())
    }).unwrap();
}

fn insert_check(
    tx: &Transaction<'_>,
    entry: &CatalogEntry,
    digest: &str,
    backend: &str,
    state: &str,
    generation: i64,
) -> rusqlite::Result<String> {
    let fingerprint = format!("historical-{backend}");
    let identity =
        json!({"engine": "historical-engine", "version": "v1", "nested": {"b": 2, "a": 1}});
    let id = catalog_self_check_id(
        &entry.id,
        digest,
        backend,
        &fingerprint,
        &identity,
        "historical-os",
        "historical-fixture",
    )
    .unwrap();
    tx.execute(
        "INSERT INTO catalog_self_checks VALUES (?1,?2,?3,?4,?5,?6,'historical-os','historical-fixture',?7,?8,123,'historical-reason')",
        params![id,entry.id,digest,backend,fingerprint,serde_json::to_string_pretty(&identity).unwrap(),state,generation],
    )?;
    Ok(id)
}

fn snapshot(store: &SynapseStore, tables: &[&str]) -> BTreeMap<String, Rows> {
    tables
        .iter()
        .map(|table| {
            let rows = store
                .store
                .with_conn(|conn| {
                    let mut stmt =
                        conn.prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))?;
                    let columns = stmt.column_count();
                    let rows = stmt.query_map([], |row| {
                        (0..columns).map(|column| row.get(column)).collect()
                    })?;
                    rows.collect()
                })
                .unwrap();
            (table.to_string(), rows)
        })
        .collect()
}

fn text(value: &SqlValue) -> &str {
    match value {
        SqlValue::Text(text) => text,
        _ => panic!("expected text: {value:?}"),
    }
}

fn expected_rekey(mut group: BTreeMap<String, Rows>, new_digest: &str) -> BTreeMap<String, Rows> {
    for (table, rows) in &mut group {
        for row in rows {
            if table == "catalog_self_checks" {
                let identity: Value = serde_json::from_str(text(&row[5])).unwrap();
                let key = json!({
                    "catalog_id": text(&row[1]), "manifest_digest": new_digest,
                    "backend": text(&row[3]), "fingerprint": text(&row[4]),
                    "engine_identity": identity, "os_build": text(&row[6]),
                    "fixture_revision": text(&row[7]),
                });
                row[0] = SqlValue::Text(sha256_hex(catalog::jcs(&key).unwrap().as_bytes()));
                row[2] = SqlValue::Text(new_digest.into());
            } else {
                row[1] = SqlValue::Text(new_digest.into());
            }
        }
    }
    group
}

fn assert_outcome(
    results: &[ManifestDigestMigration],
    id: &str,
    backend: &str,
    expected: MigrationOutcome,
) {
    let matching = results
        .iter()
        .filter(|r| r.catalog_id == id && r.backend == backend)
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);
    assert_eq!(matching[0].outcome, expected);
}

fn assert_empty(group: &BTreeMap<String, Rows>) {
    assert!(group.values().all(Vec::is_empty), "{group:?}");
}

#[test]
fn migration_rekeys_verified_groups_and_preserves_history_and_second_run() {
    let fixture = Fixture::new("migration-rekeys", 1);
    let history = snapshot(&fixture.store, &HISTORY);
    assert_eq!(history["jobs"].len(), 1);
    let new = fixture.entries[0].manifest_digest();
    let expected = ["ane", "metal"]
        .map(|backend| expected_rekey(fixture.group(0, &fixture.old_digests[0], backend), &new));
    let results = fixture.run().unwrap();
    assert_eq!(results.len(), 2);
    for (backend, expected) in ["ane", "metal"].into_iter().zip(expected) {
        assert_outcome(
            &results,
            &fixture.entries[0].id,
            backend,
            MigrationOutcome::Rekeyed,
        );
        assert_empty(&fixture.group(0, &fixture.old_digests[0], backend));
        assert_eq!(fixture.group(0, &new, backend), expected);
    }
    assert_eq!(snapshot(&fixture.store, &HISTORY), history);
    let before_second = snapshot(&fixture.store, &TABLES);
    assert!(fixture.run().unwrap().is_empty());
    assert_eq!(snapshot(&fixture.store, &TABLES), before_second);
    assert_eq!(snapshot(&fixture.store, &HISTORY), history);
}

#[test]
fn migration_altered_or_missing_shared_blob_misses_only_affected_groups() {
    for missing in [false, true] {
        let fixture = Fixture::new("migration-bad-blob", 2);
        let history = snapshot(&fixture.store, &HISTORY);
        let old_groups =
            ["ane", "metal"].map(|backend| fixture.group(0, &fixture.old_digests[0], backend));
        let blob = fixture.cache.blob_path(&fixture.entries[0].files[2].sha256);
        if missing {
            fs::remove_file(blob).unwrap();
        } else {
            let mut bytes = fs::read(&blob).unwrap();
            bytes[0] ^= 1;
            fs::write(blob, bytes).unwrap();
        }
        let intact_digest = fixture.entries[1].manifest_digest();
        let intact_expected = ["ane", "metal"].map(|backend| {
            expected_rekey(
                fixture.group(1, &fixture.old_digests[1], backend),
                &intact_digest,
            )
        });
        let results = fixture.run().unwrap();
        assert_eq!(results.len(), 4);
        for (backend, old) in ["ane", "metal"].into_iter().zip(old_groups) {
            assert_outcome(
                &results,
                &fixture.entries[0].id,
                backend,
                MigrationOutcome::Miss,
            );
            assert_eq!(fixture.group(0, &fixture.old_digests[0], backend), old);
            assert_empty(&fixture.group(0, &fixture.entries[0].manifest_digest(), backend));
        }
        for (backend, expected) in ["ane", "metal"].into_iter().zip(intact_expected) {
            assert_outcome(
                &results,
                &fixture.entries[1].id,
                backend,
                MigrationOutcome::Rekeyed,
            );
            assert_eq!(fixture.group(1, &intact_digest, backend), expected);
            assert_empty(&fixture.group(1, &fixture.old_digests[1], backend));
        }
        assert_eq!(snapshot(&fixture.store, &HISTORY), history);
    }
}

#[test]
fn migration_size_mismatch_or_inexact_member_set_is_a_miss() {
    for variant in ["size", "missing-member", "extra-member", "wrong-digest"] {
        let fixture = Fixture::new("migration-invalid-group", 1);
        let entry = &fixture.entries[0];
        let digest = &fixture.old_digests[0];
        match variant {
            "size" => fs::write(fixture.cache.blob_path(&entry.files[0].sha256), b"short").unwrap(),
            "missing-member" => {
                fixture.store.store.with_conn_fenced(|tx| {
                    tx.execute("DELETE FROM catalog_install_members WHERE catalog_id=?1 AND backend='ane' AND path='ane-model.bin'", [&entry.id])?;
                    Ok(())
                }).unwrap();
            }
            "extra-member" => {
                fixture.store.store.with_conn_fenced(|tx| {
                    tx.execute("INSERT INTO catalog_install_members VALUES (?1,?2,'ane','extra','unknown')", params![entry.id,digest])?;
                    Ok(())
                }).unwrap();
            }
            "wrong-digest" => {
                fixture.store.store.with_conn_fenced(|tx| {
                    tx.execute("UPDATE catalog_install_members SET digest='wrong' WHERE catalog_id=?1 AND backend='ane' AND path='ane-model.bin'", [&entry.id])?;
                    Ok(())
                }).unwrap();
            }
            _ => unreachable!(),
        }
        let before = fixture.group(0, digest, "ane");
        let results = fixture.run().unwrap();
        assert_outcome(&results, &entry.id, "ane", MigrationOutcome::Miss);
        assert_outcome(&results, &entry.id, "metal", MigrationOutcome::Rekeyed);
        assert_eq!(fixture.group(0, digest, "ane"), before);
        assert_empty(&fixture.group(0, &entry.manifest_digest(), "ane"));
    }
}

#[test]
fn migration_quarantine_orphan_does_not_synthesize_an_install() {
    let fixture = Fixture::new("migration-orphan", 1);
    let entry = &fixture.entries[0];
    fixture.store.store.with_conn_fenced(|tx| {
        tx.execute("DELETE FROM catalog_installs WHERE catalog_id=?1 AND backend='ane'", [&entry.id])?;
        tx.execute("DELETE FROM catalog_install_members WHERE catalog_id=?1 AND backend='ane' AND path='ane-model.bin'", [&entry.id])?;
        Ok(())
    }).unwrap();
    fs::remove_file(fixture.cache.blob_path(&entry.files[0].sha256)).unwrap();
    let orphan = fixture.group(0, &fixture.old_digests[0], "ane");
    assert!(orphan["catalog_installs"].is_empty());
    assert_eq!(orphan["catalog_install_members"].len(), 1);
    assert_eq!(orphan["catalog_self_checks"].len(), 1);
    let new = entry.manifest_digest();
    let metal = expected_rekey(fixture.group(0, &fixture.old_digests[0], "metal"), &new);
    let results = fixture.run().unwrap();
    assert_outcome(&results, &entry.id, "ane", MigrationOutcome::Miss);
    assert_outcome(&results, &entry.id, "metal", MigrationOutcome::Rekeyed);
    assert_eq!(fixture.group(0, &fixture.old_digests[0], "ane"), orphan);
    assert_empty(&fixture.group(0, &new, "ane"));
    assert_eq!(fixture.group(0, &new, "metal"), metal);
    let after = snapshot(&fixture.store, &TABLES);
    let second = fixture.run().unwrap();
    assert_eq!(second.len(), 1);
    assert_outcome(&second, &entry.id, "ane", MigrationOutcome::Miss);
    assert_eq!(snapshot(&fixture.store, &TABLES), after);
}

#[test]
fn migration_current_install_replaces_only_its_legacy_group() {
    let fixture = Fixture::new("migration-install-collision", 1);
    let entry = &fixture.entries[0];
    let new = entry.manifest_digest();
    seed_group(&fixture.store, entry, &new, "ane");
    fixture.store.store.with_conn_fenced(|tx| {
        tx.execute("UPDATE catalog_self_checks SET generation=99, state='failed', reason='newer-repair' WHERE catalog_id=?1 AND manifest_digest=?2", params![entry.id,new])?;
        Ok(())
    }).unwrap();
    let post = fixture.group(0, &new, "ane");
    let metal = expected_rekey(fixture.group(0, &fixture.old_digests[0], "metal"), &new);
    let history = snapshot(&fixture.store, &HISTORY);
    let results = fixture.run().unwrap();
    assert_outcome(&results, &entry.id, "ane", MigrationOutcome::Replaced);
    assert_outcome(&results, &entry.id, "metal", MigrationOutcome::Rekeyed);
    assert_empty(&fixture.group(0, &fixture.old_digests[0], "ane"));
    assert_eq!(fixture.group(0, &new, "ane"), post);
    assert_eq!(fixture.group(0, &new, "metal"), metal);
    assert_eq!(snapshot(&fixture.store, &HISTORY), history);
}

#[test]
fn migration_check_id_collision_preserves_existing_row_without_an_install() {
    let fixture = Fixture::new("migration-check-collision", 1);
    let entry = &fixture.entries[0];
    let new = entry.manifest_digest();
    fixture
        .store
        .store
        .with_conn_fenced(|tx| {
            insert_check(tx, entry, &new, "ane", "failed", 99)?;
            Ok(())
        })
        .unwrap();
    let existing_checks = fixture.group(0, &new, "ane")["catalog_self_checks"].clone();
    let mut expected = expected_rekey(fixture.group(0, &fixture.old_digests[0], "ane"), &new);
    expected.insert("catalog_self_checks".into(), existing_checks);
    let results = fixture.run().unwrap();
    assert_outcome(&results, &entry.id, "ane", MigrationOutcome::Rekeyed);
    assert_eq!(fixture.group(0, &new, "ane"), expected);
    assert_empty(&fixture.group(0, &fixture.old_digests[0], "ane"));
}

#[test]
fn migration_sql_fault_after_first_rekey_rolls_back_every_group() {
    let fixture = Fixture::new("migration-rollback", 1);
    fixture.store.store.with_conn_fenced(|tx| {
        tx.execute_batch("CREATE TRIGGER fail_second_rekey BEFORE UPDATE ON catalog_installs
            WHEN OLD.backend='metal' BEGIN SELECT RAISE(ABORT, 'injected second rekey failure'); END;")?;
        Ok(())
    }).unwrap();
    let before = snapshot(&fixture.store, &TABLES);
    let history = snapshot(&fixture.store, &HISTORY);
    let error = fixture.run().unwrap_err();
    assert!(
        error.to_string().contains("injected second rekey failure"),
        "{error}"
    );
    assert_eq!(snapshot(&fixture.store, &TABLES), before);
    assert_eq!(snapshot(&fixture.store, &HISTORY), history);
    for backend in ["ane", "metal"] {
        assert_empty(&fixture.group(0, &fixture.entries[0].manifest_digest(), backend));
    }
}

#[test]
fn migration_cache_io_error_rolls_back_already_rekeyed_groups() {
    let fixture = Fixture::new("migration-cache-io", 1);
    let path = fixture.cache.blob_path(&fixture.entries[0].files[1].sha256);
    fs::remove_file(&path).unwrap();
    fs::create_dir(path).unwrap();
    let before = snapshot(&fixture.store, &TABLES);
    let history = snapshot(&fixture.store, &HISTORY);
    let error = fixture.run().unwrap_err();
    assert!(
        error.to_string().contains("verifying cached catalog file"),
        "{error}"
    );
    assert_eq!(snapshot(&fixture.store, &TABLES), before);
    assert_eq!(snapshot(&fixture.store, &HISTORY), history);
}

#[test]
fn compiled_migration_empty_cache_reports_qwen3_miss_and_preserves_all_rows() {
    let (root, descriptor) = crate::tests::test_storage_descriptor("compiled-migration-miss");
    let cleanup = TempRoot(root.clone());
    let store = SynapseStore::open(&descriptor).unwrap();
    let cache = ModelCache::new(root.join("empty-cache"));
    let entry = catalog::compiled_catalog()
        .unwrap()
        .entry("qwen3-embedding-0.6b")
        .unwrap();
    let old = "ae666f103da54abdb6745b803fe49888aad0352fba2c708c449760fec043de30";
    seed_group(&store, entry, old, "ane");
    seed_history(&store);
    let before = snapshot(&store, &TABLES);
    let history = snapshot(&store, &HISTORY);
    assert_eq!(history["jobs"].len(), 1);
    let results = migrate_compiled_catalog_digests(&store, &cache).unwrap();
    assert_eq!(results.len(), 1);
    assert_outcome(&results, &entry.id, "ane", MigrationOutcome::Miss);
    assert_eq!(snapshot(&store, &TABLES), before);
    assert_eq!(snapshot(&store, &HISTORY), history);
    drop(cache);
    drop(store);
    drop(cleanup);
}
