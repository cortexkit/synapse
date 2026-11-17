use super::*;
use crate::store::*;
use cortexkit_store_types::{Isolation, StorageBackend, StorageDescriptor};
#[cfg(unix)]
use rusqlite::params;
#[cfg(unix)]
use std::collections::BTreeSet;

static JOB_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[test]
fn certification_lookup_distinguishes_profiles_with_ane_subtypes() {
    let (root, descriptor) = temp_descriptor("ane-subtype-cert");
    let store = SynapseStore::open(&descriptor).unwrap();
    let fingerprint = Fingerprint("ane-fingerprint".to_string());
    let profile_without_ane = MachineProfile {
        os_build: "23G93".to_string(),
        arch: "aarch64".to_string(),
        chip_model: "Apple M5 Max".to_string(),
        ram_class: "le_64_gib".to_string(),
        ane_subtype: None,
        engine_identities: Vec::new(),
    };
    let mut profile_with_ane = profile_without_ane.clone();
    profile_with_ane.ane_subtype = Some("h17(map)".to_string());
    let without_hash = profile_without_ane.hash();
    let with_hash = profile_with_ane.hash();
    assert_ne!(without_hash, with_hash);

    store
        .store_class_scoped_cert_row(&ClassScopedCertificationRow {
            certification_class: CertificationClass::Embedding,
            assurance_class: AssuranceClass::Measured,
            status: CertificationStatus::Certified,
            key_hash: without_hash.clone(),
            machine_profile_hash: Some(without_hash.clone()),
            remote_profile_hash: None,
            identity_revision: None,
            numeric_profile_id: Some(NumericProfileId("ane-numeric".to_string())),
            fingerprint: fingerprint.clone(),
            certified_at_ms: 10,
            os_build: "23G93".to_string(),
            module_generation: 1,
            evidence: serde_json::json!({"ane_subtype": null}),
        })
        .unwrap();

    assert!(store
        .get_cert_row(CertificationClass::Embedding, &without_hash, &fingerprint)
        .unwrap()
        .is_some());
    assert!(store
        .get_cert_row(CertificationClass::Embedding, &with_hash, &fingerprint)
        .unwrap()
        .is_none());
    assert!(store
        .has_stale_cert_row(CertificationClass::Embedding, &with_hash, &fingerprint)
        .unwrap());
    let refused = crate::ensure_fingerprint_certified(
        &store,
        CertificationClass::Embedding,
        &with_hash,
        &fingerprint,
    )
    .expect_err("a changed ANE profile must require re-probing");
    assert_eq!(refused.code, "not_certified");
    assert!(refused.message.contains("stale certification rows"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn uncertified_reprobe_demotes_the_machine_fingerprint_pair() {
    let (root, descriptor) = temp_descriptor("uncertified-reprobe");
    let store = SynapseStore::open(&descriptor).unwrap();
    let fingerprint = Fingerprint("decode-fp".to_string());
    let mut row = ClassScopedCertificationRow {
        certification_class: CertificationClass::Embedding,
        assurance_class: AssuranceClass::Measured,
        status: CertificationStatus::Certified,
        key_hash: "machine-a".to_string(),
        machine_profile_hash: Some("machine-a".to_string()),
        remote_profile_hash: None,
        identity_revision: None,
        numeric_profile_id: Some(NumericProfileId("decode-np".to_string())),
        fingerprint: fingerprint.clone(),
        certified_at_ms: 10,
        os_build: "os-a".to_string(),
        module_generation: 1,
        evidence: serde_json::json!({"gate": "token_exact"}),
    };
    store.store_class_scoped_cert_row(&row).unwrap();
    assert!(store
        .get_cert_row(CertificationClass::Embedding, "machine-a", &fingerprint)
        .unwrap()
        .is_some());

    row.status = CertificationStatus::Uncertified;
    row.certified_at_ms = 20;
    row.evidence = serde_json::json!({
        "blocking_reason": "token_mismatch",
        "mismatches": [{"prompt": "corrupted fixture"}],
    });
    store.store_class_scoped_cert_row(&row).unwrap();

    assert!(store
        .get_cert_row(CertificationClass::Embedding, "machine-a", &fingerprint)
        .unwrap()
        .is_none());
    let probe = store
        .get_probe_row(CertificationClass::Embedding, "machine-a", &fingerprint)
        .unwrap()
        .unwrap();
    assert_eq!(probe.status, CertificationStatus::Uncertified);
    assert_eq!(probe.evidence, row.evidence);
    let refused = crate::ensure_fingerprint_certified(
        &store,
        CertificationClass::Embedding,
        "machine-a",
        &fingerprint,
    )
    .expect_err("an uncertified decode outcome must refuse future owned requests");
    assert_eq!(refused.code, "not_certified");
    assert!(refused.message.contains("token_mismatch"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn embedding_certification_cannot_certify_rerank_with_shared_fingerprint() {
    let (root, descriptor) = temp_descriptor("cross-class-certification");
    let store = SynapseStore::open(&descriptor).unwrap();
    let fingerprint = Fingerprint("shared-fingerprint".to_string());
    store
        .store_class_scoped_cert_row(&ClassScopedCertificationRow {
            certification_class: CertificationClass::Embedding,
            assurance_class: AssuranceClass::Measured,
            status: CertificationStatus::Certified,
            key_hash: "legacy-profile-hash".to_string(),
            machine_profile_hash: Some("legacy-profile-hash".to_string()),
            remote_profile_hash: None,
            identity_revision: None,
            numeric_profile_id: Some(NumericProfileId("embedding-profile".to_string())),
            fingerprint: fingerprint.clone(),
            certified_at_ms: 1,
            os_build: "test-os".to_string(),
            module_generation: 1,
            evidence: serde_json::json!({"task": "embed"}),
        })
        .unwrap();
    assert!(store
        .get_cert_row(
            CertificationClass::Embedding,
            "legacy-profile-hash",
            &fingerprint,
        )
        .unwrap()
        .is_some());
    assert!(store
        .get_cert_row(
            CertificationClass::Rerank,
            "legacy-profile-hash",
            &fingerprint,
        )
        .unwrap()
        .is_none());
    let refused = crate::ensure_fingerprint_certified(
        &store,
        CertificationClass::Rerank,
        "legacy-profile-hash",
        &fingerprint,
    )
    .expect_err("embedding evidence must not admit a rerank lane");
    assert_eq!(refused.code, "not_certified");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn legacy_hash_keeps_pre_upgrade_non_owned_certification_visible() {
    let (root, descriptor) = temp_descriptor("legacy-hash-upgrade");
    let store = SynapseStore::open(&descriptor).unwrap();
    let profile = MachineProfile {
        os_build: "test-os".to_string(),
        arch: "aarch64".to_string(),
        chip_model: "test-chip".to_string(),
        ram_class: "test-ram".to_string(),
        ane_subtype: None,
        engine_identities: Vec::new(),
    };
    let (legacy_hash, revisioned_hash) = crate::module_state_machine_profile_hashes(&profile);
    assert_ne!(legacy_hash, revisioned_hash);
    let fingerprint = Fingerprint("pre-upgrade-fingerprint".to_string());
    store
        .store_class_scoped_cert_row(&ClassScopedCertificationRow {
            certification_class: CertificationClass::Embedding,
            assurance_class: AssuranceClass::Measured,
            status: CertificationStatus::Certified,
            key_hash: legacy_hash.clone(),
            machine_profile_hash: Some(legacy_hash.clone()),
            remote_profile_hash: None,
            identity_revision: None,
            numeric_profile_id: Some(NumericProfileId("pre-upgrade".to_string())),
            fingerprint: fingerprint.clone(),
            certified_at_ms: 1,
            os_build: profile.os_build,
            module_generation: 1,
            evidence: serde_json::json!({"source": "pre-upgrade"}),
        })
        .unwrap();
    assert!(store
        .get_cert_row(CertificationClass::Embedding, &legacy_hash, &fingerprint)
        .unwrap()
        .is_some());
    assert!(store
        .get_cert_row(
            CertificationClass::Embedding,
            &revisioned_hash,
            &fingerprint,
        )
        .unwrap()
        .is_none());
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
fn verification_fixture() -> (
    std::path::PathBuf,
    SynapseStore,
    CatalogInstallRecord,
    std::path::PathBuf,
    String,
) {
    let (root, descriptor) = temp_descriptor("catalog-verification");
    let store = SynapseStore::open(&descriptor).unwrap();
    let path = root.join("weights");
    std::fs::write(&path, b"original").unwrap();
    let digest = crate::sha256_file(&path).unwrap();
    let install = CatalogInstallRecord {
        catalog_id: "model".into(),
        manifest_digest: "manifest".into(),
        backend: "metal".into(),
        members: vec![CatalogInstallMember {
            path: "weights".into(),
            digest: digest.clone(),
        }],
    };
    store.sqlite().with_conn_fenced(|tx| {
            tx.execute("INSERT INTO catalog_installs VALUES ('model','manifest','metal')", [])?;
            tx.execute("INSERT INTO catalog_install_members VALUES ('model','manifest','metal','weights',?1)", params![digest])?;
            Ok(())
        }).unwrap();
    (root, store, install, path, digest)
}

#[cfg(unix)]
#[test]
fn catalog_verification_unchanged_cold_load_skips_hash_and_removal_clears_row() {
    let (root, store, install, path, digest) = verification_fixture();
    let calls = std::cell::Cell::new(0);
    for _ in 0..2 {
        let result =
            crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
                calls.set(calls.get() + 1);
                crate::sha256_file(p).ok()
            })
            .unwrap();
        assert_eq!(result.as_deref(), Some(digest.as_str()));
    }
    assert_eq!(calls.get(), 1);
    assert_eq!(table_count(&store, "catalog_file_verifications"), 1);
    store
        .sqlite()
        .with_conn_fenced(|tx| {
            tx.execute("DELETE FROM catalog_install_members", [])?;
            Ok(())
        })
        .unwrap();
    assert_eq!(table_count(&store, "catalog_file_verifications"), 0);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn catalog_verification_touch_forces_rehash() {
    let (root, store, install, path, digest) = verification_fixture();
    crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
        crate::sha256_file(p).ok()
    })
    .unwrap();
    let old = std::fs::metadata(&path).unwrap().modified().unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(old + std::time::Duration::from_secs(1))
        .unwrap();
    let calls = std::cell::Cell::new(0);
    let result =
        crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
            calls.set(calls.get() + 1);
            crate::sha256_file(p).ok()
        })
        .unwrap();
    assert_eq!(
        calls.get(),
        1,
        "a changed file identity reused verification"
    );
    assert_eq!(result.as_deref(), Some(digest.as_str()));
    std::fs::remove_dir_all(root).unwrap();
}

/// A file that changes while it is being hashed can't vouch for the bytes
/// that are loaded afterwards. The hash closure here returns the expected
/// digest (as it would if it read the file before the write) but bumps the
/// file's mtime mid-hash. The result must not count as verified, and no
/// verification row may be recorded, so the next load hashes in full.
#[cfg(unix)]
#[test]
fn catalog_verification_file_changed_during_hash_is_not_recorded() {
    let (root, store, install, path, digest) = verification_fixture();
    let result =
        crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
            let old = std::fs::metadata(p).unwrap().modified().unwrap();
            std::fs::File::options()
                .write(true)
                .open(p)
                .unwrap()
                .set_modified(old + std::time::Duration::from_secs(1))
                .unwrap();
            Some(digest.clone())
        })
        .unwrap();
    assert_eq!(result, None, "a file changed mid-hash must not verify");
    let calls = std::cell::Cell::new(0);
    crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
        calls.set(calls.get() + 1);
        crate::sha256_file(p).ok()
    })
    .unwrap();
    assert_eq!(calls.get(), 1, "no verification row may have been recorded");
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn catalog_verification_content_change_with_restored_mtime_is_refused() {
    let (root, store, install, path, digest) = verification_fixture();
    crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
        crate::sha256_file(p).ok()
    })
    .unwrap();
    let old = std::fs::metadata(&path).unwrap().modified().unwrap();
    std::fs::write(&path, b"tampered").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(old)
        .unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), old);
    let calls = std::cell::Cell::new(0);
    let result =
        crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
            calls.set(calls.get() + 1);
            crate::sha256_file(p).ok()
        })
        .unwrap();
    assert_ne!(
        result.as_deref(),
        Some(digest.as_str()),
        "corrupted artifact must be refused"
    );
    assert_eq!(calls.get(), 1);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn catalog_verification_startup_discards_orphaned_rows() {
    let (root, store, install, path, digest) = verification_fixture();
    crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
        crate::sha256_file(p).ok()
    })
    .unwrap();
    store.sqlite().with_conn_fenced(|tx| {
            tx.execute("INSERT INTO catalog_file_verifications VALUES ('orphan','manifest','metal','weights',?1,'1','2','3','4','5')", params![digest])?;
            Ok(())
        }).unwrap();
    assert_eq!(table_count(&store, "catalog_file_verifications"), 2);
    store
        .reconcile_persisted_registrations(&|_| false, &FakeBlobCache::default())
        .unwrap();
    assert_eq!(table_count(&store, "catalog_file_verifications"), 1);
    assert!(store
        .catalog_file_verified(
            &install,
            "weights",
            &digest,
            &crate::catalog_file_stamp(&path).unwrap()
        )
        .unwrap());
    std::fs::remove_dir_all(root).unwrap();
}

fn temp_descriptor(label: &str) -> (std::path::PathBuf, StorageDescriptor) {
    let root = std::env::temp_dir().join(format!(
        "synapse-store-{label}-{}-{}",
        std::process::id(),
        JOB_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let descriptor = StorageDescriptor {
        module_id: "synapse-test".to_string(),
        storage_namespace: "default".to_string(),
        isolation: Isolation::Module,
        backend: StorageBackend::Sqlite {
            path: root.join("store.db").to_string_lossy().to_string(),
        },
    };
    (root, descriptor)
}

#[cfg(unix)]
fn table_count(store: &SynapseStore, table: &str) -> i64 {
    store
        .sqlite()
        .with_conn(|connection| {
            connection.query_row(&format!("SELECT COUNT(*) FROM \"{table}\""), [], |row| {
                row.get(0)
            })
        })
        .unwrap()
}

/// In-memory stand-in for the model cache: blob sizes by digest, pinned
/// digests and job ids with a staging directory.
#[derive(Default)]
#[cfg(unix)]
struct FakeBlobCache {
    blobs: std::cell::RefCell<BTreeMap<String, u64>>,
    pins: std::cell::RefCell<BTreeSet<String>>,
    staging: std::cell::RefCell<BTreeSet<String>>,
}

#[cfg(unix)]
impl CatalogBlobCache for FakeBlobCache {
    fn blob_size(&self, digest: &str) -> std::io::Result<Option<u64>> {
        Ok(self.blobs.borrow().get(digest).copied())
    }

    fn is_pinned(&self, digest: &str) -> std::io::Result<bool> {
        Ok(self.pins.borrow().contains(digest))
    }

    fn remove_pins(&self, digest: &str) -> std::io::Result<()> {
        self.pins.borrow_mut().remove(digest);
        Ok(())
    }

    fn delete_blob(&self, digest: &str) -> std::io::Result<u64> {
        Ok(self.blobs.borrow_mut().remove(digest).unwrap_or(0))
    }

    fn delete_staging(&self, job_id: &str) -> std::io::Result<()> {
        self.staging.borrow_mut().remove(job_id);
        Ok(())
    }
}
