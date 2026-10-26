use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use cortexkit_log::{Config, SegmentRetention};
use cortexkit_store_types::{Isolation, StorageBackend, StorageDescriptor};
use rusqlite::params;
use synapse_module::{
    SynapseStore, RECLAIM_FREELIST_MIN_BYTES, RECLAIM_FREELIST_PAGE_RATIO_DIVISOR,
};

static TEST_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "synapse-vac-log-{label}-{}-{}",
        std::process::id(),
        TEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

fn test_descriptor(root: &Path) -> StorageDescriptor {
    StorageDescriptor {
        module_id: "synapse-test".to_string(),
        storage_namespace: "default".to_string(),
        isolation: Isolation::Module,
        backend: StorageBackend::Sqlite {
            path: root.join("store.db").to_string_lossy().to_string(),
        },
    }
}

/// The page counts a VACUUM log line reports, as `(before, after)`.
fn logged_page_counts(log: &str) -> Vec<(u64, u64)> {
    let field = |line: &str, name: &str| {
        line.split_whitespace()
            .find_map(|token| token.strip_prefix(name))
            .and_then(|value| value.trim_matches('"').parse::<u64>().ok())
    };
    log.lines()
        .filter_map(|line| {
            Some((
                field(line, "before_page_count=")?,
                field(line, "after_page_count=")?,
            ))
        })
        .collect()
}

#[test]
fn maintenance_log_absent_below_threshold_and_present_above_threshold() {
    let log_dir = unique_dir("logs");
    let log_handle = cortexkit_log::init(Config {
        module_id: "synapse".to_string(),
        logs_dir: log_dir.clone(),
        bound: Vec::new(),
        spec: Some("debug".to_string()),
        retention: SegmentRetention::default(),
        redactor: None,
        clock: Some(Arc::new(SystemTime::now)),
    })
    .expect("fleet logger initializes");

    // Case 1: Store below threshold reopened -> assert no VACUUM ran and log line absent.
    let below_root = unique_dir("below");
    let below_desc = test_descriptor(&below_root);
    let store_below = SynapseStore::open(&below_desc).unwrap();
    store_below
        .reclaim_freelist_if_needed()
        .expect("initial reclaim");

    // Seed 512 KiB bloat and delete
    store_below
        .pub_connection_for_test(|conn| {
            conn.execute(
                "CREATE TABLE small_bloat (id INTEGER PRIMARY KEY, data BLOB)",
                [],
            )?;
            let chunk = vec![0x42u8; 1024];
            let mut stmt = conn.prepare("INSERT INTO small_bloat (data) VALUES (?1)")?;
            for _ in 0..512 {
                stmt.execute(params![&chunk])?;
            }
            drop(stmt);
            conn.execute("DELETE FROM small_bloat", [])?;
            Ok(())
        })
        .unwrap();

    let freelist_before = store_below.freelist_count().unwrap();
    let page_count_before = store_below.page_count().unwrap();
    let page_size = store_below.page_size().unwrap();
    assert!(freelist_before >= 100);
    assert!(freelist_before >= page_count_before / RECLAIM_FREELIST_PAGE_RATIO_DIVISOR);
    assert!(freelist_before * page_size < RECLAIM_FREELIST_MIN_BYTES);
    drop(store_below);

    let reopened_below = SynapseStore::open(&below_desc).unwrap();
    let freelist_after = reopened_below.freelist_count().unwrap();
    assert_eq!(freelist_after, freelist_before);
    drop(reopened_below);

    // No VACUUM ran, so nothing reported reclaimed pages.
    let log_content = fs::read_to_string(log_handle.path()).unwrap_or_default();
    assert!(
        logged_page_counts(&log_content).is_empty(),
        "expected no VACUUM report when below threshold, found:\n{log_content}"
    );

    // Case 2: Store above threshold reopened -> assert VACUUM ran and log line present.
    let above_root = unique_dir("above");
    let above_desc = test_descriptor(&above_root);
    let store_above = SynapseStore::open(&above_desc).unwrap();

    // Seed 66 MiB bloat and delete
    store_above
        .pub_connection_for_test(|conn| {
            conn.execute(
                "CREATE TABLE bloat_seed (id INTEGER PRIMARY KEY, data BLOB)",
                [],
            )?;
            let chunk = vec![0xfeu8; 1024 * 1024];
            let mut stmt = conn.prepare("INSERT INTO bloat_seed (data) VALUES (?1)")?;
            for _ in 0..66 {
                stmt.execute(params![&chunk])?;
            }
            drop(stmt);
            conn.execute("DELETE FROM bloat_seed", [])?;
            Ok(())
        })
        .unwrap();

    let freelist_above_before = store_above.freelist_count().unwrap();
    let page_count_above_before = store_above.page_count().unwrap();
    let page_size = store_above.page_size().unwrap();
    assert!(freelist_above_before * page_size >= RECLAIM_FREELIST_MIN_BYTES);
    assert!(freelist_above_before >= page_count_above_before / RECLAIM_FREELIST_PAGE_RATIO_DIVISOR);
    drop(store_above);

    let db_path = above_root.join("store.db");
    let file_size_before = fs::metadata(&db_path).unwrap().len();
    assert!(file_size_before >= 64 * 1024 * 1024);

    let reopened_above = SynapseStore::open(&above_desc).unwrap();
    let freelist_above_after = reopened_above.freelist_count().unwrap();
    let file_size_after = fs::metadata(&db_path).unwrap().len();
    assert_eq!(freelist_above_after, 0);
    assert!(
        file_size_after < file_size_before / 10,
        "file shrank from {file_size_before} to {file_size_after}"
    );
    drop(reopened_above);

    // Exactly one VACUUM was reported, with the page counts it really saw.
    let log_content_after = fs::read_to_string(log_handle.path()).unwrap_or_default();
    let reports = logged_page_counts(&log_content_after);
    assert_eq!(
        reports.len(),
        1,
        "expected exactly 1 VACUUM report, found:\n{log_content_after}"
    );
    let (logged_before, logged_after) = reports[0];
    assert_eq!(logged_before, page_count_above_before);
    assert!(
        logged_after < logged_before / 10,
        "logged page counts must show the shrink: {logged_before} -> {logged_after}"
    );

    let _ = fs::remove_dir_all(below_root);
    let _ = fs::remove_dir_all(above_root);
    let _ = fs::remove_dir_all(log_dir);
}
