use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use cortexkit_log::{Config, Lane, Retention};
use cortexkit_store_types::{Isolation, StorageBackend, StorageDescriptor};
use rusqlite::params;
use synapse_module::{
    SynapseStore, LOG_TAGS, RECLAIM_FREELIST_MIN_BYTES, RECLAIM_FREELIST_PAGE_RATIO_DIVISOR,
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

#[test]
fn maintenance_log_absent_below_threshold_and_present_above_threshold() {
    let log_dir = unique_dir("logs");
    cortexkit_log::declared_tags(LOG_TAGS);
    let log_handle = cortexkit_log::init(Config {
        module_id: "synapse".to_string(),
        logs_dir: log_dir.clone(),
        lane: Lane::Module,
        spec: Some("debug".to_string()),
        retention: Retention::default(),
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

    // Verify maintenance log line is absent
    let log_content = fs::read_to_string(log_handle.path()).unwrap_or_default();
    assert!(
        !log_content.contains("tag=maintenance"),
        "expected no tag=maintenance line when below threshold, found:\n{log_content}"
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

    // Verify maintenance log line is present with expected fields
    let log_content_after = fs::read_to_string(log_handle.path()).unwrap_or_default();
    let maintenance_lines = log_content_after
        .lines()
        .filter(|line| line.contains("tag=maintenance"))
        .collect::<Vec<_>>();
    assert_eq!(
        maintenance_lines.len(),
        1,
        "expected exactly 1 tag=maintenance line, found:\n{log_content_after}"
    );
    let line = maintenance_lines[0];
    assert!(
        line.contains("before_page_count="),
        "missing before_page_count: {line}"
    );
    assert!(
        line.contains("after_page_count="),
        "missing after_page_count: {line}"
    );
    assert!(line.contains("wall_ms="), "missing wall_ms: {line}");

    let _ = fs::remove_dir_all(below_root);
    let _ = fs::remove_dir_all(above_root);
    let _ = fs::remove_dir_all(log_dir);
}
