//! Durable job admission, page commit and terminal-state tests.

use super::tests::{admit, temp_descriptor};
use super::*;

#[test]
fn request_key_rejects_a_different_digest_and_same_digest_creates_a_fresh_attempt() {
    let (root, descriptor) = temp_descriptor("request-key");
    let store = SynapseStore::open(&descriptor).unwrap();
    let generation = store.next_module_generation().unwrap();
    let first = admit(&store, "same-key", "digest-a", generation, 100, 1_000, 500);
    let first_id = first.record().job_id.clone();
    let same = admit(&store, "same-key", "digest-a", generation, 101, 1_000, 500);
    assert!(matches!(same, JobAdmission::Existing(_)));
    assert_eq!(same.record().job_id, first_id);

    let conflict = store
        .admit_job(
            "same-key",
            "digest-b",
            "embed.batch",
            generation,
            &serde_json::json!({"items": 2}),
            102,
            1_000,
            500,
        )
        .unwrap_err();
    assert!(matches!(
        conflict,
        SynapseStoreError::IdempotencyConflict { .. }
    ));

    store
        .fail_job(
            &first_id,
            true,
            &serde_json::json!({"code": "module_restarted"}),
            103,
        )
        .unwrap();
    let fresh = admit(&store, "same-key", "digest-a", generation, 104, 1_000, 500);
    assert!(matches!(fresh, JobAdmission::Admitted(_)));
    assert_ne!(fresh.record().job_id, first_id);
    assert!(store.get_job_at(&first_id, 105).unwrap().is_some());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn page_commit_rolls_back_as_a_unit_and_resume_skips_committed_ids() {
    let (root, descriptor) = temp_descriptor("atomic-page");
    let store = SynapseStore::open(&descriptor).unwrap();
    let generation = store.next_module_generation().unwrap();
    let job_id = admit(
        &store,
        "page-key",
        "page-digest",
        generation,
        1_000,
        1_000,
        100,
    )
    .record()
    .job_id
    .clone();
    assert!(store.mark_job_running(&job_id, generation, 1_001).unwrap());
    let active_attempt_id = store
        .get_job_at(&job_id, 1_001)
        .unwrap()
        .unwrap()
        .active_attempt_id
        .unwrap();
    assert_ne!(active_attempt_id, job_id);
    assert!(matches!(
        store.claim_job_attempt(&job_id, generation, 1_001).unwrap(),
        JobAttemptClaim::Attached {
            active_attempt_id: attached,
            ..
        } if attached == active_attempt_id
    ));
    assert!(store
        .commit_job_page(&job_id, 0, br#"{"page":0}"#, &[], 1_002)
        .is_err());
    let duplicate_items = [
        CheckpointItem {
            item_id: "item-1".to_string(),
            result: vec![1],
        },
        CheckpointItem {
            item_id: "item-1".to_string(),
            result: vec![2],
        },
    ];
    assert!(store
        .commit_job_page(&job_id, 0, br#"{"page":0}"#, &duplicate_items, 1_002)
        .is_err());
    assert_eq!(store.checkpoint_count("page-digest").unwrap(), 0);
    assert_eq!(
        store
            .get_job_at(&job_id, 1_003)
            .unwrap()
            .unwrap()
            .page_count,
        0
    );
    assert!(store.get_job_page(&job_id, 0).unwrap().is_none());

    store
        .commit_job_page(
            &job_id,
            0,
            br#"{"page":0}"#,
            &[CheckpointItem {
                item_id: "item-1".to_string(),
                result: vec![1],
            }],
            1_004,
        )
        .unwrap();
    store
        .fail_job(
            &job_id,
            true,
            &serde_json::json!({"code": "module_restarted"}),
            1_005,
        )
        .unwrap();
    let resumed = admit(
        &store,
        "page-key",
        "page-digest",
        generation,
        1_006,
        1_000,
        100,
    );
    assert_eq!(resumed.record().page_count, 1);
    assert_eq!(
        store.committed_item_ids("page-digest").unwrap(),
        ["item-1".to_string()].into_iter().collect()
    );
    assert!(store
        .get_job_page(&resumed.record().job_id, 0)
        .unwrap()
        .is_some());
    let _ = std::fs::remove_dir_all(root);
}

/// A job past its execution TTL that no poll or sweep has expired yet must
/// still refuse a late page, and neither a late finish nor a late failure may
/// replace the deadline outcome the TTL implies.
#[test]
fn a_job_past_its_ttl_refuses_late_pages_and_keeps_its_deadline_outcome() {
    let (root, descriptor) = temp_descriptor("job-ttl-fence");
    let store = SynapseStore::open(&descriptor).unwrap();
    let generation = store.next_module_generation().unwrap();
    let job_id = admit(
        &store,
        "ttl-key",
        "ttl-digest",
        generation,
        1_000,
        100,
        10_000,
    )
    .record()
    .job_id
    .clone();
    assert!(store.mark_job_running(&job_id, generation, 1_001).unwrap());
    let item = CheckpointItem {
        item_id: "item-1".to_string(),
        result: vec![1],
    };
    // Expiry falls at 1_100; every write below comes later, with no sweep first.
    assert!(store
        .commit_job_page(
            &job_id,
            0,
            br#"{"page":0}"#,
            std::slice::from_ref(&item),
            1_200
        )
        .is_err());
    assert!(!store
        .finish_job(&job_id, &serde_json::json!({"state": "done"}), 1_201)
        .unwrap());
    store
        .fail_job(
            &job_id,
            false,
            &serde_json::json!({"code": "engine_crashed"}),
            1_202,
        )
        .unwrap();
    let record = store.get_job_at(&job_id, 1_203).unwrap().unwrap();
    assert_eq!(record.state, JOB_STATE_FAILED_TRANSIENT);
    assert_eq!(record.page_count, 0);
    assert_eq!(
        record
            .error_json
            .as_ref()
            .map(|error| error["code"].clone()),
        Some(serde_json::json!("deadline_exceeded"))
    );
    assert_eq!(store.checkpoint_count("ttl-digest").unwrap(), 0);
    drop(store);
    let _ = std::fs::remove_dir_all(root);
}

/// Two request keys for the same content share one digest and so one page
/// sequence. A job admitted before the other key's job committed must start
/// its attempt from every committed page, not from its stale admission count,
/// or it would append over them or finish showing none.
#[test]
fn a_claim_counts_pages_another_key_committed_after_admission() {
    let (root, descriptor) = temp_descriptor("shared-digest-claim");
    let store = SynapseStore::open(&descriptor).unwrap();
    let generation = store.next_module_generation().unwrap();
    let first = admit(&store, "key-a", "shared", generation, 1_000, 10_000, 10_000);
    let second = admit(&store, "key-b", "shared", generation, 1_001, 10_000, 10_000);
    let (first, second) = (
        first.record().job_id.clone(),
        second.record().job_id.clone(),
    );
    assert!(store.mark_job_running(&first, generation, 1_002).unwrap());
    let item = CheckpointItem {
        item_id: "item-1".to_string(),
        result: vec![1],
    };
    store
        .commit_job_page(
            &first,
            0,
            br#"{"page":0}"#,
            std::slice::from_ref(&item),
            1_003,
        )
        .unwrap();
    assert!(store
        .finish_job(&first, &serde_json::json!({"state": "done"}), 1_004)
        .unwrap());
    let claimed = match store.claim_job_attempt(&second, generation, 1_005).unwrap() {
        JobAttemptClaim::Claimed(record) => record,
        _ => panic!("the second key's job was not claimable"),
    };
    assert_eq!(claimed.page_count, 1);
    drop(store);
    let _ = std::fs::remove_dir_all(root);
}
