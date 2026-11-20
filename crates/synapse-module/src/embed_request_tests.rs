//! Regression tests for embed and rerank request validation, admission order
//! and profile self-check retries.

use super::*;

/// Results and checkpoints are keyed by item id, so a batch whose explicit
/// id collides with the id synthesized for a bare string must be refused
/// before it can be admitted.
#[test]
fn embed_batch_refuses_an_explicit_id_that_collides_with_a_positional_id() {
    let items: Vec<EmbedBatchItemParam> =
        serde_json::from_value(json!([{"id": "1", "text": "first"}, "second"])).unwrap();
    let error = batch_items(items, Vec::new()).unwrap_err();
    assert!(error.contains("'1'"), "{error}");
}

/// An over-count catalog rerank refuses on its fixed candidate limit before
/// composing and tokenizing any pair, so an over-long candidate in it is
/// never measured.
#[cfg(unix)]
#[tokio::test]
async fn catalog_rerank_candidate_limit_refuses_before_composing_pairs() {
    use super::tests::{response_result, test_storage_descriptor};
    let (dir, descriptor) = test_storage_descriptor("catalog-rerank-count-first");
    let state =
        super::tests::catalog_test_state(&dir, &descriptor, "qwen3-reranker-0.6b.owned-vulkan");
    let too_long = std::iter::repeat_n("a", 8193).collect::<Vec<_>>().join(" ");
    let mut candidates = vec![too_long];
    candidates.resize(state.runtime.inline.max_items + 1, "short".to_string());
    let result = response_result(
        rerank_score(
            state.clone(),
            json!({"model": "catalog-sequence-test", "query": "", "candidates": candidates}),
        )
        .await,
        "rerank.score",
    );
    drop(state);
    fs::remove_dir_all(dir).unwrap();
    assert_eq!(result["error"]["code"], "queue_full", "{result}");
}

/// A worker error while the profile self-check runs its reference inputs
/// compares no output. The check must stay pending, so the next attempt runs
/// it again and passes once the worker answers, instead of finding a persisted
/// failure that disables the lane for its whole self-check key.
#[cfg(unix)]
#[tokio::test]
async fn a_worker_error_during_the_profile_self_check_leaves_it_retryable() {
    use super::tests::{finish_ane_lane_fixture, packing_lane_fixture, PACKING_TEST_LANE};
    let (fixture, peer) = packing_lane_fixture(ModuleConfig::default(), false).await;
    let state = fixture.state.clone();
    *state.runtime.profile_check_passes_for_test.lock().unwrap() = true;
    peer.lock().unwrap().fail_next_inference = true;
    let first = run_lane_check(&state, PACKING_TEST_LANE).await;
    let second = run_lane_check(&state, PACKING_TEST_LANE).await;
    let passed = profile_check_passed(&state, PACKING_TEST_LANE).map(|model| model.is_some());
    drop(state);
    finish_ane_lane_fixture(fixture).await;
    let error = match first {
        Err(error) => error,
        Ok(_) => panic!("the injected worker error must fail the first check"),
    };
    // The worker's own error comes back unchanged, not a self-check failure.
    assert_eq!(error.code, "injected", "{error:?}");
    assert!(second.is_ok(), "{:?}", second.err());
    assert!(matches!(passed, Ok(true)), "{passed:?}");
}

/// Windows cold loads reuse persisted file verification only when a file has
/// a stamp, and the stamp must change when the file's bytes do.
#[cfg(windows)]
#[test]
fn catalog_file_stamp_identifies_and_tracks_files_on_windows() {
    use std::io::Write;
    let (dir, _) = super::tests::test_storage_descriptor("windows-file-stamp");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("member.bin");
    fs::write(&path, b"weights").unwrap();
    let before = catalog_file_stamp(&path);
    std::thread::sleep(Duration::from_millis(50));
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b" changed")
        .unwrap();
    let after = catalog_file_stamp(&path);
    let _ = fs::remove_dir_all(&dir);
    let before = before.expect("an NTFS file must have a stamp");
    assert_ne!(Some(before), after);
}
