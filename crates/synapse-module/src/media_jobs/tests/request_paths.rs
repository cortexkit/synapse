//! Request-path regressions: lane waiters, admission locking, retention
//! housekeeping, unload fencing and startup revalidation.
use super::*;
use std::sync::Condvar;

/// A gate a test opens once; blocking closures wait on it.
#[derive(Clone, Default)]
struct Release(Arc<(Mutex<bool>, Condvar)>);

impl Release {
    fn wait(&self) {
        let (open, wake) = &*self.0;
        let mut open = open.lock().unwrap();
        while !*open {
            open = wake.wait(open).unwrap();
        }
    }

    fn open(&self) {
        *self.0 .0.lock().unwrap() = true;
        self.0 .1.notify_all();
    }
}

fn job_object(state: &ModuleState, id: &str) -> String {
    state
        .store
        .sqlite()
        .with_conn(|c| {
            c.query_row("SELECT object_id FROM media_jobs WHERE id=?1", [id], |r| {
                r.get(0)
            })
        })
        .unwrap()
}

fn rows(state: &ModuleState, table: &str) -> u64 {
    state
        .store
        .sqlite()
        .with_conn(|c| c.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0)))
        .unwrap()
}

fn pcm16_wav(samples: usize) -> Vec<u8> {
    let data = vec![0u8; samples * 2];
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);
    wav
}

#[tokio::test]
async fn media_cancelled_queued_job_releases_its_guards_without_waiting_for_the_lane() {
    let (root, state) = fixture("media-cancel-waiter", true);
    let lane = catalog::lane_id(MODEL, &first(&state));
    // Stand in for a slow transcription that owns the lane.
    let busy = lane_cache(&state.runtime, &lane).lock_owned().await;
    let id = call(&state, "alice", "media.submit", request("waiting"))
        .await
        .unwrap()["job_id"]
        .clone();
    let serving = lane_serving_gate(&state.runtime, &lane);
    assert!(
        serving.clone().try_write_owned().is_err(),
        "a queued job holds its serving guard"
    );
    call(&state, "alice", "media.cancel", json!({"job_id":id}))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while serving.clone().try_write_owned().is_err() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("a cancelled queued job must drop its guards while the lane is still busy");
    drop(busy);
    cleanup(root, state).await;
}

#[tokio::test]
async fn media_expired_jobs_keep_only_a_bounded_replay_tombstone() {
    let (root, state) = fixture("media-tombstone", true);
    let start = now_ms();
    *state.runtime.media.clock.lock().unwrap() = Some(start);
    let id = call(&state, "alice", "media.submit", request("replay"))
        .await
        .unwrap()["job_id"]
        .clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    persistence::sweep(&state, start + RETENTION_MS + 1).unwrap();
    // Inside the tombstone window a replay still names the original job.
    assert_eq!(
        call(&state, "alice", "media.submit", request("replay"))
            .await
            .unwrap()["job_id"],
        id
    );
    persistence::sweep(&state, start + RETENTION_MS + TOMBSTONE_MS + 1).unwrap();
    assert_eq!(rows(&state, "media_jobs"), 0);
    assert_eq!(rows(&state, "media_objects"), 0);
    // After the window the client_job_id is free and starts new work.
    let replayed = call(&state, "alice", "media.submit", request("replay"))
        .await
        .unwrap()["job_id"]
        .clone();
    assert_ne!(replayed, id);
    done(&state, replayed.as_str().unwrap()).await;
    cleanup(root, state).await;
}

#[tokio::test]
async fn media_submit_reads_and_inspects_audio_outside_shared_locks() {
    let (root, state) = fixture("media-admission-locks", true);
    let (entered, reached) = std::sync::mpsc::channel();
    let entered = Mutex::new(entered);
    let release = Release::default();
    let held = release.clone();
    *state.runtime.media.admission_probe.lock().unwrap() = Some(Arc::new(move || {
        entered.lock().unwrap().send(()).unwrap();
        held.wait();
    }));
    let submit = tokio::spawn({
        let state = state.clone();
        async move { call(&state, "alice", "media.submit", request("slow-read")).await }
    });
    tokio::task::spawn_blocking(move || reached.recv().unwrap())
        .await
        .unwrap();
    // The submit now holds its recording; neither shared lock may be held.
    let disk_free = state.runtime.catalog_disk.try_lock().is_ok();
    let status = tokio::time::timeout(
        Duration::from_secs(10),
        call(&state, "alice", "media.status", json!({"job_id":"missing"})),
    )
    .await;
    // Release before asserting, so a failure reports instead of hanging on
    // the blocked admission at runtime shutdown.
    release.open();
    assert!(
        disk_free,
        "catalog-disk lock must be free while audio is read"
    );
    let unknown = status
        .expect("media.status must not wait for another submit's audio")
        .unwrap_err();
    assert_eq!(unknown.0["code"], "unknown_job");
    let id = submit.await.unwrap().unwrap()["job_id"].clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    *state.runtime.media.admission_probe.lock().unwrap() = None;
    cleanup(root, state).await;
}

#[tokio::test]
async fn media_interrupted_unload_keeps_the_lane_locked_until_the_host_retires() {
    let (root, state) = fixture("media-unload-fence", true);
    let selected = first(&state);
    let lane = catalog::lane_id(MODEL, &selected);
    let profile = transcribe_host::TranscribeProfile::pinned(
        state
            .runtime
            .release_catalog
            .entry(MODEL)
            .unwrap()
            .backend(&selected)
            .unwrap()
            .profile
            .as_deref()
            .unwrap(),
    )
    .unwrap();
    let (entered, reached) = std::sync::mpsc::channel();
    let release = Release::default();
    let held = release.clone();
    lane_cache(&state.runtime, &lane).lock().await.host = Some(
        transcribe_host::TranscribeHost::in_process_for_test(profile, move |_| unreachable!())
            .with_unload_hook_for_test(move || {
                entered.send(()).unwrap();
                held.wait();
            }),
    );
    let unloading = tokio::spawn({
        let state = state.clone();
        async move { unload(&state.runtime, None, false, ()).await }
    });
    tokio::task::spawn_blocking(move || reached.recv().unwrap())
        .await
        .unwrap();
    // The caller stops waiting (a pause deadline, resume or cancelled request)
    // while the old host is still retiring.
    unloading.abort();
    let _ = unloading.await;
    let cache = lane_cache(&state.runtime, &lane);
    let locked = cache.try_lock().is_err();
    // Release before asserting, so a failure reports instead of hanging on
    // the blocked unload at runtime shutdown.
    release.open();
    assert!(
        locked,
        "the lane must stay locked until its old host has retired"
    );
    let retired = tokio::time::timeout(Duration::from_secs(10), cache.lock())
        .await
        .expect("the unload finishes once released");
    assert!(retired.host.is_none());
    drop(retired);
    cleanup(root, state).await;
}

#[tokio::test]
async fn media_unremovable_expired_audio_does_not_block_other_calls() {
    let (root, state) = fixture("media-stuck-unlink", true);
    let start = now_ms();
    *state.runtime.media.clock.lock().unwrap() = Some(start);
    let id = call(&state, "alice", "media.submit", request("stuck"))
        .await
        .unwrap()["job_id"]
        .clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    let object = job_object(&state, id.as_str().unwrap());
    // A directory at the object's path makes its unlink fail, as a file held
    // open without delete sharing does on Windows.
    let path = persistence::directory(&state).unwrap().join(&object);
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    fs::write(path.join("held"), b"x").unwrap();
    *state.runtime.media.clock.lock().unwrap() = Some(start + RETENTION_MS + 1);
    call(
        &state,
        "alice",
        "media.upload_begin",
        json!({"mime":"audio/wav","bytes":1,"sha256":"0".repeat(64)}),
    )
    .await
    .expect("an unrelated call must not fail on another object's unlink");
    let (object_state, expired): (String, bool) = state
        .store
        .sqlite()
        .with_conn(|c| {
            c.query_row(
                "SELECT o.state, j.expired FROM media_objects o JOIN media_jobs j ON j.object_id=o.id WHERE j.id=?1",
                [id.as_str().unwrap()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
        })
        .unwrap();
    // The stuck object stays charged and is retried later; the rest of the
    // sweep still ran.
    assert_eq!(object_state, "committed");
    assert!(expired);
    cleanup(root, state).await;
}

#[tokio::test]
async fn startup_revalidation_checks_speech_lanes_through_the_speech_self_check() {
    let (root, state) = fixture("media-revalidate", true);
    let (entry, backend) = select(&state, MODEL).unwrap();
    let loads = Arc::new(Mutex::new(Vec::new()));
    let seen = loads.clone();
    *state.runtime.media.loader.lock().unwrap() = Some(Arc::new(move |lane: &str| {
        seen.lock().unwrap().push(lane.to_owned());
        Err(transcribe_host::TranscribeHostError::Profile(
            "test loader".into(),
        ))
    }));
    let error = revalidate_catalog_lane(state.clone(), entry, backend.clone())
        .await
        .unwrap_err();
    assert_eq!(
        *loads.lock().unwrap(),
        [catalog::lane_id(MODEL, &backend.backend)]
    );
    assert_eq!(error.code, "engine_crashed");
    cleanup(root, state).await;
}

#[tokio::test]
async fn media_audio_shorter_than_one_mel_hop_is_permanent_invalid_audio() {
    let (root, state) = fixture("media-sub-hop", true);
    let wav = pcm16_wav(100);
    let mut p = request("sub-hop");
    p["input"]["audio"] = media(&wav);
    let error = call(&state, "alice", "media.submit", p).await.unwrap_err();
    assert_eq!(error.0["code"], "invalid_params");
    assert_eq!(error.0["detail"]["class"], "permanent");
    assert_eq!(error.0["detail"]["field"], "audio");
    assert!(error.0["detail"].get("retry_after_ms").is_none());
    assert_eq!(rows(&state, "media_jobs"), 0);
    // Direct host preparation refuses the same clip as audio, too.
    let (_, backend) = select(&state, MODEL).unwrap();
    let profile =
        transcribe_host::TranscribeProfile::pinned(backend.profile.as_deref().unwrap()).unwrap();
    assert!(matches!(
        transcribe_host::prepare_wav(&profile, &wav, transcribe_host::ObservationMode::Serving),
        Err(transcribe_host::TranscribeHostError::Audio(_))
    ));
    cleanup(root, state).await;
}
