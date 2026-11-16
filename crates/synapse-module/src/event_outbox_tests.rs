use super::*;
use crate::store::{CheckpointItem, DownloadJobRequest};
use serde::Deserialize;
use serde_json::json;
use sha2::Digest;
use std::{collections::BTreeMap, sync::Mutex};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobPage {
    event_id: String,
    job_id: String,
    page_seq: u32,
    job_state: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DownloadState {
    event_id: String,
    download_id: String,
    state: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FindingsChanged {
    event_id: String,
    generation: u64,
    worst_severity: String,
}

fn fixture(label: &str) -> (PathBuf, Arc<SynapseStore>) {
    let (root, descriptor) = crate::tests::test_storage_descriptor(label);
    (root, Arc::new(SynapseStore::open(&descriptor).unwrap()))
}

fn sql(store: &SynapseStore, sql: &str) {
    store
        .store
        .with_conn_fenced(|tx| tx.execute_batch(sql))
        .unwrap();
}

fn count(store: &SynapseStore, table: &str) -> u64 {
    store
        .store
        .with_conn(|c| c.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0)))
        .unwrap()
}

fn fail_after_insert(store: &SynapseStore) {
    sql(store, "CREATE TRIGGER refuse_event AFTER INSERT ON event_outbox BEGIN SELECT RAISE(ABORT,'fail after outbox insert'); END;");
}

fn uuid_v4(id: &str) {
    assert_eq!(id.len(), 36);
    assert_eq!(
        [
            id.as_bytes()[8],
            id.as_bytes()[13],
            id.as_bytes()[18],
            id.as_bytes()[23]
        ],
        [b'-'; 4]
    );
    assert_eq!(id.as_bytes()[14], b'4');
    assert!(b"89ab".contains(&id.as_bytes()[19]));
    assert!(id
        .bytes()
        .enumerate()
        .all(|(i, b)| [8, 13, 18, 23].contains(&i) || b.is_ascii_hexdigit()));
}

#[test]
fn embed_page_event_is_atomic_and_has_contract_body() {
    let (root, store) = fixture("outbox-embed");
    let id = store
        .admit_job(
            "embed-key",
            "embed-digest",
            "embed.batch",
            1,
            &json!({}),
            1000,
            10000,
            10000,
        )
        .unwrap()
        .record()
        .job_id
        .clone();
    let item = CheckpointItem {
        item_id: "item".into(),
        result: vec![1],
    };
    fail_after_insert(&store);
    assert!(store
        .commit_job_page(&id, 0, b"{}", std::slice::from_ref(&item), 1001)
        .is_err());
    assert_eq!(count(&store, "result_pages"), 0);
    assert_eq!(count(&store, "checkpoints"), 0);
    assert_eq!(count(&store, "event_outbox"), 0);
    sql(&store, "DROP TRIGGER refuse_event;");
    store.commit_job_page(&id, 0, b"{}", &[item], 1001).unwrap();
    let event = store.next_event().unwrap().unwrap();
    assert_eq!(event.subject, "event.synapse.job_page.v1");
    let body: JobPage = serde_json::from_str(&event.body).unwrap();
    assert_eq!(body.event_id, event.event_id);
    uuid_v4(&body.event_id);
    assert_eq!(
        (body.job_id, body.page_seq, body.job_state),
        (id, 0, "queued".into())
    );
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn media_terminal_event_is_atomic_and_has_contract_body() {
    let (root, store) = fixture("outbox-media");
    sql(&store, "INSERT INTO media_jobs(id,owner,client_id,binding,lane,object_id,state,accepted_ms,retained_ms) VALUES('media','owner','client','{}','lane','object','queued',1,100000);");
    fail_after_insert(&store);
    assert!(
        store
            .store
            .with_conn_fenced(
                |tx| tx.execute("UPDATE media_jobs SET state='done' WHERE id='media'", [])
            )
            .is_err()
    );
    assert_eq!(count(&store, "event_outbox"), 0);
    let state: String = store
        .store
        .with_conn(|c| c.query_row("SELECT state FROM media_jobs", [], |r| r.get(0)))
        .unwrap();
    assert_eq!(state, "queued");
    sql(&store, "DROP TRIGGER refuse_event; UPDATE media_jobs SET state='done' WHERE id='media'; UPDATE media_jobs SET state='done' WHERE id='media';");
    let event = store.next_event().unwrap().unwrap();
    let body: JobPage = serde_json::from_str(&event.body).unwrap();
    assert_eq!(event.subject, "event.synapse.job_page.v1");
    assert_eq!(body.event_id, event.event_id);
    uuid_v4(&body.event_id);
    assert_eq!(
        (body.job_id, body.page_seq, body.job_state),
        ("media".into(), 0, "done".into())
    );
    assert_eq!(
        count(&store, "event_outbox"),
        1,
        "no-op state updates are silent"
    );
    sql(&store, "DELETE FROM media_jobs;");
    assert_eq!(
        count(&store, "event_outbox"),
        1,
        "retention deletion is silent"
    );
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn download_event_is_atomic_and_has_contract_body() {
    let (root, store) = fixture("outbox-download");
    let params = json!({"catalog_id":"catalog","manifest_digest":"manifest"});
    let request = DownloadJobRequest {
        request_key: "download-key",
        request_digest: "download-digest",
        module_generation: 1,
        params_json: &params,
        now_ms: 1000,
        result_retention_ttl_ms: 10000,
        entry_complete: false,
    };
    fail_after_insert(&store);
    assert!(store.admit_download_job(&request).is_err());
    assert_eq!(count(&store, "jobs"), 0);
    assert_eq!(count(&store, "download_key_bindings"), 0);
    assert_eq!(count(&store, "event_outbox"), 0);
    sql(&store, "DROP TRIGGER refuse_event;");
    let id = store
        .admit_download_job(&request)
        .unwrap()
        .record()
        .job_id
        .clone();
    let event = store.next_event().unwrap().unwrap();
    let body: DownloadState = serde_json::from_str(&event.body).unwrap();
    assert_eq!(event.subject, "event.synapse.download_state.v1");
    assert_eq!(body.event_id, event.event_id);
    uuid_v4(&body.event_id);
    assert_eq!(
        (body.download_id, body.state),
        (id.clone(), "queued".into())
    );
    fail_after_insert(&store);
    assert!(store
        .advance_download_job(&id, "downloading", 1001)
        .is_err());
    assert_eq!(count(&store, "event_outbox"), 1);
    sql(&store, "DROP TRIGGER refuse_event;");
    store
        .advance_download_job(&id, "downloading", 1001)
        .unwrap();
    store
        .advance_download_job(&id, "downloading", 1002)
        .unwrap();
    assert_eq!(
        count(&store, "event_outbox"),
        2,
        "progress repeats do not emit"
    );
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn findings_event_is_atomic_and_has_contract_body() {
    let (root, store) = fixture("outbox-findings");
    fail_after_insert(&store);
    assert!(store
        .record_findings_digest("stale", "error", 1000)
        .is_err());
    assert_eq!(count(&store, "event_outbox"), 0);
    let generation: u64 = store
        .store
        .with_conn(|c| {
            c.query_row("SELECT findings_generation FROM findings_state", [], |r| {
                r.get(0)
            })
        })
        .unwrap();
    assert_eq!(generation, 0);
    sql(&store, "DROP TRIGGER refuse_event;");
    assert!(store
        .record_findings_digest("stale", "error", 1000)
        .unwrap());
    assert!(!store
        .record_findings_digest("stale", "error", 1001)
        .unwrap());
    let event = store.next_event().unwrap().unwrap();
    let body: FindingsChanged = serde_json::from_str(&event.body).unwrap();
    assert_eq!(event.subject, "event.synapse.findings_changed.v1");
    assert_eq!(body.event_id, event.event_id);
    uuid_v4(&body.event_id);
    assert_eq!((body.generation, body.worst_severity), (1, "error".into()));
    assert_eq!(count(&store, "event_outbox"), 1);
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}

#[derive(Clone)]
struct Attempt {
    event_id: String,
    subject: String,
    body: String,
    headers: BTreeMap<String, String>,
}

fn capture(event: &Event, content: Content) -> Attempt {
    let (body, mut headers) = match content {
        Content::Inline => (event.body.clone(), Headers::new()),
        Content::ClaimCheck {
            digest,
            mut headers,
        } => {
            headers.insert("Ck-Content-Digest".into(), digest.to_string());
            (String::new(), headers)
        }
    };
    headers.insert("Nats-Msg-Id".into(), event.event_id.clone());
    Attempt {
        event_id: event.event_id.clone(),
        subject: event.subject.clone(),
        body,
        headers,
    }
}

fn outcome_json(outcome: subc_client_rs::HandlerOutcome) -> serde_json::Value {
    match outcome {
        subc_client_rs::HandlerOutcome::Response(body) => serde_json::from_slice(&body).unwrap(),
        subc_client_rs::HandlerOutcome::Error { code, message } => {
            json!({"error":{"code":code,"message":message}})
        }
        other => panic!("unexpected event response: {other:?}"),
    }
}

const SAMPLE_KEY: &str = "00000000-0000-4000-8000-000000000001";
const SAMPLE_BODY: &str = r#"{"event_id":"00000000-0000-4000-8000-000000000001","generation":1,"worst_severity":"error"}"#;

fn insert_sample(store: &SynapseStore) {
    store.store.with_conn_fenced(|tx| {
        tx.execute("INSERT INTO event_outbox(event_id,subject,body,created_at) VALUES (?1,?2,?3,1700000000000)",params![SAMPLE_KEY,FINDINGS_SUBJECT,SAMPLE_BODY])?;
        Ok(())
    }).unwrap();
}

fn module_state(store: &Arc<SynapseStore>) -> Arc<crate::ModuleState> {
    let profile = crate::tests::test_machine_profile("test-os");
    store.observe_profile(&profile, 1, 1).unwrap();
    crate::tests::test_module_state(store.clone(), profile)
}

#[test]
fn events_get_preserves_stored_bytes_and_refuses_other_events() {
    let (root, store) = fixture("outbox-events-get");
    insert_sample(&store);
    let state = module_state(&store);
    let request = json!({"event_key":SAMPLE_KEY,"event_name":"findings_changed","version":1});
    let before = outcome_json(events_get(&state, request.clone()));
    assert_eq!(
        before["result"]["body"], SAMPLE_BODY,
        "fetch works before publish ACK"
    );
    assert_eq!(before["result"]["published_at"], "2023-11-14T22:13:20.000Z");
    let event = Event {
        event_id: SAMPLE_KEY.into(),
        subject: FINDINGS_SUBJECT.into(),
        body: SAMPLE_BODY.into(),
    };
    let notice = capture(&event, content(&event));
    assert!(notice.body.is_empty());
    assert_eq!(
        before["result"]["digest"],
        format!("{:x}", sha2::Sha256::digest(SAMPLE_BODY.as_bytes()))
    );
    assert_eq!(
        before["result"]["digest"],
        notice.headers["Ck-Content-Digest"]
    );
    store
        .mark_event_published(SAMPLE_KEY, 1700000060000)
        .unwrap();
    store
        .record_findings_digest("new-state", "ok", 1700000070000)
        .unwrap();
    assert_eq!(
        outcome_json(events_get(&state, request)),
        before,
        "new findings and ACK time never change the old body or timestamp"
    );
    sql(&store,"INSERT INTO event_outbox(event_id,subject,body,created_at) VALUES ('inline','event.synapse.job_page.v1','{}',1);");
    for (key, name, version) in [
        ("unknown", "findings_changed", 1),
        (SAMPLE_KEY, "job_page", 1),
        (SAMPLE_KEY, "findings_changed", 2),
        ("inline", "findings_changed", 1),
    ] {
        let result = outcome_json(events_get(
            &state,
            json!({"event_key":key,"event_name":name,"version":version}),
        ));
        assert_eq!(result["error"]["code"], "event_not_visible");
    }
    drop(state);
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn findings_changed_samples_v1_are_current() {
    let (root, store) = fixture("outbox-event-samples");
    insert_sample(&store);
    let state = module_state(&store);
    let event = Event {
        event_id: SAMPLE_KEY.into(),
        subject: FINDINGS_SUBJECT.into(),
        body: SAMPLE_BODY.into(),
    };
    let notice = capture(&event, content(&event));
    let names = cortexkit_bus_naming::AccountNames::derive("box_test").unwrap();
    let mut samples = BTreeMap::from([(
        "notice.json",
        json!({"name":"notice","subject":names.event_subject("synapse","findings_changed",1).unwrap(),"body":notice.body,"headers":notice.headers}),
    )]);
    for (name, key, event_name, version) in [
        ("events-get-initial", SAMPLE_KEY, "findings_changed", 1),
        ("events-get-unknown-key", "unknown", "findings_changed", 1),
        ("events-get-wrong-name", SAMPLE_KEY, "job_page", 1),
        (
            "events-get-wrong-version",
            SAMPLE_KEY,
            "findings_changed",
            2,
        ),
    ] {
        let params = json!({"event_key":key,"event_name":event_name,"version":version});
        let response = outcome_json(events_get(&state, params.clone()));
        let file = match name {
            "events-get-initial" => "events-get-initial.json",
            "events-get-unknown-key" => "events-get-unknown-key.json",
            "events-get-wrong-name" => "events-get-wrong-name.json",
            "events-get-wrong-version" => "events-get-wrong-version.json",
            _ => unreachable!(),
        };
        samples.insert(file,json!({"name":name,"request":{"method":"events_get","params":params},"response":response}));
    }
    store
        .record_findings_digest("changed", "ok", 1700000070000)
        .unwrap();
    let params = json!({"event_key":SAMPLE_KEY,"event_name":"findings_changed","version":1});
    samples.insert("events-get-after-change.json",json!({"name":"events-get-after-change","request":{"method":"events_get","params":params.clone()},"response":outcome_json(events_get(&state,params))}));
    let directory =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/samples/findings-changed-v1");
    let regenerate = std::env::var_os("SYNAPSE_REGENERATE_EVENT_SAMPLES").is_some();
    let mut sums = String::new();
    for (file, value) in samples {
        let text = format!("{}\n", serde_json::to_string_pretty(&value).unwrap());
        sums.push_str(&format!(
            "{:x}  {file}\n",
            sha2::Sha256::digest(text.as_bytes())
        ));
        if regenerate {
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join(file), &text).unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(directory.join(file)).unwrap_or_default(),
            text,
            "stale sample {file}; regenerate with SYNAPSE_REGENERATE_EVENT_SAMPLES=1"
        );
    }
    if regenerate {
        std::fs::write(directory.join("SHA256SUMS"), &sums).unwrap();
    }
    assert_eq!(
        std::fs::read_to_string(directory.join("SHA256SUMS")).unwrap_or_default(),
        sums,
        "sample hashes are stale"
    );
    drop(state);
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}

struct FakeBus {
    attempts: Arc<Mutex<Vec<Attempt>>>,
    refuse: Arc<std::sync::atomic::AtomicBool>,
    refuse_first: bool,
}

impl EventBus for FakeBus {
    async fn publish(&mut self, event: &Event, content: Content) -> Result<bool, String> {
        self.attempts.lock().unwrap().push(capture(event, content));
        if std::mem::take(&mut self.refuse_first)
            || self.refuse.load(std::sync::atomic::Ordering::SeqCst)
        {
            Err("publish refused".into())
        } else {
            Ok(false)
        }
    }
}

#[tokio::test(start_paused = true)]
async fn publisher_orders_retries_claim_checks_and_prunes_after_seven_days() {
    let (root, store) = fixture("outbox-publisher");
    sql(&store,"INSERT INTO event_outbox(event_id,subject,body,created_at) VALUES ('third','event.synapse.findings_changed.v1','{\"event_id\":\"third\",\"generation\":2,\"worst_severity\":\"warn\"}',30),('first','event.synapse.job_page.v1','{\"page\":0}',10),('second','event.synapse.download_state.v1','{\"state\":\"queued\"}',20);");
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let refuse = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let task = tokio::spawn(drain(
        store.clone(),
        FakeBus {
            attempts: attempts.clone(),
            refuse: refuse.clone(),
            refuse_first: true,
        },
        Clock::new(100000),
    ));
    tokio::task::yield_now().await;
    assert_eq!(attempts.lock().unwrap().len(), 1);
    assert_eq!(count(&store, "event_outbox"), 3);
    assert_eq!(store.next_event().unwrap().unwrap().event_id, "first");
    tokio::time::advance(Duration::from_millis(250)).await;
    tokio::task::yield_now().await;
    let delivered = attempts.lock().unwrap().clone();
    assert_eq!(
        delivered
            .iter()
            .map(|attempt| attempt.event_id.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "first", "second", "third"]
    );
    for attempt in &delivered {
        assert_eq!(attempt.headers.get("Nats-Msg-Id"), Some(&attempt.event_id));
        assert!(attempt.subject.starts_with("event.synapse."));
    }
    assert_eq!(delivered[1].body, r#"{"page":0}"#);
    assert_eq!(delivered[2].body, r#"{"state":"queued"}"#);
    let findings = &delivered[3];
    assert_eq!(findings.body, "");
    assert_eq!(findings.headers["event_key"], "third");
    let digest = format!(
        "{:x}",
        sha2::Sha256::digest(br#"{"event_id":"third","generation":2,"worst_severity":"warn"}"#)
    );
    assert_eq!(findings.headers["Ck-Content-Digest"], digest);
    assert_eq!(
        count(&store, "event_outbox"),
        3,
        "retries reuse the persisted rows"
    );
    assert!(store.next_event().unwrap().is_none());
    refuse.store(true, std::sync::atomic::Ordering::SeqCst);
    sql(&store,"INSERT INTO event_outbox(event_id,subject,body,created_at) VALUES ('pending','event.synapse.job_page.v1','{}',40);");
    tokio::time::advance(Duration::from_secs(6 * 24 * 60 * 60)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        count(&store, "event_outbox"),
        4,
        "all published history is kept at day six"
    );
    tokio::time::advance(Duration::from_secs(2 * 24 * 60 * 60)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        count(&store, "event_outbox"),
        1,
        "expired published history is pruned, unpublished rows remain"
    );
    assert_eq!(store.next_event().unwrap().unwrap().event_id, "pending");
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}
