use super::*;
use std::collections::VecDeque;

struct Stub {
    replies: VecDeque<Value>,
    calls: Vec<(String, Value)>,
}
impl Client for Stub {
    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.calls.push((method.into(), params));
        Ok(self.replies.pop_front().expect("unexpected request"))
    }
}
fn stub(replies: Vec<Value>) -> Stub {
    Stub {
        replies: replies.into(),
        calls: vec![],
    }
}
fn pin() -> Pin {
    Pin {
        lane_id: "parakeet-tdt-0.6b-v3-cuda".into(),
        fingerprint: "pinned".into(),
        task: "transcribe".into(),
    }
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
fn descriptor(raw: &[u8]) -> Value {
    json!({"bytes":raw.len(),"sha256":sha256_hex(raw),"pages":raw.len().div_ceil(524288)})
}

#[tokio::test]
async fn transcribe_pages_reassemble_every_page_and_verify_both_seals() {
    let raw = vec![7; 528384];
    let mut client = stub(
        raw.chunks(524288)
            .map(|b| json!({"data":STANDARD.encode(b)}))
            .collect(),
    );
    let (actual, seal) = read_part(
        &mut client,
        "job",
        "encoder_output",
        &descriptor(&raw),
        deadline(),
    )
    .await
    .unwrap();
    assert_eq!(actual, raw);
    assert_eq!(seal.reassembled_bytes, 528384);
    assert_eq!(seal.pages, 2);
    assert!(client.replies.is_empty());
    for (index, (method, params)) in client.calls.iter().enumerate() {
        assert_eq!(method, "certify.transcribe_observation");
        assert_eq!(
            params,
            &json!({"job_id":"job","part":"encoder_output","page":index})
        );
    }
    for part in ["encoder_output", "decode_trace"] {
        let mut bad = descriptor(&[1, 2, 3]);
        bad["sha256"] = json!("a".repeat(64));
        let mut client = stub(vec![json!({"data":STANDARD.encode([1,2,3])})]);
        assert!(read_part(&mut client, "job", part, &bad, deadline())
            .await
            .unwrap_err()
            .to_string()
            .contains("sha256"));
        for key in ["bytes", "pages"] {
            let mut bad = descriptor(&[1, 2, 3]);
            bad[key] = json!(2);
            let mut client = stub(vec![json!({"data":STANDARD.encode([1,2,3])})]);
            assert!(read_part(&mut client, "job", part, &bad, deadline())
                .await
                .is_err());
        }
        let mut client = stub(vec![json!({"data":STANDARD.encode([1,2])})]);
        assert!(read_part(
            &mut client,
            "job",
            part,
            &descriptor(&[1, 2, 3]),
            deadline()
        )
        .await
        .is_err());
    }
}

#[tokio::test]
async fn transcribe_every_observation_must_match_lane_and_pin() {
    let mut response = json!({"text":"","observation":{"lane_id":pin().lane_id,"fingerprint":"pinned","valid_encoder_frames":1,"triples":"","encoder_output":descriptor(&[0;4096]),"decode_trace":descriptor(&[])}});
    let mut client = stub(vec![json!({"data":STANDARD.encode([0;4096])})]);
    let (got, _, _) = observation(&mut client, &pin(), "job", &response, deadline())
        .await
        .unwrap();
    assert_eq!(got.valid_encoder_frames, Some(1));
    for field in ["lane_id", "fingerprint"] {
        let old = response["observation"][field].clone();
        response["observation"][field] = json!("wrong");
        assert!(
            observation(&mut client, &pin(), "job", &response, deadline())
                .await
                .unwrap_err()
                .to_string()
                .contains("pin")
        );
        response["observation"][field] = old;
    }
    assert_eq!(client.calls.len(), 1);
}

#[tokio::test]
async fn transcribe_models_list_requires_pass_and_exact_single_fingerprint() {
    let good = json!({"models":[{"model_id":pin().lane_id,"self_check":{"state":"passed"},"fingerprints":["pinned"]}]});
    let mut client = stub(vec![good.clone()]);
    self_check(&mut client, &pin(), deadline()).await.unwrap();
    for (field, value) in [
        ("fingerprints", json!([])),
        ("fingerprints", json!(["pinned", "pinned"])),
        ("fingerprints", json!(["other"])),
        ("self_check", json!({"state":"running"})),
    ] {
        let mut bad = good.clone();
        bad["models"][0][field] = value;
        let mut client = stub(vec![bad]);
        assert!(self_check(&mut client, &pin(), deadline()).await.is_err());
    }
}

#[tokio::test]
async fn transcribe_submission_pins_lane_instead_of_catalog_default() {
    let mut client = stub(vec![json!({"job_id":"job"})]);
    submit(
        &mut client,
        &pin(),
        "clip",
        json!({"object_id":"object"}),
        deadline(),
    )
    .await
    .unwrap();
    assert_eq!(client.calls[0].0, "media.submit");
    assert_eq!(client.calls[0].1["model"], pin().lane_id);
    assert_eq!(client.calls[0].1["kind"], "transcribe");
    assert_eq!(client.calls[0].1["input"]["audio"]["object_id"], "object");
}

#[tokio::test]
async fn transcribe_missing_measurements_name_the_surface_and_field() {
    let good = json!({"self_check_wall_ms":1,"longest_progress_interval_ms":2,"ane_shapes":[],"peak_device_memory_bytes":3,"peak_storage_buffer_bytes":4});
    for field in [
        "self_check_wall_ms",
        "longest_progress_interval_ms",
        "ane_shapes",
        "peak_device_memory_bytes",
        "peak_storage_buffer_bytes",
    ] {
        let mut bad = good.clone();
        bad.as_object_mut().unwrap().remove(field);
        let mut client = stub(vec![bad]);
        let error = measurements(&mut client, "maximum-job", "vulkan", deadline())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("certify.transcribe_measurements.{field}")),
            "{error}"
        );
        assert_eq!(client.calls[0].1, json!({"job_id":"maximum-job"}));
    }
}

#[tokio::test(start_paused = true)]
async fn transcribe_above_limit_requires_c4_details_and_no_engine_or_shape_work() {
    let manifest = synapse_certify::self_check::manifest().unwrap();
    let mut profile = manifest.profiles["parakeet-tdt-0.6b-v3.ane-direct-worker"]
        .transcribe
        .clone()
        .unwrap();
    profile.max_duration_ms = 81850;
    let fixture = SpeechFixture {
        id: "ladder-1025".into(),
        sample_count: 1310880,
        duration_ms: 81930,
        valid_encoder_frames: 1025,
        trace: speech::Artifact {
            path: Some("unused".into()),
            cache_path: None,
            sha256: "a".repeat(64),
            bytes: 0,
        },
        encoder: None,
        text: "".into(),
        split: None,
        ground_truth: None,
    };
    let good = json!({"error":{"code":"input_too_large","detail":{"field":"input.audio","limit":"max_duration_ms","duration_ms":81930,"max_duration_ms":81850,"valid_encoder_frames":1025,"max_encoder_frames":1024}}});
    let counters = json!({"available":true,"worker_requests":{"lane":5},"admitted_count":3});
    let source = CounterSource::Worker;
    let mut client = stub(vec![counters.clone()]);
    let got = refusal(
        &mut client,
        &fixture,
        &good,
        &profile,
        source,
        (5, 3),
        deadline(),
    )
    .await
    .unwrap();
    assert_eq!(got.worker_requests, 0);
    assert_eq!(got.shape_admissions, 0);
    for field in [
        "duration_ms",
        "max_duration_ms",
        "valid_encoder_frames",
        "max_encoder_frames",
        "field",
        "limit",
    ] {
        let mut bad = good.clone();
        bad["error"]["detail"][field] = json!(0);
        let mut client = stub(vec![counters.clone()]);
        assert!(refusal(
            &mut client,
            &fixture,
            &bad,
            &profile,
            source,
            (5, 3),
            deadline()
        )
        .await
        .is_err());
    }
    for (worker, shape) in [(6, 3), (5, 4)] {
        let mut changed = counters.clone();
        changed["worker_requests"]["lane"] = json!(worker);
        changed["admitted_count"] = json!(shape);
        let mut client = stub(vec![changed]);
        assert!(refusal(
            &mut client,
            &fixture,
            &good,
            &profile,
            source,
            (5, 3),
            deadline()
        )
        .await
        .is_err());
    }
    let mut client = stub(vec![json!({"status":"done"})]);
    assert!(refusal(
        &mut client,
        &fixture,
        &json!({"job_id":"silently-accepted"}),
        &profile,
        source,
        (5, 3),
        deadline()
    )
    .await
    .is_err());

    let submitted = json!({"job_id":"limit-refusal"});
    let failed = json!({"status":"failed","failure":good["error"]});
    let source = CounterSource::InProcess("metal-lane");
    for (count, passes) in [(Some(7), true), (Some(8), false), (None, false)] {
        let mut after = json!({"available":true,"worker_requests":{},"admitted_count":0});
        if let Some(count) = count {
            after["in_process_transcribe_engine_invocations"] = json!({"metal-lane":count});
        }
        let mut client = stub(vec![json!({"status":"running"}), failed.clone(), after]);
        let result = refusal(
            &mut client,
            &fixture,
            &submitted,
            &profile,
            source,
            (7, 0),
            deadline(),
        )
        .await;
        if passes {
            let got = result.unwrap();
            assert_eq!(got.worker_requests, 0);
            assert_eq!(got.shape_admissions, 0);
        } else {
            assert!(result.is_err());
        }
        assert_eq!(
            client.calls,
            [
                ("media.status".into(), submitted.clone()),
                ("media.status".into(), submitted.clone()),
                ("certify.observations".into(), json!({})),
            ]
        );
    }
}
