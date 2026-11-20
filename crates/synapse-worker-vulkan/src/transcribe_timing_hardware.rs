//! Isolate one production encoder request from grading, capture and decode.
use super::{env_path, load_hardware, speech, Artifact, LoadedEngine, PathBuf, Value};
use crate::runtime::speech_timing::{self, CHILD};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;

#[test]
#[ignore = "requires pinned Vulkan speech package and ladder assets; run the AMD timing kit"]
fn vulkan_speech_time_ladder_encoder() {
    let id =
        std::env::var("SYNAPSE_VULKAN_SPEECH_FILTER").expect("set the exact ladder fixture id");
    let frames = match id.as_str() {
        "ladder-2048" => 2048,
        "ladder-4096" => 4096,
        "ladder-600000ms" => 7500,
        _ => panic!("expected ladder-2048, ladder-4096 or ladder-600000ms, got {id}"),
    };
    if std::env::var_os(CHILD).is_none() {
        let name = module_path!().split_once("::").unwrap().1;
        speech_timing::supervise(
            &format!("{name}::vulkan_speech_time_ladder_encoder"),
            "encoder_finished",
        );
        return;
    }

    speech_timing::event(
        json!({"event":"phase_start", "phase":"fixture_preparation", "fixture":id}),
    );
    let started = Instant::now();
    let parity = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
    let fixtures = speech::load_transcribe_fixtures(&parity, &crate::manifest()).unwrap();
    let fixture = fixtures
        .iter()
        .find(|f| f.id == id)
        .expect("fixture not pinned");
    assert_eq!(fixture.valid_encoder_frames, frames as u64);
    let reference: Value = serde_json::from_slice(
        &std::fs::read(parity.join("fixtures/parakeet-tdt-0.6b-v3/reference.json")).unwrap(),
    )
    .unwrap();
    let row = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == id)
        .unwrap();
    let mel_rows = row["mel_frames"].as_u64().unwrap() as usize;
    let valid = row["valid_mel_frames"].as_u64().unwrap() as usize;
    let mel = if !row["log_mel"].is_null() {
        let artifact: Artifact = serde_json::from_value(row["log_mel"].clone()).unwrap();
        let mirror = env_path("SYNAPSE_VULKAN_SPEECH_MIRROR");
        synapse_core::decode_f32_frame(&artifact.read(&parity, &mirror).unwrap()).unwrap()
    } else {
        let audio = env_path("SYNAPSE_VULKAN_SPEECH_AUDIO");
        let index: Value =
            serde_json::from_slice(&std::fs::read(audio.join("inputs.json")).unwrap()).unwrap();
        let indexed = &index[&id];
        let bytes = std::fs::read(audio.join(indexed["path"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            indexed["sha256"].as_str().unwrap()
        );
        let features = synapse_speech::Frontend::new().process_wav(&bytes).unwrap();
        assert_eq!(
            (features.mel_frames, features.valid_mel_frames),
            (mel_rows, valid)
        );
        features.values
    };
    speech_timing::event(json!({"event":"fixture_ready", "mel_rows":mel_rows,
        "valid_mel_rows":valid, "wall_ms":started.elapsed().as_secs_f64() * 1e3}));
    speech_timing::event(json!({"event":"phase_start", "phase":"load", "fixture":id}));
    let started = Instant::now();
    let mut loaded = load_hardware();
    let LoadedEngine::Speech(engine) = &mut loaded.engine else {
        panic!("wrong engine");
    };
    speech_timing::event(json!({"event":"model_loaded", "decoder":engine.decoder(),
        "wall_ms":started.elapsed().as_secs_f64() * 1e3}));
    speech_timing::event(json!({"event":"phase_start", "phase":"encoder_setup", "fixture":id}));
    engine
        .debug_time_encoder(&mel, mel_rows, valid, frames)
        .unwrap();
}
