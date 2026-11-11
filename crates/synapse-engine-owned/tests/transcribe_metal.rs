#![cfg(target_os = "macos")]

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use synapse_core::{RuntimeConfig, TranscribeEngine, TranscribeRequest, ValidatedArtifact};
use synapse_engine_owned::transcribe::{OwnedMetalTranscribeEngine, Progress};
use synapse_parity::speech::{decode_trace, grade_encoder, grade_trace, grade_wer};

fn parity() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity")
}
fn json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn sealed(seal: &Value) -> Vec<u8> {
    let path = if let Some(p) = seal["path"].as_str() {
        parity().join(p)
    } else {
        parity()
            .join(".cache-out")
            .join(seal["cache_path"].as_str().unwrap())
    };
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len() as u64, seal["bytes"].as_u64().unwrap());
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        seal["sha256"].as_str().unwrap()
    );
    bytes
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}
fn cases() -> Vec<Value> {
    let index = json(parity().join("fixtures/index.json"));
    let seal = &index["parakeet-tdt-0.6b-v3.ref-v1.transformers-5.16.1.seed-0"];
    serde_json::from_slice::<Value>(&sealed(seal)).unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}
fn load(maximum: usize) -> (OwnedMetalTranscribeEngine, synapse_core::LoadedModel) {
    let path = std::env::var("SYNAPSE_METAL_SPEECH_MODEL")
        .expect("set SYNAPSE_METAL_SPEECH_MODEL to the pinned HF package directory");
    let mut engine = OwnedMetalTranscribeEngine::new();
    let cfg = RuntimeConfig {
        values: [
            ("model_path".into(), path),
            ("max_encoder_frames".into(), maximum.to_string()),
        ]
        .into(),
    };
    let model = engine
        .load(
            &ValidatedArtifact {
                format: "safetensors-package".into(),
                digest: "test".into(),
            },
            &cfg,
        )
        .unwrap();
    (engine, model)
}
fn input(case: &Value, audio: Option<(&Path, &Value)>) -> TranscribeRequest {
    let mel = if case["log_mel"].is_object() {
        floats(&sealed(&case["log_mel"]))
    } else {
        let (root, index) = audio.expect("set SYNAPSE_METAL_SPEECH_AUDIO for corpus WAVs");
        let entry = &index[case["id"].as_str().unwrap()];
        let bytes = fs::read(root.join(entry["path"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            entry["sha256"].as_str().unwrap()
        );
        let features = synapse_speech::Frontend::new().process_wav(&bytes).unwrap();
        assert_eq!(
            features.samples_16khz as u64,
            case["sample_count"].as_u64().unwrap()
        );
        assert_eq!(
            features.valid_encoder_frames as u64,
            case["valid_encoder_frames"].as_u64().unwrap()
        );
        features.values
    };
    TranscribeRequest {
        mel,
        mel_frames: case["mel_frames"].as_u64().unwrap() as usize,
        valid_mel_frames: case["valid_mel_frames"].as_u64().unwrap() as usize,
        observe: true,
    }
}

unsafe extern "C" {
    fn synapse_parakeet_resident_bytes() -> u64;
    fn synapse_parakeet_peak_resident_bytes() -> u64;
}
fn resident() -> u64 {
    unsafe { synapse_parakeet_resident_bytes() }
}

struct Monitor {
    done: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}
impl Drop for Monitor {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        self.join.take().unwrap().join().unwrap();
    }
}
fn monitor(start: Instant, completed: Arc<AtomicU64>) -> Monitor {
    let done = Arc::new(AtomicBool::new(false));
    let stop = done.clone();
    let join = std::thread::spawn(move || {
        let mut last_print = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(1));
            let idle = (start.elapsed().as_millis() as u64)
                .saturating_sub(completed.load(Ordering::Relaxed));
            let rss = resident();
            if idle >= 110_000 || rss > 48 * 1024 * 1024 * 1024 {
                eprintln!("STOP: progress_gap_ms={idle} resident_bytes={rss}");
                std::process::exit(3);
            }
            // Diagnostic output is not an engine progress event or watchdog reset.
            if last_print.elapsed() >= Duration::from_secs(30) {
                eprintln!(
                    "monitor elapsed_s={:.1} resident_bytes={rss} completed_work_gap_ms={idle}",
                    start.elapsed().as_secs_f64()
                );
                last_print = Instant::now();
            }
        }
    });
    Monitor {
        done,
        join: Some(join),
    }
}

/// Explicit opt-in: these tests use the shared GPU and the pinned 2.51 GB model.
#[test]
#[ignore = "requires pinned model, tensor cache and explicit Metal GPU opt-in"]
fn transcribe_metal_c6_parity() {
    assert_eq!(
        std::env::var("SYNAPSE_METAL_SPEECH_GPU").as_deref(),
        Ok("1")
    );
    let filter = std::env::var("SYNAPSE_METAL_SPEECH_CASE").unwrap_or_default();
    let start = Instant::now();
    let completed = Arc::new(AtomicU64::new(0));
    let _monitor = monitor(start, completed.clone());
    let audio = std::env::var_os("SYNAPSE_METAL_SPEECH_AUDIO").map(PathBuf::from);
    let audio_index = audio.as_ref().map(|p| json(p.join("inputs.json")));
    let manifest = json(parity().join("models.json"));
    let tau = manifest["profiles"]["parakeet-tdt-0.6b-v3.owned-metal"]["transcribe"]["tau_margin"]
        .as_f64()
        .unwrap();
    let evidence =
        std::env::var_os("SYNAPSE_METAL_SPEECH_EVIDENCE").map(|p| fs::File::create(p).unwrap());
    let mut evidence = evidence;
    let (mut engine, model) = load(7500);
    let mut all = cases();
    all.sort_by_key(|c| {
        (
            if c["id"].as_str().unwrap().starts_with("ladder-") {
                2
            } else if c["encoder"].is_object() {
                0
            } else {
                1
            },
            c["duration_ms"].as_u64().unwrap(),
        )
    });
    let mut count = 0;
    let mut tensors = 0;
    let mut exempt = 0;
    let mut refused = 0;
    let mut wer = vec![];
    let mut maximum = 0;
    let mut peak = 0;
    let mut longest_gap = 0;
    for case in all
        .iter()
        .filter(|c| c["id"].as_str().unwrap().contains(&filter))
    {
        let case_start = Instant::now();
        let mut progress = vec![];
        let mut previous = Instant::now();
        let mut gap = 0;
        completed.store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
        let request = input(case, audio.as_deref().zip(audio_index.as_ref()));
        if case["valid_encoder_frames"].as_u64().unwrap() > 7500 {
            let result = engine.transcribe_with_progress(&model, request, |_| {
                panic!("above-limit request performed GPU work")
            });
            assert!(result.unwrap_err().message.contains("max_encoder_frames"));
            refused += 1;
            let row = serde_json::json!({"id":case["id"],"expected_refusal":true});
            eprintln!("{row}");
            if let Some(f) = evidence.as_mut() {
                writeln!(f, "{row}").unwrap();
                f.flush().unwrap();
            }
            continue;
        }
        let output = engine
            .transcribe_with_progress(&model, request.clone(), |p| {
                gap = gap.max(previous.elapsed().as_millis() as u64);
                previous = Instant::now();
                completed.store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                progress.push(p);
            })
            .unwrap();
        gap = gap.max(previous.elapsed().as_millis() as u64);
        longest_gap = longest_gap.max(gap);
        peak = peak.max(resident());
        assert_eq!(
            output.valid_encoder_frames,
            case["valid_encoder_frames"].as_u64().unwrap() as usize
        );
        let obs = output.observation.as_ref().unwrap();
        assert_eq!(obs.encoder_output.len(), output.valid_encoder_frames * 1024);
        assert!(obs.encoder_output.iter().all(|x| x.is_finite()));
        let cosine = case["encoder"].is_object().then(|| {
            grade_encoder(
                &floats(&sealed(&case["encoder"])),
                &obs.encoder_output,
                output.valid_encoder_frames,
            )
            .unwrap()
        });
        let got: Vec<_> = output
            .triples
            .iter()
            .map(|t| [t.token_id, t.frame_index, t.duration])
            .collect();
        let trace =
            decode_trace(&synapse_core::encode_decode_trace_frame(&obs.decode_trace)).unwrap();
        let margin_exempt = grade_trace(
            &decode_trace(&sealed(&case["trace"])).unwrap(),
            &trace,
            &got,
            output.valid_encoder_frames as u64,
            tau,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case["id"]));
        exempt += usize::from(margin_exempt);
        if case["encoder"].is_object() {
            tensors += 1;
            let mut plain_request = request;
            plain_request.observe = false;
            let plain = engine
                .transcribe_with_progress(&model, plain_request, |_| {
                    completed.store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                })
                .unwrap();
            assert_eq!(output.triples, plain.triples);
            assert_eq!(output.valid_encoder_frames, plain.valid_encoder_frames);
            assert!(plain.observation.is_none());
        }
        if let Some(split) = case["split"].as_str() {
            let ids: Vec<_> = output.triples.iter().map(|t| t.token_id as u32).collect();
            let text = synapse_speech::Tokenizer::pinned().decode(&ids).unwrap();
            wer.push((
                split.to_owned(),
                case["ground_truth"].as_str().unwrap().to_owned(),
                case["text"].as_str().unwrap().to_owned(),
                text,
            ));
        }
        let expected_progress: Vec<_> = (1..=24)
            .map(Progress::EncoderLayersCompleted)
            .chain(
                (125..=output.valid_encoder_frames)
                    .step_by(125)
                    .map(Progress::DecodedFrames),
            )
            .collect();
        assert_eq!(progress, expected_progress);
        count += 1;
        if case["id"].as_str().unwrap().starts_with("ladder-") {
            maximum = maximum.max(case["duration_ms"].as_u64().unwrap());
        }
        let row = serde_json::json!({"id":case["id"],"valid_encoder_frames":output.valid_encoder_frames,"min_cosine":cosine,"margin_exempt":margin_exempt,"seconds":case_start.elapsed().as_secs_f64(),"progress_gap_ms":gap,"resident_bytes":resident()});
        eprintln!("{row}");
        if let Some(f) = evidence.as_mut() {
            writeln!(f, "{row}").unwrap();
            f.flush().unwrap();
        }
    }
    assert!(count > 0);
    let split_wer = grade_wer(
        wer.iter()
            .map(|(a, b, c, d)| (a.as_str(), b.as_str(), c.as_str(), d.as_str())),
    )
    .unwrap();
    if filter.is_empty() {
        assert_eq!(count, 1244);
        assert_eq!(refused, 3);
        assert_eq!(tensors, 76);
        assert_eq!(split_wer.len(), 10);
        assert_eq!(maximum, 600000);
    }
    peak = peak.max(unsafe { synapse_parakeet_peak_resident_bytes() });
    let summary = serde_json::json!({"graded":count,"expected_refusals":refused,"tensor_observe_pairs":tensors,"margin_exempt":exempt,"split_wer":split_wer,"largest_ladder_ms":maximum,"maximum_encoder_frames":7500,"peak_resident_bytes":peak,"longest_progress_gap_ms":longest_gap,"seconds":start.elapsed().as_secs_f64(),"kernel_revision":synapse_core::METAL_TRANSCRIBE_KERNEL_REVISION});
    eprintln!("SUMMARY {summary}");
    if let Some(f) = evidence.as_mut() {
        writeln!(f, "{summary}").unwrap();
    }
    engine.unload(&model);
}
