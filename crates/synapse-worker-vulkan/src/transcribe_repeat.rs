//! Repeat one hardware fixture on the same engine without grading it against the reference.
use super::{exchange, Loaded, LoadedEngine};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, rc::Rc, time::Instant};

#[path = "transcribe_repeat_load.rs"]
mod host_load;

const REPEAT: &str = "SYNAPSE_VULKAN_SPEECH_REPEAT";
const CAPTURE: &str = "SYNAPSE_VULKAN_SPEECH_CAPTURE_LAYERS";
const LIMIT: &str = "SYNAPSE_VULKAN_SPEECH_DISPATCH_LIMIT";
const DISJOINT: &str = "SYNAPSE_VULKAN_SPEECH_DISJOINT_BINDINGS";

pub(super) struct Probe {
    count: usize,
    capture: bool,
    dispatch_limit: Option<usize>,
    disjoint: bool,
    host_load: host_load::Config,
}

fn env(name: &str) -> Option<String> {
    std::env::var_os(name).map(|value| {
        value
            .into_string()
            .unwrap_or_else(|_| panic!("{name} must be UTF-8"))
    })
}

fn enabled(value: Option<&str>, name: &str) -> bool {
    match value {
        None | Some("0") => false,
        Some("1") => true,
        _ => panic!("{name} must be 0 or 1"),
    }
}

impl Probe {
    pub(super) fn from_env() -> Option<Self> {
        let capture = env(CAPTURE);
        let dispatch_limit = env(LIMIT);
        let disjoint = env(DISJOINT);
        let host_load = host_load::Config::from_env();
        let Some(count) = env(REPEAT) else {
            assert!(
                capture.is_none()
                    && dispatch_limit.is_none()
                    && disjoint.is_none()
                    && host_load.is_none(),
                "encoder probe switches require {REPEAT}"
            );
            return None;
        };
        let count = count
            .parse()
            .ok()
            .filter(|&n| n >= 2)
            .unwrap_or_else(|| panic!("{REPEAT} must be an integer of at least 2"));
        Some(Self {
            count,
            capture: enabled(capture.as_deref(), CAPTURE),
            dispatch_limit: dispatch_limit.map(|value| {
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("{LIMIT} must be an integer"))
            }),
            disjoint: enabled(disjoint.as_deref(), DISJOINT),
            host_load: host_load.unwrap_or_default(),
        })
    }

    pub(super) fn run(
        &self,
        loaded: &mut Loaded,
        id: &str,
        mel: &[f32],
        frames: usize,
        valid: usize,
    ) {
        let LoadedEngine::Speech(engine) = &mut loaded.engine else {
            panic!("wrong engine");
        };
        engine.debug_clear_capture();
        engine
            .debug_set_encoder_probe(self.dispatch_limit, self.disjoint)
            .unwrap();
        let readback = engine.debug_encoder_readback_path();
        let captured = Rc::new(RefCell::new(Vec::new()));
        let mut names = Vec::new();
        if self.capture {
            names.push("subsampling".to_owned());
            names.extend((0..24).map(|i| format!("layer-{i:02}")));
            let captured = Rc::clone(&captured);
            engine.debug_capture_layers(names.iter().cloned().collect(), move |name, values| {
                captured
                    .borrow_mut()
                    .push((name.to_owned(), encoder_hash(values)));
                Ok(())
            });
        }

        let mut first_encoder = None;
        let mut first_layers = Vec::new();
        let load = host_load::Running::start(self.host_load).unwrap();
        for run in 1..=self.count {
            let started = Instant::now();
            // Keep the normal final observation so the hash is available even
            // when no intermediate readbacks are enabled.
            let (output, _, _) = exchange(loaded, id, mel, frames, valid, true, None);
            let LoadedEngine::Speech(engine) = &loaded.engine else {
                unreachable!("wrong engine")
            };
            let (submission_stats, next_dispatch_limit) = engine.debug_submission_stats();
            let hash = encoder_hash(&output.encoder_output.expect("missing final encoder"));
            let layers = captured.take();
            assert!(layers.iter().map(|(name, _)| name).eq(&names));
            let first_difference = if run == 1 {
                None
            } else {
                first_differing_layer(&first_layers, &layers)
            };
            println!(
                "{}",
                json!({"vulkan_speech_repeat": {
                    "fixture": id,
                    "run": run,
                    "encoder_sha256": hash,
                    "encoder_matches_run_1": first_encoder.as_ref().is_none_or(|first| first == &hash),
                    "encoder_readback": readback,
                    "capture_layers": self.capture,
                    "first_differing_layer": first_difference,
                    "dispatch_limit_override": self.dispatch_limit,
                    "next_dispatch_limit": next_dispatch_limit,
                    "encoder_submissions": submission_stats,
                    "disjoint_bindings": self.disjoint,
                    "elapsed_s": started.elapsed().as_secs_f64(),
                }})
            );
            if run == 1 {
                first_encoder = Some(hash);
                first_layers = layers;
            }
        }
        if let Some(load) = load {
            load.report().unwrap();
        }
    }
}

pub(super) fn encoder_hash(values: &[f32]) -> String {
    format!(
        "{:x}",
        Sha256::digest(synapse_core::encode_f32_frame(values))
    )
}

fn first_differing_layer<'a>(
    first: &[(String, String)],
    current: &'a [(String, String)],
) -> Option<&'a str> {
    first
        .iter()
        .zip(current)
        .find_map(|((_, first_hash), (name, hash))| (first_hash != hash).then_some(name.as_str()))
}

#[test]
fn speech_repeat_uses_execution_order_for_first_differing_layer() {
    let layer = |name: &str, value| (name.to_owned(), encoder_hash(&[value]));
    let baseline = vec![layer("subsampling", 0.0), layer("layer-00", 1.0)];
    let changed = vec![layer("subsampling", -0.0), layer("layer-00", 2.0)];
    assert_eq!(first_differing_layer(&baseline, &baseline), None);
    assert_eq!(
        first_differing_layer(&baseline, &changed),
        Some("subsampling")
    );
    assert_eq!(
        first_differing_layer(&baseline[1..], &changed[1..]),
        Some("layer-00")
    );
}
