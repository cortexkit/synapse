use burn_onnx::ModelGen;
use std::env;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

const QWEN_CACHE_REPO: &str = "models--onnx-community--Qwen3-Embedding-0.6B-ONNX";
const MINILM_CACHE_REPO: &str = "models--Qdrant--all-MiniLM-L6-v2-onnx";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SYNAPSE_BURN_QWEN_ONNX");
    println!("cargo:rerun-if-env-changed=SYNAPSE_BURN_MINILM_ONNX");
    println!("cargo:rerun-if-env-changed=HOME");

    let qwen_path = env::var_os("SYNAPSE_BURN_QWEN_ONNX")
        .map(PathBuf::from)
        .or_else(find_qwen_model);
    let minilm_path = env::var_os("SYNAPSE_BURN_MINILM_ONNX")
        .map(PathBuf::from)
        .or_else(find_minilm_model);

    if let Some(path) = &qwen_path {
        println!("cargo:rerun-if-changed={}", path.display());
        if let Some(sidecar) = qwen_sidecar(path) {
            println!("cargo:rerun-if-changed={}", sidecar.display());
        }
    }
    if let Some(path) = &minilm_path {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let qwen_probe = match qwen_path {
        Some(path) => probe_qwen_import(&path),
        None => ProbeOutcome::Skipped(
            "Qwen ONNX not found in ~/.cache/huggingface or SYNAPSE_BURN_QWEN_ONNX".into(),
        ),
    };

    let minilm_path = match minilm_path {
        Some(path) => path,
        None => {
            panic!(
                "all-MiniLM-L6-v2 ONNX not found. Put model.onnx at ~/.cache/huggingface/hub/{MINILM_CACHE_REPO}/snapshots/*/model.onnx or set SYNAPSE_BURN_MINILM_ONNX. Qwen probe: {}",
                qwen_probe.summary()
            );
        }
    };

    generate_model(&minilm_path);
    write_model_info(&minilm_path, &qwen_probe).expect("write model_info.rs");
}

fn probe_qwen_import(path: &Path) -> ProbeOutcome {
    if !path.exists() {
        return ProbeOutcome::Skipped(format!("Qwen ONNX path does not exist: {}", path.display()));
    }

    let probe_dir = "qwen-probe/";
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut gen = ModelGen::new();
        gen.input(path.to_str().expect("utf-8 qwen path"))
            .out_dir(probe_dir)
            .run_from_script();
    }));

    match result {
        Ok(()) => ProbeOutcome::Imported(format!(
            "Qwen import unexpectedly succeeded for {}",
            path.display()
        )),
        Err(payload) => ProbeOutcome::Failed(payload_to_string(payload)),
    }
}

fn generate_model(path: &Path) {
    if !path.exists() {
        panic!("MiniLM ONNX path does not exist: {}", path.display());
    }
    let mut gen = ModelGen::new();
    gen.input(path.to_str().expect("utf-8 minilm path"))
        .out_dir("model/")
        .run_from_script();
}

fn write_model_info(minilm_path: &Path, qwen_probe: &ProbeOutcome) -> std::io::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let model_info = out_dir.join("model_info.rs");
    let qwen_error = match qwen_probe {
        ProbeOutcome::Failed(msg) => format!("Some({msg:?})"),
        _ => "None".into(),
    };
    let build_notes = match qwen_probe {
        ProbeOutcome::Imported(msg) => format!(
            "Qwen probe: {msg}. Lane is wired to the validated MiniLM fallback until a Qwen-specific runtime wrapper is implemented."
        ),
        ProbeOutcome::Failed(msg) => format!(
            "Qwen probe failed in burn-onnx; exact error: {msg}. Falling back to all-MiniLM-L6-v2, which burn-onnx validates upstream."
        ),
        ProbeOutcome::Skipped(msg) => format!(
            "Qwen probe skipped: {msg}. Building the validated all-MiniLM-L6-v2 fallback."
        ),
    };
    fs::write(
        model_info,
        format!(
            r#"pub const COMPILED_TARGET: &str = "all-MiniLM-L6-v2";
pub const COMPILED_MODEL_PATH: &str = {compiled_model_path:?};
pub const BUILD_NOTES: &str = {build_notes:?};
pub const QWEN_IMPORT_ERROR: Option<&str> = {qwen_error};
pub const WEIGHTS_PATH: &str = concat!(env!("OUT_DIR"), "/model/model.bpk");

pub mod generated_model {{
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}}
"#,
            compiled_model_path = minilm_path.display().to_string(),
            build_notes = build_notes,
            qwen_error = qwen_error,
        ),
    )
}

fn payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "burn-onnx panicked with a non-string payload".into(),
        },
    }
}

fn qwen_sidecar(path: &Path) -> Option<PathBuf> {
    let sidecar = path.with_file_name("model.onnx_data");
    sidecar.exists().then_some(sidecar)
}

fn find_qwen_model() -> Option<PathBuf> {
    find_latest_snapshot(QWEN_CACHE_REPO, &["onnx", "model.onnx"])
}

fn find_minilm_model() -> Option<PathBuf> {
    find_latest_snapshot(MINILM_CACHE_REPO, &["model.onnx"])
}

fn find_latest_snapshot(repo: &str, suffix: &[&str]) -> Option<PathBuf> {
    let home = env::var_os("HOME")?;
    let snapshots_dir = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| entry.path())
        .collect();
    entries.sort();
    entries.reverse();
    for entry in entries {
        let candidate = suffix.iter().fold(entry.clone(), |path, segment| path.join(segment));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

enum ProbeOutcome {
    Imported(String),
    Failed(String),
    Skipped(String),
}

impl ProbeOutcome {
    fn summary(&self) -> &str {
        match self {
            ProbeOutcome::Imported(msg) | ProbeOutcome::Failed(msg) | ProbeOutcome::Skipped(msg) => {
                msg.as_str()
            }
        }
    }
}
