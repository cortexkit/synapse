use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{ensure, Result};
use clap::Parser;
use stt_common::{
    measurement::{Transcript, WorkerResult},
    Request,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    backend: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite worker result");
    let request: Request = serde_json::from_slice(&fs::read(&args.request)?)?;
    let mut result = benchmark(&request, &args.backend, || {
        let mut engine = stt_synapse::load(&args.backend, &args.model)?;
        let frontend = synapse_speech::Frontend::new();
        let tokenizer = synapse_speech::Tokenizer::pinned();
        Ok(move |path: &Path| {
            let features = frontend.process_wav(&fs::read(path)?)?;
            tokenizer
                .decode(&engine.tokens(features)?)
                .map_err(Into::into)
        })
    })?;
    result.peak_rss_bytes = peak_rss()?;
    fs::write(args.output, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}

fn benchmark<F: FnMut(&Path) -> Result<String>>(
    request: &Request,
    backend: &str,
    load: impl FnOnce() -> Result<F>,
) -> Result<WorkerResult> {
    ensure!(!request.utterances.is_empty(), "empty corpus");
    let start = Instant::now();
    let mut transcribe = load()?;
    let memory_probe = stt_common::memory::loaded_barrier()?;
    transcribe(&request.utterances[0].path)?;
    let cold_start_seconds = start.elapsed().saturating_sub(memory_probe).as_secs_f64();
    let warm = Instant::now();
    let mut transcripts = Vec::new();
    for utterance in &request.utterances {
        let start = Instant::now();
        let text = transcribe(&utterance.path)?;
        transcripts.push(Transcript {
            id: utterance.id.clone(),
            text: Some(text),
            seconds: Some(start.elapsed().as_secs_f64()),
            error: None,
        });
    }
    Ok(WorkerResult {
        engine: "synapse".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        commit: option_env!("SYNAPSE_BUILD_REV").unwrap_or("unknown").into(),
        backend: backend.into(),
        cold_start_seconds,
        warm_wall_seconds: warm.elapsed().as_secs_f64(),
        peak_rss_bytes: 0,
        gpu_peak_bytes: None,
        energy_joules: None,
        transcripts,
    })
}

#[cfg(unix)]
fn peak_rss() -> Result<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    ensure!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
        "getrusage failed"
    );
    Ok(unsafe { usage.assume_init() }.ru_maxrss as u64
        * if cfg!(target_os = "macos") { 1 } else { 1024 })
}

#[cfg(windows)]
fn peak_rss() -> Result<u64> {
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };
    let mut counters = std::mem::MaybeUninit::<PROCESS_MEMORY_COUNTERS>::zeroed();
    ensure!(
        unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                counters.as_mut_ptr(),
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
        } != 0,
        "GetProcessMemoryInfo failed"
    );
    Ok(unsafe { counters.assume_init() }.PeakWorkingSetSize as u64)
}

#[test]
fn stt_synapse_discards_exactly_one_warmup_and_measures_full_pass() {
    use stt_common::input::Utterance;
    let request = Request {
        utterances: ["a", "b"]
            .iter()
            .map(|id| Utterance {
                id: (*id).into(),
                path: (*id).into(),
                split: "test".into(),
                locale: "en-US".into(),
                duration_seconds: 1.0,
            })
            .collect(),
    };
    let mut calls = Vec::new();
    let result = benchmark(&request, "fake", || {
        Ok(|path: &Path| {
            calls.push(path.to_owned());
            Ok(format!("call {}", calls.len()))
        })
    })
    .unwrap();
    assert_eq!(calls, [Path::new("a"), Path::new("a"), Path::new("b")]);
    assert_eq!(result.transcripts.len(), 2);
    assert_eq!(result.transcripts[0].text.as_deref(), Some("call 2"));
    assert_eq!(result.transcripts[1].text.as_deref(), Some("call 3"));
    assert!(result.transcripts.iter().all(|t| t.seconds.unwrap() > 0.0));
    assert!(result.cold_start_seconds > 0.0 && result.warm_wall_seconds > 0.0);
}
