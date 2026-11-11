#[cfg(not(feature = "native"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("build with --features metal, cuda, or vulkan before running this lane")
}

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    runner::run()
}

#[cfg(feature = "native")]
mod runner {
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    use anyhow::{ensure, Result};
    use clap::Parser;
    use stt_common::{
        input::read_pcm,
        measurement::{Transcript, WorkerResult},
        Request,
    };
    use transcribe_cpp::{
        init_backends_with, Backend, BackendMask, Model, ModelOptions, RunOptions,
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

    pub fn run() -> Result<()> {
        let args = Args::parse();
        let backend = match args.backend.as_str() {
            "metal" => Backend::Metal,
            "cuda" => Backend::Cuda,
            "vulkan" => Backend::Vulkan,
            _ => anyhow::bail!("backend must be metal, cuda or vulkan"),
        };
        let request: Request = serde_json::from_slice(&fs::read(&args.request)?)?;
        ensure!(!request.utterances.is_empty(), "empty corpus");
        let start = Instant::now();
        init_backends_with(None::<&std::path::Path>, BackendMask::for_backend(backend))?;
        let model = Model::load_with(
            &args.model,
            &ModelOptions {
                backend,
                ..Default::default()
            },
        )?;
        ensure!(
            model.backend().eq_ignore_ascii_case(&args.backend),
            "requested {} but model uses {}",
            args.backend,
            model.backend()
        );
        let mut session = model.session()?;
        let options = RunOptions::default();
        let first = read_pcm(&request.utterances[0].path)?;
        session.run(&first, &options)?;
        let cold = start.elapsed().as_secs_f64();
        let warm = Instant::now();
        let mut transcripts = Vec::new();
        for utterance in &request.utterances {
            let before = Instant::now();
            let pcm = read_pcm(&utterance.path)?;
            let transcript = session.run(&pcm, &options)?;
            transcripts.push(Transcript {
                id: utterance.id.clone(),
                text: Some(transcript.text),
                seconds: Some(before.elapsed().as_secs_f64()),
                error: None,
            });
        }
        let result = WorkerResult {
            engine: "transcribe.cpp (Hex fork)".into(),
            version: transcribe_cpp::version(),
            commit: "bcb639faa4c74a2300ba19e30360cb2355b1663a".into(),
            backend: args.backend,
            cold_start_seconds: cold,
            warm_wall_seconds: warm.elapsed().as_secs_f64(),
            peak_rss_bytes: peak_rss()?,
            gpu_peak_bytes: None,
            energy_joules: None,
            transcripts,
        };
        fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
        Ok(())
    }

    #[cfg(unix)]
    fn peak_rss() -> Result<u64> {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // getrusage initializes the supplied structure on success.
        ensure!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
            "getrusage failed"
        );
        let usage = unsafe { usage.assume_init() };
        let unit = if cfg!(target_os = "macos") { 1 } else { 1024 };
        Ok(usage.ru_maxrss as u64 * unit)
    }

    #[cfg(windows)]
    fn peak_rss() -> Result<u64> {
        use windows_sys::Win32::System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            Threading::GetCurrentProcess,
        };
        let mut counters = std::mem::MaybeUninit::<PROCESS_MEMORY_COUNTERS>::zeroed();
        // Windows reports PeakWorkingSetSize in bytes for the current process.
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
}
