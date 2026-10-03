use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use synapse_worker_vulkan::{
    admission::{probe, requirements, Required},
    protocol, KERNEL_REVISION, MANIFEST_DIGEST,
};

#[derive(Parser)]
#[command(name = "ck-synapse-worker-vulkan")]
struct Args {
    #[arg(long)]
    socket: Option<PathBuf>,
    #[cfg(windows)]
    #[arg(long)]
    pipe: Option<String>,
    #[arg(long)]
    nonce: Option<String>,
    #[arg(long)]
    probe_floor: bool,
    #[arg(long)]
    model: Option<String>,
}

fn main() -> Result<()> {
    if std::env::args().skip(1).any(|a| a == "--version") {
        println!(
            "ck-synapse-worker-vulkan {} features={} manifest_digest={} kernel_revision={}",
            env!("CARGO_PKG_VERSION"),
            if cfg!(feature = "vulkan") {
                "vulkan"
            } else {
                "none (vulkan disabled)"
            },
            MANIFEST_DIGEST,
            KERNEL_REVISION
        );
        return Ok(());
    }
    let args = Args::parse();
    if args.probe_floor {
        let result = match requirements(&synapse_worker_vulkan::manifest(), args.model.as_deref()) {
            Ok(required) => probe(required, synapse_worker_vulkan::enumerate),
            Err(code) => synapse_worker_vulkan::admission::Probe {
                status: "refused",
                code: Some(code),
                required: Required {
                    min_storage_buffer_range: 0,
                    min_device_local_bytes: 0,
                },
                observed: None,
            },
        };
        println!("{}", serde_json::to_string(&result)?);
        std::process::exit(result.exit_code());
    }
    let nonce = args.nonce.context("worker requires --nonce")?;
    #[cfg(unix)]
    {
        let socket = args.socket.context("worker requires --socket")?;
        let mut stream = std::os::unix::net::UnixStream::connect(&socket)
            .with_context(|| format!("connect {}", socket.display()))?;
        protocol::session(&mut stream, &nonce, synapse_worker_vulkan::enumerate)
    }
    #[cfg(windows)]
    {
        let pipe = args.pipe.context("worker requires --pipe")?;
        let hello = protocol::hello(&nonce);
        let (mut stream, max_frame) =
            synapse_core::worker_transport::windows_client::connect_and_handshake(
                &pipe,
                &hello,
                synapse_core::DEFAULT_MAX_FRAME_BYTES,
            )?;
        protocol::request_loop(&mut stream, max_frame, synapse_worker_vulkan::enumerate)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = nonce;
        anyhow::bail!("unsupported worker transport")
    }
}
