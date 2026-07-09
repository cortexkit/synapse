#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
mod mlx_main;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    mlx_main::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("synapse-worker-mlx is only supported on macOS");
}
