#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
include!("mlx_main.rs");

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("synapse-worker-mlx is only supported on macOS");
}
