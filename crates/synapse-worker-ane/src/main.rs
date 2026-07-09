#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    macos::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("synapse-worker-ane is only supported on macOS");
}
