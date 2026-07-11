#![forbid(unsafe_code)]

// Provenance probe: print and exit 0 before any runtime-required argument or
// platform gate, so harnesses can identify the binary without side effects.
fn version_probe() -> bool {
    if std::env::args().skip(1).any(|arg| arg == "--version") {
        println!(concat!(
            env!("CARGO_BIN_NAME"),
            " ",
            env!("CARGO_PKG_VERSION")
        ));
        return true;
    }
    false
}

#[cfg(target_os = "macos")]
mod mlx_main;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    mlx_main::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    anyhow::bail!("synapse-worker-mlx is only supported on macOS");
}
