#![cfg_attr(not(target_os = "macos"), forbid(unsafe_code))]

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
mod runner;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    runner::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    anyhow::bail!("ck-synapse-worker-decode requires macOS Metal")
}
