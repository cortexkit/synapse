#![forbid(unsafe_code)]

// Provenance probe: print and exit 0 before the runner parses its
// runtime-required arguments, so harnesses can identify the binary without
// side effects.
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

#[cfg(all(not(target_os = "macos"), unix))]
mod runner;

#[cfg(all(not(target_os = "macos"), unix))]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    runner::main()
}

#[cfg(windows)]
mod runner;

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    runner::main()
}
