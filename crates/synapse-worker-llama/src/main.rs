#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
mod runner;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    runner::main()
}

#[cfg(all(not(target_os = "macos"), unix))]
mod runner;

#[cfg(all(not(target_os = "macos"), unix))]
fn main() -> anyhow::Result<()> {
    runner::main()
}

#[cfg(windows)]
mod runner;

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    runner::main()
}
