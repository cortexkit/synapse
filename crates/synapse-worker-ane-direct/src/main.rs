#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod modernbert;
#[cfg(target_os = "macos")]
mod profile;
#[cfg(target_os = "macos")]
mod qwen;
#[cfg(target_os = "macos")]
mod worker;
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    worker::main()
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("{{\"status\":\"refused\",\"code\":\"ane_private_api_unavailable\",\"required\":null,\"observed\":null}}");
    std::process::exit(2);
}
