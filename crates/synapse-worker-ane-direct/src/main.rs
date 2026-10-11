#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod modernbert;
// Fixed-slot multi-row layouts remain experimental. Serving compiles the kernel
// that packs rows into 256 columns separately from single-row kernels, charging
// its 28 layer executables to the same 100-executable ANE residency budget.
#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
mod multirow;
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
