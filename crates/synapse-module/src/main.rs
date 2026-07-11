#![forbid(unsafe_code)]

#[tokio::main]
async fn main() {
    // Provenance probe: must print and exit 0 before any runtime-required
    // argument (--subc) is parsed, so harnesses can identify the binary
    // without side effects.
    if std::env::args().skip(1).any(|arg| arg == "--version") {
        println!(concat!(
            env!("CARGO_BIN_NAME"),
            " ",
            env!("CARGO_PKG_VERSION")
        ));
        return;
    }
    if let Err(error) = synapse_module::run_from_env().await {
        if matches!(error, synapse_module::ModuleError::SingletonHeld(_)) {
            std::process::exit(1);
        }
        panic!("synapse module failed: {error}");
    }
}
