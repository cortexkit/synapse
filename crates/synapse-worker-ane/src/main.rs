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
mod macos;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    macos::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    if version_probe() {
        return Ok(());
    }
    anyhow::bail!("synapse-worker-ane is only supported on macOS");
}

#[cfg(test)]
mod tests {
    use synapse_core::worker_engine_names::ANE_WORKER_ENGINE;

    #[test]
    fn swift_hello_identity_matches_rust_constant_and_has_protocol_control() {
        let artifact = std::path::Path::new(env!("SYNAPSE_ANE_SWIFT_WORKER"));
        let bytes = if artifact.is_file() {
            std::fs::read(artifact).expect("read built Swift worker")
        } else {
            include_bytes!("../swift/ane_worker.swift").to_vec()
        };
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains(ANE_WORKER_ENGINE),
            "Swift worker HELLO must contain the canonical engine identity"
        );
        assert!(
            text.contains("module rejected worker handshake"),
            "Swift worker scan positive control is missing"
        );
    }
}
