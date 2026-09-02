use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=swift/ane_worker.swift");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    // Same file name the launcher looks for beside itself in installed layouts,
    // so release packaging can copy this artifact next to the launcher as-is.
    let output = out_dir.join("ck-synapse-worker-ane-swift");
    println!(
        "cargo:rustc-env=SYNAPSE_ANE_SWIFT_WORKER={}",
        output.display()
    );

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    // The Swift worker answers `--version` with the crate version; swiftc has
    // no value-carrying define, so the constant is generated as a source file.
    let version_source = out_dir.join("crate_version.swift");
    std::fs::write(
        &version_source,
        format!(
            "let crateVersion = \"{}\"\n",
            env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION is set")
        ),
    )
    .expect("write generated Swift version source");

    let status = Command::new("swiftc")
        .arg("-O")
        .arg("-parse-as-library")
        .arg("swift/ane_worker.swift")
        .arg(&version_source)
        .arg("-o")
        .arg(&output)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!("swiftc failed with status {status}"),
        Err(error) => panic!("failed to run swiftc: {error}"),
    }
}
