use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=swift/ane_worker.swift");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let output = out_dir.join("synapse-worker-ane-swift");
    println!(
        "cargo:rustc-env=SYNAPSE_ANE_SWIFT_WORKER={}",
        output.display()
    );

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let status = Command::new("swiftc")
        .arg("-O")
        .arg("-parse-as-library")
        .arg("swift/ane_worker.swift")
        .arg("-o")
        .arg(&output)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!("swiftc failed with status {status}"),
        Err(error) => panic!("failed to run swiftc: {error}"),
    }
}
