fn main() {
    synapse_build_stamp::emit_git_provenance();
    let manifest = "../../parity/models.json";
    println!("cargo:rerun-if-changed={manifest}");
    std::fs::copy(
        manifest,
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("models.json"),
    )
    .unwrap();
}
