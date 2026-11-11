fn main() {
    println!("cargo:rerun-if-changed=src/memory_macos.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=proc");
        cc::Build::new()
            .file("src/memory_macos.c")
            .compile("stt_memory");
    }
}
