fn main() {
    println!("cargo:rerun-if-changed=src/metal_mpsgraph.m");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    cc::Build::new()
        .file("src/metal_mpsgraph.m")
        .flag("-fobjc-exceptions")
        .compile("synapse_unified_rt_mpsgraph");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
}
