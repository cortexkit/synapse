fn main() {
    for source in [
        "src/metal_mpsgraph.m",
        "src/qwen3_mpsgraph.m",
        "src/modernbert_mpsgraph.m",
        "src/mpsgraph_runtime.m",
        "src/mpsgraph_runtime.h",
    ] {
        println!("cargo:rerun-if-changed={source}");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    cc::Build::new()
        .file("src/mpsgraph_runtime.m")
        .file("src/metal_mpsgraph.m")
        .file("src/qwen3_mpsgraph.m")
        .file("src/modernbert_mpsgraph.m")
        .flag("-fobjc-exceptions")
        .compile("synapse_owned_mpsgraph");

    for framework in [
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}
