fn main() {
    println!("cargo:rerun-if-changed=src/metal_mpsgraph.m");
    println!("cargo:rerun-if-changed=src/qwen3_mpsgraph.m");
    println!("cargo:rerun-if-changed=src/modernbert_mpsgraph.m");
    println!("cargo:rerun-if-changed=src/mpsgraph_runtime.m");
    println!("cargo:rerun-if-changed=src/mpsgraph_runtime.h");
    println!("cargo:rerun-if-changed=src/cuda_minilm.cu");
    println!("cargo:rerun-if-changed=src/cuda_minilm.h");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        cc::Build::new()
            .file("src/mpsgraph_runtime.m")
            .file("src/metal_mpsgraph.m")
            .file("src/qwen3_mpsgraph.m")
            .file("src/modernbert_mpsgraph.m")
            .flag("-fobjc-exceptions")
            .compile("synapse_unified_rt_mpsgraph");

        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }

    if target_os == "linux" && std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let cuda_root = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("shared")
            .file("src/cuda_minilm.cu")
            .include(format!("{cuda_root}/include"))
            .flag("-O3")
            .flag("-Wno-deprecated-gpu-targets");
        if std::env::var("PROFILE").as_deref() == Ok("release") {
            build.flag("-lineinfo");
        }
        build.compile("synapse_unified_rt_cuda");
        println!("cargo:rustc-link-search=native={cuda_root}/lib64");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
    }
}
