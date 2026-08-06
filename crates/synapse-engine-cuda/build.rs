fn main() {
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PROFILE");

    for source in [
        "src/port/cuda_family_common.cuh",
        "src/port/cuda_minilm.h",
        "src/port/cuda_minilm.cu",
        "src/port/cuda_modernbert.cu",
        "src/port/cuda_qwen3.cu",
    ] {
        println!("cargo:rerun-if-changed={source}");
    }

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:warning=owned CUDA is disabled on macOS");
        return;
    }

    let cuda_root = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/usr/local/cuda"));
    let include = cuda_root.join("include");

    let mut build = cc::Build::new();
    build
        .compiler(cuda_root.join("bin/nvcc"))
        .cpp(true)
        .no_default_flags(true)
        .warnings(false)
        .extra_warnings(false)
        .flag("-Xcompiler=-fPIC")
        .include("src/port")
        .include(&include)
        // V1 distributes virtual PTX only. Do not add an sm_* SASS image here.
        .flag("-gencode=arch=compute_75,code=compute_75")
        .flag("-O3")
        .flag("-Wno-deprecated-gpu-targets")
        .file("src/port/cuda_minilm.cu")
        .file("src/port/cuda_modernbert.cu")
        .file("src/port/cuda_qwen3.cu");
    if std::env::var("PROFILE").as_deref() == Ok("release") {
        build.flag("-lineinfo");
    }
    build.compile("synapse_owned_cuda");

    let lib_dir = if target_os == "windows" {
        cuda_root.join("lib/x64")
    } else {
        cuda_root.join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudart");
}
