fn main() {
    println!("cargo:rerun-if-changed=src/metal_mpsgraph.m");
    println!("cargo:rerun-if-changed=src/qwen3_mpsgraph.m");
    println!("cargo:rerun-if-changed=src/qwen3_decode_mpsgraph.m");
    println!("cargo:rerun-if-changed=src/qwen3_decode_metal_step.m");
    println!("cargo:rerun-if-changed=src/qwen3_decode_metal_step.metal");
    println!("cargo:rerun-if-changed=src/modernbert_mpsgraph.m");
    println!("cargo:rerun-if-changed=src/mpsgraph_runtime.m");
    println!("cargo:rerun-if-changed=src/mpsgraph_runtime.h");
    println!("cargo:rerun-if-changed=src/cuda_minilm.cu");
    println!("cargo:rerun-if-changed=src/cuda_minilm.h");
    println!("cargo:rerun-if-changed=src/cuda_family_common.cuh");
    println!("cargo:rerun-if-changed=src/cuda_modernbert.cu");
    println!("cargo:rerun-if-changed=src/cuda_qwen3.cu");
    println!("cargo:rerun-if-changed=src/cuda_qwen3_decode.cu");
    println!("cargo:rerun-if-changed=src/cuda_lfm2.cu");
    println!("cargo:rerun-if-changed=src/cuda_ops.cu");
    println!("cargo:rerun-if-changed=src/cpu_hand_kernel.c");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_arch == "x86_64" && target_env != "msvc" {
        cc::Build::new()
            .file("src/cpu_hand_kernel.c")
            .flag_if_supported("-O3")
            .compile("synapse_unified_rt_cpu_hand");
    }
    if target_os == "macos" {
        cc::Build::new()
            .file("src/mpsgraph_runtime.m")
            .file("src/metal_mpsgraph.m")
            .file("src/qwen3_mpsgraph.m")
            .file("src/qwen3_decode_mpsgraph.m")
            .file("src/qwen3_decode_metal_step.m")
            .file("src/modernbert_mpsgraph.m")
            .flag("-fobjc-exceptions")
            .compile("synapse_unified_rt_mpsgraph");

        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");

        let out_dir = std::path::PathBuf::from(
            std::env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"),
        );
        let air_path = out_dir.join("qwen3_decode_metal_step.air");
        let metallib_path = out_dir.join("qwen3_decode_metal_step.metallib");
        let metal_available = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "--find", "metal"])
            .output()
            .is_ok_and(|output| output.status.success());
        if metal_available {
            let metal_status = std::process::Command::new("xcrun")
                .args([
                    "-sdk",
                    "macosx",
                    "metal",
                    "-std=macos-metal2.3",
                    "-c",
                    "src/qwen3_decode_metal_step.metal",
                    "-o",
                ])
                .arg(&air_path)
                .status()
                .expect("run xcrun metal for the Metal step kernels");
            assert!(
                metal_status.success(),
                "xcrun metal failed for Metal step kernels"
            );
            let metallib_status = std::process::Command::new("xcrun")
                .args(["-sdk", "macosx", "metallib"])
                .arg(&air_path)
                .arg("-o")
                .arg(&metallib_path)
                .status()
                .expect("run xcrun metallib for the Metal step kernels");
            assert!(
                metallib_status.success(),
                "xcrun metallib failed for Metal step kernels"
            );
            // Cargo places the executable three parents above OUT_DIR: target/profile.
            // Keeping a copy beside it lets the runtime load the library without
            // compiling Metal source or relying on a relocatable build-script path.
            if let Some(profile_dir) = out_dir.ancestors().nth(3) {
                let executable_library = profile_dir.join("qwen3_decode_metal_step.metallib");
                std::fs::copy(&metallib_path, executable_library)
                    .expect("copy Metal step metallib beside the executable");
            }
        } else {
            println!(
                "cargo:warning=Metal developer tools unavailable; Metal step metallib will be built by a macOS toolchain"
            );
        }
        println!(
            "cargo:rustc-env=SYNAPSE_UNIFIED_RT_METAL_STEP_LIB={}",
            metallib_path.display()
        );
    }

    if target_os == "linux" && std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let cuda_root = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("shared")
            .file("src/cuda_minilm.cu")
            .file("src/cuda_modernbert.cu")
            .file("src/cuda_qwen3.cu")
            .file("src/cuda_qwen3_decode.cu")
            .file("src/cuda_lfm2.cu")
            .file("src/cuda_ops.cu")
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
