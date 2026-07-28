fn main() {
    for source in [
        "src/metal_mpsgraph.m",
        "src/qwen3_mpsgraph.m",
        "src/modernbert_mpsgraph.m",
        "src/mpsgraph_runtime.m",
        "src/mpsgraph_runtime.h",
        // Owned decode engine: Metal step kernels and their Objective-C drivers.
        "owned-decode-engine/src/qwen3_decode_metal_step.m",
        "owned-decode-engine/src/lfm2_decode_metal_step.m",
        "owned-decode-engine/src/qwen3_decode_metal_step.metal",
        "owned-decode-engine/src/lfm2_decode_metal_step.metal",
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

    // Owned decode engine: compile the Metal step Objective-C drivers into a
    // separate static library so the FFI symbols are available to the
    // owned_decode_engine module. These drivers bind the Metal step kernels
    // (compiled to metallib below) and expose the C FFI that the Rust engines
    // call.
    cc::Build::new()
        .file("owned-decode-engine/src/qwen3_decode_metal_step.m")
        .file("owned-decode-engine/src/lfm2_decode_metal_step.m")
        .flag("-fobjc-exceptions")
        .compile("synapse_owned_decode_metal_step");

    for framework in [
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }

    // Compile the Metal step kernels into metallibs. The runtime loads them
    // beside the executable (relocatable) or from the OUT_DIR path (build-time).
    let out_dir =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));

    let metal_available = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "--find", "metal"])
        .output()
        .is_ok_and(|output| output.status.success());

    if metal_available {
        // Qwen3 step metallib.
        let qwen3_air_path = out_dir.join("qwen3_decode_metal_step.air");
        let qwen3_metallib_path = out_dir.join("qwen3_decode_metal_step.metallib");
        let qwen3_metal_status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=macos-metal2.3",
                "-c",
                "owned-decode-engine/src/qwen3_decode_metal_step.metal",
                "-o",
            ])
            .arg(&qwen3_air_path)
            .status()
            .expect("run xcrun metal for the Qwen3 step kernels");
        assert!(
            qwen3_metal_status.success(),
            "xcrun metal failed for the Qwen3 step kernels"
        );
        let qwen3_metallib_status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(&qwen3_air_path)
            .arg("-o")
            .arg(&qwen3_metallib_path)
            .status()
            .expect("run xcrun metallib for the Qwen3 step kernels");
        assert!(
            qwen3_metallib_status.success(),
            "xcrun metallib failed for the Qwen3 step kernels"
        );
        if let Some(profile_dir) = out_dir.ancestors().nth(3) {
            let executable_library = profile_dir.join("qwen3_decode_metal_step.metallib");
            std::fs::copy(&qwen3_metallib_path, executable_library)
                .expect("copy Qwen3 step metallib beside the executable");
        }

        // LFM2 step metallib: the conv step kernel plus a reused IEEE-strict
        // copy of the Qwen3 step kernels (RMSNorm, QKV matvec, QK-norm+RoPE,
        // GQA attention, matvec+residual, SwiGLU, LM head, argmax, embedding
        // gather) for the attention layers. The Qwen3 source is compiled a
        // second time, IEEE-strict, into the LFM2 metallib so the whole LFM2
        // forward runs under the same fast-math discipline the conv step
        // requires for bit-exactness vs the CPU reference.
        let lfm2_air_path = out_dir.join("lfm2_decode_metal_step.air");
        let lfm2_metallib_path = out_dir.join("lfm2_decode_metal_step.metallib");
        let lfm2_metal_status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=macos-metal2.3",
                // IEEE-strict math: the conv step must reproduce the CPU
                // reference bit-for-bit, so disable Metal's default fast-math
                // (reassociation) and FMA contraction.
                "-fno-fast-math",
                "-ffp-contract=off",
                "-c",
                "owned-decode-engine/src/lfm2_decode_metal_step.metal",
                "-o",
            ])
            .arg(&lfm2_air_path)
            .status()
            .expect("run xcrun metal for the LFM2 step kernels");
        assert!(
            lfm2_metal_status.success(),
            "xcrun metal failed for the LFM2 step kernels"
        );
        // Reused Qwen3 step kernels compiled IEEE-strict into the LFM2 metallib.
        let lfm2_reused_air_path = out_dir.join("lfm2_reused_qwen3_step.air");
        let lfm2_reused_metal_status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=macos-metal2.3",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-c",
                "owned-decode-engine/src/qwen3_decode_metal_step.metal",
                "-o",
            ])
            .arg(&lfm2_reused_air_path)
            .status()
            .expect("run xcrun metal for the reused Qwen3 step kernels (LFM2 lib)");
        assert!(
            lfm2_reused_metal_status.success(),
            "xcrun metal failed for the reused Qwen3 step kernels (LFM2 lib)"
        );
        let lfm2_metallib_status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(&lfm2_air_path)
            .arg(&lfm2_reused_air_path)
            .arg("-o")
            .arg(&lfm2_metallib_path)
            .status()
            .expect("run xcrun metallib for the LFM2 step kernels");
        assert!(
            lfm2_metallib_status.success(),
            "xcrun metallib failed for the LFM2 step kernels"
        );
        if let Some(profile_dir) = out_dir.ancestors().nth(3) {
            let executable_library = profile_dir.join("lfm2_decode_metal_step.metallib");
            std::fs::copy(&lfm2_metallib_path, executable_library)
                .expect("copy LFM2 step metallib beside the executable");
        }
    } else {
        println!(
            "cargo:warning=Metal developer tools unavailable; Metal step metallib will be built by a macOS toolchain"
        );
    }

    // Expose the metallib paths to the Rust engines so they can fall back to
    // the build-time path if the beside-executable copy is not found.
    println!(
        "cargo:rustc-env=SYNAPSE_OWNED_DECODE_QWEN3_STEP_LIB={}",
        out_dir.join("qwen3_decode_metal_step.metallib").display()
    );
    println!(
        "cargo:rustc-env=SYNAPSE_OWNED_DECODE_LFM2_STEP_LIB={}",
        out_dir.join("lfm2_decode_metal_step.metallib").display()
    );
}
