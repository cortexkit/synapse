fn main() {
    embed_manifest();
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

    // CUDACXX names the compiler outright (cross/toolchain installs); otherwise
    // resolve nvcc under the toolkit root. Windows needs the .exe suffix:
    // `bin/nvcc` does not exist there and cc::Build reports it as a missing
    // compiler rather than falling back to the host C++ compiler.
    let nvcc = std::env::var_os("CUDACXX")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let name = if target_os == "windows" {
                "nvcc.exe"
            } else {
                "nvcc"
            };
            cuda_root.join("bin").join(name)
        });

    let mut build = cc::Build::new();
    build
        .compiler(&nvcc)
        .cpp(true)
        .no_default_flags(true)
        .warnings(false)
        .extra_warnings(false)
        .include("src/port")
        .include(&include)
        // V1 distributes virtual PTX only. Do not add an sm_* SASS image here.
        .flag("-gencode=arch=compute_75,code=compute_75")
        .flag("-O3")
        .flag("-Wno-deprecated-gpu-targets")
        .file("src/port/cuda_minilm.cu")
        .file("src/port/cuda_modernbert.cu")
        .file("src/port/cuda_qwen3.cu");
    // Position-independent host code is an ELF concern. Forwarded to MSVC it is
    // an unknown-option error out of cl, which nvcc surfaces as a build failure,
    // so the flag is applied only where the host toolchain accepts it.
    if target_os != "windows" {
        build.flag("-Xcompiler=-fPIC");
    }
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
    if target_os == "windows" {
        // CUDA 13's redist archives place cublas's import library beside its
        // DLL in bin/x64 (the 12.x layout keeps it under lib/x64). Search
        // both so cublas/cublasLt resolve on either toolkit line; duplicated
        // search paths are harmless to link.exe.
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_root.join("bin/x64").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_root.join("bin").display()
        );
    }

    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudart");
}

fn embed_manifest() {
    use sha2::{Digest, Sha256};
    let path = "../../bench/parity/models.json";
    println!("cargo:rerun-if-changed={path}");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("read model manifest"))
            .expect("parse model manifest");
    fn canonical(value: &serde_json::Value, out: &mut Vec<u8>) {
        use serde_json::Value;
        match value {
            Value::Object(map) => {
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();
                out.push(b'{');
                for (i, key) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend(serde_json::to_vec(key).unwrap());
                    out.push(b':');
                    canonical(&map[key], out);
                }
                out.push(b'}');
            }
            Value::Array(items) => {
                out.push(b'[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    canonical(v, out);
                }
                out.push(b']');
            }
            scalar => out.extend(serde_json::to_vec(scalar).unwrap()),
        }
    }
    let mut bytes = Vec::new();
    canonical(&manifest, &mut bytes);
    println!(
        "cargo:rustc-env=SYNAPSE_CUDA_MANIFEST_DIGEST={:x}",
        Sha256::digest(&bytes)
    );
    std::fs::write(
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("models.json"),
        bytes,
    )
    .expect("write canonical manifest");
}
