use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};

const SHADERS: [(&str, &str); 2] = [("plain", "vulkan1.2"), ("cooperative", "vulkan1.3")];

fn verify_spirv(bytes: &[u8]) {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V word alignment");
    let words: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(words.first(), Some(&0x07230203), "SPIR-V magic");
    let mut offset = 5;
    while offset < words.len() {
        let instruction = words[offset];
        let length = (instruction >> 16) as usize;
        let opcode = instruction & 0xffff;
        assert!(
            length > 0 && offset + length <= words.len(),
            "SPIR-V instruction bounds"
        );
        assert!(
            !matches!(opcode, 2 | 5 | 6 | 7 | 8 | 317),
            "SPIR-V contains source/debug information"
        );
        assert!(
            opcode != 3 || length <= 3,
            "SPIR-V contains an embedded source path"
        );
        offset += length;
    }
}

fn main() {
    let manifest_path = PathBuf::from("../../bench/parity/models.json");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let raw = fs::read(&manifest_path).expect("read model manifest");
    let value: synapse_parity::manifest::Manifest =
        synapse_parity::manifest::Manifest::from_slice(&raw).expect("parse model manifest");
    for (slug, model) in &value.models {
        synapse_parity::arch::check_schema(slug, &model.architecture)
            .expect("embedded architecture must be complete and match its class");
    }
    let canonical = synapse_parity::canonical::canonical_bytes_of(&value);
    fs::write(out.join("manifest.json"), &canonical).expect("embed manifest");
    println!(
        "cargo:rustc-env=VULKAN_MANIFEST_DIGEST={:x}",
        Sha256::digest(&canonical)
    );
    println!("cargo:rerun-if-changed=shaders/common.glsl");
    let enabled = env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let sdk = enabled.then(|| {
        let sdk = PathBuf::from(
            env::var_os("VULKAN_SDK").expect("vulkan requires Vulkan SDK 1.4.357.0 via VULKAN_SDK"),
        );
        assert!(
            sdk.components().any(|c| c.as_os_str() == "1.4.357.0"),
            "VULKAN_SDK must select SDK 1.4.357.0"
        );
        sdk
    });
    let mut hash = Sha256::new();
    let mut declarations = String::from("pub static SHADERS: &[(&str, &[u8])] = &[\n");
    for (name, target) in SHADERS {
        let source = format!("shaders/{name}.comp");
        println!("cargo:rerun-if-changed={source}");
        let binary = out.join(format!("{name}.spv"));
        if let Some(sdk) = &sdk {
            let compiler = sdk.join(if cfg!(windows) {
                "Bin/glslc.exe"
            } else {
                "bin/glslc"
            });
            let version = Command::new(&compiler)
                .arg("--version")
                .output()
                .expect("read pinned glslc version");
            let version_text = String::from_utf8_lossy(&version.stdout);
            assert!(
                version.status.success()
                    && version_text.contains("shaderc v2026.3")
                    && version_text.contains("glslang 11.1.0-1493-g168d452a")
                    && version_text.contains("spirv-tools v2026.3"),
                "glslc does not match SDK 1.4.357.0: {version_text}"
            );
            let output = Command::new(compiler)
                .current_dir("shaders")
                .args([
                    "-O",
                    "-fshader-stage=compute",
                    &format!("--target-env={target}"),
                ])
                .arg(format!("{name}.comp"))
                .arg("-o")
                .arg(&binary)
                .output()
                .expect("run pinned glslc");
            assert!(
                output.status.success(),
                "glslc: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let bytes = fs::read(&binary).expect("read compiled shader");
            verify_spirv(&bytes);
            hash.update(&bytes);
            declarations.push_str(&format!("({name:?}, include_bytes!({:?})),\n", binary));
        }
    }
    declarations.push_str("];\n");
    fs::write(out.join("shaders.rs"), declarations).expect("embed SPIR-V set");
    println!(
        "cargo:rustc-env=VULKAN_KERNEL_REVISION={:x}",
        hash.finalize()
    );
}
