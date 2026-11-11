use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};

// Attention has its own pipeline because its shared-memory tiles would
// otherwise be reserved by every other kernel and lower their occupancy.
const SHADERS: [(&str, &str); 5] = [
    ("plain", "vulkan1.2"),
    ("cooperative", "vulkan1.3"),
    ("attention", "vulkan1.2"),
    ("transcribe", "vulkan1.2"),
    ("transcribe_f32", "vulkan1.2"),
];

fn verify_spirv(bytes: &[u8]) {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V word alignment");
    let words: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
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
        if name == "transcribe_f32"
            && env::var_os("CARGO_FEATURE_TRANSCRIBE_F32_MEASUREMENT").is_none()
        {
            continue;
        }
        let source_name = if name == "transcribe_f32" {
            "transcribe"
        } else {
            name
        };
        let source = format!("shaders/{source_name}.comp");
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
            // Generate only the measurement shader. The original fp16 source
            // and compiler arguments remain unchanged, preserving its SPIR-V.
            let input = if name == "transcribe_f32" {
                let mut text = fs::read_to_string(&source).expect("read speech shader");
                for (from, to) in [
                    ("#extension GL_EXT_shader_16bit_storage : require\n", ""),
                    (
                        "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n",
                        "",
                    ),
                    ("float16_t w[]", "float w[]"),
                    ("unpackHalf2x16(b[i/2])[i%2]", "uintBitsToFloat(b[i])"),
                    ("unpackHalf2x16(c[i/2])[i%2]", "uintBitsToFloat(c[i])"),
                    ("unpackHalf2x16(z[i/2])[i%2]", "uintBitsToFloat(z[i])"),
                    ("p.op", "(p.op-100u)"),
                ] {
                    assert!(
                        text.contains(from),
                        "speech shader transformation anchor disappeared: {from}"
                    );
                    text = text.replace(from, to);
                }
                let path = out.join("transcribe_f32.comp");
                fs::write(&path, text).expect("write measurement shader");
                path
            } else {
                PathBuf::from(format!("{name}.comp"))
            };
            let output = Command::new(compiler)
                .current_dir("shaders")
                .args([
                    "-O",
                    "-fshader-stage=compute",
                    &format!("--target-env={target}"),
                ])
                .arg(input)
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
            // Speech kernels have their own load-time revision. Excluding them
            // preserves the embedding/reranking shader-set identity.
            if !name.starts_with("transcribe") {
                hash.update(&bytes);
            }
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
