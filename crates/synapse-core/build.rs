use serde_json::Value;

pub fn collect_cuda_floor(manifest: &Value) -> Result<[u32; 3], String> {
    let profiles = manifest["profiles"]
        .as_object()
        .ok_or("manifest has no profiles object")?;
    let mut common: Option<(&str, [u32; 3])> = None;
    for (name, profile) in profiles {
        if profile["lane"] != "owned-cuda" {
            continue;
        }
        let integer = |key: &str| -> Result<u32, String> {
            profile[key]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("CUDA profile `{name}` lacks a valid integer `{key}`"))
        };
        let floor = [
            integer("cuda_min_driver_api")?,
            integer("cuda_min_compute_major")?,
            integer("cuda_min_compute_minor")?,
        ];
        if let Some((first, expected)) = common {
            if floor != expected {
                return Err(format!("CUDA profile `{name}` has floor {floor:?}, disagreeing with `{first}` floor {expected:?}"));
            }
        } else {
            common = Some((name, floor));
        }
    }
    common
        .map(|(_, floor)| floor)
        .ok_or_else(|| "manifest has no owned-cuda profiles".into())
}

fn main() {
    let path = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../../bench/parity/models.json");
    println!("cargo:rerun-if-changed={}", path.display());
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read canonical model manifest"))
            .expect("parse canonical model manifest");
    let [driver, major, minor] =
        collect_cuda_floor(&manifest).unwrap_or_else(|error| panic!("{error}"));
    for (key, value) in [
        ("SYNAPSE_CUDA_MIN_DRIVER_API", driver),
        ("SYNAPSE_CUDA_MIN_COMPUTE_MAJOR", major),
        ("SYNAPSE_CUDA_MIN_COMPUTE_MINOR", minor),
    ] {
        println!("cargo:rustc-env={key}={value}");
    }
}
