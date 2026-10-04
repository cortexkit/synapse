use sha2::{Digest, Sha256};
fn main() {
    let path = "../../bench/parity/models.json";
    println!("cargo:rerun-if-changed={path}");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    fn canonical(value: &serde_json::Value, out: &mut Vec<u8>) {
        match value {
            serde_json::Value::Object(map) => {
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
            serde_json::Value::Array(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    canonical(item, out);
                }
                out.push(b']');
            }
            scalar => out.extend(serde_json::to_vec(scalar).unwrap()),
        }
    }
    let mut bytes = Vec::new();
    canonical(&manifest, &mut bytes);
    println!(
        "cargo:rustc-env=ANE_MANIFEST_DIGEST={:x}",
        Sha256::digest(&bytes)
    );
    std::fs::write(
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("models.json"),
        bytes,
    )
    .unwrap();
}
