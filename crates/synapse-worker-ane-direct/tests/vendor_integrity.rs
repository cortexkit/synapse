use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;
fn files(path: &Path, root: &Path, out: &mut BTreeSet<String>) {
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            // A tool that treats the vendored crate as its own package (for
            // example rust-analyzer) writes build output to vendor/ane/target.
            // That is never part of the vendored source; git ignores it too.
            if path == root.join("target") {
                continue;
            }
            files(&path, root, out);
        } else {
            let name = path
                .strip_prefix(root)
                .unwrap()
                .to_str()
                .unwrap()
                .replace('\\', "/");
            if name != ".cargo-checksum.json" {
                out.insert(name);
            }
        }
    }
}
#[test]
fn vendored_binding_checksums_match() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/ane");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(".cargo-checksum.json")).unwrap()).unwrap();
    let expected = manifest["files"].as_object().unwrap();
    let mut inventory = BTreeSet::new();
    files(&root, &root, &mut inventory);
    assert_eq!(
        inventory,
        expected.keys().cloned().collect(),
        "vendored file inventory changed"
    );
    for (name, digest) in expected {
        let actual = format!(
            "{:x}",
            Sha256::digest(std::fs::read(root.join(name)).unwrap())
        );
        assert_eq!(
            actual,
            digest.as_str().unwrap(),
            "vendored file changed: {name}"
        );
    }
}
