//! Canonical JSON bytes and SHA-256 helpers.
//!
//! Canonical form: object keys sorted by their UTF-8 bytes at every depth, no
//! insignificant whitespace, strings escaped as `serde_json` escapes them, and
//! numbers written as `serde_json` writes the parsed value (so `1e-05` and
//! `1e-5` are the same number and the same bytes). Every manifest digest is the
//! SHA-256 of these bytes, so reformatting `models.json` never changes one.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Canonical bytes of an arbitrary JSON value.
pub fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_canonical(value, &mut out);
    out
}

/// Canonical bytes of any serializable value, through its JSON form.
pub fn canonical_bytes_of<T: Serialize>(value: &T) -> Vec<u8> {
    let value = serde_json::to_value(value).expect("manifest values always serialize");
    canonical_bytes(&value)
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Object(map) => {
            // Sorting here, rather than trusting the map type, keeps the form
            // independent of whether serde_json's `preserve_order` feature is
            // switched on somewhere in the dependency graph.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push(b'{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                out.extend(serde_json::to_vec(key).expect("string keys serialize"));
                out.push(b':');
                write_canonical(&map[*key], out);
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_canonical(item, out);
            }
            out.push(b']');
        }
        scalar => out.extend(serde_json::to_vec(scalar).expect("scalars serialize")),
    }
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Lowercase hex SHA-256 of a file's bytes, streamed so multi-gigabyte
/// checkpoints are not held in memory.
pub fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// True when `value` is 64 lowercase hex characters.
pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn key_order_and_whitespace_do_not_change_canonical_bytes() {
        let a: Value =
            serde_json::from_str(r#"{ "b": [1, {"y": 2, "x": 1}], "a": 1e-05 }"#).unwrap();
        let b = json!({"a": 1e-5, "b": [1, {"x": 1, "y": 2}]});
        assert_eq!(canonical_bytes(&a), canonical_bytes(&b));
        assert_eq!(
            String::from_utf8(canonical_bytes(&b)).unwrap(),
            r#"{"a":0.00001,"b":[1,{"x":1,"y":2}]}"#
        );
    }

    #[test]
    fn array_order_is_significant() {
        assert_ne!(
            canonical_bytes(&json!([1, 2])),
            canonical_bytes(&json!([2, 1]))
        );
    }
}
