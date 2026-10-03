pub mod admission;
pub mod allocation;
pub mod protocol;
#[cfg(feature = "vulkan")]
pub mod runtime;

use synapse_parity::manifest::Manifest;

pub const MANIFEST_DIGEST: &str = env!("VULKAN_MANIFEST_DIGEST");
pub const KERNEL_REVISION: &str = env!("VULKAN_KERNEL_REVISION");
include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

pub fn manifest() -> Manifest {
    Manifest::from_slice(include_bytes!(concat!(env!("OUT_DIR"), "/manifest.json")))
        .expect("build-validated embedded manifest")
}

pub fn enumerate() -> Result<Vec<admission::Adapter>, String> {
    #[cfg(feature = "vulkan")]
    {
        runtime::enumerate()
    }
    #[cfg(not(feature = "vulkan"))]
    {
        Err("vulkan_no_device".into())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Padded {
    pub ids: Vec<i32>,
    pub mask: Vec<i32>,
    pub lengths: Vec<usize>,
    pub width: usize,
}

pub fn pad(sequences: &[Vec<i32>], pad_id: i32) -> Result<Padded, String> {
    let width = sequences.iter().map(Vec::len).max().unwrap_or(0);
    if width > 8192 {
        return Err("sequence_too_long".into());
    }
    if width == 0 || sequences.iter().any(Vec::is_empty) {
        return Err("invalid_tokens".into());
    }
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for sequence in sequences {
        ids.extend_from_slice(sequence);
        ids.resize(ids.len() + width - sequence.len(), pad_id);
        mask.extend(std::iter::repeat_n(1, sequence.len()));
        mask.resize(mask.len() + width - sequence.len(), 0);
    }
    Ok(Padded {
        ids,
        mask,
        lengths: sequences.iter().map(Vec::len).collect(),
        width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn embedded_binding_is_derived_from_bytes() {
        assert_eq!(
            MANIFEST_DIGEST,
            format!(
                "{:x}",
                Sha256::digest(include_bytes!(concat!(env!("OUT_DIR"), "/manifest.json")))
            )
        );
        let mut hash = Sha256::new();
        for (_, bytes) in SHADERS {
            hash.update(bytes);
        }
        assert_eq!(KERNEL_REVISION, format!("{:x}", hash.finalize()));
        let original = Sha256::digest(b"shader-set-a");
        assert_ne!(original, Sha256::digest(b"shader-set-b"));
    }
    #[test]
    fn batch_longest_padding_golden() {
        #[derive(serde::Deserialize)]
        struct Golden {
            sequences: Vec<Vec<i32>>,
            pad_id: i32,
            ids: Vec<i32>,
            mask: Vec<i32>,
            lengths: Vec<usize>,
            width: usize,
        }
        let golden: Golden =
            serde_json::from_str(include_str!("../tests/fixtures/padded-vulkan.json")).unwrap();
        let padded = pad(&golden.sequences, golden.pad_id).unwrap();
        assert_eq!(
            padded,
            Padded {
                ids: golden.ids,
                mask: golden.mask,
                lengths: golden.lengths,
                width: golden.width
            }
        );
    }
}
