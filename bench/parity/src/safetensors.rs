//! Minimal safetensors reader and the canonical package writer.
//!
//! The writer is deliberately not a general serializer: it produces exactly
//! one byte layout (see `write_package`) so that a converted package's SHA-256
//! is reproducible from the pinned checkpoint alone.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::{perr, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDType {
    F16,
    Bf16,
    F32,
}

impl StDType {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            other => Err(perr!("unsupported safetensors dtype `{other}`")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: StDType,
    pub shape: Vec<u64>,
    pub data_offsets: (usize, usize),
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
}

/// A parsed safetensors header: tensors by name, the metadata map, and the
/// order tensor entries appear in the header JSON.
#[derive(Clone, Debug)]
pub struct Header {
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: Option<Map<String, Value>>,
    pub key_order: Vec<String>,
}

/// Parse a raw header JSON (the bytes between the length prefix and the data).
pub fn parse_header_json(bytes: &[u8]) -> Result<Header> {
    // Parse with key order preserved by walking the raw object ourselves, so
    // the writer's lexicographic-order rule can be checked on real files.
    let text = std::str::from_utf8(bytes).map_err(|_| perr!("safetensors header is not UTF-8"))?;
    let value: Value = serde_json::from_str(text.trim_end())
        .map_err(|error| perr!("safetensors header: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| perr!("safetensors header is not an object"))?;
    let mut tensors = BTreeMap::new();
    let mut metadata = None;
    for (name, entry) in object {
        if name == "__metadata__" {
            metadata = Some(
                entry
                    .as_object()
                    .cloned()
                    .ok_or_else(|| perr!("__metadata__ is not an object"))?,
            );
            continue;
        }
        let dtype = StDType::parse(
            entry
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| perr!("tensor `{name}` has no dtype"))?,
        )?;
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| perr!("tensor `{name}` has no shape"))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .ok_or_else(|| perr!("tensor `{name}` shape is not integral"))
            })
            .collect::<Result<Vec<u64>>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| perr!("tensor `{name}` has no data_offsets"))?;
        let [start, end] = offsets.as_slice() else {
            return Err(perr!("tensor `{name}` data_offsets is not a pair"));
        };
        let start = start.as_u64().ok_or_else(|| perr!("bad offset"))? as usize;
        let end = end.as_u64().ok_or_else(|| perr!("bad offset"))? as usize;
        let info = TensorInfo {
            dtype,
            shape,
            data_offsets: (start, end),
        };
        if end - start != info.elements() * dtype.size() {
            return Err(perr!(
                "tensor `{name}` byte length does not match its shape"
            ));
        }
        tensors.insert(name.clone(), info);
    }
    let key_order = header_key_order(text)?;
    Ok(Header {
        tensors,
        metadata,
        key_order,
    })
}

/// Top-level keys of a JSON object in the order they appear in the text.
fn header_key_order(text: &str) -> Result<Vec<String>> {
    let mut stream = serde_json::Deserializer::from_str(text.trim_end()).into_iter::<OrderedKeys>();
    match stream.next() {
        Some(Ok(keys)) => Ok(keys.0),
        _ => Err(perr!("safetensors header key order could not be read")),
    }
}

struct OrderedKeys(Vec<String>);

impl<'de> serde::Deserialize<'de> for OrderedKeys {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = OrderedKeys;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<OrderedKeys, A::Error> {
                let mut keys = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    map.next_value::<serde::de::IgnoredAny>()?;
                    keys.push(key);
                }
                Ok(OrderedKeys(keys))
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

/// Split a whole safetensors file into its raw header bytes and data section.
pub fn split_file(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    if bytes.len() < 8 {
        return Err(perr!("safetensors file is shorter than its length prefix"));
    }
    let length = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) as usize;
    let header_end = 8usize
        .checked_add(length)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| perr!("safetensors header length runs past the file"))?;
    Ok((&bytes[8..header_end], &bytes[header_end..]))
}

/// Read only the raw header bytes of a safetensors file on disk.
pub fn read_header_bytes(path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|error| perr!("open {}: {error}", path.display()))?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| perr!("read {}: {error}", path.display()))?;
    let mut header = vec![0u8; u64::from_le_bytes(prefix) as usize];
    file.read_exact(&mut header)
        .map_err(|error| perr!("read {}: {error}", path.display()))?;
    Ok(header)
}

/// A loaded checkpoint: header plus data, with element access widened to f32
/// (exact for f16, bf16 and f32 sources).
pub struct Checkpoint {
    pub header: Header,
    pub data: Vec<u8>,
}

impl Checkpoint {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let (header_bytes, _) = split_file(&bytes)?;
        let header = parse_header_json(header_bytes)?;
        let data_start = 8 + header_bytes.len();
        let mut data = bytes;
        data.drain(..data_start);
        for (name, info) in &header.tensors {
            if info.data_offsets.1 > data.len() {
                return Err(perr!("tensor `{name}` runs past the end of the file"));
            }
        }
        Ok(Self { header, data })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.header
            .tensors
            .get(name)
            .ok_or_else(|| perr!("checkpoint has no tensor `{name}`"))
    }

    pub fn raw(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self.info(name)?;
        Ok((info, &self.data[info.data_offsets.0..info.data_offsets.1]))
    }

    /// Values widened to f32. Every source dtype here widens exactly.
    pub fn values_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (info, raw) = self.raw(name)?;
        Ok(match info.dtype {
            StDType::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            StDType::F16 => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            StDType::Bf16 => raw
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        })
    }
}

/// One tensor of a package, already encoded.
#[derive(Clone, Debug, PartialEq)]
pub struct PackageTensor {
    pub dtype: StDType,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

/// Write the canonical package bytes.
///
/// Layout: the 8-byte little-endian header length; a compact JSON header whose
/// first key is `__metadata__` (exactly `conversion_rule` then `profile`) and
/// whose remaining keys are tensor names in lexicographic byte order, each
/// `{"dtype","shape","data_offsets"}`; the header padded with ASCII spaces to a
/// multiple of 8 bytes; then each tensor's bytes, contiguous, in that same
/// order.
pub fn write_package(profile: &str, tensors: &BTreeMap<String, PackageTensor>) -> Vec<u8> {
    let metadata = [("conversion_rule", "v1"), ("profile", profile)];
    write_package_with(&metadata, tensors.iter().collect())
}

/// Shared by `write_package` and the mutation tests, which pass a different
/// metadata set or tensor order to prove the verifier and digest catch them.
pub(crate) fn write_package_with(
    metadata: &[(&str, &str)],
    tensors: Vec<(&String, &PackageTensor)>,
) -> Vec<u8> {
    let mut header = String::from("{\"__metadata__\":{");
    for (index, (key, value)) in metadata.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        header.push_str(&serde_json::to_string(key).expect("string"));
        header.push(':');
        header.push_str(&serde_json::to_string(value).expect("string"));
    }
    header.push('}');
    let mut offset = 0usize;
    for (name, tensor) in &tensors {
        let end = offset + tensor.bytes.len();
        let shape: Vec<String> = tensor.shape.iter().map(u64::to_string).collect();
        header.push_str(&format!(
            ",{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
            serde_json::to_string(name).expect("string"),
            tensor.dtype.as_str(),
            shape.join(","),
        ));
        offset = end;
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend((header.len() as u64).to_le_bytes());
    out.extend(header.as_bytes());
    for (_, tensor) in tensors {
        out.extend(&tensor.bytes);
    }
    out
}

/// Structural rules every converted package must meet, independent of its
/// values: metadata is exactly `{profile, conversion_rule: "v1"}`, tensor keys
/// appear in lexicographic order, and the data is laid out contiguously in
/// that order with no gaps. Value-level faults (a wrong rounding mode) are
/// caught by comparing the reconverted digest instead.
pub fn verify_package_structure(bytes: &[u8], profile: &str) -> Result<()> {
    let (header_bytes, data) = split_file(bytes)?;
    if header_bytes.len() % 8 != 0 {
        return Err(perr!("package header is not padded to 8 bytes"));
    }
    let header = parse_header_json(header_bytes)?;
    let metadata = header
        .metadata
        .as_ref()
        .ok_or_else(|| perr!("package has no __metadata__"))?;
    let mut expected = Map::new();
    expected.insert("conversion_rule".into(), Value::from("v1"));
    expected.insert("profile".into(), Value::from(profile));
    if metadata != &expected {
        return Err(perr!(
            "package __metadata__ must be exactly {{profile, conversion_rule: v1}}, found {}",
            Value::Object(metadata.clone())
        ));
    }
    let names: Vec<&String> = header
        .key_order
        .iter()
        .filter(|k| *k != "__metadata__")
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    if names != sorted {
        return Err(perr!("package tensors are not in lexicographic order"));
    }
    let mut offset = 0usize;
    for name in names {
        let info = &header.tensors[name];
        if info.data_offsets.0 != offset {
            return Err(perr!(
                "package tensor `{name}` is not laid out in key order"
            ));
        }
        offset = info.data_offsets.1;
    }
    if offset != data.len() {
        return Err(perr!("package data section has trailing or missing bytes"));
    }
    Ok(())
}
