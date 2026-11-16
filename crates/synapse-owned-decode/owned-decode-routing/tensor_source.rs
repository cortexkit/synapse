//! Header-only inventory and bounded, one-tensor reads of HF checkpoints.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

use anyhow::{ensure, Context, Result};
use safetensors::tensor::Dtype;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
pub(super) struct TensorRecord {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    data_offsets: [u64; 2],
    #[serde(skip)]
    path: PathBuf,
    #[serde(skip)]
    data_start: u64,
}

pub(super) struct TensorSource {
    pub root: PathBuf,
    pub tensors: BTreeMap<String, TensorRecord>,
}

impl TensorSource {
    pub fn open(source: &Path) -> Result<Self> {
        let path = if source.is_dir() {
            let index = source.join("model.safetensors.index.json");
            if index.is_file() {
                index
            } else {
                source.join("model.safetensors")
            }
        } else {
            source.to_path_buf()
        };
        let root = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let mut tensors = BTreeMap::new();
        if path.extension().is_some_and(|ext| ext == "json") {
            #[derive(Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&fs::read(&path)?)?;
            ensure!(
                !index.weight_map.is_empty(),
                "empty safetensors shard index"
            );
            for shard in index.weight_map.values().collect::<BTreeSet<_>>() {
                let mut components = Path::new(shard).components();
                ensure!(
                    matches!(components.next(), Some(Component::Normal(_)))
                        && components.next().is_none(),
                    "invalid shard filename {shard}"
                );
                for (name, tensor) in read_header(&root.join(shard))? {
                    ensure!(
                        index.weight_map.get(&name) == Some(shard),
                        "shard index disagrees for {name}"
                    );
                    ensure!(
                        tensors.insert(name, tensor).is_none(),
                        "duplicate shard tensor"
                    );
                }
            }
            ensure!(
                tensors.len() == index.weight_map.len(),
                "shard index names missing tensors"
            );
        } else {
            tensors = read_header(&path)?;
        }
        Ok(Self { root, tensors })
    }

    pub fn resolve(&self, base: &str) -> Result<&str> {
        for name in [
            base.to_string(),
            format!("bert.{base}"),
            format!("model.{base}"),
            format!("model.bert.{base}"),
            format!("lfm.{base}"),
            format!("model.lfm.{base}"),
        ] {
            if let Some((key, _)) = self.tensors.get_key_value(&name) {
                return Ok(key);
            }
        }
        anyhow::bail!("missing Q8 tensor {base}")
    }

    pub fn read(&self, name: &str) -> Result<(TensorRecord, Vec<u8>)> {
        let record = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?
            .clone();
        let mut file = File::open(&record.path)?;
        file.seek(SeekFrom::Start(record.data_start + record.data_offsets[0]))?;
        let length = usize::try_from(record.data_offsets[1] - record.data_offsets[0])?;
        let mut bytes = vec![0; length];
        file.read_exact(&mut bytes)
            .with_context(|| format!("read tensor {name}"))?;
        Ok((record, bytes))
    }
}

fn read_header(path: &Path) -> Result<BTreeMap<String, TensorRecord>> {
    let mut file = File::open(path).with_context(|| format!("open shard {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut size = [0; 8];
    file.read_exact(&mut size)?;
    let length = u64::from_le_bytes(size);
    ensure!(
        length <= 16 * 1024 * 1024 && length + 8 <= file_len,
        "invalid safetensors header size"
    );
    let mut header = vec![0; length as usize];
    file.read_exact(&mut header)?;
    let mut values: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
    values.remove("__metadata__");
    let mut records = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, value) in values {
        let mut record: TensorRecord = serde_json::from_value(value)?;
        let [start, end] = record.data_offsets;
        let elements = record
            .shape
            .iter()
            .try_fold(1_usize, |n, dim| n.checked_mul(*dim))
            .context("tensor shape overflow")?;
        let bits = elements
            .checked_mul(record.dtype.bitsize())
            .context("tensor size overflow")?;
        ensure!(bits % 8 == 0, "sub-byte tensor size is not byte aligned");
        let bytes = bits / 8;
        ensure!(
            start <= end && end <= file_len - 8 - length && end - start == bytes as u64,
            "invalid tensor bounds for {name}"
        );
        ranges.push((start, end));
        record.path = path.to_path_buf();
        record.data_start = 8 + length;
        records.insert(name, record);
    }
    ranges.sort_unstable();
    let mut end = 0;
    for (start, next) in ranges {
        ensure!(start == end, "non-contiguous safetensors data");
        end = next;
    }
    ensure!(end == file_len - 8 - length, "trailing safetensors data");
    Ok(records)
}
