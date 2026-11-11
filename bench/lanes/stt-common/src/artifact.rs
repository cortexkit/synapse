use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize)]
pub struct Artifact {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

pub fn artifacts(path: &Path) -> Result<Vec<Artifact>> {
    let mut result = Vec::new();
    collect(path, &mut result)?;
    result.sort_by(|a, b| a.path.cmp(&b.path));
    ensure!(!result.is_empty(), "empty model directory");
    Ok(result)
}

fn collect(path: &Path, result: &mut Vec<Artifact>) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    // Hugging Face snapshots use file symlinks into their content-addressed
    // cache. Follow files but not directory links, which could introduce cycles.
    if path.is_file() {
        let mut file = File::open(path)?;
        let mut digest = Sha256::new();
        let mut count = 0;
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let length = file.read(&mut buffer)?;
            if length == 0 {
                break;
            }
            digest.update(&buffer[..length]);
            count += length as u64;
        }
        result.push(Artifact {
            path: path.to_path_buf(),
            bytes: count,
            sha256: format!("{:x}", digest.finalize()),
        });
    } else {
        ensure!(
            metadata.is_dir() && !metadata.is_symlink(),
            "not a file or directory: {}",
            path.display()
        );
        for entry in fs::read_dir(path)? {
            collect(&entry?.path(), result)?;
        }
    }
    Ok(())
}
