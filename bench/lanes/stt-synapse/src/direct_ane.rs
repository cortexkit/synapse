use super::*;
use anyhow::Context;

struct DirectAne {
    model: backend::Model,
    os: String,
    _lock: std::fs::File,
}

pub fn load(path: &Path) -> Result<Box<dyn Engine>> {
    ensure!(
        std::env::var_os("TMPDIR").is_none(),
        "unset TMPDIR before direct ANE execution"
    );
    ensure!(
        std::env::var_os("SYNAPSE_ANE_PROFILE_DIR").is_none(),
        "disable ANE profiling for benchmark execution"
    );
    let home = std::env::var_os("HOME").context("missing HOME")?;
    let lock_path =
        std::path::PathBuf::from(home).join("Library/Caches/ck-synapse/ane-direct.lock");
    std::fs::create_dir_all(lock_path.parent().unwrap())?;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)?;
    lock.try_lock()
        .context("ane_lane_busy: another ANE client holds the machine-wide lock")?;
    let profile = backend::Profile::select(ANE_PROFILE, "transcribe")?;
    let digest = profile.numeric["converted_package_digest"]
        .as_str()
        .context("missing ANE package pin")?
        .to_owned();
    let model = backend::Model::load(profile, &package(path), &digest)?;
    let os = std::process::Command::new("sw_vers")
        .arg("-buildVersion")
        .output()?;
    ensure!(os.status.success(), "cannot identify macOS build");
    Ok(Box::new(DirectAne {
        model,
        os: String::from_utf8(os.stdout)?.trim().into(),
        _lock: lock,
    }))
}

impl Engine for DirectAne {
    fn tokens(&mut self, features: Features) -> Result<Vec<u32>> {
        let shape = backend::rung(features.valid_encoder_frames)?;
        self.model.admit(shape, &self.os)?;
        let output = self.model.transcribe_progress(
            &features.values,
            features.mel_frames,
            features.valid_mel_frames,
            false,
            &mut || Ok(()),
        )?;
        ensure!(
            output.encoder.is_empty() && output.decoded.trace.is_empty(),
            "benchmark engine returned observations"
        );
        Ok(output.decoded.triples.iter().map(|t| t[0]).collect())
    }
}
