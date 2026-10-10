//! Live certification CLI; candidate and producer provenance are checked separately.
use crate::{live, refuse};
use std::{ffi::OsString, path::Path, process::Output};
use synapse_certify::Result;

pub const LIVE_OPTIONS_MISSING: &str =
    "certify run requires --assets <extracted-dir> --checkout <root> --weights <model-dir>";
pub const USAGE: &str = "usage: ckdev-synapse-certify run --row <row-id> --model <slug> --assets <extracted-dir> --checkout <root> --weights <model-dir>";

#[derive(Debug, PartialEq, Eq)]
pub struct Command {
    row: String,
    model: String,
    options: Option<live::Options>,
}

pub fn parse(arguments: &[OsString]) -> Result<Command> {
    let bad = || refuse(USAGE);
    if arguments.first().is_none_or(|arg| arg != "run")
        || arguments.len() < 5
        || arguments.len().is_multiple_of(2)
    {
        return Err(bad());
    }
    let mut row = None;
    let mut model = None;
    let mut assets = None;
    let mut checkout = None;
    let mut weights = None;
    for pair in arguments[1..].as_chunks::<2>().0 {
        let value = pair[1]
            .to_str()
            .filter(|s| !s.is_empty() && !s.starts_with('-'))
            .ok_or_else(bad)?;
        match pair[0].to_str() {
            Some("--row") if row.is_none() => row = Some(value.to_string()),
            Some("--model") if model.is_none() => model = Some(value.to_string()),
            Some("--assets") if assets.is_none() => assets = Some(value.into()),
            Some("--checkout") if checkout.is_none() => checkout = Some(value.into()),
            Some("--weights") if weights.is_none() => weights = Some(value.into()),
            _ => return Err(bad()),
        }
    }
    let row = row.ok_or_else(bad)?;
    let model = model.ok_or_else(bad)?;
    if !synapse_certify::ROWS.contains(&row.as_str())
        || !synapse_certify::MODELS.contains(&model.as_str())
    {
        return Err(refuse("unknown certification row or model"));
    }
    let options = match (assets, checkout, weights) {
        (Some(assets), Some(checkout), Some(weights)) => Some(live::Options {
            assets,
            checkout,
            weights,
        }),
        (None, None, None) => None,
        _ => return Err(refuse(LIVE_OPTIONS_MISSING)),
    };
    Ok(Command {
        row,
        model,
        options,
    })
}

pub fn dispatch(command: Command, source: Option<&str>) -> Result<serde_json::Value> {
    let executable = std::env::current_exe().map_err(|error| refuse(error.to_string()))?;
    let digest = synapse_parity::canonical::sha256_file(&executable)
        .map_err(|error| refuse(error.to_string()))?;
    // Log the separate runner's path, commit and digest; records still describe only candidate assets.
    eprintln!(
        "{}",
        serde_json::json!({"runner_path": executable, "runner_commit": source, "runner_sha256": digest})
    );
    let options = command
        .options
        .ok_or_else(|| refuse(LIVE_OPTIONS_MISSING))?;
    let source = source.ok_or_else(|| refuse("runner was built from a dirty tree or without git; evidence cannot be bound to a commit"))?;
    let candidate = options
        .assets
        .join(format!("ck-synapse{}", std::env::consts::EXE_SUFFIX));
    verify_candidate_source(
        &candidate,
        &options
            .checkout
            .join("crates/synapse-certify-runner/.live/source"),
        source,
    )?;
    let mut runner = live::LiveRunner::new(options.clone(), source)?;
    let record =
        synapse_certify::produce(&mut runner, &options.assets, &command.row, &command.model)?;
    synapse_certify::write_record(&record, &options.checkout)?;
    serde_json::to_value(record).map_err(|error| refuse(error.to_string()))
}

pub fn verify_candidate_source(candidate: &Path, scratch: &Path, source: &str) -> Result<()> {
    // Probe the candidate's exact bytes through a ckdev-* hard link, never an executable copy.
    let alias = synapse_core::dev_binary::ckdev_binary_hard_link(candidate, scratch)
        .map_err(|error| refuse(error.to_string()))?;
    let output = synapse_core::without_launch_nonce(std::process::Command::new(alias))
        .args(["certify", "source"])
        .output()
        .map_err(|error| refuse(error.to_string()))?;
    verify_source_output(&output, source)
}

fn verify_source_output(output: &Output, source: &str) -> Result<()> {
    if !output.status.success() {
        return Err(refuse(format!(
            "candidate source probe refused: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let candidate = std::str::from_utf8(&output.stdout)
        .map_err(|error| refuse(error.to_string()))?
        .trim();
    if candidate != source {
        return Err(refuse(
            "candidate source commit does not match the clean runner build",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn probe(revision: Option<&str>, tree: Option<&str>) -> Output {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt;
        use synapse_certify::command::{dispatch, source_stamp, Command};
        match dispatch(Command::Source, source_stamp(revision, tree)) {
            Ok(source) => Output {
                status: std::process::ExitStatus::from_raw(0),
                stdout: format!("{}\n", source.as_str().unwrap()).into_bytes(),
                stderr: Vec::new(),
            },
            Err(error) => Output {
                status: std::process::ExitStatus::from_raw(256),
                stdout: Vec::new(),
                stderr: error.to_string().into_bytes(),
            },
        }
    }

    #[test]
    fn clean_matching_candidate_source_is_accepted() {
        verify_source_output(&probe(Some(COMMIT), Some("clean")), COMMIT).unwrap();
    }

    #[test]
    fn dirty_candidate_source_is_refused() {
        let error = verify_source_output(&probe(Some(COMMIT), Some("dirty")), COMMIT).unwrap_err();
        assert!(error.to_string().contains("candidate source probe refused"));
    }

    #[test]
    fn missing_candidate_stamp_is_refused() {
        let error = verify_source_output(&probe(None, None), COMMIT).unwrap_err();
        assert!(error.to_string().contains("candidate source probe refused"));
    }

    #[test]
    fn candidate_source_mismatch_is_refused() {
        let error =
            verify_source_output(&probe(Some(&"b".repeat(40)), Some("clean")), COMMIT).unwrap_err();
        assert!(error.to_string().contains("does not match"));
    }

    #[cfg(unix)]
    #[test]
    fn source_probe_executes_a_hard_link_under_a_development_name() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let root =
            std::env::temp_dir().join(format!("certify-source-probe-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let candidate = root.join("ck-synapse");
        std::fs::write(&candidate, format!("#!/bin/sh\n[ \"$1 $2\" = 'certify source' ] || exit 2\nprintf '%s\\n' '{COMMIT}'\n")).unwrap();
        std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o755)).unwrap();
        verify_candidate_source(&candidate, &root.join("probe"), COMMIT).unwrap();
        let alias_dir = std::fs::read_dir(root.join("probe"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let alias = alias_dir.join("ckdev-synapse");
        assert_eq!(
            std::fs::metadata(&candidate).unwrap().ino(),
            std::fs::metadata(&alias).unwrap().ino()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_requires_candidate_paths() {
        let command = parse(&args(&[
            "run",
            "--row",
            "metal-m5",
            "--model",
            "gte-modernbert-base",
        ]))
        .unwrap();
        assert_eq!(
            dispatch(command, None).unwrap_err().to_string(),
            format!("certification_refused: {LIVE_OPTIONS_MISSING}")
        );
    }

    #[test]
    fn parses_live_candidate_paths() {
        let command = parse(&args(&[
            "run",
            "--row",
            "metal-m5",
            "--model",
            "gte-modernbert-base",
            "--assets",
            "assets",
            "--checkout",
            "root",
            "--weights",
            "weights",
        ]))
        .unwrap();
        assert_eq!(
            command.options.unwrap(),
            live::Options {
                assets: "assets".into(),
                checkout: "root".into(),
                weights: "weights".into()
            }
        );
    }

    #[test]
    fn bad_arguments_are_typed_refusals() {
        for values in [
            vec!["run"],
            vec!["validate"],
            vec!["run", "--row", "ane-m5", "--row", "ane-m5"],
            vec![
                "run",
                "--row",
                "metal-m5",
                "--model",
                "gte-modernbert-base",
                "--assets",
                "assets",
            ],
            vec![
                "run",
                "--row",
                "metal-m5",
                "--model",
                "gte-modernbert-base",
                "--assets",
            ],
        ] {
            assert!(parse(&args(&values)).is_err());
        }
    }
}
