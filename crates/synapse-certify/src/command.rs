//! Candidate certification command parsing and validator dispatch.
use crate::{refuse, validate_checkout, Result};
use std::ffi::OsString;
use std::path::PathBuf;

pub const LIVE_OPTIONS_MISSING: &str =
    "certify run requires --assets <extracted-dir> --checkout <root> --weights <model-dir>";

pub const USAGE: &str = "usage: ck-synapse certify run --row <row-id> --model <slug> --assets <extracted-dir> --checkout <root> --weights <model-dir> | ck-synapse certify validate --assets <dir> <checkout-root>";

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Run {
        row: String,
        model: String,
        options: Option<crate::live::Options>,
    },
    Validate {
        assets: PathBuf,
        checkout: PathBuf,
    },
}

pub fn parse(arguments: &[OsString]) -> Result<Option<Command>> {
    if arguments.first().is_none_or(|arg| arg != "certify") {
        return Ok(None);
    }
    let bad = || refuse(USAGE);
    match arguments.get(1).and_then(|s| s.to_str()) {
        Some("run") if arguments.len() >= 6 && arguments.len().is_multiple_of(2) => {
            let mut row = None;
            let mut model = None;
            let mut assets = None;
            let mut checkout = None;
            let mut weights = None;
            for pair in arguments[2..].as_chunks::<2>().0 {
                let value = pair[1]
                    .to_str()
                    .filter(|s| !s.is_empty() && !s.starts_with('-'))
                    .ok_or_else(bad)?;
                match pair[0].to_str() {
                    Some("--row") if row.is_none() => row = Some(value.to_string()),
                    Some("--model") if model.is_none() => model = Some(value.to_string()),
                    Some("--assets") if assets.is_none() => assets = Some(PathBuf::from(value)),
                    Some("--checkout") if checkout.is_none() => {
                        checkout = Some(PathBuf::from(value))
                    }
                    Some("--weights") if weights.is_none() => weights = Some(PathBuf::from(value)),
                    _ => return Err(bad()),
                }
            }
            let row = row.ok_or_else(bad)?;
            let model = model.ok_or_else(bad)?;
            crate::combination(&row, &model)?;
            let options = match (assets, checkout, weights) {
                (Some(assets), Some(checkout), Some(weights)) => Some(crate::live::Options {
                    assets,
                    checkout,
                    weights,
                }),
                (None, None, None) => None,
                _ => return Err(refuse(LIVE_OPTIONS_MISSING)),
            };
            Ok(Some(Command::Run {
                row,
                model,
                options,
            }))
        }
        Some("validate") if arguments.len() == 5 && arguments[2] == "--assets" => {
            if arguments[3..]
                .iter()
                .any(|s| s.is_empty() || s.to_string_lossy().starts_with('-'))
            {
                return Err(bad());
            }
            Ok(Some(Command::Validate {
                assets: PathBuf::from(&arguments[3]),
                checkout: PathBuf::from(&arguments[4]),
            }))
        }
        _ => Err(bad()),
    }
}

pub fn dispatch(command: Command, source: Option<&str>) -> Result<serde_json::Value> {
    match command {
        Command::Run {
            row,
            model,
            options,
        } => {
            let options = options.ok_or_else(|| refuse(LIVE_OPTIONS_MISSING))?;
            crate::combination(&row, &model)?;
            crate::live::require_observation_support(&row)?;
            let source = source.ok_or_else(|| refuse("candidate was built from a dirty tree or without git; evidence cannot be bound to a commit"))?;
            let candidate = options
                .assets
                .join(format!("ck-synapse{}", std::env::consts::EXE_SUFFIX));
            let executable = std::env::current_exe().map_err(|error| refuse(error.to_string()))?;
            if synapse_parity::canonical::sha256_file(&candidate)
                .map_err(|error| refuse(error.to_string()))?
                != synapse_parity::canonical::sha256_file(&executable)
                    .map_err(|error| refuse(error.to_string()))?
            {
                return Err(refuse(
                    "executing producer does not match the named candidate ck-synapse",
                ));
            }
            let mut runner = crate::live::LiveRunner::new(options.clone(), source)?;
            let record = crate::produce(&mut runner, &options.assets, &row, &model)?;
            crate::write_record(&record, &options.checkout)?;
            serde_json::to_value(record).map_err(|error| refuse(error.to_string()))
        }
        Command::Validate { assets, checkout } => {
            let source = source.ok_or_else(|| refuse("candidate was built from a dirty tree or without git; evidence cannot be bound to a commit"))?;
            let eligible = validate_checkout(&checkout, &assets, source)?;
            Ok(serde_json::json!({"source_commit": source, "eligible": eligible}))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ane_live_run_refuses_missing_supervisor_integration_and_writes_nothing() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!(".live/worker-refusal-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        for row in ["ane-m5"] {
            let options = crate::live::Options {
                assets: root.join("absent-assets"),
                checkout: root.join("checkout"),
                weights: root.join("absent-weights"),
            };
            let error = super::dispatch(
                super::Command::Run {
                    row: row.into(),
                    model: crate::MODELS[0].into(),
                    options: Some(options),
                },
                Some(&"1".repeat(40)),
            )
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("production residency supervisor integration"),
                "{row}: {error}"
            );
            assert!(
                !root.join("checkout").exists(),
                "{row} wrote a record before refusing"
            );
        }
        for row in crate::ROWS {
            if row != "ane-m5" {
                assert!(
                    crate::live::require_observation_support(row).is_ok(),
                    "{row}"
                );
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    use super::*;
    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }
    #[test]
    fn parses_certification_commands() {
        assert!(matches!(
            parse(&args(&[
                "certify",
                "run",
                "--row",
                "metal-m5",
                "--model",
                "gte-modernbert-base"
            ]))
            .unwrap(),
            Some(Command::Run { .. })
        ));
        assert!(matches!(
            parse(&args(&[
                "certify", "validate", "--assets", "assets", "checkout"
            ]))
            .unwrap(),
            Some(Command::Validate { .. })
        ));
        assert_eq!(parse(&args(&["restore-import"])).unwrap(), None);
    }
    #[test]
    fn run_requires_candidate_paths() {
        let command = parse(&args(&[
            "certify",
            "run",
            "--row",
            "metal-m5",
            "--model",
            "gte-modernbert-base",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            dispatch(command, None).unwrap_err().to_string(),
            format!("certification_refused: {LIVE_OPTIONS_MISSING}")
        );
    }
    #[test]
    fn bad_arguments_are_typed_refusals() {
        for values in [
            vec!["certify"],
            vec!["certify", "validate", "--assets"],
            vec!["certify", "run", "--row", "ane-m5", "--row", "ane-m5"],
            vec!["certify", "validate", "--assets", "--bad", "root"],
        ] {
            assert!(parse(&args(&values))
                .unwrap_err()
                .to_string()
                .starts_with("certification_refused:"));
        }
    }
}
