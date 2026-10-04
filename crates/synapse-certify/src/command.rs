//! Candidate certification command parsing and validator dispatch.
use crate::{refuse, validate_checkout, Result};
use std::ffi::OsString;
use std::path::PathBuf;

pub const LIVE_RUNNER_MISSING: &str = "live runner not yet integrated: needs module sequence_too_long, preload profiles, ane-direct routing and the Metal Qwen reranker";

pub const USAGE: &str = "usage: ck-synapse certify run --row <row-id> --model <slug> | ck-synapse certify validate --assets <dir> <checkout-root>";

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Run { row: String, model: String },
    Validate { assets: PathBuf, checkout: PathBuf },
}

pub fn parse(arguments: &[OsString]) -> Result<Option<Command>> {
    if arguments.first().is_none_or(|arg| arg != "certify") {
        return Ok(None);
    }
    let bad = || refuse(USAGE);
    match arguments.get(1).and_then(|s| s.to_str()) {
        Some("run") if arguments.len() == 6 => {
            let mut row = None;
            let mut model = None;
            for pair in arguments[2..].as_chunks::<2>().0 {
                let value = pair[1]
                    .to_str()
                    .filter(|s| !s.is_empty() && !s.starts_with('-'))
                    .ok_or_else(bad)?;
                match pair[0].to_str() {
                    Some("--row") if row.is_none() => row = Some(value.to_string()),
                    Some("--model") if model.is_none() => model = Some(value.to_string()),
                    _ => return Err(bad()),
                }
            }
            let row = row.ok_or_else(bad)?;
            let model = model.ok_or_else(bad)?;
            crate::combination(&row, &model)?;
            Ok(Some(Command::Run { row, model }))
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

pub fn dispatch(command: Command) -> Result<serde_json::Value> {
    match command {
        Command::Run { .. } => Err(refuse(LIVE_RUNNER_MISSING)),
        Command::Validate { assets, checkout } => {
            let source = env!("SYNAPSE_CERTIFICATION_SOURCE");
            let eligible = validate_checkout(&checkout, &assets, source)?;
            Ok(serde_json::json!({"source_commit": source, "eligible": eligible}))
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn run_refuses_missing_live_runner() {
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
            dispatch(command).unwrap_err().to_string(),
            format!("certification_refused: {LIVE_RUNNER_MISSING}")
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
