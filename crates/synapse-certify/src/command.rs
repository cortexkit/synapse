//! Candidate certification command parsing and validator dispatch.
use crate::{refuse, validate_checkout, Result};
use std::ffi::OsString;
use std::path::PathBuf;

pub const USAGE: &str = "usage: ck-synapse certify source | ck-synapse certify validate --assets <dir> <checkout-root> (run with ckdev-synapse-certify)";

/// Evidence is keyed by the embedded Git commit, never by a dirty or missing build identity.
pub fn source_stamp<'a>(revision: Option<&'a str>, tree: Option<&str>) -> Option<&'a str> {
    revision.filter(|revision| {
        tree == Some("clean")
            && revision.len() == 40
            && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Source,
    Validate { assets: PathBuf, checkout: PathBuf },
}

pub fn parse(arguments: &[OsString]) -> Result<Option<Command>> {
    if arguments.first().is_none_or(|arg| arg != "certify") {
        return Ok(None);
    }
    let bad = || refuse(USAGE);
    match arguments.get(1).and_then(|s| s.to_str()) {
        Some("source") if arguments.len() == 2 => Ok(Some(Command::Source)),
        Some("run") => Err(refuse("certify run moved to ckdev-synapse-certify run")),
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
        Command::Source => {
            let source = source.ok_or_else(|| refuse("candidate was built from a dirty tree or without git; source commit unavailable"))?;
            Ok(serde_json::Value::String(source.to_string()))
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
    use super::*;
    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }
    #[test]
    fn parses_certification_commands() {
        assert!(matches!(
            parse(&args(&["certify", "source"])).unwrap(),
            Some(Command::Source)
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
    fn run_is_not_a_shipped_command() {
        assert!(parse(&args(&["certify", "run"]))
            .unwrap_err()
            .to_string()
            .contains("ckdev-synapse-certify"));
    }

    #[test]
    fn clean_source_is_reported() {
        let commit = "a".repeat(40);
        let source = source_stamp(Some(&commit), Some("clean"));
        assert_eq!(dispatch(Command::Source, source).unwrap(), commit);
    }

    #[test]
    fn dirty_source_is_refused() {
        let commit = "a".repeat(40);
        let source = source_stamp(Some(&commit), Some("dirty"));
        assert!(dispatch(Command::Source, source).is_err());
    }

    #[test]
    fn missing_source_is_refused() {
        for (revision, tree) in [
            (None, None),
            (None, Some("clean")),
            (Some("bad"), Some("clean")),
        ] {
            assert!(dispatch(Command::Source, source_stamp(revision, tree)).is_err());
        }
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
