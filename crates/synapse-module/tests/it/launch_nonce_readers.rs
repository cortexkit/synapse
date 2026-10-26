//! Every read of the launch nonce must go through the SDK's one accessor,
//! `subc_client_rs::launch_nonce()`.
//!
//! The daemon hands a module its launch nonce on descriptor 3, named by
//! `SUBC_LAUNCH_NONCE_FD`, and for now also sets the `SUBC_LAUNCH_NONCE`
//! environment copy. The copy is going away. Code that reads it directly works
//! today and breaks silently the day the daemon stops setting it; code that
//! reads the descriptor itself would race the SDK, because the first read
//! closes the descriptor and its number is then reused for something else.
//! This test scans the source of every workspace crate and fails on any code
//! that names either variable, or the SDK constants that spell them, so the
//! only way to the nonce left is the accessor, whose call does not name them.
//!
//! Comment lines are skipped: explaining the variables is fine, reading them
//! is not. A string that merely mentions a name in a message is flagged too;
//! that is deliberate, since a `const` holding the name can be passed to
//! `env::var` from anywhere, and the rule stays simple to state.

use std::path::{Path, PathBuf};

/// Names whose presence in code means the nonce is being read around the SDK.
/// `SUBC_LAUNCH_NONCE` also matches `SUBC_LAUNCH_NONCE_FD` and the protocol
/// crate's `SUBC_LAUNCH_NONCE_ENV`; the other two are the subc-os constants
/// for the same two variables.
const FORBIDDEN: &[&str] = &[
    "SUBC_LAUNCH_NONCE",
    "LAUNCH_NONCE_ENV",
    "LAUNCH_NONCE_FD_ENV",
];

/// Lines of `source` that name a forbidden variable outside a comment, as
/// `(line number, line)`.
fn direct_nonce_reads(source: &str) -> Vec<(usize, String)> {
    source
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let code = code_part(line);
            FORBIDDEN
                .iter()
                .any(|name| code.contains(name))
                .then(|| (index + 1, line.trim().to_string()))
        })
        .collect()
}

/// The part of `line` before a `//` comment. A `//` inside a string literal
/// (a URL, say) ends the scan early; that can only hide text after it, and a
/// line that reads a variable names it before any URL in practice.
fn code_part(line: &str) -> &str {
    match line.find("//") {
        Some(start) => &line[..start],
        None => line,
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

/// Every `.rs` file under a `src` directory in `crates/` and `bench/`, which
/// between them hold every workspace member (nested members included).
fn shipped_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for top in ["crates", "bench"] {
        collect(&root.join(top), false, &mut found);
    }
    found.sort();
    found
}

fn collect(directory: &Path, inside_src: bool, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if matches!(name.as_ref(), "target" | ".git" | "node_modules") {
                continue;
            }
            collect(&path, inside_src || name == "src", found);
        } else if inside_src && name.ends_with(".rs") {
            found.push(path);
        }
    }
}

/// Every direct nonce read under `root`, as `path:line: text` relative to it.
fn tree_nonce_reads(root: &Path) -> Vec<String> {
    let mut offenders = Vec::new();
    for path in &shipped_sources(root) {
        let source = std::fs::read_to_string(path).expect("read source file");
        let relative = path.strip_prefix(root).unwrap_or(path);
        for (line, text) in nonce_reads_in_file(relative, &source) {
            offenders.push(format!("{}:{line}: {text}", relative.display()));
        }
    }
    offenders
}

#[test]
fn no_workspace_source_reads_the_launch_nonce_around_the_sdk() {
    let root = workspace_root();
    let sources = shipped_sources(&root);
    assert!(
        sources.len() > 50,
        "the scan found only {} source files under {}; it is not looking where the code is",
        sources.len(),
        root.display()
    );

    let offenders = tree_nonce_reads(&root);
    assert!(
        offenders.is_empty(),
        "read the launch nonce through subc_client_rs::launch_nonce(), not the \
         variables directly:\n{}",
        offenders.join("\n")
    );
}

/// The probe test stub reports whether a child inherited the nonce variables,
/// so reading them is its whole job. It builds only with `test-support` and is
/// never shipped.
const PROBE_STUB: &str = "crates/synapse-module/src/bin/probe_stub.rs";

/// A file whose first code line is `#![cfg(test)]` is compiled only for tests,
/// the whole-file form of the `#[cfg(test)]` items the command scan skips.
fn test_only_file(source: &str) -> bool {
    source
        .lines()
        .map(|line| code_part(line).trim())
        .find(|code| !code.is_empty())
        == Some("#![cfg(test)]")
}

// Only child_process.rs's remove_launch_nonce helper may name the SDK constants,
// and only in these exact removal statements. Reads in that file must still go
// through the SDK. Test-only files and the probe test stub are skipped.
fn nonce_reads_in_file(path: &Path, source: &str) -> Vec<(usize, String)> {
    if test_only_file(source) || path == Path::new(PROBE_STUB) {
        return Vec::new();
    }
    direct_nonce_reads(source)
        .into_iter()
        .filter(|(_, text)| {
            !(path == Path::new("crates/synapse-core/src/child_process.rs")
                && matches!(
                    text.as_str(),
                    "command.env_remove(subc_os::launch_nonce::LAUNCH_NONCE_ENV);"
                        | "command.env_remove(subc_os::launch_nonce::LAUNCH_NONCE_FD_ENV);"
                ))
        })
        .collect()
}

// Production constructors must be wrapped in without_launch_nonce (or its tokio
// counterpart) on the same or immediately preceding line (rustfmt wraps long
// initializers). Skip #[cfg(test)] items by balanced braces. The inventory only
// includes src directories, not tests/ fixtures, and this scan excludes bench/.
// This intentionally enforces a spelling convention, not dataflow.
fn unstripped_commands(path: &Path, source: &str) -> Vec<String> {
    if test_only_file(source) {
        return Vec::new();
    }
    let mut offenders = Vec::new();
    let mut test_item = false;
    let mut depth = 0isize;
    let mut opened = false;
    let mut previous = "";
    for (index, line) in source.lines().enumerate() {
        let code = code_part(line).trim();
        if code == "#[cfg(test)]" {
            test_item = true;
            opened = false;
            depth = 0;
        }
        if test_item {
            opened |= code.contains('{');
            depth += code.matches('{').count() as isize - code.matches('}').count() as isize;
            if (opened && depth == 0) || (!opened && code.ends_with(';')) {
                test_item = false;
            }
        } else if code.contains("Command::new(")
            && !code.contains("without_launch_nonce(")
            && !code.contains("without_launch_nonce_tokio(")
            && !previous.ends_with("without_launch_nonce(")
            && !previous.ends_with("without_launch_nonce_tokio(")
        {
            offenders.push(format!("{}:{}: {}", path.display(), index + 1, code));
        }
        previous = code;
    }
    offenders
}

#[test]
fn every_shipped_command_strips_the_launch_nonce() {
    let root = workspace_root();
    let sources = shipped_sources(&root);
    assert!(sources.len() > 50, "source inventory is unexpectedly empty");
    let mut offenders = Vec::new();
    for path in sources {
        let relative = path.strip_prefix(&root).unwrap();
        // The build-stamp crate is linked only into Cargo build scripts, which
        // run at compile time and never receive a daemon launch nonce, so its
        // git commands have nothing to strip. build.rs files themselves sit
        // outside src/ and are never collected.
        if relative.starts_with("bench") || relative.starts_with("crates/synapse-build-stamp") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read source file");
        offenders.extend(unstripped_commands(relative, &source));
    }
    assert!(
        offenders.is_empty(),
        "child commands must use without_launch_nonce:\n{}",
        offenders.join("\n")
    );
}

/// Which scanner a control case runs, with the path it pretends to scan.
#[derive(Debug)]
enum Scan {
    /// The line matcher alone.
    Line,
    /// The nonce-read scan with its per-file exemptions.
    File(&'static str),
    /// The unstripped-`Command` scan.
    Command(&'static str),
}

/// Line numbers each scanner reports for `source`.
fn flagged_lines(scan: &Scan, source: &str) -> Vec<usize> {
    match scan {
        Scan::Line => direct_nonce_reads(source)
            .into_iter()
            .map(|(line, _)| line)
            .collect(),
        Scan::File(path) => nonce_reads_in_file(Path::new(path), source)
            .into_iter()
            .map(|(line, _)| line)
            .collect(),
        Scan::Command(path) => unstripped_commands(Path::new(path), source)
            .iter()
            .map(|offender| {
                offender[path.len() + 1..]
                    .split(':')
                    .next()
                    .and_then(|line| line.parse().ok())
                    .unwrap_or_else(|| panic!("unparseable offender: {offender}"))
            })
            .collect(),
    }
}

/// Controls for the two scans above: each planted source must be flagged on
/// exactly the expected lines, so a scanner that stopped matching (or started
/// matching everything) fails here rather than passing the tree scans silently.
#[test]
fn scanner_controls_flag_planted_reads_and_honour_only_the_listed_exemptions() {
    const HELPER: &str = "crates/synapse-core/src/child_process.rs";
    const OTHER: &str = "crates/control/src/lib.rs";
    const REMOVAL: &str = "command.env_remove(subc_os::launch_nonce::LAUNCH_NONCE_ENV);";
    const READ: &str = "std::env::var_os(subc_protocol::SUBC_LAUNCH_NONCE_ENV);";
    let lead_test_only = format!("#![cfg(test)]\n{READ}");
    let trail_test_only = format!("{READ}\n#![cfg(test)]");
    let cases: Vec<(Scan, &str, &[usize])> = vec![
        // Direct reads by name or by constant are flagged, comment or not.
        (
            Scan::Line,
            r#"let nonce = std::env::var("SUBC_LAUNCH_NONCE").ok();"#,
            &[1],
        ),
        (
            Scan::Line,
            r#"let fd = env::var_os("SUBC_LAUNCH_NONCE_FD");"#,
            &[1],
        ),
        (
            Scan::Line,
            "let nonce = env::var_os(SUBC_LAUNCH_NONCE_ENV);",
            &[1],
        ),
        (
            Scan::Line,
            "let fd = std::env::var(subc_os::launch_nonce::LAUNCH_NONCE_FD_ENV);",
            &[1],
        ),
        (
            Scan::Line,
            r#"let nonce = env::var("SUBC_LAUNCH_NONCE"); // read the nonce"#,
            &[1],
        ),
        // Comments and the SDK accessor are fine.
        (
            Scan::Line,
            "// The SUBC_LAUNCH_NONCE copy is going away.",
            &[],
        ),
        (
            Scan::Line,
            "/// Reads `SUBC_LAUNCH_NONCE_FD` through the SDK.",
            &[],
        ),
        (
            Scan::Line,
            "let nonce = subc_client_rs::launch_nonce();",
            &[],
        ),
        // Only the helper file may name the constants, and only to remove them.
        (Scan::File(HELPER), REMOVAL, &[]),
        (Scan::File(OTHER), REMOVAL, &[1]),
        (
            Scan::File(HELPER),
            "std::env::var(subc_os::launch_nonce::LAUNCH_NONCE_ENV);",
            &[1],
        ),
        // The exemption is the exact probe stub path, not every test binary.
        (Scan::File(PROBE_STUB), READ, &[]),
        (
            Scan::File("crates/synapse-module/src/bin/timeout_worker.rs"),
            READ,
            &[1],
        ),
        // A test-only file is skipped only when the attribute leads the file.
        (Scan::File(OTHER), &lead_test_only, &[]),
        (Scan::File(OTHER), &trail_test_only, &[1]),
        // Unwrapped constructors are flagged; wrapped ones, including one
        // rustfmt split across lines, and #[cfg(test)] items are not.
        (
            Scan::Command(OTHER),
            "fn launch() {\n    let child = Command::new(\"worker\").spawn();\n}\n",
            &[2],
        ),
        (
            Scan::Command(OTHER),
            "let command = without_launch_nonce(Command::new(\"worker\"));",
            &[],
        ),
        (
            Scan::Command(OTHER),
            "let command = without_launch_nonce(\nCommand::new(\"worker\"));",
            &[],
        ),
        (
            Scan::Command(OTHER),
            "#[cfg(test)]\nmod tests {\nlet command = Command::new(\"worker\");\n}\n",
            &[],
        ),
        (
            Scan::Command(OTHER),
            "// test file\n#![cfg(test)]\nlet command = Command::new(\"worker\");\n",
            &[],
        ),
        (
            Scan::Command(OTHER),
            "let command = Command::new(\"worker\");\n#![cfg(test)]\n",
            &[1],
        ),
    ];
    for (scan, source, expected) in &cases {
        assert_eq!(
            &flagged_lines(scan, source),
            expected,
            "{scan:?} over:\n{source}"
        );
    }
    // The command scan names the file, line and code it flagged.
    assert_eq!(
        unstripped_commands(Path::new(OTHER), "let child = Command::new(\"worker\");"),
        vec![format!("{OTHER}:1: let child = Command::new(\"worker\");")]
    );

    // The decode worker passes its sidecar a nonce of its own on `--nonce`.
    // That argument is not the module's launch nonce (the worker mints it to
    // recognise its own sidecar), so it must not trip the scan; this reads the
    // real file so the check follows the code if it moves.
    let runner = workspace_root().join("crates/synapse-worker-decode/src/runner.rs");
    let source = std::fs::read_to_string(&runner).expect("read the decode worker runner");
    assert!(
        source.contains("\"--nonce\""),
        "the decode worker no longer passes --nonce; update this check"
    );
    assert!(shipped_sources(&workspace_root()).contains(&runner.canonicalize().unwrap()));
    assert!(direct_nonce_reads(&source).is_empty());

    // The whole tree scan, inventory walk included, over a scratch workspace
    // with a read planted in a nested member's `src/` and in places the walk
    // must skip: the line controls prove the matcher, this proves the walk
    // reaches the files the matcher is meant to see.
    let root = std::env::temp_dir().join(format!(
        "synapse-tests/nonce-scan-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let read = r#"let nonce = std::env::var("SUBC_LAUNCH_NONCE");"#;
    for (file, body) in [
        ("crates/outer/inner/src/deep/lib.rs", read),
        ("bench/lane/src/main.rs", "fn main() {}"),
        ("crates/outer/target/debug/build/gen.rs", read),
        ("crates/outer/tests/fixture.rs", read),
    ] {
        let path = root.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
    }
    let offenders = tree_nonce_reads(&root);
    std::fs::remove_dir_all(&root).unwrap();
    // Built from components so the expected path uses the platform separator,
    // the same way the scan renders the path it walked.
    let planted: PathBuf = ["crates", "outer", "inner", "src", "deep", "lib.rs"]
        .iter()
        .collect();
    assert_eq!(offenders, vec![format!("{}:1: {read}", planted.display())]);
}
