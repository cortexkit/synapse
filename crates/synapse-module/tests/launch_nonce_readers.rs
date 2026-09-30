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

    let mut offenders = Vec::new();
    for path in &sources {
        let source = std::fs::read_to_string(path).expect("read source file");
        for (line, text) in direct_nonce_reads(&source) {
            let relative = path.strip_prefix(&root).unwrap_or(path);
            offenders.push(format!("{}:{line}: {text}", relative.display()));
        }
    }
    assert!(
        offenders.is_empty(),
        "read the launch nonce through subc_client_rs::launch_nonce(), not the \
         variables directly:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_scan_flags_direct_reads_and_leaves_other_nonces_alone() {
    for read in [
        r#"let nonce = std::env::var("SUBC_LAUNCH_NONCE").ok();"#,
        r#"let fd = env::var_os("SUBC_LAUNCH_NONCE_FD");"#,
        "let nonce = env::var_os(SUBC_LAUNCH_NONCE_ENV);",
        "let fd = std::env::var(subc_os::launch_nonce::LAUNCH_NONCE_FD_ENV);",
        r#"let nonce = env::var("SUBC_LAUNCH_NONCE"); // read the nonce"#,
    ] {
        assert_eq!(direct_nonce_reads(read).len(), 1, "not flagged: {read}");
    }
    for fine in [
        "// The SUBC_LAUNCH_NONCE copy is going away.",
        "/// Reads `SUBC_LAUNCH_NONCE_FD` through the SDK.",
        "let nonce = subc_client_rs::launch_nonce();",
    ] {
        assert!(direct_nonce_reads(fine).is_empty(), "flagged: {fine}");
    }

    // The decode worker passes its sidecar a nonce of its own on `--nonce`.
    // That argument is not the module's launch nonce (the worker mints it to
    // recognise its own sidecar), so it must not trip the scan; this reads the real
    // file so the check follows the code if it moves.
    let runner = workspace_root().join("crates/synapse-worker-decode/src/runner.rs");
    let source = std::fs::read_to_string(&runner).expect("read the decode worker runner");
    assert!(
        source.contains("\"--nonce\""),
        "the decode worker no longer passes --nonce; update this check"
    );
    assert!(shipped_sources(&workspace_root()).contains(&runner.canonicalize().unwrap()));
    assert!(direct_nonce_reads(&source).is_empty());
}
