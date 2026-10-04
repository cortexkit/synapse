use std::process::Command;

fn main() {
    for name in ["HEAD", "refs"] {
        let output = Command::new("git")
            .args(["rev-parse", "--git-path", name])
            .output()
            .expect("resolve candidate Git metadata");
        assert!(
            output.status.success(),
            "cannot locate candidate Git metadata"
        );
        println!(
            "cargo:rerun-if-changed={}",
            String::from_utf8(output.stdout)
                .expect("Git metadata path is UTF-8")
                .trim()
        );
    }
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("certification candidate must be built from a Git checkout");
    assert!(
        output.status.success(),
        "cannot identify candidate source commit"
    );
    let source = String::from_utf8(output.stdout).expect("Git commit is UTF-8");
    println!(
        "cargo:rustc-env=SYNAPSE_CERTIFICATION_SOURCE={}",
        source.trim()
    );
}
