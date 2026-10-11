#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod modernbert;
// Fixed-slot multi-row layouts remain experimental. Serving compiles the kernel
// that packs rows into 256 columns separately from single-row kernels, charging
// its 28 layer executables to the same 100-executable ANE residency budget.
#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
mod multirow;
#[cfg(target_os = "macos")]
mod profile;
#[cfg(target_os = "macos")]
mod qwen;
#[cfg(target_os = "macos")]
mod worker;
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    if std::env::args_os()
        .skip(1)
        .any(|argument| argument == "--manifest")
    {
        print_manifest();
        return Ok(());
    }
    worker::main()
}
#[cfg(not(target_os = "macos"))]
fn main() {
    if std::env::args_os()
        .skip(1)
        .any(|argument| argument == "--manifest")
    {
        print_manifest();
        return;
    }
    eprintln!("{{\"status\":\"refused\",\"code\":\"ane_private_api_unavailable\",\"required\":null,\"observed\":null}}");
    std::process::exit(2);
}

fn print_manifest() {
    use subc_protocol::manifest::{
        build_provenance_from_source, BuildGitShaSource, GitTreeState, ModuleManifest,
    };

    let source = match (
        option_env!("SYNAPSE_BUILD_REV"),
        option_env!("SYNAPSE_BUILD_TREE"),
    ) {
        (Some(revision), Some(tree)) => BuildGitShaSource::Git {
            revision,
            tree_state: if tree == "clean" {
                GitTreeState::Clean
            } else {
                GitTreeState::Dirty
            },
        },
        _ => BuildGitShaSource::NoGitDir,
    };
    let provenance = build_provenance_from_source(source, None, None)
        .expect("build provenance uses the canonical Git revision emitted by build.rs");
    let manifest = ModuleManifest::builder("synapse-worker-ane-direct", env!("CARGO_PKG_VERSION"))
        .provenance(Some(provenance))
        .build();
    println!(
        "{}",
        serde_json::to_string(&manifest).expect("the SDK worker manifest is serializable")
    );
}
