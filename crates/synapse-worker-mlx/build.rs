use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    require_xcrun_tool("metal");
    require_xcrun_tool("metallib");
}

fn require_xcrun_tool(tool: &str) {
    match Command::new("xcrun").args(["--find", tool]).output() {
        Ok(output) if output.status.success() && !output.stdout.is_empty() => {}
        Ok(output) => panic!(
            "synapse-worker-mlx requires Apple's Metal toolchain, but `xcrun --find {tool}` failed with status {}. Install full Xcode with Metal support and build with DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer (or switch xcode-select to that path). Cargo will not set DEVELOPER_DIR automatically.",
            output.status
        ),
        Err(error) => panic!(
            "synapse-worker-mlx requires Apple's Metal toolchain, but `xcrun --find {tool}` could not run: {error}. Install full Xcode with Metal support and build with DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer (or switch xcode-select to that path). Cargo will not set DEVELOPER_DIR automatically."
        ),
    }
}
