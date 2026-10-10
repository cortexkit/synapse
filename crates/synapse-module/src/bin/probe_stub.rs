#![forbid(unsafe_code)]

//! A real executable fixture for bounded worker-probe tests on every host.

use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
    thread,
    time::Duration,
};

fn main() {
    // Tests launch each shared copy once with --warm before timing it: macOS
    // assesses a new executable on its first launch, which under load can take
    // longer than the probe deadlines under test. Warming logs nothing.
    if std::env::args().nth(1).as_deref() == Some("--warm") {
        return;
    }
    // Many tests, in several processes at once, run this same executable file
    // (a new executable path costs a macOS security assessment). The caller's
    // state directory therefore holds this copy's configuration and log, named
    // after the copy's file name, rather than files beside the executable.
    let path = std::env::current_exe().unwrap();
    let state = PathBuf::from(
        std::env::var_os("SYNAPSE_PROBE_STUB_STATE")
            .expect("SYNAPSE_PROBE_STUB_STATE names the probe stub's state directory"),
    );
    let files = state.join(path.file_name().unwrap());
    let config: Value =
        serde_json::from_slice(&fs::read(files.with_extension("json")).unwrap()).unwrap();
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(files.with_extension("log"))
        .unwrap();
    // Concurrent probes append to the same log, and the tests parse each line
    // as JSON. One write of the whole line keeps lines from interleaving;
    // `writeln!` would split it into several writes.
    let mut line = json!({"args": args, "nonce": std::env::var_os(subc_protocol::SUBC_LAUNCH_NONCE_ENV).is_some(), "nonce_fd": std::env::var_os("SUBC_LAUNCH_NONCE_FD").is_some(), "pid": std::process::id()}).to_string();
    line.push('\n');
    log.write_all(line.as_bytes()).unwrap();
    drop(log);
    if args == ["--assert-reaped"] {
        // Parents send this fixture its probe PID after the bounded runner returns.
        // No signal is sent: the assertion only observes whether the child remains.
        let mut text = String::new();
        std::io::stdin().read_to_string(&mut text).unwrap();
        let pid = text.trim().parse::<u32>().unwrap();
        #[cfg(unix)]
        {
            let result = synapse_core::without_launch_nonce(std::process::Command::new("kill"))
                .args(["-0", &pid.to_string()])
                .status()
                .unwrap();
            assert!(!result.success(), "probe child still exists: {pid}");
        }
        #[cfg(windows)]
        {
            let output = synapse_core::without_launch_nonce(std::process::Command::new("tasklist"))
                .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
                .output()
                .unwrap();
            assert!(
                !String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\"")),
                "probe child still exists: {pid}"
            );
        }
        return;
    }
    assert!(
        args == ["--version"] || args == ["--probe-floor"],
        "unexpected argv: {args:?}"
    );
    let kind = if args[0] == "--version" {
        "version"
    } else {
        "floor"
    };
    let behavior = &config[kind];
    if let Some(ready) = behavior["ready"].as_str() {
        fs::write(ready, "ready").unwrap();
        let release = PathBuf::from(behavior["release"].as_str().unwrap());
        while !release.exists() {
            thread::sleep(Duration::from_millis(5));
        }
    }
    if behavior["sleep"].as_bool() == Some(true) {
        thread::sleep(Duration::from_secs(60));
    }
    if let Some(text) = behavior["stdout"].as_str() {
        print!("{text}");
    }
    if let Some(envelope) = behavior.get("envelope") {
        println!("{envelope}");
    }
    if let Some(text) = behavior["stderr"].as_str() {
        eprint!("{text}");
    }
    std::process::exit(behavior["exit"].as_i64().unwrap_or(0) as i32);
}
