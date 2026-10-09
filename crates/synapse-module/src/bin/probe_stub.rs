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
    let path = std::env::current_exe().unwrap();
    let config: Value =
        serde_json::from_slice(&fs::read(path.with_extension("json")).unwrap()).unwrap();
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.with_extension("log"))
        .unwrap();
    writeln!(log, "{}", json!({"args": args, "nonce": std::env::var_os(subc_protocol::SUBC_LAUNCH_NONCE_ENV).is_some(), "nonce_fd": std::env::var_os("SUBC_LAUNCH_NONCE_FD").is_some(), "pid": std::process::id()})).unwrap();
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
