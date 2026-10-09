use std::fmt;
#[cfg(any(target_os = "macos", test))]
use std::io::Read;
#[cfg(any(target_os = "macos", test))]
use std::process::{Command, Stdio};
#[cfg(any(target_os = "macos", test))]
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::EngineIdentity;

/// How long one command probe may take before it is abandoned. `sw_vers`
/// and `sysctl` answer in single-digit milliseconds on a healthy host,
/// so this is a generous ceiling rather than a tuning knob.
#[cfg(any(target_os = "macos", test))]
const PROBE_BUDGET: Duration = Duration::from_millis(2_000);
#[cfg(any(target_os = "macos", test))]
const PROBE_POLL: Duration = Duration::from_millis(10);

/// A machine-identity input that could not be established.
///
/// Deliberately a hard error rather than a default. Every field it guards feeds
/// the machine-profile hash, and that hash gates serving: a substituted
/// placeholder rotates the profile, fails every certified lane closed, and
/// rotates BACK on the next boot that happens to succeed. A silent,
/// self-reverting identity change is far worse to operate than a refusal to
/// start, which the daemon surfaces and retries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileProbeError {
    program: String,
    reason: String,
}

impl ProfileProbeError {
    fn new(program: &str, reason: impl Into<String>) -> Self {
        Self {
            program: program.to_string(),
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn program(&self) -> &str {
        &self.program
    }
}

impl fmt::Display for ProfileProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "machine identity probe `{}` {}; refusing to derive a machine profile from a substituted value",
            self.program, self.reason
        )
    }
}

impl std::error::Error for ProfileProbeError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineProfileBase {
    pub os_build: String,
    pub arch: String,
    pub chip_model: String,
    pub ram_class: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ane_subtype: Option<String>,
}

pub trait MachineProfileCollector {
    fn collect_base_profile(&self) -> Result<MachineProfileBase, ProfileProbeError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemMachineProfileCollector;

/// One identity probe. A function pointer rather than a closure so the
/// collector stays trivially copyable, and an internal seam rather than a
/// configuration knob: production always passes `command_stdout`, and only
/// tests substitute a failing prober to prove the refusal is real.
#[cfg(target_os = "macos")]
type Prober = fn(&str, &[&str]) -> Result<String, ProfileProbeError>;

impl MachineProfileCollector for SystemMachineProfileCollector {
    fn collect_base_profile(&self) -> Result<MachineProfileBase, ProfileProbeError> {
        #[cfg(target_os = "macos")]
        {
            collect_base_profile_with(command_stdout)
        }
        #[cfg(target_os = "linux")]
        {
            collect_linux_profile_with(native_uname, native_sysinfo)
        }
        #[cfg(target_os = "windows")]
        {
            collect_windows_profile_with(
                native_windows_version,
                native_windows_arch,
                native_windows_memory,
            )
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            Err(ProfileProbeError::new(
                "native platform",
                "cannot establish os_build, chip_model and ram_class on this unsupported platform",
            ))
        }
    }
}

#[cfg(target_os = "macos")]
fn collect_base_profile_with(probe: Prober) -> Result<MachineProfileBase, ProfileProbeError> {
    let chip_model = chip_model(probe)?;
    Ok(MachineProfileBase {
        os_build: os_build(probe)?,
        // A compile-time constant, identical on every run of this binary, so it
        // cannot rotate the profile the way a probed value can.
        arch: std::env::consts::ARCH.to_string(),
        chip_model: chip_model.clone(),
        ram_class: ram_class(probe)?,
        // Derived from the chip model rather than probed, and legitimately
        // absent on hardware without a Neural Engine.
        ane_subtype: ane_subtype(&chip_model),
    })
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn native_source_error(source: &str, fields: &str, reason: impl fmt::Display) -> ProfileProbeError {
    ProfileProbeError::new(source, format!("could not establish {fields}: {reason}"))
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn native_profile(os_build: String, chip_model: String, ram_class: String) -> MachineProfileBase {
    MachineProfileBase {
        os_build,
        arch: std::env::consts::ARCH.to_string(),
        chip_model,
        ram_class,
        ane_subtype: None,
    }
}

#[cfg(any(target_os = "linux", test))]
struct UnameReading {
    sysname: String,
    release: String,
    machine: String,
}

#[cfg(any(target_os = "linux", test))]
struct SysinfoReading {
    totalram: u64,
    mem_unit: u32,
}

#[cfg(any(target_os = "linux", test))]
fn linux_identity(reading: UnameReading) -> Result<(String, String), ProfileProbeError> {
    if reading.sysname.is_empty() || reading.release.is_empty() || reading.machine.is_empty() {
        return Err(native_source_error(
            "uname(2)",
            "os_build and chip_model",
            "returned an empty identity",
        ));
    }
    Ok((
        format!("{} {}", reading.sysname, reading.release),
        reading.machine,
    ))
}

#[cfg(any(target_os = "linux", test))]
fn linux_ram_class(reading: SysinfoReading) -> Result<String, ProfileProbeError> {
    let bytes = reading
        .totalram
        .checked_mul(u64::from(reading.mem_unit))
        .ok_or_else(|| {
            native_source_error("sysinfo(2)", "ram_class", "totalram * mem_unit overflowed")
        })?;
    native_ram_class("sysinfo(2)", bytes)
}

/// Each function supplies one source, so a refusal can be injected without
/// invoking the OS or starving an unrelated identity field.
#[cfg(any(target_os = "linux", test))]
fn collect_linux_profile_with(
    uname: impl FnOnce() -> Result<UnameReading, String>,
    sysinfo: impl FnOnce() -> Result<SysinfoReading, String>,
) -> Result<MachineProfileBase, ProfileProbeError> {
    // One reading feeds both fields, and a refusal must name both consumers.
    let reading = uname()
        .map_err(|reason| native_source_error("uname(2)", "os_build and chip_model", reason))?;
    let (os_build, chip_model) = linux_identity(reading)?;
    let memory =
        sysinfo().map_err(|reason| native_source_error("sysinfo(2)", "ram_class", reason))?;
    Ok(native_profile(
        os_build,
        chip_model,
        linux_ram_class(memory)?,
    ))
}

#[cfg(target_os = "linux")]
fn native_uname() -> Result<UnameReading, String> {
    let reading = rustix::system::uname();
    Ok(UnameReading {
        sysname: reading
            .sysname()
            .to_str()
            .map_err(|error| error.to_string())?
            .to_string(),
        release: reading
            .release()
            .to_str()
            .map_err(|error| error.to_string())?
            .to_string(),
        machine: reading
            .machine()
            .to_str()
            .map_err(|error| error.to_string())?
            .to_string(),
    })
}

#[cfg(target_os = "linux")]
fn native_sysinfo() -> Result<SysinfoReading, String> {
    let reading = rustix::system::sysinfo();
    #[cfg(target_pointer_width = "64")]
    let totalram = reading.totalram;
    #[cfg(not(target_pointer_width = "64"))]
    let totalram = u64::from(reading.totalram);
    Ok(SysinfoReading {
        totalram,
        mem_unit: reading.mem_unit,
    })
}

#[cfg(any(target_os = "windows", test))]
fn windows_os_build((major, minor, build): (u32, u32, u32)) -> String {
    format!("Windows {major}.{minor}.{build}")
}

#[cfg(any(target_os = "windows", test))]
fn windows_chip_model(architecture: u16) -> Result<String, ProfileProbeError> {
    match architecture {
        9 => Ok("x86_64".to_string()),
        12 => Ok("aarch64".to_string()),
        other => Err(native_source_error(
            "GetNativeSystemInfo",
            "chip_model",
            format!("unsupported architecture {other}"),
        )),
    }
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn native_ram_class(source: &str, bytes: u64) -> Result<String, ProfileProbeError> {
    if bytes == 0 {
        return Err(native_source_error(
            source,
            "ram_class",
            "returned zero physical memory",
        ));
    }
    Ok(ram_class_from_bytes(bytes))
}

#[cfg(any(target_os = "windows", test))]
fn collect_windows_profile_with(
    version: impl FnOnce() -> Result<(u32, u32, u32), String>,
    architecture: impl FnOnce() -> Result<u16, String>,
    memory: impl FnOnce() -> Result<u64, String>,
) -> Result<MachineProfileBase, ProfileProbeError> {
    let version =
        version().map_err(|reason| native_source_error("RtlGetVersion", "os_build", reason))?;
    let architecture = architecture()
        .map_err(|reason| native_source_error("GetNativeSystemInfo", "chip_model", reason))?;
    let bytes = memory()
        .map_err(|reason| native_source_error("GlobalMemoryStatusEx", "ram_class", reason))?;
    Ok(native_profile(
        windows_os_build(version),
        windows_chip_model(architecture)?,
        native_ram_class("GlobalMemoryStatusEx", bytes)?,
    ))
}

#[cfg(target_os = "windows")]
fn native_windows_version() -> Result<(u32, u32, u32), String> {
    synapse_native_probe::windows::version().map_err(|error| error.to_string())
}

#[cfg(target_os = "windows")]
fn native_windows_arch() -> Result<u16, String> {
    synapse_native_probe::windows::processor_architecture().map_err(|error| error.to_string())
}

#[cfg(target_os = "windows")]
fn native_windows_memory() -> Result<u64, String> {
    synapse_native_probe::windows::total_physical_memory().map_err(|error| error.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineProfile {
    pub os_build: String,
    pub arch: String,
    pub chip_model: String,
    pub ram_class: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ane_subtype: Option<String>,
    #[serde(default)]
    pub engine_identities: Vec<EngineIdentity>,
}

impl MachineProfile {
    pub fn collect<C, I>(collector: &C, engine_identities: I) -> Result<Self, ProfileProbeError>
    where
        C: MachineProfileCollector,
        I: IntoIterator<Item = EngineIdentity>,
    {
        let base = collector.collect_base_profile()?;
        let mut engine_identities = engine_identities.into_iter().collect::<Vec<_>>();
        engine_identities.sort_by(|left, right| {
            left.engine
                .cmp(&right.engine)
                .then_with(|| left.version.cmp(&right.version))
                .then_with(|| left.build_flags.cmp(&right.build_flags))
        });
        Ok(Self {
            os_build: base.os_build,
            arch: base.arch,
            chip_model: base.chip_model,
            ram_class: base.ram_class,
            ane_subtype: base.ane_subtype,
            engine_identities,
        })
    }

    #[must_use]
    pub fn stable_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("machine profile should always serialize")
    }

    #[must_use]
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.stable_bytes());
        hex::encode(hasher.finalize())
    }

    /// Hash used by new certification records. The explicit revision prevents
    /// future profile-field additions from silently reusing old evidence.
    #[must_use]
    pub fn revisioned_hash(&self) -> String {
        crate::revisioned_machine_profile_hash(&self.stable_bytes())
    }
}

#[cfg(target_os = "macos")]
fn os_build(probe: Prober) -> Result<String, ProfileProbeError> {
    #[cfg(target_os = "macos")]
    {
        probe("sw_vers", &["-buildVersion"])
    }
}

#[cfg(target_os = "macos")]
fn chip_model(probe: Prober) -> Result<String, ProfileProbeError> {
    #[cfg(target_os = "macos")]
    {
        // hw.model is the documented fallback for hosts where the brand string
        // is absent, so a failure of the first probe is not yet a refusal.
        match sysctl_value(probe, "machdep.cpu.brand_string") {
            Ok(value) => Ok(value),
            Err(_) => sysctl_value(probe, "hw.model"),
        }
    }
}

#[cfg(target_os = "macos")]
fn ram_class(probe: Prober) -> Result<String, ProfileProbeError> {
    #[cfg(target_os = "macos")]
    {
        let raw = sysctl_value(probe, "hw.memsize")?;
        let bytes = raw.parse::<u64>().map_err(|_| {
            ProfileProbeError::new(
                "sysctl",
                format!("returned an unparseable hw.memsize {raw:?}"),
            )
        })?;
        Ok(ram_class_from_bytes(bytes))
    }
}

/// The ANE subtype, carrying HOW it was learned in the value itself.
///
/// The `(map)` suffix is provenance: this came from the static chip-identity
/// table, not from a device probe. Nothing parses it today.
///
/// READ THIS BEFORE ADDING A PROBED SUBTYPE. The suffix means the two spellings
/// are distinct strings, and this value feeds the machine-profile hash that
/// gates serving, so the day a probe lands someone must decide whether `"h16"`
/// and `"h16(map)"` are the same machine. There is no safe default:
///
/// - Treat them as THE SAME and certification evidence gathered while we were
///   guessing transfers to a machine where the guess was verified. Usually
///   harmless, and wrong exactly when the static map was wrong about a chip --
///   which is the case a probe exists to catch.
/// - Treat them as DIFFERENT and every Apple machine's fingerprint rotates when
///   the probe ships, costing a full re-certification for no physical change.
///
/// Whoever adds the probe owns that call and should record it here rather than
/// letting the string comparison decide silently.
#[cfg(target_os = "macos")]
fn ane_subtype(chip_model: &str) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        // Public IORegistry inspection on current Apple silicon exposes ANE
        // firmware functions but no stable subtype property. Keep private
        // _ANEDeviceInfo out of the daemon and use the static chip-identity
        // mapping until macOS exposes a supported read-only subtype value.
        mapped_ane_subtype(chip_model)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = chip_model;
        None
    }
}

#[cfg(any(target_os = "macos", test))]
fn mapped_ane_subtype(chip_model: &str) -> Option<String> {
    let chip_model = chip_model.trim().to_ascii_lowercase();
    if chip_model == "apple m5 max" {
        Some("h17(map)".to_string())
    } else if chip_model == "apple m5"
        || chip_model == "apple m4"
        || chip_model.starts_with("apple m4 ")
    {
        Some("h16(map)".to_string())
    } else {
        None
    }
}

fn ram_class_from_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    let gib = bytes.div_ceil(GIB).max(1);
    for bucket in [4_u64, 8, 16, 32, 64, 128, 256] {
        if gib <= bucket {
            return format!("le_{bucket}_gib");
        }
    }
    "gt_256_gib".to_string()
}

#[cfg(target_os = "macos")]
fn sysctl_value(probe: Prober, name: &str) -> Result<String, ProfileProbeError> {
    probe("sysctl", &["-n", name])
}

/// Runs one identity probe under a deadline.
///
/// `Command::output()` is an unbounded wait on a process this module does not
/// own, and on macOS 27 a program that touches a path under policy evaluation
/// can block in the kernel indefinitely. On expiry the child is killed and then
/// ABANDONED rather than waited on: a process blocked in the kernel does not
/// answer a signal either, so reaping it would restore the very hang the
/// deadline exists to escape.
#[cfg(any(target_os = "macos", test))]
fn command_stdout(program: &str, args: &[&str]) -> Result<String, ProfileProbeError> {
    command_stdout_within(program, args, PROBE_BUDGET)
}

/// The budget is a parameter so the deadline path can be exercised in a test
/// without a two-second wait. It is not a configuration knob: the only
/// production caller passes `PROBE_BUDGET`.
#[cfg(any(target_os = "macos", test))]
fn command_stdout_within(
    program: &str,
    args: &[&str],
    budget: Duration,
) -> Result<String, ProfileProbeError> {
    let mut child = crate::without_launch_nonce(Command::new(program))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            ProfileProbeError::new(program, format!("could not be spawned: {error}"))
        })?;

    let deadline = Instant::now() + budget;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    return Err(ProfileProbeError::new(
                        program,
                        format!("did not answer within {}ms", budget.as_millis()),
                    ));
                }
                std::thread::sleep(PROBE_POLL);
            }
            Err(error) => {
                let _ = child.kill();
                return Err(ProfileProbeError::new(
                    program,
                    format!("could not be waited on: {error}"),
                ));
            }
        }
    };

    if !status.success() {
        return Err(ProfileProbeError::new(
            program,
            format!("exited with {status}"),
        ));
    }

    // Read only after exit. These probes emit tens of bytes, far below the pipe
    // buffer, so the child cannot have blocked on a full pipe before exiting.
    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .ok_or_else(|| ProfileProbeError::new(program, "produced no stdout handle"))?
        .read_to_end(&mut raw)
        .map_err(|error| {
            ProfileProbeError::new(program, format!("stdout could not be read: {error}"))
        })?;
    let value = String::from_utf8(raw)
        .map_err(|_| ProfileProbeError::new(program, "returned non-UTF-8 output"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ProfileProbeError::new(program, "returned an empty value"));
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use super::*;

    #[cfg(target_os = "macos")]
    thread_local! {
        /// Which probes the current case starves: a program, and the specific
        /// arguments to starve (empty means every call to it). Thread-local
        /// because `Prober` is a function pointer on the production path and
        /// cannot capture.
        static STARVED: RefCell<Option<(String, Vec<String>)>> = const { RefCell::new(None) };
    }

    struct FakeCollector {
        ane_subtype: Option<&'static str>,
    }

    impl MachineProfileCollector for FakeCollector {
        fn collect_base_profile(&self) -> Result<MachineProfileBase, ProfileProbeError> {
            Ok(MachineProfileBase {
                os_build: "23G93".to_string(),
                arch: "aarch64".to_string(),
                chip_model: "Apple M3".to_string(),
                ram_class: "le_32_gib".to_string(),
                ane_subtype: self.ane_subtype.map(str::to_string),
            })
        }
    }

    #[test]
    fn machine_profile_hash_is_stable_and_sorts_engines() {
        let mut flags = BTreeMap::new();
        flags.insert("execution_provider".to_string(), "cpu".to_string());
        let ort = EngineIdentity {
            engine: "ort".to_string(),
            version: "2.0".to_string(),
            build_flags: flags,
        };
        let llama = EngineIdentity {
            engine: "llama.cpp".to_string(),
            version: "1.0".to_string(),
            build_flags: BTreeMap::new(),
        };

        let collector = FakeCollector { ane_subtype: None };
        let left = MachineProfile::collect(&collector, [ort.clone(), llama.clone()])
            .expect("fake collector never fails");
        let right =
            MachineProfile::collect(&collector, [llama, ort]).expect("fake collector never fails");
        assert_eq!(left, right);
        assert_eq!(left.hash(), right.hash());
        assert_eq!(left.engine_identities[0].engine, "llama.cpp");
    }

    /// A probe that never answers must refuse within its budget rather than
    /// stalling the boot. `sleep 30` stands in for the macOS 27 condition where
    /// a program touching a path under policy evaluation blocks in the kernel.
    #[test]
    fn a_hanging_probe_refuses_within_its_budget() {
        #[cfg(not(target_os = "windows"))]
        let (program, args): (&str, &[&str]) = ("sleep", &["30"]);
        // Windows ping sends one request per second. Invoke it directly so the
        // deadline test owns its only child rather than a shell's grandchild.
        #[cfg(target_os = "windows")]
        let (program, args): (&str, &[&str]) = ("ping", &["-n", "31", "127.0.0.1"]);
        let started = Instant::now();
        let error = command_stdout_within(program, args, Duration::from_millis(150))
            .expect_err("a hanging probe must refuse");
        let elapsed = started.elapsed();

        assert_eq!(error.program(), program);
        assert!(
            error.to_string().contains("did not answer within 150ms"),
            "the refusal must name the budget it exceeded: {error}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the deadline must bound the wait, took {elapsed:?}"
        );
    }

    #[test]
    fn a_missing_or_failing_probe_refuses_and_names_the_program() {
        let missing = command_stdout("synapse-no-such-identity-probe", &[])
            .expect_err("an unspawnable probe must refuse");
        assert_eq!(missing.program(), "synapse-no-such-identity-probe");
        assert!(missing.to_string().contains("could not be spawned"));

        #[cfg(not(target_os = "windows"))]
        let (program, args): (&str, &[&str]) = ("false", &[]);
        #[cfg(target_os = "windows")]
        let (program, args): (&str, &[&str]) = ("cmd", &["/D", "/C", "exit /B 1"]);
        let failed = command_stdout(program, args).expect_err("a failing probe must refuse");
        assert_eq!(failed.program(), program);
        assert!(failed.to_string().contains("exited with"));

        // An empty answer is the subtle one: it is well-formed and useless, and
        // the old code turned it into a placeholder.
        #[cfg(not(target_os = "windows"))]
        let (program, args): (&str, &[&str]) = ("true", &[]);
        #[cfg(target_os = "windows")]
        let (program, args): (&str, &[&str]) = ("cmd", &["/D", "/C", "exit /B 0"]);
        let empty = command_stdout(program, args).expect_err("an empty answer must refuse");
        assert!(empty.to_string().contains("returned an empty value"));
    }

    /// The reason all of the above are refusals rather than defaults: any
    /// substituted value changes the identity hash that gates serving.
    #[test]
    fn a_substituted_identity_input_would_rotate_the_profile_hash() {
        struct Fixed(&'static str);
        impl MachineProfileCollector for Fixed {
            fn collect_base_profile(&self) -> Result<MachineProfileBase, ProfileProbeError> {
                Ok(MachineProfileBase {
                    os_build: self.0.to_string(),
                    arch: "aarch64".to_string(),
                    chip_model: "Apple M5 Max".to_string(),
                    ram_class: "le_128_gib".to_string(),
                    ane_subtype: None,
                })
            }
        }

        let real = MachineProfile::collect(&Fixed("26A428"), std::iter::empty::<EngineIdentity>())
            .expect("fixed collector never fails");
        let placeholder = MachineProfile::collect(
            &Fixed("macos-unknown"),
            std::iter::empty::<EngineIdentity>(),
        )
        .expect("fixed collector never fails");

        assert_ne!(
            real.hash(),
            placeholder.hash(),
            "a placeholder os_build must not hash the same as the real one"
        );
    }

    /// The test that catches a restored placeholder, per FIELD rather than in
    /// aggregate. An all-refusing prober is not enough: the assembly evaluates
    /// one field first, so it would refuse on that one no matter what the others
    /// do, and a default restored on any later field would go unnoticed. Each
    /// prober below refuses exactly one program and answers the rest.
    ///
    /// `chip_model` needs BOTH of its probes starved, not one: it tries
    /// `machdep.cpu.brand_string` and falls back to `hw.model`, which is
    /// deliberate (the brand string is absent on some hosts), so starving only
    /// the first exercises the fallback rather than the refusal.
    #[cfg(target_os = "macos")]
    #[test]
    fn every_probed_identity_field_refuses_rather_than_substituting() {
        fn answer_for(program: &str, args: &[&str]) -> String {
            match (program, args.last().copied()) {
                ("sysctl", Some("hw.memsize")) => "137438953472".to_string(),
                _ => "probe-answer".to_string(),
            }
        }

        // Each case names the program to starve and the field it feeds. Every
        // one of these fields is an input to the identity hash.
        // Each case starves the probes of exactly ONE field. Starving a whole
        // program would be easier and would not attribute: `sysctl` serves both
        // chip_model and ram_class, so starving it entirely proves only that
        // whichever is evaluated first refuses.
        #[cfg(target_os = "macos")]
        let cases: &[(&str, &[&str], &str)] = &[
            ("sw_vers", &[], "os_build"),
            ("sysctl", &["hw.memsize"], "ram_class"),
            (
                "sysctl",
                &["machdep.cpu.brand_string", "hw.model"],
                "chip_model",
            ),
        ];

        for (starved_program, starved_args, field) in cases {
            // A thread-local rather than a capture, because Prober is a plain
            // function pointer and must stay one for the production path.
            STARVED.with(|cell| {
                *cell.borrow_mut() = Some((
                    starved_program.to_string(),
                    starved_args.iter().map(|a| a.to_string()).collect(),
                ))
            });

            fn selective(program: &str, args: &[&str]) -> Result<String, ProfileProbeError> {
                let starve = STARVED.with(|cell| cell.borrow().clone());
                if let Some((target_program, target_args)) = starve {
                    let program_matches = program == target_program;
                    // An empty arg list starves every call to that program.
                    let arg_matches = target_args.is_empty()
                        || args
                            .last()
                            .copied()
                            .is_some_and(|last| target_args.iter().any(|a| a == last));
                    if program_matches && arg_matches {
                        return Err(ProfileProbeError::new(program, "refused for this test"));
                    }
                }
                Ok(answer_for(program, args))
            }

            let result = collect_base_profile_with(selective);
            assert!(
                result.is_err(),
                "{field} substituted a default when its probe refused; a placeholder \
                 there silently rotates the identity hash"
            );
        }

        STARVED.with(|cell| *cell.borrow_mut() = None);
    }

    /// The companion that makes the `chip_model` case above attributable: with
    /// only the brand string starved, the documented `hw.model` fallback must
    /// carry the profile rather than refusing. If this ever fails, the case
    /// above is passing for the wrong reason.
    #[cfg(target_os = "macos")]
    #[test]
    fn chip_model_falls_back_to_hw_model_before_it_refuses() {
        STARVED.with(|cell| {
            *cell.borrow_mut() = Some((
                "sysctl".to_string(),
                vec!["machdep.cpu.brand_string".to_string()],
            ))
        });

        fn selective(program: &str, args: &[&str]) -> Result<String, ProfileProbeError> {
            let starve = STARVED.with(|cell| cell.borrow().clone());
            if let Some((target_program, target_args)) = starve {
                if program == target_program
                    && args
                        .last()
                        .copied()
                        .is_some_and(|last| target_args.iter().any(|a| a == last))
                {
                    return Err(ProfileProbeError::new(program, "refused for this test"));
                }
            }
            match (program, args.last().copied()) {
                ("sysctl", Some("hw.memsize")) => Ok("137438953472".to_string()),
                ("sysctl", Some("hw.model")) => Ok("Mac16,6".to_string()),
                _ => Ok("probe-answer".to_string()),
            }
        }

        let base = collect_base_profile_with(selective)
            .expect("a starved brand string must fall back to hw.model, not refuse");
        assert_eq!(base.chip_model, "Mac16,6");

        STARVED.with(|cell| *cell.borrow_mut() = None);
    }

    /// And the converse: a prober that answers yields a profile, so the refusal
    /// above is the probe failing rather than the seam being broken.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_system_collector_accepts_answers_from_its_prober() {
        fn answers(program: &str, args: &[&str]) -> Result<String, ProfileProbeError> {
            match (program, args.last().copied()) {
                ("sysctl", Some("hw.memsize")) => Ok("137438953472".to_string()),
                _ => Ok("probe-answer".to_string()),
            }
        }

        let base =
            collect_base_profile_with(answers).expect("an answering prober yields a profile");
        assert_eq!(base.arch, std::env::consts::ARCH);
        assert!(!base.os_build.is_empty());
        assert!(!base.ram_class.is_empty());
    }

    #[test]
    fn a_probe_refusal_propagates_instead_of_yielding_a_profile() {
        struct Refusing;
        impl MachineProfileCollector for Refusing {
            fn collect_base_profile(&self) -> Result<MachineProfileBase, ProfileProbeError> {
                Err(ProfileProbeError::new("sw_vers", "did not answer"))
            }
        }

        let error = MachineProfile::collect(&Refusing, std::iter::empty::<EngineIdentity>())
            .expect_err("a refused probe must not produce a profile");
        assert_eq!(error.program(), "sw_vers");
    }

    #[test]
    fn ane_subtype_mapping_marks_chip_identity_provenance() {
        assert_eq!(
            mapped_ane_subtype("Apple M5 Max"),
            Some("h17(map)".to_string())
        );
        assert_eq!(mapped_ane_subtype("Apple M5"), Some("h16(map)".to_string()));
        assert_eq!(
            mapped_ane_subtype("Apple M4 Max"),
            Some("h16(map)".to_string())
        );
        assert_eq!(mapped_ane_subtype("Apple M3 Max"), None);
    }

    #[test]
    fn ane_subtype_changes_profile_hash_and_none_keeps_legacy_shape() {
        let without_ane = MachineProfile::collect(
            &FakeCollector { ane_subtype: None },
            std::iter::empty::<EngineIdentity>(),
        )
        .expect("fake collector never fails");
        let with_ane = MachineProfile::collect(
            &FakeCollector {
                ane_subtype: Some("h17(map)"),
            },
            std::iter::empty::<EngineIdentity>(),
        )
        .expect("fake collector never fails");

        assert_ne!(without_ane.hash(), with_ane.hash());
        assert_eq!(
            without_ane.hash(),
            "883d3caf3aa4da4277fe8744fefa4829ee9d1c00bde0722c41d6b6ce959427c0"
        );
        assert_eq!(without_ane.ane_subtype, None);
        assert_eq!(with_ane.ane_subtype.as_deref(), Some("h17(map)"));
        assert!(!String::from_utf8(without_ane.stable_bytes())
            .unwrap()
            .contains("ane_subtype"));
    }

    #[test]
    fn ram_class_buckets_to_stable_labels() {
        assert_eq!(ram_class_from_bytes(7 * 1024 * 1024 * 1024), "le_8_gib");
        assert_eq!(ram_class_from_bytes(300 * 1024 * 1024 * 1024), "gt_256_gib");
    }

    fn uname_answer() -> Result<UnameReading, String> {
        Ok(UnameReading {
            sysname: "Linux".to_string(),
            release: "6.8.0-test".to_string(),
            machine: "x86_64".to_string(),
        })
    }

    fn sysinfo_answer() -> Result<SysinfoReading, String> {
        Ok(SysinfoReading {
            totalram: 4,
            mem_unit: 1 << 30,
        })
    }

    #[test]
    fn native_formatting_and_mapping_are_host_independent() {
        assert_eq!(
            linux_identity(uname_answer().unwrap()).unwrap(),
            ("Linux 6.8.0-test".to_string(), "x86_64".to_string())
        );
        assert_eq!(
            linux_ram_class(sysinfo_answer().unwrap()).unwrap(),
            "le_4_gib"
        );
        assert_eq!(windows_os_build((10, 0, 22631)), "Windows 10.0.22631");
        assert_eq!(windows_chip_model(9).unwrap(), "x86_64");
        assert_eq!(windows_chip_model(12).unwrap(), "aarch64");
        for architecture in [0, 0xFFFF] {
            let error =
                windows_chip_model(architecture).expect_err("unsupported native architecture");
            assert!(error.to_string().contains("GetNativeSystemInfo"));
            assert!(error.to_string().contains("chip_model"));
        }
    }

    #[test]
    fn invalid_native_readings_refuse_with_field_attribution() {
        let mut uname = uname_answer().unwrap();
        uname.machine.clear();
        let error = linux_identity(uname).unwrap_err().to_string();
        assert!(
            error.contains("uname(2)")
                && error.contains("os_build")
                && error.contains("chip_model")
        );
        for reading in [
            SysinfoReading {
                totalram: 0,
                mem_unit: 1,
            },
            SysinfoReading {
                totalram: 4,
                mem_unit: 0,
            },
            SysinfoReading {
                totalram: u64::MAX,
                mem_unit: 2,
            },
        ] {
            let error = linux_ram_class(reading).unwrap_err().to_string();
            assert!(error.contains("sysinfo(2)") && error.contains("ram_class"));
        }
        let error = native_ram_class("GlobalMemoryStatusEx", 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("GlobalMemoryStatusEx") && error.contains("ram_class"));
    }

    /// Starve one source at a time and answer all others. Both platforms' seams
    /// run on every non-macOS host, without touching any native provider.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn every_probed_identity_field_refuses_rather_than_substituting() {
        for source in ["uname(2)", "sysinfo(2)"] {
            let error = collect_linux_profile_with(
                || {
                    if source == "uname(2)" {
                        Err("starved".to_string())
                    } else {
                        uname_answer()
                    }
                },
                || {
                    if source == "sysinfo(2)" {
                        Err("starved".to_string())
                    } else {
                        sysinfo_answer()
                    }
                },
            )
            .expect_err("a starved native source must refuse");
            assert_eq!(error.program(), source);
            assert!(
                error.to_string().contains(source),
                "refusal must name {source}: {error}"
            );
            let fields: &[&str] = if source == "uname(2)" {
                &["os_build", "chip_model"]
            } else {
                &["ram_class"]
            };
            for field in fields {
                assert!(
                    error.to_string().contains(field),
                    "{source} must name {field}: {error}"
                );
            }
        }
        for (source, field) in [
            ("RtlGetVersion", "os_build"),
            ("GetNativeSystemInfo", "chip_model"),
            ("GlobalMemoryStatusEx", "ram_class"),
        ] {
            let error = collect_windows_profile_with(
                || {
                    if source == "RtlGetVersion" {
                        Err("starved".to_string())
                    } else {
                        Ok((10, 0, 22631))
                    }
                },
                || {
                    if source == "GetNativeSystemInfo" {
                        Err("starved".to_string())
                    } else {
                        Ok(9)
                    }
                },
                || {
                    if source == "GlobalMemoryStatusEx" {
                        Err("starved".to_string())
                    } else {
                        Ok(4 << 30)
                    }
                },
            )
            .expect_err("a starved native source must refuse");
            assert_eq!(error.program(), source);
            assert!(
                error.to_string().contains(source),
                "refusal must name {source}: {error}"
            );
            assert!(
                error.to_string().contains(field),
                "{source} must name {field}: {error}"
            );
        }
    }

    #[test]
    fn native_collectors_accept_answers_from_each_source() {
        let linux = collect_linux_profile_with(uname_answer, sysinfo_answer).unwrap();
        assert_eq!(linux.os_build, "Linux 6.8.0-test");
        assert_eq!(linux.chip_model, "x86_64");
        assert_eq!(linux.ram_class, "le_4_gib");
        let windows =
            collect_windows_profile_with(|| Ok((10, 0, 22631)), || Ok(9), || Ok(4 << 30)).unwrap();
        assert_eq!(windows.os_build, "Windows 10.0.22631");
        assert_eq!(windows.chip_model, "x86_64");
        assert_eq!(windows.ram_class, "le_4_gib");
        for base in [linux, windows] {
            assert_eq!(base.arch, std::env::consts::ARCH);
            assert_eq!(base.ane_subtype, None);
        }
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn native_collection_succeeds_and_matches_native_identity() {
        let base = SystemMachineProfileCollector
            .collect_base_profile()
            .expect("native collection");
        assert_eq!(base.arch, std::env::consts::ARCH);
        assert_eq!(base.ane_subtype, None);
        assert!([
            "le_4_gib",
            "le_8_gib",
            "le_16_gib",
            "le_32_gib",
            "le_64_gib",
            "le_128_gib",
            "le_256_gib",
            "gt_256_gib"
        ]
        .contains(&base.ram_class.as_str()));
        #[cfg(target_os = "linux")]
        {
            let reading = rustix::system::uname();
            assert_eq!(
                base.os_build,
                format!(
                    "{} {}",
                    reading.sysname().to_str().unwrap(),
                    reading.release().to_str().unwrap()
                )
            );
            assert_eq!(base.chip_model, reading.machine().to_str().unwrap());
        }
        #[cfg(target_os = "windows")]
        {
            let (major, minor, build) = synapse_native_probe::windows::version().unwrap();
            assert_eq!(base.os_build, format!("Windows {major}.{minor}.{build}"));
            let architecture = synapse_native_probe::windows::processor_architecture().unwrap();
            assert_eq!(
                base.chip_model,
                match architecture {
                    9 => "x86_64",
                    12 => "aarch64",
                    _ => panic!("unsupported native architecture"),
                }
            );
        }
    }
}
