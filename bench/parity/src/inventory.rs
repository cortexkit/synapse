//! Committed certification inputs beside the manifest: the per-row module
//! config fixtures (`rows/`), the machine registry (`machines.json`), the
//! release asset inventory (`release-assets.json`) and the preload regression
//! inputs (`preload/`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{perr, Result};

/// The eight certification rows.
pub const ROW_IDS: [&str; 8] = [
    "metal-m5",
    "ane-m5",
    "cuda-linux-nvidia",
    "cuda-windows-nvidia",
    "vulkan-linux-amd",
    "vulkan-linux-nvidia",
    "vulkan-windows-amd",
    "vulkan-windows-nvidia",
];

pub const OS_ARCHES: [&str; 3] = ["darwin-arm64", "linux-x64", "windows-x64"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineRegistry {
    pub schema: String,
    pub machines: BTreeMap<String, Machine>,
    pub rows: BTreeMap<String, Row>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Machine {
    pub description: String,
    pub os_arch: Vec<String>,
    pub rented: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_identifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub machine: String,
    pub os_arch: String,
    pub lane: String,
    /// Path, relative to `bench/parity/`, of the row's module config fixture.
    pub module_config: String,
    /// Whether `certify run` must see an `ok` floor probe on this row.
    pub floor_probe: bool,
}

pub fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|error| perr!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| perr!("parse {}: {error}", path.display()))
}

/// Binaries each lane identity executes, per the lane table.
pub fn lane_binaries(lane: &str) -> Result<&'static [&'static str]> {
    Ok(match lane {
        "owned-metal" => &["ck-synapse"],
        "owned-cuda" => &["ck-synapse", "ck-synapse-worker-cuda"],
        "owned-vulkan" => &["ck-synapse", "ck-synapse-worker-vulkan"],
        "ane-direct-worker" => &["ck-synapse", "ck-synapse-worker-ane-direct"],
        other => return Err(perr!("`{other}` is not a certification lane")),
    })
}

/// Every binary some certification lane executes: the union of
/// `lane_binaries` over the four lanes.
pub const CERTIFICATION_BINARIES: [&str; 4] = [
    "ck-synapse",
    "ck-synapse-worker-cuda",
    "ck-synapse-worker-vulkan",
    "ck-synapse-worker-ane-direct",
];

pub fn check_machine_registry(registry: &MachineRegistry, parity_dir: &Path) -> Result<()> {
    if registry.schema != "synapse-parity-machines-v1" {
        return Err(perr!(
            "machines.json schema `{}` is not synapse-parity-machines-v1",
            registry.schema
        ));
    }
    let rows: BTreeSet<&str> = registry.rows.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = ROW_IDS.into_iter().collect();
    if rows != expected {
        return Err(perr!(
            "machines.json rows must be exactly {expected:?}, found {rows:?}"
        ));
    }
    for (id, row) in &registry.rows {
        let machine = registry
            .machines
            .get(&row.machine)
            .ok_or_else(|| perr!("row `{id}` names unknown machine `{}`", row.machine))?;
        if !machine.os_arch.contains(&row.os_arch) {
            return Err(perr!(
                "row `{id}` runs {} on machine `{}`, which lists {:?}",
                row.os_arch,
                row.machine,
                machine.os_arch
            ));
        }
        lane_binaries(&row.lane).map_err(|error| perr!("row `{id}`: {error}"))?;
        let backend = id.split('-').next().unwrap_or_default();
        let lane_backend = match row.lane.as_str() {
            "owned-metal" => "metal",
            "ane-direct-worker" => "ane",
            "owned-cuda" => "cuda",
            "owned-vulkan" => "vulkan",
            _ => "",
        };
        if backend != lane_backend {
            return Err(perr!("row `{id}` is on lane `{}`", row.lane));
        }
        // Every row except the in-process Metal one has a floor probe.
        if row.floor_probe != (row.lane != "owned-metal") {
            return Err(perr!(
                "row `{id}` floor_probe must be {}",
                row.lane != "owned-metal"
            ));
        }
        if row.module_config != format!("rows/{id}.json") {
            return Err(perr!("row `{id}` module config must be rows/{id}.json"));
        }
        if !parity_dir.join(&row.module_config).is_file() {
            return Err(perr!(
                "row `{id}` module config {} is missing",
                row.module_config
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module defaults, read from the module's own source.

/// The `inline` and `jobs` defaults of `ck-synapse`'s module config, read from
/// `crates/synapse-module/src/lib.rs`: each field's `#[serde(default = "f")]`
/// function, the constant `f` returns, and that constant's value. Reading the
/// source keeps the row fixtures tied to the defaults actually compiled into
/// the module, not to a copy that could drift.
pub fn module_defaults(lib_rs: &str) -> Result<BTreeMap<String, BTreeMap<String, u64>>> {
    let mut out = BTreeMap::new();
    for (section, structure) in [("inline", "InlineConfig"), ("jobs", "JobConfig")] {
        out.insert(section.to_string(), struct_defaults(lib_rs, structure)?);
    }
    Ok(out)
}

fn struct_defaults(source: &str, name: &str) -> Result<BTreeMap<String, u64>> {
    let start = source
        .find(&format!("struct {name} {{"))
        .ok_or_else(|| perr!("module source has no `struct {name}`"))?;
    let body_end = source[start..]
        .find("\n}")
        .ok_or_else(|| perr!("`struct {name}` is not closed"))?;
    let body = &source[start..start + body_end];
    let mut fields = BTreeMap::new();
    let mut pending: Option<String> = None;
    for line in body.lines().skip(1) {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("#[serde(default = \"") {
            let function = rest.split('"').next().unwrap_or_default();
            pending = Some(function.to_string());
        } else if line.starts_with("#[") || line.starts_with("//") || line.is_empty() {
            continue;
        } else if let Some((field, _)) = line.split_once(':') {
            let function = pending
                .take()
                .ok_or_else(|| perr!("`{name}.{field}` has no serde default function"))?;
            fields.insert(
                field.trim().to_string(),
                function_constant(source, &function)?,
            );
        }
    }
    Ok(fields)
}

fn function_constant(source: &str, function: &str) -> Result<u64> {
    let start = source
        .find(&format!("fn {function}() ->"))
        .ok_or_else(|| perr!("module source has no `fn {function}`"))?;
    let open = source[start..]
        .find('{')
        .ok_or_else(|| perr!("`fn {function}` has no body"))?;
    let close = source[start + open..]
        .find('}')
        .ok_or_else(|| perr!("`fn {function}` is not closed"))?;
    let constant = source[start + open + 1..start + open + close].trim();
    if !constant
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(perr!(
            "`fn {function}` does not return a plain constant: `{constant}`"
        ));
    }
    constant_value(source, constant)
}

fn constant_value(source: &str, constant: &str) -> Result<u64> {
    let marker = format!("const {constant}:");
    let start = source
        .find(&marker)
        .ok_or_else(|| perr!("module source has no `{marker}`"))?;
    let line = source[start..].lines().next().unwrap_or_default();
    let expression = line
        .split_once('=')
        .and_then(|(_, rest)| rest.split_once(';'))
        .map(|(expression, _)| expression.trim())
        .ok_or_else(|| perr!("`{constant}` is not a one-line constant"))?;
    expression.split('*').try_fold(1u64, |product, factor| {
        let factor = factor.trim().replace('_', "");
        let value: u64 = factor.parse().map_err(|_| {
            perr!("`{constant}` = `{expression}` is not a product of integer literals")
        })?;
        Ok(product * value)
    })
}

/// A row fixture must set `inline` and `jobs` to exactly the module defaults:
/// every field present, every value equal, nothing extra.
pub fn check_row_fixture(
    row_id: &str,
    fixture: &Value,
    defaults: &BTreeMap<String, BTreeMap<String, u64>>,
) -> Result<()> {
    let object = fixture
        .as_object()
        .ok_or_else(|| perr!("row `{row_id}` fixture is not a JSON object"))?;
    for (section, expected) in defaults {
        let values = object
            .get(section)
            .and_then(Value::as_object)
            .ok_or_else(|| perr!("row `{row_id}` fixture has no `{section}` object"))?;
        let actual: BTreeMap<String, Option<u64>> = values
            .iter()
            .map(|(key, value)| (key.clone(), value.as_u64()))
            .collect();
        let wanted: BTreeMap<String, Option<u64>> = expected
            .iter()
            .map(|(key, value)| (key.clone(), Some(*value)))
            .collect();
        if actual != wanted {
            return Err(perr!(
                "row `{row_id}` `{section}` values {actual:?} differ from the module defaults {wanted:?}"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Release asset inventory.

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseAssets {
    pub schema: String,
    /// Each zip's `.sha256` sidecar travels with the zip and is covered by its
    /// entry.
    pub sidecar_suffix: String,
    pub assets: Vec<AssetEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetEntry {
    pub asset: String,
    pub os_arch: String,
    pub binding: Binding,
    /// File name of the executable inside the zip (absent for non-zip assets).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary: Option<String>,
    /// Rows whose `passed` records must carry this binary's extracted SHA-256.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<String>,
    /// For the Windows CUDA zip: the DLLs bound to its rows are the set the
    /// zip's `manifest.json` lists in `runtime_files` at tag time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_files_from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Binding {
    Certification,
    Exempt,
}

/// Binaries each platform's release-candidate job builds.
pub fn candidate_binaries(os_arch: &str) -> &'static [&'static str] {
    match os_arch {
        "darwin-arm64" => &[
            "ck-synapse",
            "ck-synapse-opctl",
            "ck-synapse-worker-llama",
            "ck-synapse-worker-ane",
            "ck-synapse-worker-ane-swift",
            "ck-synapse-worker-decode",
            "ck-synapse-worker-ane-direct",
        ],
        "linux-x64" | "windows-x64" => &[
            "ck-synapse",
            "ck-synapse-opctl",
            "ck-synapse-worker-llama",
            "ck-synapse-worker-cuda",
            "ck-synapse-worker-vulkan",
        ],
        _ => &[],
    }
}

/// Non-zip candidate assets.
pub const EXTRA_ASSETS: [&str; 1] = ["release-manifest.json"];

/// Check the inventory against the candidate asset set and the rows.
pub fn check_release_assets(inventory: &ReleaseAssets, registry: &MachineRegistry) -> Result<()> {
    if inventory.schema != "synapse-parity-release-assets-v1"
        || inventory.sidecar_suffix != ".sha256"
    {
        return Err(perr!("release-assets.json header is not synapse-parity-release-assets-v1 with .sha256 sidecars"));
    }
    let mut candidates: BTreeSet<String> = EXTRA_ASSETS.iter().map(|s| s.to_string()).collect();
    for os_arch in OS_ARCHES {
        for binary in candidate_binaries(os_arch) {
            candidates.insert(format!("{binary}-{os_arch}.zip"));
        }
    }
    let mut seen = BTreeSet::new();
    for entry in &inventory.assets {
        if !seen.insert(entry.asset.clone()) {
            return Err(perr!("asset `{}` has more than one entry", entry.asset));
        }
        if !candidates.contains(&entry.asset) {
            return Err(perr!("entry `{}` names no candidate asset", entry.asset));
        }
        check_asset_entry(entry, registry)
            .map_err(|error| perr!("asset `{}`: {error}", entry.asset))?;
    }
    if let Some(missing) = candidates.difference(&seen).next() {
        return Err(perr!("candidate asset `{missing}` has no entry"));
    }
    Ok(())
}

fn check_asset_entry(entry: &AssetEntry, registry: &MachineRegistry) -> Result<()> {
    let stem = entry.asset.strip_suffix(&format!("-{}.zip", entry.os_arch));
    let expected_binary = stem.map(|stem| {
        if entry.os_arch == "windows-x64" {
            format!("{stem}.exe")
        } else {
            stem.to_string()
        }
    });
    if stem.is_some() && entry.binary != expected_binary {
        return Err(perr!("binary must be {expected_binary:?}"));
    }
    if stem.is_none() && entry.binary.is_some() {
        return Err(perr!("a non-zip asset names no binary"));
    }
    match entry.binding {
        Binding::Exempt => {
            if entry.reason.as_deref().is_none_or(str::is_empty) {
                return Err(perr!("an exempt entry gives a reason"));
            }
            if !entry.rows.is_empty() || entry.runtime_files_from.is_some() {
                return Err(perr!("an exempt entry binds no rows"));
            }
            if let Some(stem) = stem {
                if CERTIFICATION_BINARIES.contains(&stem) {
                    return Err(perr!("`{stem}` is an executed artifact of a certification lane and cannot be exempt"));
                }
            }
        }
        Binding::Certification => {
            let stem =
                stem.ok_or_else(|| perr!("only a zipped binary can be certification-bound"))?;
            if !CERTIFICATION_BINARIES.contains(&stem) {
                return Err(perr!("`{stem}` is not executed by any certification lane"));
            }
            if entry.reason.is_some() {
                return Err(perr!(
                    "a certification-bound entry gives no exemption reason"
                ));
            }
            // Bound rows: exactly the rows on this platform whose lane runs
            // this binary.
            let expected: BTreeSet<&str> = registry
                .rows
                .iter()
                .filter(|(_, row)| row.os_arch == entry.os_arch)
                .filter(|(_, row)| lane_binaries(&row.lane).is_ok_and(|bins| bins.contains(&stem)))
                .map(|(id, _)| id.as_str())
                .collect();
            let rows: BTreeSet<&str> = entry.rows.iter().map(String::as_str).collect();
            if rows != expected || rows.len() != entry.rows.len() {
                return Err(perr!(
                    "rows must be exactly {expected:?}, found {:?}",
                    entry.rows
                ));
            }
            let windows_cuda = stem == "ck-synapse-worker-cuda" && entry.os_arch == "windows-x64";
            if windows_cuda != (entry.runtime_files_from.as_deref() == Some("manifest.json")) {
                return Err(perr!(
                    "only the Windows CUDA zip binds its runtime DLLs through manifest.json"
                ));
            }
        }
    }
    Ok(())
}
