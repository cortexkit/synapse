//! Candidate extraction paths and the single worker alias used for a run.
use crate::refuse;
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
use synapse_certify::{ArtifactFile, Result};

pub(crate) fn os_arch(row: &str) -> Result<String> {
    let bindings: Value =
        serde_json::from_slice(include_bytes!("../../../bench/parity/release-assets.json"))
            .map_err(|e| refuse(e.to_string()))?;
    bindings["assets"]
        .as_array()
        .and_then(|assets| {
            assets.iter().find(|asset| {
                asset["binary"] == binary_file(row, "ck-synapse")
                    && asset["rows"]
                        .as_array()
                        .is_some_and(|rows| rows.iter().any(|r| r == row))
            })
        })
        .and_then(|asset| asset["os_arch"].as_str())
        .map(str::to_owned)
        .ok_or_else(|| refuse("unknown certification asset row"))
}

pub(crate) fn binary_file(row: &str, role: &str) -> String {
    format!(
        "{role}{}",
        if row.contains("windows") { ".exe" } else { "" }
    )
}

pub(crate) struct Layout {
    pub root: PathBuf,
    pub candidate: PathBuf,
    pub worker: Option<PathBuf>,
    pub runtime: PathBuf,
    pub artifacts: Vec<ArtifactFile>,
    engine: &'static str,
}

impl Layout {
    pub fn new(assets: &Path, scratch: &Path, row: &str) -> Result<Self> {
        let os_arch = os_arch(row)?;
        let extracted = assets.join(&os_arch);
        std::fs::create_dir_all(scratch).map_err(|e| refuse(e.to_string()))?;
        let link = |role| {
            synapse_core::dev_binary::ckdev_binary_hard_link(
                extracted.join(binary_file(row, role)),
                scratch.join(role),
            )
            .map_err(|e| refuse(e.to_string()))
        };
        let candidate = link("ck-synapse")?;
        let mut artifacts = vec![ArtifactFile {
            role: "ck-synapse".into(),
            file: format!("{os_arch}/{}", binary_file(row, "ck-synapse")),
        }];
        let worker = crate::live::worker(row).map(link).transpose()?;
        if let Some(role) = crate::live::worker(row) {
            artifacts.push(ArtifactFile {
                role: role.into(),
                file: format!("{os_arch}/{}", binary_file(row, role)),
            });
        }
        if row == "cuda-windows-nvidia" {
            let manifest: Value = serde_json::from_slice(
                &std::fs::read(extracted.join("manifest.json"))
                    .map_err(|e| refuse(e.to_string()))?,
            )
            .map_err(|e| refuse(e.to_string()))?;
            let mut names = BTreeSet::new();
            for runtime in manifest["runtime_files"]
                .as_array()
                .ok_or_else(|| refuse("CUDA runtime_files missing"))?
            {
                let file = runtime["file"]
                    .as_str()
                    .filter(|file| {
                        !file.is_empty()
                            && !file.contains(['/', '\\'])
                            && *file != "."
                            && *file != ".."
                    })
                    .ok_or_else(|| refuse("invalid CUDA runtime file name"))?;
                if !names.insert(file) {
                    return Err(refuse("duplicate CUDA runtime file"));
                }
                std::fs::hard_link(
                    extracted.join(file),
                    worker
                        .as_ref()
                        .expect("CUDA worker")
                        .parent()
                        .expect("alias directory")
                        .join(file),
                )
                .map_err(|e| refuse(e.to_string()))?;
                artifacts.push(ArtifactFile {
                    role: format!("cuda-runtime:{file}"),
                    file: format!("{os_arch}/{file}"),
                });
            }
        }
        let runtime = scratch.join("ipc");
        std::fs::create_dir_all(&runtime).map_err(|e| refuse(e.to_string()))?;
        Ok(Self {
            root: scratch.into(),
            candidate,
            worker,
            runtime,
            artifacts,
            engine: crate::live::lane(row),
        })
    }

    pub fn environment(&self) -> Vec<(String, Option<PathBuf>)> {
        [
            "owned-cuda",
            "owned-vulkan",
            "ane-direct-worker",
            "owned-metal",
            "ane",
            "llama",
            "owned-decode",
        ]
        .into_iter()
        .flat_map(|engine| {
            let selected = engine == self.engine && self.worker.is_some();
            [
                (
                    synapse_core::cuda::worker_binary_env_var(engine),
                    selected.then(|| self.worker.clone().unwrap()),
                ),
                // The candidate runs in root; a relative IPC path avoids Unix socket path limits in long checkouts.
                (
                    synapse_core::cuda::worker_runtime_dir_env_var(engine),
                    selected.then(|| {
                        PathBuf::from(self.runtime.file_name().expect("scratch IPC directory"))
                    }),
                ),
            ]
        })
        .collect()
    }
}

pub(crate) fn config(endpoint: &str) -> Value {
    // Allow one token beyond the model's 8192-token limit into the queue. The probe
    // must fail the model's length check, not the queue's total-token budget.
    json!({"hf_endpoint": endpoint, "certify_observation": true,
        "inline": {"deadline_ms": 3600000, "max_queue_ms": 3600000, "max_items": 64, "max_tokens": 8193}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_certify::{Admission, Catalog, Outcome, Parity, RunEvidence, Runner, GATES};

    #[test]
    fn windows_cuda_layout_and_record_round_trip_use_the_extraction_root() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!(".live/layout-test-{}", std::process::id()));
        let assets = root.join("assets");
        let extracted = assets.join("windows-x64");
        std::fs::create_dir_all(&extracted).unwrap();
        for file in [
            "ck-synapse.exe",
            "ck-synapse-worker-cuda.exe",
            "cublas64_12.dll",
            "cudart64_12.dll",
        ] {
            std::fs::write(extracted.join(file), file.as_bytes()).unwrap();
        }
        std::fs::write(
            extracted.join("manifest.json"),
            serde_json::to_vec(
                &json!({"runtime_files":[{"file":"cublas64_12.dll"},{"file":"cudart64_12.dll"}]}),
            )
            .unwrap(),
        )
        .unwrap();
        let layout = Layout::new(&assets, &root.join("scratch"), "cuda-windows-nvidia").unwrap();
        let alias = layout.worker.as_ref().unwrap();
        assert_eq!(alias.file_name().unwrap(), "ckdev-synapse-worker-cuda.exe");
        let names = std::fs::read_dir(alias.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names,
            [
                "ckdev-synapse-worker-cuda.exe",
                "cublas64_12.dll",
                "cudart64_12.dll"
            ]
            .into_iter()
            .map(String::from)
            .collect()
        );
        for file in ["cublas64_12.dll", "cudart64_12.dll"] {
            let source = extracted.join(file);
            let linked = alias.parent().unwrap().join(file);
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(
                    (
                        std::fs::metadata(&source).unwrap().dev(),
                        std::fs::metadata(&source).unwrap().ino()
                    ),
                    (
                        std::fs::metadata(&linked).unwrap().dev(),
                        std::fs::metadata(&linked).unwrap().ino()
                    )
                );
            }
            // Changing the source also changes the alias on Windows, proving these are not executable-directory copies.
            std::fs::write(&source, b"shared file identity").unwrap();
            assert_eq!(std::fs::read(&linked).unwrap(), b"shared file identity");
        }
        let env = layout
            .environment()
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            env["SYNAPSE_OWNED_CUDA_WORKER_BIN"].as_ref().unwrap(),
            alias
        );
        let runtime = layout.root.join(
            env["SYNAPSE_OWNED_CUDA_WORKER_RUNTIME_DIR"]
                .as_ref()
                .unwrap(),
        );
        assert_eq!(runtime, layout.runtime);
        assert!(runtime.is_dir());
        assert!(!runtime.starts_with(&assets));
        assert!(env
            .iter()
            .filter(|(name, _)| !name.contains("OWNED_CUDA"))
            .all(|(_, value)| value.is_none()));
        assert_eq!(
            config("http://localhost:1")
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["certify_observation", "hf_endpoint", "inline"]
        );
        assert_eq!(config("http://localhost:1")["inline"]["max_tokens"], 8193);

        let model = "gte-modernbert-base";
        let manifest = synapse_certify::self_check::manifest().unwrap();
        let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixtures = synapse_parity::evaluator::FixtureSet::load(
            &checkout.join("bench/parity"),
            &manifest,
            model,
        )
        .unwrap();
        let profile = format!("{model}.owned-cuda");
        let parity = Parity {
            model: model.into(),
            operation: "embed".into(),
            profile_id: profile.clone(),
            fingerprint: "f".repeat(64),
            fixture_set_id: synapse_parity::evaluator::fixture_set_id(&manifest, model),
            completed_fixtures: fixtures.cases().len(),
            expected_fixtures: fixtures.cases().len(),
            tolerance_class: "fp16".into(),
            metrics: json!({}),
            gates: GATES.iter().map(|gate| ((*gate).into(), true)).collect(),
            failures: vec![],
        };
        struct Mock(RunEvidence);
        impl Runner for Mock {
            fn probe_floor(&mut self, _: &str, _: &str) -> Result<String> {
                Ok("ok".into())
            }
            fn observe(&mut self, _: &str, _: &str) -> Result<RunEvidence> {
                Ok(self.0.clone())
            }
        }
        let mut mock = Mock(RunEvidence {
            source_commit: "a".repeat(40),
            machine: json!({"system_product_name":"fixture", "gpu":{"uuid_sha256": synapse_certify::gpu_uuid_sha256("GPU-fixture")}}),
            artifacts: layout.artifacts,
            operation: "embed".into(),
            profile_id: profile,
            fingerprint: parity.fingerprint.clone(),
            fixture_set_id: parity.fixture_set_id.clone(),
            expected_fixtures: parity.expected_fixtures,
            tolerance_class: parity.tolerance_class.clone(),
            parity: Some(parity),
            admission: Admission {
                tokens_8192: Outcome {
                    outcome: "processed".into(),
                    truncated: false,
                    diverted: false,
                    worker_requests: 1,
                },
                tokens_8193: Outcome {
                    outcome: "sequence_too_long".into(),
                    truncated: false,
                    diverted: false,
                    worker_requests: 0,
                },
            },
            layer_count: 0,
            inventories: vec![],
            admitted_count: 0,
        });
        let record =
            synapse_certify::produce(&mut mock, &assets, "cuda-windows-nvidia", model).unwrap();
        assert_eq!(
            record
                .executed_artifacts
                .iter()
                .filter(|artifact| artifact.role.starts_with("cuda-runtime:"))
                .map(|artifact| (artifact.role.as_str(), artifact.file.as_str()))
                .collect::<BTreeSet<_>>(),
            [
                (
                    "cuda-runtime:cublas64_12.dll",
                    "windows-x64/cublas64_12.dll"
                ),
                (
                    "cuda-runtime:cudart64_12.dll",
                    "windows-x64/cudart64_12.dll"
                )
            ]
            .into_iter()
            .collect()
        );
        assert!(record
            .executed_artifacts
            .iter()
            .all(|artifact| artifact.file.starts_with("windows-x64/")));
        let round_trip = serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
        let catalog = Catalog {
            lanes: vec![(
                model.into(),
                "cuda".into(),
                "owned-cuda".into(),
                "f".repeat(64),
            )],
            tasks: [(model.into(), "embed".into())].into_iter().collect(),
        };
        synapse_certify::validate_record(&round_trip, &checkout, &assets, &catalog).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
