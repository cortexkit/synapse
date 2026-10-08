//! Private production-candidate setup shared by the Qwen comparison examples.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{path::Path, sync::Arc, time::Duration};
use subc_client_rs::{CallOptions, ConsumerOptions, SubcConsumer};
use subc_daemon::{
    daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth,
};
use subc_protocol::{BindIdentity, RouteTarget};
use subc_transport::{
    generate_daemon_id, generate_key, write_atomic, ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use synapse_core::dev_binary::ckdev_binary_hard_link;
use synapse_parity::{canonical::sha256_file, manifest::Manifest};
use tokio::net::TcpListener;

pub const MODEL: &str = "qwen3-embedding-0.6b";

/// The embedding model under test. Defaults to Qwen3; set
/// SYNAPSE_COMPARE_MODEL=gte-modernbert-base to run the same harness on gte.
pub fn model() -> String {
    std::env::var("SYNAPSE_COMPARE_MODEL").unwrap_or_else(|_| MODEL.into())
}

/// gte-modernbert is CLS-pooled; Qwen3-Embedding takes the last token.
fn pooling(model: &str) -> &'static str {
    if model.starts_with("gte-") {
        "cls"
    } else {
        "last"
    }
}

pub fn verify_checkpoint(weights: &Path) -> Result<Manifest> {
    let manifest = Manifest::from_slice(include_bytes!("../../../../bench/parity/models.json"))?;
    for (file, digest) in &manifest.model(&model())?.files {
        ensure!(
            sha256_file(&weights.join(file))? == *digest,
            "original checkpoint digest mismatch: {file}"
        );
    }
    Ok(manifest)
}

pub async fn raw_call(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    method: &str,
    params: Value,
) -> Result<Value> {
    let bytes = consumer
        .call(
            RouteTarget::ManagementSurface {
                module_id: "synapse".into(),
            },
            identity.clone(),
            serde_json::to_vec(&json!({"method":method,"params":params}))?,
            CallOptions {
                timeout: Duration::from_secs(600),
                ..CallOptions::default()
            },
        )
        .await?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub async fn call(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    method: &str,
    params: Value,
) -> Result<Value> {
    let response = raw_call(consumer, identity, method, params).await?;
    ensure!(
        !response["error"].is_object() && !response["result"]["error"].is_object(),
        "{method} refused: {response}"
    );
    Ok(response["result"].clone())
}

pub struct Candidate {
    pub consumer: SubcConsumer,
    pub identity: BindIdentity,
    child: tokio::process::Child,
    daemon: tokio::task::JoinHandle<()>,
}

impl Candidate {
    pub async fn start(
        checkout: &Path,
        assets: &Path,
        weights: &Path,
        label: &str,
        model_ids: [&str; 2],
        inline_tokens: usize,
    ) -> Result<Self> {
        let manifest = verify_checkpoint(weights)?;
        let model = model();
        let pooling = pooling(&model);
        let root = checkout
            .join("target")
            .join(format!("{label}-{}", std::process::id()));
        std::fs::create_dir_all(root.join("data"))?;
        let package = root.join("qwen-ane.safetensors");
        std::fs::write(
            &package,
            synapse_parity::convert::convert_profile_file(
                &manifest,
                &format!("{model}.ane-direct-worker"),
                &weights.join("model.safetensors"),
            )?,
        )?;
        let module = ckdev_binary_hard_link(assets.join("ck-synapse"), &root)?;
        let worker = ckdev_binary_hard_link(assets.join("ck-synapse-worker-ane-direct"), &root)?;
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let conn = ConnectionInfo {
            schema: SCHEMA_VERSION,
            endpoints: vec![Endpoint {
                host: "127.0.0.1".into(),
                port: listener.local_addr()?.port(),
            }],
            key: generate_key()?,
            daemon_id: generate_daemon_id()?,
            pid: std::process::id(),
            daemon_ver: label.into(),
            wire_version: Some(subc_protocol::PROTOCOL_VERSION),
        };
        let conn_path = root.join("connection.json");
        write_atomic(&conn_path, &conn)?;
        let router = Arc::new(Router::with_control_handler(Arc::new(
            ControlHandler::new(Arc::new(Registry::default())).with_storage_config(Some(
                StorageConfig::Sqlite {
                    data_home: root.join("data"),
                },
            )),
        )));
        let config = root.join("config.json");
        std::fs::write(
            &config,
            serde_json::to_vec(&json!({"preload_models":[
            {"model_id":model_ids[0],"engine":"ane-direct-worker","profile":format!("{model}.ane-direct-worker"),"task":"embed","model_path":package,"tokenizer_path":weights.join("tokenizer.json"),"pooling":pooling,"normalize":true,"worker_bin":worker,"execution":"explicit","attention_units":8192*8192},
            {"model_id":model_ids[1],"engine":"owned-metal","profile":format!("{model}.owned-metal"),"task":"embed","model_path":weights.join("model.safetensors"),"tokenizer_path":weights.join("tokenizer.json"),"pooling":pooling,"normalize":true,"execution":"explicit","attention_units":8192*8192}],
            "inline":{"max_items":64,"max_tokens":inline_tokens,"deadline_ms":600000,"max_queue_ms":600000,"max_concurrent_workers":2}}))?,
        )?;
        let child = synapse_core::without_launch_nonce_tokio(tokio::process::Command::new(module))
            .arg("--subc")
            .arg(&conn_path)
            .env("SUBC_MODULE_ID", "synapse")
            .env("SYNAPSE_CONFIG_PATH", config)
            .env("XDG_DATA_HOME", root.join("data"))
            .env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
            .env("CORTEXKIT_STORE_ROOT", root.join("store"))
            .kill_on_drop(true)
            .spawn()?;
        let daemon = tokio::spawn(async move {
            let _ = serve_listener(
                listener,
                router,
                ServerAuth::new(conn.key, conn.daemon_id, conn.daemon_ver),
            )
            .await;
        });
        let consumer = match SubcConsumer::connect(&conn_path, ConsumerOptions::default()).await {
            Ok(consumer) => consumer,
            Err(error) => {
                daemon.abort();
                return Err(error.into());
            }
        };
        let identity = BindIdentity::new(
            checkout.to_path_buf(),
            label,
            format!("{label}-{}", std::process::id()),
        );
        let mut candidate = Self {
            consumer,
            identity,
            child,
            daemon,
        };
        // Registration polls only this private candidate, never daemon discovery.
        tokio::time::timeout(Duration::from_secs(600), async {
            loop {
                if call(
                    &candidate.consumer,
                    &candidate.identity,
                    "models.list",
                    json!({}),
                )
                .await
                .is_ok()
                {
                    break;
                }
                ensure!(
                    candidate.child.try_wait()?.is_none(),
                    "candidate exited before registering"
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(candidate)
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.consumer.close().await;
        if self.child.try_wait()?.is_none() {
            self.child.start_kill()?;
        }
        self.child.wait().await?;
        self.daemon.abort();
        Ok(())
    }
}

impl Drop for Candidate {
    fn drop(&mut self) {
        // Error exits must not leave the private daemon running. The child has kill_on_drop.
        self.daemon.abort();
    }
}
