use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::CatalogFile;
use crate::catalog_runtime_tests::isolated_catalog_state;
use crate::tests::response_result;
use crate::{models_download, sha256_file, sha256_hex, store};

#[tokio::test]
async fn catalog_multishard_download_installs_only_pins_and_checks_every_file() {
    // Use a runnable catalog entry with the Qwen3-ASR file names to exercise
    // repeated file roles without adding a new model to the release catalog.
    let manifest = synapse_parity::manifest::Manifest::from_slice(include_bytes!(
        "../../../../bench/parity/models.json"
    ))
    .unwrap();
    let source = manifest.model("qwen3-asr-1.7b").unwrap();
    for corrupt in std::iter::once(None).chain((0..source.files.len()).map(Some)) {
        let (root, mut state) = isolated_catalog_state("multishard-download", true);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
        runtime.hf_endpoint = endpoint;
        let entry = runtime
            .release_catalog
            .models
            .iter_mut()
            .find(|e| e.id == "gte-modernbert-base")
            .unwrap();
        entry.backends.retain(|backend| backend.backend == "metal");
        entry.files = source
            .files
            .keys()
            .map(|name| {
                let body = format!("catalog-only bytes for {name}");
                CatalogFile {
                    path: name.clone(),
                    role: if name.ends_with(".safetensors") {
                        "model"
                    } else if name == "vocab.json" {
                        "tokenizer"
                    } else {
                        "config"
                    }
                    .into(),
                    sha256: sha256_hex(body.as_bytes()),
                    size_bytes: body.len() as u64,
                    backends: vec!["metal".into()],
                }
            })
            .collect();
        let files = entry.files.clone();
        let responses = files
            .iter()
            .enumerate()
            .map(|(index, file)| {
                let path = format!(
                    "/{}/resolve/{}/{}",
                    entry.upstream.hf_repo, entry.upstream.revision, file.path
                );
                let body = if corrupt == Some(index) {
                    "x".repeat(file.size_bytes as usize)
                } else {
                    format!("catalog-only bytes for {}", file.path)
                };
                (path, body)
            })
            .collect::<BTreeMap<_, _>>();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let requests = seen.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0u8; 4096];
                let size = stream.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]);
                let path = request.split_whitespace().nth(1).unwrap().to_string();
                let body = responses
                    .get(&path)
                    .expect("only catalog-pinned paths may be requested");
                requests.lock().unwrap().push(path);
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                stream.write_all(body.as_bytes()).await.unwrap();
            }
        });
        let accepted = response_result(
            models_download(
                state.clone(),
                json!({"catalog_id":"gte-modernbert-base","request_key":"pinned"}),
            )
            .await,
            "models.download",
        );
        let job = accepted["job_id"].as_str().unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        let terminal = loop {
            let record = state.store.get_job(job).unwrap().unwrap();
            if !store::is_download_non_terminal(&record.state) {
                break record;
            }
            assert!(tokio::time::Instant::now() < deadline, "{record:?}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        if let Some(index) = corrupt {
            let error = terminal.error_json.unwrap();
            assert_eq!(error["code"], "artifact_invalid");
            assert_eq!(error["details"]["file"], files[index].path);
            assert!(state
                .store
                .catalog_installs("gte-modernbert-base")
                .unwrap()
                .is_empty());
            assert!(!state.model_cache.blob_path(&files[index].sha256).exists());
        } else {
            assert_eq!(terminal.state, "committed");
            assert_eq!(seen.lock().unwrap().len(), files.len());
            let installs = state.store.catalog_installs("gte-modernbert-base").unwrap();
            assert_eq!(installs.len(), 1);
            assert_eq!(installs[0].members.len(), files.len());
            for file in &files {
                assert!(installs[0]
                    .members
                    .iter()
                    .any(|member| member.path == file.path && member.digest == file.sha256));
                assert_eq!(
                    sha256_file(&state.model_cache.blob_path(&file.sha256)).unwrap(),
                    file.sha256
                );
            }
        }
        server.abort();
        let _ = server.await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while Arc::strong_count(&state) > 1 {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }
}
