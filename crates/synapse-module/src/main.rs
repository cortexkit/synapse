#![forbid(unsafe_code)]

#[tokio::main]
async fn main() {
    if let Err(error) = synapse_module::run_from_env().await {
        if matches!(error, synapse_module::ModuleError::SingletonHeld(_)) {
            std::process::exit(1);
        }
        panic!("synapse module failed: {error}");
    }
}
