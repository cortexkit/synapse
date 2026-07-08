#![forbid(unsafe_code)]

#[tokio::main]
async fn main() -> Result<(), synapse_module::ModuleError> {
    synapse_module::run_from_env().await
}
