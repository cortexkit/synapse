fn main() {
    let backends = [
        ("cpu", std::env::var_os("CARGO_FEATURE_CPU").is_some()),
        ("cuda", std::env::var_os("CARGO_FEATURE_CUDA").is_some()),
        ("vulkan", std::env::var_os("CARGO_FEATURE_VULKAN").is_some()),
    ];
    let enabled: Vec<_> = backends
        .iter()
        .filter_map(|(name, enabled)| enabled.then_some(*name))
        .collect();
    if enabled.len() != 1 {
        panic!(
            "ck-synapse-worker-llama requires exactly one backend feature (got {})",
            if enabled.is_empty() {
                "none".to_string()
            } else {
                enabled.join(", ")
            }
        );
    }
}
