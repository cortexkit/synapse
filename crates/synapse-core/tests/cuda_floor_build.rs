#[allow(dead_code)]
#[path = "../build.rs"]
mod floor_build;

fn manifest() -> serde_json::Value {
    serde_json::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap()
}

#[test]
fn all_four_cuda_profiles_agree_with_public_floor_constants() {
    let manifest = manifest();
    assert_eq!(
        manifest["profiles"]
            .as_object()
            .unwrap()
            .values()
            .filter(|p| p["lane"] == "owned-cuda")
            .count(),
        4
    );
    let floor = floor_build::collect_cuda_floor(&manifest).unwrap();
    assert_eq!(floor, [13000, 7, 5]);
    assert_eq!(synapse_core::OWNED_CUDA_MINIMUM_DRIVER_API, floor[0]);
    assert_eq!(
        synapse_core::OWNED_CUDA_MINIMUM_DEVICE_CC,
        floor[1] as f32 + floor[2] as f32 / 10.0
    );
}

#[test]
fn missing_or_disagreeing_cuda_floors_name_the_offending_profile() {
    let name = "qwen3-reranker-0.6b.owned-cuda";
    for key in [
        "cuda_min_driver_api",
        "cuda_min_compute_major",
        "cuda_min_compute_minor",
    ] {
        let mut missing = manifest();
        missing["profiles"][name]
            .as_object_mut()
            .unwrap()
            .remove(key);
        let error = floor_build::collect_cuda_floor(&missing).unwrap_err();
        assert!(error.contains(name) && error.contains(key), "{error}");
        let mut disagreeing = manifest();
        let value = disagreeing["profiles"][name][key].as_u64().unwrap();
        disagreeing["profiles"][name][key] = (value + 1).into();
        let error = floor_build::collect_cuda_floor(&disagreeing).unwrap_err();
        assert!(
            error.contains(name) && error.contains("disagreeing"),
            "{error}"
        );
    }
}
