#[cfg(all(feature = "cuda", not(target_os = "macos")))]
mod enabled {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use crate::model::{ModernBertLayer, Qwen3Layer};
    use crate::Precision;

    #[repr(C)]
    struct MiniLmLayerParams {
        query_weight: *const f32,
        query_bias: *const f32,
        key_weight: *const f32,
        key_bias: *const f32,
        value_weight: *const f32,
        value_bias: *const f32,
        attention_output_weight: *const f32,
        attention_output_bias: *const f32,
        attention_ln_weight: *const f32,
        attention_ln_bias: *const f32,
        intermediate_weight: *const f32,
        intermediate_bias: *const f32,
        output_weight: *const f32,
        output_bias: *const f32,
        output_ln_weight: *const f32,
        output_ln_bias: *const f32,
    }

    #[repr(C)]
    struct ModernBertLayerParams {
        qkv_weight: *const f32,
        attention_output_weight: *const f32,
        attention_norm_weight: *const f32,
        mlp_input_weight: *const f32,
        mlp_output_weight: *const f32,
        mlp_norm_weight: *const f32,
        attention_type: i32,
    }

    #[repr(C)]
    struct Qwen3LayerParams {
        input_norm: *const f32,
        post_attention_norm: *const f32,
        q_weight: *const f32,
        q_norm: *const f32,
        k_weight: *const f32,
        k_norm: *const f32,
        v_weight: *const f32,
        o_weight: *const f32,
        gate_weight: *const f32,
        up_weight: *const f32,
        down_weight: *const f32,
    }

    struct DeviceBinding {
        runtime_device: i32,
        driver_device: i32,
        context: NonNull<c_void>,
    }

    impl DeviceBinding {
        fn capture() -> Result<Self> {
            cuda_driver_check(unsafe { cuInit(0) }, "cuInit")?;
            let mut runtime_device = 0;
            cuda_runtime_check(
                unsafe { cudaGetDevice(&mut runtime_device) },
                "cudaGetDevice",
            )?;
            cuda_runtime_check(unsafe { cudaSetDevice(runtime_device) }, "cudaSetDevice")?;
            let mut driver_device = 0;
            cuda_driver_check(
                unsafe { cuCtxGetDevice(&mut driver_device) },
                "cuCtxGetDevice",
            )?;
            let mut context = std::ptr::null_mut();
            cuda_driver_check(
                unsafe { cuDevicePrimaryCtxRetain(&mut context, driver_device) },
                "cuDevicePrimaryCtxRetain",
            )?;
            let binding = Self {
                runtime_device,
                driver_device,
                context: NonNull::new(context)
                    .ok_or_else(|| anyhow::anyhow!("CUDA primary context is null"))?,
            };
            binding.bind()?;
            Ok(binding)
        }

        fn bind(&self) -> Result<()> {
            cuda_driver_check(
                unsafe { cuCtxSetCurrent(self.context.as_ptr()) },
                "cuCtxSetCurrent",
            )?;
            cuda_runtime_check(
                unsafe { cudaSetDevice(self.runtime_device) },
                "cudaSetDevice",
            )
        }
    }

    impl Drop for DeviceBinding {
        fn drop(&mut self) {
            unsafe {
                cuDevicePrimaryCtxRelease(self.driver_device);
            }
        }
    }

    pub fn ensure_available() -> Result<()> {
        cuda_driver_check(unsafe { cuInit(0) }, "cuInit")?;
        let version = unsafe { synapse_cuda_cublaslt_version() };
        ensure!(version > 0, "cuBLASLt did not report a version");
        Ok(())
    }

    pub struct MiniLmContext {
        binding: DeviceBinding,
        raw: NonNull<c_void>,
    }

    impl MiniLmContext {
        pub fn new(graphs: bool) -> Result<Self> {
            let binding = DeviceBinding::capture()?;
            let raw = unsafe { synapse_cuda_context_new(i32::from(graphs)) };
            Ok(Self {
                binding,
                raw: NonNull::new(raw).ok_or_else(last_error)?,
            })
        }

        pub fn forward(
            &mut self,
            hidden_states: &[u16],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            epsilon: f32,
            layers: &[crate::model::MiniLmLayer],
        ) -> Result<Vec<Vec<f32>>> {
            self.binding.bind()?;
            ensure!(batch > 0 && seq > 0 && hidden > 0 && heads > 0);
            ensure!(hidden_states.len() == batch * seq * hidden);
            ensure!(attention_mask.len() == batch * seq);
            let params = layers
                .iter()
                .map(|layer| MiniLmLayerParams {
                    query_weight: layer.query.weight.as_ptr(),
                    query_bias: layer.query.bias.as_ptr(),
                    key_weight: layer.key.weight.as_ptr(),
                    key_bias: layer.key.bias.as_ptr(),
                    value_weight: layer.value.weight.as_ptr(),
                    value_bias: layer.value.bias.as_ptr(),
                    attention_output_weight: layer.attention_output.weight.as_ptr(),
                    attention_output_bias: layer.attention_output.bias.as_ptr(),
                    attention_ln_weight: layer.attention_norm.weight.as_ptr(),
                    attention_ln_bias: layer.attention_norm.bias.as_ptr(),
                    intermediate_weight: layer.intermediate.weight.as_ptr(),
                    intermediate_bias: layer.intermediate.bias.as_ptr(),
                    output_weight: layer.output.weight.as_ptr(),
                    output_bias: layer.output.bias.as_ptr(),
                    output_ln_weight: layer.output_norm.weight.as_ptr(),
                    output_ln_bias: layer.output_norm.bias.as_ptr(),
                })
                .collect::<Vec<_>>();
            let mut output = vec![0.0f32; batch * hidden];
            let status = unsafe {
                synapse_cuda_encoder_forward(
                    self.raw.as_ptr(),
                    batch as u64,
                    seq as u64,
                    hidden as u64,
                    heads as u64,
                    intermediate as u64,
                    params.len() as u64,
                    epsilon,
                    hidden_states.as_ptr(),
                    attention_mask.as_ptr(),
                    output.as_mut_ptr(),
                    params.as_ptr(),
                )
            };
            check_status(status, "CUDA MiniLM encoder")?;
            Ok(output.chunks_exact(hidden).map(<[f32]>::to_vec).collect())
        }
    }

    impl Drop for MiniLmContext {
        fn drop(&mut self) {
            let _ = self.binding.bind();
            unsafe { synapse_cuda_context_free(self.raw.as_ptr()) }
        }
    }

    pub struct ModernBertContext {
        binding: DeviceBinding,
        raw: NonNull<c_void>,
        precision: Precision,
    }

    impl ModernBertContext {
        pub fn new(graphs: bool, precision: Precision) -> Result<Self> {
            let binding = DeviceBinding::capture()?;
            let raw = unsafe {
                synapse_cuda_modernbert_context_new(i32::from(graphs), precision_code(precision))
            };
            Ok(Self {
                binding,
                raw: NonNull::new(raw).ok_or_else(last_error)?,
                precision,
            })
        }

        pub fn forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            epsilon: f32,
            global_theta: f32,
            local_theta: f32,
            local_half_window: usize,
            layers: &[ModernBertLayer],
            final_norm: &[f32],
        ) -> Result<()> {
            self.binding.bind()?;
            ensure!(hidden_states.len() == batch * seq * hidden);
            ensure!(attention_mask.len() == batch * seq);
            ensure!(final_norm.len() == hidden);
            let params = layers
                .iter()
                .map(|layer| ModernBertLayerParams {
                    qkv_weight: layer.qkv_weight.as_ptr(),
                    attention_output_weight: layer.attention_output_weight.as_ptr(),
                    attention_norm_weight: layer
                        .attention_norm_weight
                        .as_ref()
                        .map_or(std::ptr::null(), Vec::as_ptr),
                    mlp_input_weight: layer.mlp_input_weight.as_ptr(),
                    mlp_output_weight: layer.mlp_output_weight.as_ptr(),
                    mlp_norm_weight: layer.mlp_norm_weight.as_ptr(),
                    attention_type: i32::from(layer.sliding_attention),
                })
                .collect::<Vec<_>>();
            let input_f16 = matches!(self.precision, Precision::F16)
                .then(|| crate::encode_f16_bits(hidden_states));
            let input = input_f16
                .as_ref()
                .map_or(hidden_states.as_ptr().cast(), |v| v.as_ptr().cast());
            let status = unsafe {
                synapse_cuda_modernbert_forward(
                    self.raw.as_ptr(),
                    batch as u64,
                    seq as u64,
                    hidden as u64,
                    heads as u64,
                    intermediate as u64,
                    params.len() as u64,
                    epsilon,
                    global_theta,
                    local_theta,
                    local_half_window as u64,
                    input,
                    attention_mask.as_ptr(),
                    params.as_ptr(),
                    final_norm.as_ptr(),
                    hidden_states.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA ModernBERT encoder")
        }
    }

    impl Drop for ModernBertContext {
        fn drop(&mut self) {
            let _ = self.binding.bind();
            unsafe { synapse_cuda_modernbert_context_free(self.raw.as_ptr()) }
        }
    }

    pub struct Qwen3Context {
        binding: DeviceBinding,
        raw: NonNull<c_void>,
    }

    impl Qwen3Context {
        pub fn new(graphs: bool, precision: Precision) -> Result<Self> {
            ensure!(
                matches!(precision, Precision::F16),
                "Qwen3 CUDA requires f16 storage"
            );
            let binding = DeviceBinding::capture()?;
            let raw = unsafe { synapse_cuda_qwen3_context_new(i32::from(graphs)) };
            Ok(Self {
                binding,
                raw: NonNull::new(raw).ok_or_else(last_error)?,
            })
        }

        pub fn forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            epsilon: f32,
            rope_theta: f32,
            layers: &[Qwen3Layer],
            final_norm: &[f32],
        ) -> Result<()> {
            self.binding.bind()?;
            ensure!(hidden_states.len() == batch * seq * hidden);
            ensure!(attention_mask.len() == batch * seq);
            ensure!(final_norm.len() == hidden);
            let params = layers
                .iter()
                .map(|layer| Qwen3LayerParams {
                    input_norm: layer.input_norm.as_ptr(),
                    post_attention_norm: layer.post_attention_norm.as_ptr(),
                    q_weight: layer.q_weight.as_ptr(),
                    q_norm: layer.q_norm.as_ptr(),
                    k_weight: layer.k_weight.as_ptr(),
                    k_norm: layer.k_norm.as_ptr(),
                    v_weight: layer.v_weight.as_ptr(),
                    o_weight: layer.o_weight.as_ptr(),
                    gate_weight: layer.gate_weight.as_ptr(),
                    up_weight: layer.up_weight.as_ptr(),
                    down_weight: layer.down_weight.as_ptr(),
                })
                .collect::<Vec<_>>();
            let input = crate::encode_f16_bits(hidden_states);
            let status = unsafe {
                synapse_cuda_qwen3_forward(
                    self.raw.as_ptr(),
                    batch as u64,
                    seq as u64,
                    hidden as u64,
                    query_heads as u64,
                    kv_heads as u64,
                    head_dim as u64,
                    intermediate as u64,
                    params.len() as u64,
                    epsilon,
                    rope_theta,
                    input.as_ptr(),
                    attention_mask.as_ptr(),
                    params.as_ptr(),
                    final_norm.as_ptr(),
                    hidden_states.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA Qwen3 encoder")
        }
    }

    impl Drop for Qwen3Context {
        fn drop(&mut self) {
            let _ = self.binding.bind();
            unsafe { synapse_cuda_qwen3_context_free(self.raw.as_ptr()) }
        }
    }

    fn precision_code(precision: Precision) -> i32 {
        match precision {
            Precision::F32 => 0,
            Precision::F16 => 1,
        }
    }

    fn check_status(status: i32, operation: &str) -> Result<()> {
        if status != 0 {
            bail!("{operation} failed with status {status}: {}", last_error());
        }
        Ok(())
    }

    fn cuda_driver_check(status: i32, operation: &str) -> Result<()> {
        if status != 0 {
            let mut raw = std::ptr::null();
            let message = unsafe {
                if cuGetErrorString(status, &mut raw) != 0 || raw.is_null() {
                    "unknown CUDA driver error".into()
                } else {
                    CStr::from_ptr(raw).to_string_lossy()
                }
            };
            bail!("{operation} failed with status {status}: {message}");
        }
        Ok(())
    }

    fn cuda_runtime_check(status: i32, operation: &str) -> Result<()> {
        if status != 0 {
            let message = unsafe {
                let raw = cudaGetErrorString(status);
                if raw.is_null() {
                    "unknown CUDA runtime error".into()
                } else {
                    CStr::from_ptr(raw).to_string_lossy()
                }
            };
            bail!("{operation} failed with status {status}: {message}");
        }
        Ok(())
    }

    fn last_error() -> anyhow::Error {
        unsafe {
            let raw = synapse_cuda_last_error();
            if raw.is_null() {
                anyhow::anyhow!("unknown CUDA error")
            } else {
                anyhow::anyhow!(CStr::from_ptr(raw).to_string_lossy().into_owned())
            }
        }
    }

    unsafe extern "C" {
        fn cuInit(flags: u32) -> i32;
        fn cuCtxGetDevice(device: *mut i32) -> i32;
        fn cuCtxSetCurrent(context: *mut c_void) -> i32;
        fn cuDevicePrimaryCtxRetain(context: *mut *mut c_void, device: i32) -> i32;
        fn cuDevicePrimaryCtxRelease(device: i32) -> i32;
        fn cuGetErrorString(status: i32, message: *mut *const c_char) -> i32;
        fn cudaGetDevice(device: *mut i32) -> i32;
        fn cudaSetDevice(device: i32) -> i32;
        fn cudaGetErrorString(status: i32) -> *const c_char;
        fn synapse_cuda_context_new(graphs_enabled: i32) -> *mut c_void;
        fn synapse_cuda_context_free(context: *mut c_void);
        fn synapse_cuda_encoder_forward(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            heads: u64,
            intermediate: u64,
            layer_count: u64,
            layer_norm_eps: f32,
            input: *const u16,
            attention_mask: *const u8,
            output: *mut f32,
            layers: *const MiniLmLayerParams,
        ) -> i32;
        fn synapse_cuda_modernbert_context_new(graphs_enabled: i32, precision: i32) -> *mut c_void;
        fn synapse_cuda_modernbert_context_free(context: *mut c_void);
        fn synapse_cuda_modernbert_forward(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            heads: u64,
            intermediate: u64,
            layer_count: u64,
            epsilon: f32,
            global_theta: f32,
            local_theta: f32,
            local_half_window: u64,
            input: *const c_void,
            attention_mask: *const u8,
            layers: *const ModernBertLayerParams,
            final_norm: *const f32,
            output: *mut f32,
        ) -> i32;
        fn synapse_cuda_qwen3_context_new(graphs_enabled: i32) -> *mut c_void;
        fn synapse_cuda_qwen3_context_free(context: *mut c_void);
        fn synapse_cuda_qwen3_forward(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            epsilon: f32,
            rope_theta: f32,
            input: *const u16,
            attention_mask: *const u8,
            layers: *const Qwen3LayerParams,
            final_norm: *const f32,
            output: *mut f32,
        ) -> i32;
        fn synapse_cuda_last_error() -> *const c_char;
        fn synapse_cuda_cublaslt_version() -> u64;
    }
}

#[cfg(any(not(feature = "cuda"), target_os = "macos"))]
mod enabled {
    use anyhow::{bail, Result};

    use crate::model::{MiniLmLayer, ModernBertLayer, Qwen3Layer};
    use crate::Precision;

    pub fn ensure_available() -> Result<()> {
        bail!("owned CUDA requires a non-macOS build with cargo feature `cuda`")
    }

    pub struct MiniLmContext;
    impl MiniLmContext {
        pub fn new(_graphs: bool) -> Result<Self> {
            bail!("owned CUDA is unavailable in this build")
        }
        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &mut self,
            _hidden_states: &[u16],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _heads: usize,
            _intermediate: usize,
            _epsilon: f32,
            _layers: &[MiniLmLayer],
        ) -> Result<Vec<Vec<f32>>> {
            bail!("owned CUDA is unavailable in this build")
        }
    }

    pub struct ModernBertContext;
    impl ModernBertContext {
        pub fn new(_graphs: bool, _precision: Precision) -> Result<Self> {
            bail!("owned CUDA is unavailable in this build")
        }
        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _heads: usize,
            _intermediate: usize,
            _epsilon: f32,
            _global_theta: f32,
            _local_theta: f32,
            _local_half_window: usize,
            _layers: &[ModernBertLayer],
            _final_norm: &[f32],
        ) -> Result<()> {
            bail!("owned CUDA is unavailable in this build")
        }
    }

    pub struct Qwen3Context;
    impl Qwen3Context {
        pub fn new(_graphs: bool, _precision: Precision) -> Result<Self> {
            bail!("owned CUDA is unavailable in this build")
        }
        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _query_heads: usize,
            _kv_heads: usize,
            _head_dim: usize,
            _intermediate: usize,
            _epsilon: f32,
            _rope_theta: f32,
            _layers: &[Qwen3Layer],
            _final_norm: &[f32],
        ) -> Result<()> {
            bail!("owned CUDA is unavailable in this build")
        }
    }
}

pub use enabled::{ensure_available, MiniLmContext, ModernBertContext, Qwen3Context};
