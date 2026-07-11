#[cfg(all(target_os = "linux", feature = "cuda"))]
mod enabled {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use super::super::{encode_f16_bits, EncoderLayer};

    #[repr(C)]
    struct SynapseCudaEncoderLayerParams {
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

    pub fn ensure_available() -> Result<()> {
        let version = unsafe { synapse_cuda_cublaslt_version() };
        ensure!(version > 0, "cuBLASLt did not report a version");
        Ok(())
    }

    pub struct CudaContext {
        raw: NonNull<c_void>,
    }

    impl CudaContext {
        pub fn new(graphs: bool) -> Result<Self> {
            let raw = unsafe { synapse_cuda_context_new(i32::from(graphs)) };
            let raw = NonNull::new(raw).ok_or_else(last_error)?;
            Ok(Self { raw })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            layer_norm_eps: f32,
            layers: &[EncoderLayer],
        ) -> Result<Vec<Vec<f32>>> {
            ensure!(
                batch > 0 && seq > 0,
                "CUDA encoder dimensions must be non-zero"
            );
            ensure!(hidden % heads == 0, "hidden size must divide heads");
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "CUDA encoder hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "CUDA encoder mask shape mismatch"
            );
            ensure!(!layers.is_empty(), "CUDA encoder requires layers");

            let params = layers
                .iter()
                .map(|layer| SynapseCudaEncoderLayerParams {
                    query_weight: layer.query.weight.data.as_ptr(),
                    query_bias: layer.query.bias.as_slice().as_ptr(),
                    key_weight: layer.key.weight.data.as_ptr(),
                    key_bias: layer.key.bias.as_slice().as_ptr(),
                    value_weight: layer.value.weight.data.as_ptr(),
                    value_bias: layer.value.bias.as_slice().as_ptr(),
                    attention_output_weight: layer.attention_output.weight.data.as_ptr(),
                    attention_output_bias: layer.attention_output.bias.as_slice().as_ptr(),
                    attention_ln_weight: layer.attention_ln_weight.as_slice().as_ptr(),
                    attention_ln_bias: layer.attention_ln_bias.as_slice().as_ptr(),
                    intermediate_weight: layer.intermediate.weight.data.as_ptr(),
                    intermediate_bias: layer.intermediate.bias.as_slice().as_ptr(),
                    output_weight: layer.output.weight.data.as_ptr(),
                    output_bias: layer.output.bias.as_slice().as_ptr(),
                    output_ln_weight: layer.output_ln_weight.as_slice().as_ptr(),
                    output_ln_bias: layer.output_ln_bias.as_slice().as_ptr(),
                })
                .collect::<Vec<_>>();
            let input = encode_f16_bits(hidden_states);
            let mut output = vec![0.0f32; batch * hidden];
            let status = unsafe {
                synapse_cuda_encoder_forward(
                    self.raw.as_ptr(),
                    batch as u64,
                    seq as u64,
                    hidden as u64,
                    heads as u64,
                    intermediate as u64,
                    layers.len() as u64,
                    layer_norm_eps,
                    input.as_ptr(),
                    attention_mask.as_ptr(),
                    output.as_mut_ptr(),
                    params.as_ptr(),
                )
            };
            if status != 0 {
                bail!("CUDA encoder failed with status {status}: {}", last_error());
            }
            Ok(output.chunks_exact(hidden).map(<[f32]>::to_vec).collect())
        }
    }

    impl Drop for CudaContext {
        fn drop(&mut self) {
            unsafe { synapse_cuda_context_free(self.raw.as_ptr()) }
        }
    }

    fn last_error() -> anyhow::Error {
        unsafe {
            let raw = synapse_cuda_last_error();
            if raw.is_null() {
                return anyhow::anyhow!("unknown CUDA error");
            }
            let message = CStr::from_ptr(raw).to_string_lossy();
            if message.is_empty() {
                anyhow::anyhow!("unknown CUDA error")
            } else {
                anyhow::anyhow!(message.into_owned())
            }
        }
    }

    unsafe extern "C" {
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
            layers: *const SynapseCudaEncoderLayerParams,
        ) -> i32;
        fn synapse_cuda_last_error() -> *const c_char;
        fn synapse_cuda_cublaslt_version() -> u64;
    }
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
mod enabled {
    use anyhow::{bail, Result};

    use super::super::EncoderLayer;

    pub fn ensure_available() -> Result<()> {
        bail!("CUDA provider requires Linux and cargo feature `cuda`")
    }

    pub struct CudaContext;

    impl CudaContext {
        pub fn new(_graphs: bool) -> Result<Self> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _heads: usize,
            _intermediate: usize,
            _layer_norm_eps: f32,
            _layers: &[EncoderLayer],
        ) -> Result<Vec<Vec<f32>>> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }
    }
}

pub use enabled::{ensure_available, CudaContext};
