#[cfg(all(target_os = "linux", feature = "cuda"))]
mod enabled {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use super::super::{encode_f16_bits, EncoderLayer, Precision};

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

    #[repr(C)]
    #[allow(dead_code)]
    pub struct ModernBertLayerParams {
        pub qkv_weight: *const f32,
        pub attention_output_weight: *const f32,
        pub attention_norm_weight: *const f32,
        pub mlp_input_weight: *const f32,
        pub mlp_output_weight: *const f32,
        pub mlp_norm_weight: *const f32,
        pub attention_type: i32,
    }

    #[repr(C)]
    #[allow(dead_code)]
    pub struct Qwen3LayerParams {
        pub input_norm: *const f32,
        pub post_attention_norm: *const f32,
        pub q_weight: *const f32,
        pub q_norm: *const f32,
        pub k_weight: *const f32,
        pub k_norm: *const f32,
        pub v_weight: *const f32,
        pub o_weight: *const f32,
        pub gate_weight: *const f32,
        pub up_weight: *const f32,
        pub down_weight: *const f32,
    }

    #[repr(C)]
    #[allow(dead_code)]
    pub struct Lfm2LayerParams {
        pub mixer_type: i32,
        pub operator_norm: *const f32,
        pub ffn_norm: *const f32,
        pub conv_in_weight: *const f32,
        pub conv_weight: *const f32,
        pub conv_out_weight: *const f32,
        pub q_weight: *const f32,
        pub q_norm: *const f32,
        pub k_weight: *const f32,
        pub k_norm: *const f32,
        pub v_weight: *const f32,
        pub attention_out_weight: *const f32,
        pub w1_weight: *const f32,
        pub w2_weight: *const f32,
        pub w3_weight: *const f32,
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
            check_status(status, "CUDA MiniLM encoder")?;
            Ok(output.chunks_exact(hidden).map(<[f32]>::to_vec).collect())
        }
    }

    impl Drop for CudaContext {
        fn drop(&mut self) {
            unsafe { synapse_cuda_context_free(self.raw.as_ptr()) }
        }
    }

    pub struct ModernBertContext {
        raw: NonNull<c_void>,
        precision: Precision,
    }

    impl ModernBertContext {
        pub fn new(graphs: bool, precision: Precision) -> Result<Self> {
            let raw = unsafe {
                synapse_cuda_modernbert_context_new(i32::from(graphs), precision_code(precision))
            };
            Ok(Self {
                raw: NonNull::new(raw).ok_or_else(last_error)?,
                precision,
            })
        }

        #[allow(clippy::too_many_arguments)]
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
            layers: &[ModernBertLayerParams],
            final_norm: &[f32],
        ) -> Result<()> {
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "ModernBERT CUDA hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "ModernBERT CUDA mask shape mismatch"
            );
            ensure!(
                final_norm.len() == hidden,
                "ModernBERT CUDA final norm shape mismatch"
            );
            let input_f16 =
                matches!(self.precision, Precision::F16).then(|| encode_f16_bits(hidden_states));
            let input = input_f16
                .as_ref()
                .map_or(hidden_states.as_ptr().cast(), |values| {
                    values.as_ptr().cast()
                });
            let status = unsafe {
                synapse_cuda_modernbert_forward(
                    self.raw.as_ptr(),
                    batch as u64,
                    seq as u64,
                    hidden as u64,
                    heads as u64,
                    intermediate as u64,
                    layers.len() as u64,
                    epsilon,
                    global_theta,
                    local_theta,
                    local_half_window as u64,
                    input,
                    attention_mask.as_ptr(),
                    layers.as_ptr(),
                    final_norm.as_ptr(),
                    hidden_states.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA ModernBERT encoder")
        }
    }

    impl Drop for ModernBertContext {
        fn drop(&mut self) {
            unsafe { synapse_cuda_modernbert_context_free(self.raw.as_ptr()) }
        }
    }

    pub struct Qwen3Context {
        raw: NonNull<c_void>,
    }

    impl Qwen3Context {
        pub fn new(graphs: bool, precision: Precision) -> Result<Self> {
            ensure!(
                matches!(precision, Precision::F16),
                "Qwen3 CUDA requires f16 storage"
            );
            let raw = unsafe { synapse_cuda_qwen3_context_new(i32::from(graphs)) };
            Ok(Self {
                raw: NonNull::new(raw).ok_or_else(last_error)?,
            })
        }

        #[allow(clippy::too_many_arguments)]
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
            layers: &[Qwen3LayerParams],
            final_norm: &[f32],
        ) -> Result<()> {
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "Qwen3 CUDA hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "Qwen3 CUDA mask shape mismatch"
            );
            let input = encode_f16_bits(hidden_states);
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
                    layers.len() as u64,
                    epsilon,
                    rope_theta,
                    input.as_ptr(),
                    attention_mask.as_ptr(),
                    layers.as_ptr(),
                    final_norm.as_ptr(),
                    hidden_states.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA Qwen3 encoder")
        }
    }

    impl Drop for Qwen3Context {
        fn drop(&mut self) {
            unsafe { synapse_cuda_qwen3_context_free(self.raw.as_ptr()) }
        }
    }

    pub struct Lfm2Context {
        raw: NonNull<c_void>,
    }

    impl Lfm2Context {
        pub fn new(graphs: bool, precision: Precision) -> Result<Self> {
            let raw = unsafe {
                synapse_cuda_lfm2_context_new(i32::from(graphs), precision_code(precision))
            };
            Ok(Self {
                raw: NonNull::new(raw).ok_or_else(last_error)?,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn full_forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            seq: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            kernel: usize,
            vocab: usize,
            epsilon: f32,
            rope_theta: f32,
            layers: &[Lfm2LayerParams],
            final_norm: &[f32],
            lm_head: &[f32],
        ) -> Result<()> {
            ensure!(
                hidden_states.len() == seq * hidden,
                "LFM2 CUDA full hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == seq,
                "LFM2 CUDA full mask shape mismatch"
            );
            let status = unsafe {
                synapse_cuda_lfm2_full_forward(
                    self.raw.as_ptr(),
                    seq as u64,
                    hidden as u64,
                    query_heads as u64,
                    kv_heads as u64,
                    head_dim as u64,
                    intermediate as u64,
                    layers.len() as u64,
                    kernel as u64,
                    vocab as u64,
                    epsilon,
                    rope_theta,
                    hidden_states.as_ptr(),
                    attention_mask.as_ptr(),
                    layers.as_ptr(),
                    final_norm.as_ptr(),
                    lm_head.as_ptr(),
                    hidden_states.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA LFM2 full forward")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn prefill(
            &mut self,
            embeddings: &[f32],
            seq: usize,
            capacity: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            kernel: usize,
            vocab: usize,
            epsilon: f32,
            rope_theta: f32,
            layers: &[Lfm2LayerParams],
            final_norm: &[f32],
            lm_head: &[f32],
        ) -> Result<Vec<f32>> {
            ensure!(
                embeddings.len() == seq * hidden,
                "LFM2 CUDA prefill shape mismatch"
            );
            let mut logits = vec![0.0; vocab];
            let status = unsafe {
                synapse_cuda_lfm2_prefill(
                    self.raw.as_ptr(),
                    seq as u64,
                    capacity as u64,
                    hidden as u64,
                    query_heads as u64,
                    kv_heads as u64,
                    head_dim as u64,
                    intermediate as u64,
                    layers.len() as u64,
                    kernel as u64,
                    vocab as u64,
                    epsilon,
                    rope_theta,
                    embeddings.as_ptr(),
                    layers.as_ptr(),
                    final_norm.as_ptr(),
                    lm_head.as_ptr(),
                    logits.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA LFM2 prefill")?;
            Ok(logits)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn decode(
            &mut self,
            embedding: &[f32],
            position: usize,
            capacity: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            kernel: usize,
            vocab: usize,
            epsilon: f32,
            rope_theta: f32,
            layers: &[Lfm2LayerParams],
            final_norm: &[f32],
            lm_head: &[f32],
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            ensure!(
                embedding.len() == hidden,
                "LFM2 CUDA decode embedding shape mismatch"
            );
            let mut output = vec![0.0; hidden];
            let mut logits = vec![0.0; vocab];
            let status = unsafe {
                synapse_cuda_lfm2_decode(
                    self.raw.as_ptr(),
                    position as u64,
                    capacity as u64,
                    hidden as u64,
                    query_heads as u64,
                    kv_heads as u64,
                    head_dim as u64,
                    intermediate as u64,
                    layers.len() as u64,
                    kernel as u64,
                    vocab as u64,
                    epsilon,
                    rope_theta,
                    embedding.as_ptr(),
                    layers.as_ptr(),
                    final_norm.as_ptr(),
                    lm_head.as_ptr(),
                    output.as_mut_ptr(),
                    logits.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA LFM2 decode")?;
            Ok((output, logits))
        }
    }

    impl Drop for Lfm2Context {
        fn drop(&mut self) {
            unsafe { synapse_cuda_lfm2_context_free(self.raw.as_ptr()) }
        }
    }

    pub struct OpsContext {
        raw: NonNull<c_void>,
    }

    impl OpsContext {
        pub fn new() -> Result<Self> {
            let raw = unsafe { synapse_cuda_ops_context_new() };
            Ok(Self {
                raw: NonNull::new(raw).ok_or_else(last_error)?,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn matmul(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            a: &[f32],
            b: &[f32],
            transpose_b: bool,
            static_rhs: bool,
            c: &mut [f32],
        ) -> Result<()> {
            ensure!(a.len() == m * k, "CUDA matmul A shape mismatch");
            ensure!(b.len() == n * k, "CUDA matmul B shape mismatch");
            ensure!(c.len() == m * n, "CUDA matmul output shape mismatch");
            let status = unsafe {
                synapse_cuda_ops_matmul(
                    self.raw.as_ptr(),
                    m as u64,
                    n as u64,
                    k as u64,
                    a.as_ptr(),
                    b.as_ptr(),
                    i32::from(transpose_b),
                    i32::from(static_rhs),
                    c.as_mut_ptr(),
                )
            };
            check_status(status, "CUDA matmul")
        }
    }

    impl Drop for OpsContext {
        fn drop(&mut self) {
            unsafe { synapse_cuda_ops_context_free(self.raw.as_ptr()) }
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
        fn synapse_cuda_lfm2_context_new(graphs_enabled: i32, precision: i32) -> *mut c_void;
        fn synapse_cuda_lfm2_context_free(context: *mut c_void);
        fn synapse_cuda_lfm2_full_forward(
            context: *mut c_void,
            seq: u64,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            kernel: u64,
            vocab: u64,
            epsilon: f32,
            rope_theta: f32,
            input: *const f32,
            mask: *const u8,
            layers: *const Lfm2LayerParams,
            final_norm: *const f32,
            lm_head: *const f32,
            output: *mut f32,
        ) -> i32;
        fn synapse_cuda_lfm2_prefill(
            context: *mut c_void,
            seq: u64,
            capacity: u64,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            kernel: u64,
            vocab: u64,
            epsilon: f32,
            rope_theta: f32,
            input: *const f32,
            layers: *const Lfm2LayerParams,
            final_norm: *const f32,
            lm_head: *const f32,
            logits: *mut f32,
        ) -> i32;
        fn synapse_cuda_lfm2_decode(
            context: *mut c_void,
            position: u64,
            capacity: u64,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            kernel: u64,
            vocab: u64,
            epsilon: f32,
            rope_theta: f32,
            embedding: *const f32,
            layers: *const Lfm2LayerParams,
            final_norm: *const f32,
            lm_head: *const f32,
            output_hidden: *mut f32,
            logits: *mut f32,
        ) -> i32;
        fn synapse_cuda_ops_context_new() -> *mut c_void;
        fn synapse_cuda_ops_context_free(context: *mut c_void);
        fn synapse_cuda_ops_matmul(
            context: *mut c_void,
            m: u64,
            n: u64,
            k: u64,
            a: *const f32,
            b: *const f32,
            transpose_b: i32,
            static_rhs: i32,
            c: *mut f32,
        ) -> i32;
        fn synapse_cuda_last_error() -> *const c_char;
        fn synapse_cuda_cublaslt_version() -> u64;
    }
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
mod enabled {
    use anyhow::{bail, Result};

    use super::super::{EncoderLayer, Precision};

    #[allow(dead_code)]
    pub struct ModernBertLayerParams {
        pub qkv_weight: *const f32,
        pub attention_output_weight: *const f32,
        pub attention_norm_weight: *const f32,
        pub mlp_input_weight: *const f32,
        pub mlp_output_weight: *const f32,
        pub mlp_norm_weight: *const f32,
        pub attention_type: i32,
    }

    #[allow(dead_code)]
    pub struct Qwen3LayerParams {
        pub input_norm: *const f32,
        pub post_attention_norm: *const f32,
        pub q_weight: *const f32,
        pub q_norm: *const f32,
        pub k_weight: *const f32,
        pub k_norm: *const f32,
        pub v_weight: *const f32,
        pub o_weight: *const f32,
        pub gate_weight: *const f32,
        pub up_weight: *const f32,
        pub down_weight: *const f32,
    }

    #[allow(dead_code)]
    pub struct Lfm2LayerParams {
        pub mixer_type: i32,
        pub operator_norm: *const f32,
        pub ffn_norm: *const f32,
        pub conv_in_weight: *const f32,
        pub conv_weight: *const f32,
        pub conv_out_weight: *const f32,
        pub q_weight: *const f32,
        pub q_norm: *const f32,
        pub k_weight: *const f32,
        pub k_norm: *const f32,
        pub v_weight: *const f32,
        pub attention_out_weight: *const f32,
        pub w1_weight: *const f32,
        pub w2_weight: *const f32,
        pub w3_weight: *const f32,
    }

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

    pub struct ModernBertContext;
    impl ModernBertContext {
        pub fn new(_graphs: bool, _precision: Precision) -> Result<Self> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
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
            _layers: &[ModernBertLayerParams],
            _final_norm: &[f32],
        ) -> Result<()> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }
    }

    pub struct Qwen3Context;
    impl Qwen3Context {
        pub fn new(_graphs: bool, _precision: Precision) -> Result<Self> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
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
            _layers: &[Qwen3LayerParams],
            _final_norm: &[f32],
        ) -> Result<()> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }
    }

    pub struct Lfm2Context;
    impl Lfm2Context {
        pub fn new(_graphs: bool, _precision: Precision) -> Result<Self> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn full_forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _seq: usize,
            _hidden: usize,
            _query_heads: usize,
            _kv_heads: usize,
            _head_dim: usize,
            _intermediate: usize,
            _kernel: usize,
            _vocab: usize,
            _epsilon: f32,
            _rope_theta: f32,
            _layers: &[Lfm2LayerParams],
            _final_norm: &[f32],
            _lm_head: &[f32],
        ) -> Result<()> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn prefill(
            &mut self,
            _embeddings: &[f32],
            _seq: usize,
            _capacity: usize,
            _hidden: usize,
            _query_heads: usize,
            _kv_heads: usize,
            _head_dim: usize,
            _intermediate: usize,
            _kernel: usize,
            _vocab: usize,
            _epsilon: f32,
            _rope_theta: f32,
            _layers: &[Lfm2LayerParams],
            _final_norm: &[f32],
            _lm_head: &[f32],
        ) -> Result<Vec<f32>> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn decode(
            &mut self,
            _embedding: &[f32],
            _position: usize,
            _capacity: usize,
            _hidden: usize,
            _query_heads: usize,
            _kv_heads: usize,
            _head_dim: usize,
            _intermediate: usize,
            _kernel: usize,
            _vocab: usize,
            _epsilon: f32,
            _rope_theta: f32,
            _layers: &[Lfm2LayerParams],
            _final_norm: &[f32],
            _lm_head: &[f32],
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }
    }

    pub struct OpsContext;
    impl OpsContext {
        pub fn new() -> Result<Self> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn matmul(
            &mut self,
            _m: usize,
            _n: usize,
            _k: usize,
            _a: &[f32],
            _b: &[f32],
            _transpose_b: bool,
            _static_rhs: bool,
            _c: &mut [f32],
        ) -> Result<()> {
            bail!("CUDA provider requires Linux and cargo feature `cuda`")
        }
    }
}

pub use enabled::{
    ensure_available, CudaContext, Lfm2Context, Lfm2LayerParams, ModernBertContext,
    ModernBertLayerParams, OpsContext, Qwen3Context, Qwen3LayerParams,
};
