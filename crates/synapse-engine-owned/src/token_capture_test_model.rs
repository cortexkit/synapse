use super::*;

struct TokenCaptureFamily {
    policy: TokenizerPolicy,
    captured: Arc<Mutex<Vec<Vec<u32>>>>,
}

impl runtime::ModelFamily for TokenCaptureFamily {
    fn family_name(&self) -> &'static str {
        "token-capture"
    }
    fn output_dim(&self) -> usize {
        1
    }
    fn tokenizer_policy(&self) -> runtime::FamilyTokenizerPolicy {
        runtime::FamilyTokenizerPolicy {
            pad_token_id: self.policy.pad_token_id,
            terminal_token_id: self.policy.terminal_token_id,
        }
    }
    fn supports_rerank(&self) -> bool {
        true
    }
    fn embed_batch(
        &self,
        _provider: &mut dyn runtime::KernelProvider,
        sequences: &[Vec<u32>],
        _shape: Option<runtime::BatchShape>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.captured.lock().unwrap().extend_from_slice(sequences);
        Ok(vec![vec![1.0]; sequences.len()])
    }
    fn rerank_batch(
        &self,
        _provider: &mut dyn runtime::KernelProvider,
        sequences: &[Vec<u32>],
        _shape: Option<runtime::BatchShape>,
    ) -> anyhow::Result<Vec<f32>> {
        self.captured.lock().unwrap().extend_from_slice(sequences);
        Ok(vec![0.5; sequences.len()])
    }
}

impl OwnedMetalEmbedEngine {
    /// Captures the sequences handed to a model family after normal terminal
    /// validation and bucket planning, without executing neural-network kernels.
    pub fn insert_token_capture_test_model(
        &mut self,
        model_id: String,
        policy: TokenizerPolicy,
        captured: Arc<Mutex<Vec<Vec<u32>>>>,
    ) -> LoadedModel {
        let config =
            runtime::MetalExecutionConfig::new(runtime::Execution::Explicit, None).unwrap();
        let provider =
            runtime::MetalProvider::new_with_config(precision(self.dtype), config).unwrap();
        self.models.insert(
            model_id.clone(),
            Arc::new(Mutex::new(OwnedLoadedModel {
                family: Box::new(TokenCaptureFamily { policy, captured }),
                provider,
                buckets: [64, 128, 256, 512, 1024, 8192]
                    .into_iter()
                    .map(|seq| runtime::BatchShape { batch: 1, seq })
                    .collect(),
                tokenizer_policy: policy,
            })),
        );
        LoadedModel { model_id }
    }
}
