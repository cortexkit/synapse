pub(super) mod classify;
pub(super) mod client;
pub(super) mod openai_compat;
pub(super) mod validator;

#[cfg(test)]
pub mod mock;

use subc_client_rs::async_trait;
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ContinuityError {
    pub message: String,
}

impl ContinuityError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[async_trait]
pub trait ContinuityCheck: Send + Sync {
    async fn check(
        &self,
        request_digest: &str,
        synapse_model_id: &str,
        logical_handle: Option<&str>,
    ) -> Result<(), ContinuityError>;
}

pub struct NoopContinuityCheck;

#[async_trait]
impl ContinuityCheck for NoopContinuityCheck {
    async fn check(
        &self,
        _request_digest: &str,
        _synapse_model_id: &str,
        _logical_handle: Option<&str>,
    ) -> Result<(), ContinuityError> {
        Ok(())
    }
}
