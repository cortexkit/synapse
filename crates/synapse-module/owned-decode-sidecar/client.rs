//! Pure sidecar completion normalization.
//!
//! This module deliberately contains no transport or routing launch. It converts
//! completed bytes into validated semantic objects so an integration layer can
//! publish a bank without giving a sidecar direct access to target decode state.

use std::str;

use serde_json::Value;
use thiserror::Error;

use crate::renderer::{FrozenObjectSchema, RenderError, RenderPolicy, SchemaValidationError};
use crate::slotter::{PerFieldPlan, PerFieldPlanError, PerFieldResult};

/// A validated sidecar value together with its deterministic runtime rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSidecarResult {
    pub value: Value,
    pub rendered_view: String,
}

/// Validate and render one whole-object sidecar completion.
pub fn prepare_whole_object(
    schema: &FrozenObjectSchema,
    raw: &[u8],
    policy: &RenderPolicy,
) -> Result<PreparedSidecarResult, SidecarResultError> {
    let value = parse_one_json_value(raw)?;
    if !value.is_object() {
        return Err(SidecarResultError::NonObject);
    }
    schema.validate_value(&value)?;
    let rendered_view = crate::renderer::render_object(schema, &value, policy)?;
    Ok(PreparedSidecarResult {
        value,
        rendered_view,
    })
}

/// Validate independent per-field completions, join usable fields in frozen order,
/// and render the result using the same runtime-owned layout as whole-object mode.
pub fn prepare_per_field(
    schema: &FrozenObjectSchema,
    plan: &PerFieldPlan,
    raw_slots: &[&[u8]],
    policy: &RenderPolicy,
) -> Result<(PreparedSidecarResult, PerFieldResult), SidecarResultError> {
    if plan.schema_identity != schema.schema_identity {
        return Err(SidecarResultError::SchemaIdentityMismatch);
    }
    let slot_results = PerFieldResult::decode(plan, raw_slots)?;
    let value = slot_results.join_object(plan)?;
    schema.validate_value(&value)?;
    let rendered_view = crate::renderer::render_object(schema, &value, policy)?;
    Ok((
        PreparedSidecarResult {
            value,
            rendered_view,
        },
        slot_results,
    ))
}

/// Parse exactly one JSON value with optional trailing JSON whitespace.
///
/// Validate UTF-8 before JSON parsing so invalid bytes produce `InvalidUtf8`
/// rather than being replaced or reported as a JSON error.
pub(crate) fn parse_one_json_value(raw: &[u8]) -> Result<Value, SidecarResultError> {
    let text = str::from_utf8(raw).map_err(|_| SidecarResultError::InvalidUtf8)?;
    if text.trim().is_empty() {
        return Err(SidecarResultError::EmptyOutput);
    }
    serde_json::from_str(text).map_err(|error| SidecarResultError::Json(error.to_string()))
}

/// Whole-object or per-field result normalization failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SidecarResultError {
    #[error("sidecar output was not valid UTF-8")]
    InvalidUtf8,
    #[error("sidecar output was empty")]
    EmptyOutput,
    #[error("sidecar output was not exactly one JSON value: {0}")]
    Json(String),
    #[error("sidecar output must be a JSON object")]
    NonObject,
    #[error("invalid per-field envelope: {0}")]
    InvalidFieldEnvelope(String),
    #[error("sidecar result schema identity did not match the frozen field plan")]
    SchemaIdentityMismatch,
    #[error(transparent)]
    Validation(#[from] SchemaValidationError),
    #[error(transparent)]
    Render(#[from] RenderError),
    #[error(transparent)]
    PerField(#[from] PerFieldPlanError),
}
