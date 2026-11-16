//! Frozen per-field fallback planning and independent slot validation.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::client::{parse_one_json_value, SidecarResultError};
use super::renderer::{FrozenObjectSchema, FrozenProperty, SchemaValidationError};

/// Versioned inputs captured when a per-field fallback plan is frozen. These
/// values contribute to the plan digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerFieldFallbackContract {
    pub prompt_template_revision: String,
    pub max_new_tokens: u32,
    pub presence_representation_revision: String,
    pub validation_revision: String,
    pub joining_revision: String,
}

/// One top-level-property sidecar slot. Nested objects and arrays remain whole
/// values in this slot; the plan never recursively decomposes them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerFieldSlot {
    pub index: usize,
    pub property: FrozenProperty,
    /// Prompt text derived from the frozen property. Callers may embed it in a
    /// model prompt; frozen-plan validation detects later plan edits.
    pub prompt: String,
}

/// A schema-order-preserving, digest-bound per-field fallback plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerFieldPlan {
    pub schema_identity: String,
    pub contract: PerFieldFallbackContract,
    pub slots: Vec<PerFieldSlot>,
    pub frozen_digest: String,
}

impl PerFieldPlan {
    /// Freeze one slot per top-level property in declaration order.
    pub fn freeze(
        schema: &FrozenObjectSchema,
        contract: PerFieldFallbackContract,
    ) -> Result<Self, PerFieldPlanError> {
        schema.validate_definition()?;
        validate_contract(&contract)?;
        let slots = schema
            .properties
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, property)| PerFieldSlot {
                index,
                prompt: slot_prompt(&schema.schema_identity, &property),
                property,
            })
            .collect();
        let mut plan = Self {
            schema_identity: schema.schema_identity.clone(),
            contract,
            slots,
            frozen_digest: String::new(),
        };
        plan.frozen_digest = plan.compute_digest();
        Ok(plan)
    }

    /// Verify that a caller did not alter a plan after it was frozen.
    pub fn validate_frozen(&self) -> Result<(), PerFieldPlanError> {
        if self.schema_identity.trim().is_empty() {
            return Err(PerFieldPlanError::InvalidPlan(
                "schema identity must not be empty".to_string(),
            ));
        }
        validate_contract(&self.contract)?;
        let schema = FrozenObjectSchema {
            schema_identity: self.schema_identity.clone(),
            properties: self
                .slots
                .iter()
                .map(|slot| slot.property.clone())
                .collect(),
        };
        schema.validate_definition()?;
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.index != index {
                return Err(PerFieldPlanError::InvalidPlan(
                    "slot indexes must match frozen schema order".to_string(),
                ));
            }
            let expected_prompt = slot_prompt(&self.schema_identity, &slot.property);
            if slot.prompt != expected_prompt {
                return Err(PerFieldPlanError::InvalidPlan(
                    "slot prompt does not match the frozen property contract".to_string(),
                ));
            }
        }
        if self.frozen_digest != self.compute_digest() {
            return Err(PerFieldPlanError::DigestMismatch);
        }
        Ok(())
    }

    #[must_use]
    pub fn compute_digest(&self) -> String {
        #[derive(Serialize)]
        struct DigestPayload<'a> {
            schema_identity: &'a str,
            contract: &'a PerFieldFallbackContract,
            slots: &'a [PerFieldSlot],
        }

        let bytes = serde_json::to_vec(&DigestPayload {
            schema_identity: &self.schema_identity,
            contract: &self.contract,
            slots: &self.slots,
        })
        .expect("frozen per-field plan serializes");
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }
}

/// The normalized state of a single per-field response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state", deny_unknown_fields)]
pub enum PerFieldSlotResult {
    Usable { value: Value },
    Omitted,
    Invalid { reason: String },
}

/// All per-field results for one frozen plan. Slot failures are intentionally
/// retained independently so an invalid optional slot cannot erase valid slots.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerFieldResult {
    pub plan_digest: String,
    pub slots: Vec<PerFieldSlotResult>,
}

impl PerFieldResult {
    /// Decode each raw field response independently, preserving every slot's
    /// result rather than failing fast on one malformed completion.
    pub fn decode(plan: &PerFieldPlan, raw_slots: &[&[u8]]) -> Result<Self, PerFieldPlanError> {
        plan.validate_frozen()?;
        let slots = plan
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let Some(raw) = raw_slots.get(index) else {
                    return PerFieldSlotResult::Invalid {
                        reason: "slot did not return a result".to_string(),
                    };
                };
                decode_slot(slot, raw)
            })
            .collect();
        Ok(Self {
            plan_digest: plan.frozen_digest.clone(),
            slots,
        })
    }

    /// Join usable fields in frozen schema order. Invalid optional slots are
    /// omitted; an invalid required slot prevents the joined object from being
    /// returned, while retaining other independent slot outcomes for reporting.
    pub fn join_object(&self, plan: &PerFieldPlan) -> Result<Value, PerFieldPlanError> {
        plan.validate_frozen()?;
        if self.plan_digest != plan.frozen_digest {
            return Err(PerFieldPlanError::ResultPlanMismatch);
        }
        if self.slots.len() != plan.slots.len() {
            return Err(PerFieldPlanError::ResultSlotCountMismatch {
                expected: plan.slots.len(),
                actual: self.slots.len(),
            });
        }

        let mut joined = Map::new();
        for (slot, result) in plan.slots.iter().zip(&self.slots) {
            match result {
                PerFieldSlotResult::Usable { value } => {
                    joined.insert(slot.property.name.clone(), value.clone());
                }
                PerFieldSlotResult::Omitted if !slot.property.required => {}
                PerFieldSlotResult::Omitted => {
                    return Err(PerFieldPlanError::RequiredSlotInvalid {
                        property: slot.property.name.clone(),
                        reason: "required slot reported present=false".to_string(),
                    })
                }
                PerFieldSlotResult::Invalid { .. } if !slot.property.required => {}
                PerFieldSlotResult::Invalid { reason } => {
                    return Err(PerFieldPlanError::RequiredSlotInvalid {
                        property: slot.property.name.clone(),
                        reason: reason.clone(),
                    })
                }
            }
        }

        let object = Value::Object(joined);
        let schema = FrozenObjectSchema {
            schema_identity: plan.schema_identity.clone(),
            properties: plan
                .slots
                .iter()
                .map(|slot| slot.property.clone())
                .collect(),
        };
        schema.validate_value(&object)?;
        Ok(object)
    }
}

/// A frozen-plan construction, integrity, or per-field joining error.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PerFieldPlanError {
    #[error(transparent)]
    Schema(#[from] SchemaValidationError),
    #[error("per-field fallback requires a non-zero output limit")]
    ZeroOutputLimit,
    #[error("per-field fallback requires a non-empty {field}")]
    MissingContractRevision { field: &'static str },
    #[error("invalid frozen per-field plan: {0}")]
    InvalidPlan(String),
    #[error("frozen per-field plan digest does not match its content")]
    DigestMismatch,
    #[error("per-field result was produced for a different frozen plan")]
    ResultPlanMismatch,
    #[error("per-field result carried {actual} slots; frozen plan requires {expected}")]
    ResultSlotCountMismatch { expected: usize, actual: usize },
    #[error("required property '{property}' has no usable field result: {reason}")]
    RequiredSlotInvalid { property: String, reason: String },
}

fn decode_slot(slot: &PerFieldSlot, raw: &[u8]) -> PerFieldSlotResult {
    let parsed = parse_one_json_value(raw).and_then(|value| parse_envelope(&value));
    match parsed {
        Ok(SlotEnvelope {
            present: true,
            value,
        }) => match slot.property.schema.validate_value(&value) {
            Ok(()) => PerFieldSlotResult::Usable { value },
            Err(error) => PerFieldSlotResult::Invalid {
                reason: error.to_string(),
            },
        },
        Ok(SlotEnvelope {
            present: false,
            value,
        }) if value.is_null() && !slot.property.required => PerFieldSlotResult::Omitted,
        Ok(SlotEnvelope { present: false, .. }) if slot.property.required => {
            PerFieldSlotResult::Invalid {
                reason: "required slot reported present=false".to_string(),
            }
        }
        Ok(SlotEnvelope { .. }) => PerFieldSlotResult::Invalid {
            reason: "present=false envelope must carry a null value".to_string(),
        },
        Err(error) => PerFieldSlotResult::Invalid {
            reason: error.to_string(),
        },
    }
}

#[derive(Debug)]
struct SlotEnvelope {
    present: bool,
    value: Value,
}

fn parse_envelope(value: &Value) -> Result<SlotEnvelope, SidecarResultError> {
    let object = value.as_object().ok_or(SidecarResultError::NonObject)?;
    if object.len() != 2 || !object.contains_key("present") || !object.contains_key("value") {
        return Err(SidecarResultError::InvalidFieldEnvelope(
            "envelope must contain exactly present and value".to_string(),
        ));
    }
    let present = object
        .get("present")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            SidecarResultError::InvalidFieldEnvelope("present must be a boolean".to_string())
        })?;
    Ok(SlotEnvelope {
        present,
        value: object.get("value").expect("checked field exists").clone(),
    })
}

fn validate_contract(contract: &PerFieldFallbackContract) -> Result<(), PerFieldPlanError> {
    if contract.max_new_tokens == 0 {
        return Err(PerFieldPlanError::ZeroOutputLimit);
    }
    for (field, value) in [
        (
            "prompt_template_revision",
            &contract.prompt_template_revision,
        ),
        (
            "presence_representation_revision",
            &contract.presence_representation_revision,
        ),
        ("validation_revision", &contract.validation_revision),
        ("joining_revision", &contract.joining_revision),
    ] {
        if value.trim().is_empty() {
            return Err(PerFieldPlanError::MissingContractRevision { field });
        }
    }
    Ok(())
}

fn slot_prompt(schema_identity: &str, property: &FrozenProperty) -> String {
    let schema = serde_json::to_string(&property.schema).expect("frozen field schema serializes");
    format!(
        "schema={schema_identity};property={};required={};schema={schema};return={{\"present\":bool,\"value\":json}}",
        property.name, property.required
    )
}
