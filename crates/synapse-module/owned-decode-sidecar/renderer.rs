//! Runtime-owned validation and deterministic JSON rendering.
//!
//! A sidecar supplies semantic values only. Property names, property order,
//! delimiters, and whitespace all come from a frozen schema plus render policy so
//! model-produced structural text cannot affect target-lane tokenization.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// The frozen top-level object schema used by sidecar result handling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenObjectSchema {
    pub schema_identity: String,
    /// The declaration order is the only traversal order used for rendering,
    /// field planning, and deterministic prompt construction.
    pub properties: Vec<FrozenProperty>,
}

impl FrozenObjectSchema {
    /// Validate a semantic object against the frozen schema.
    pub fn validate_value(&self, value: &Value) -> Result<(), SchemaValidationError> {
        self.validate_definition()?;
        validate_object(&self.properties, value, "$")
    }

    /// Reject malformed frozen schemas before they can define an ambiguous
    /// traversal order or a silently permissive validation rule.
    pub fn validate_definition(&self) -> Result<(), SchemaValidationError> {
        if self.schema_identity.trim().is_empty() {
            return Err(SchemaValidationError::InvalidSchema {
                path: "$".to_string(),
                message: "schema identity must not be empty".to_string(),
            });
        }
        validate_properties(&self.properties, "$")
    }

    /// Return the schema property with this name.
    #[must_use]
    pub fn property(&self, name: &str) -> Option<&FrozenProperty> {
        self.properties
            .iter()
            .find(|property| property.name == name)
    }
}

/// One named property of a frozen object schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenProperty {
    pub name: String,
    pub required: bool,
    pub schema: FrozenSchema,
}

/// The supported semantic schema nodes for runtime validation and rendering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrozenSchema {
    Object {
        properties: Vec<FrozenProperty>,
    },
    Array {
        items: Box<FrozenSchema>,
    },
    Scalar {
        scalar_type: FrozenScalarType,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        enumeration: Option<Vec<Value>>,
    },
}

impl FrozenSchema {
    /// Validate one semantic value against this node, including the node's own
    /// enum and nested-schema definition. Per-field fallback uses this without
    /// inventing a second validator for top-level property values.
    pub fn validate_value(&self, value: &Value) -> Result<(), SchemaValidationError> {
        validate_schema(self, "$")?;
        validate_node(self, value, "$")
    }
}

/// JSON scalar type accepted by a [`FrozenSchema::Scalar`] node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrozenScalarType {
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

/// Target-lane whitespace decisions frozen before prompt tuning or measurement.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderPolicy {
    #[serde(default)]
    pub leading_whitespace: String,
    #[serde(default)]
    pub colon_whitespace: String,
    #[serde(default)]
    pub comma_whitespace: String,
    #[serde(default)]
    pub trailing_whitespace: String,
}

impl RenderPolicy {
    /// Return the default compact JSON layout with no additional whitespace.
    #[must_use]
    pub fn compact() -> Self {
        Self::default()
    }

    /// Canonically identify the frozen layout. Struct field order is fixed, so
    /// the serialized bytes do not depend on a map iteration order.
    #[must_use]
    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("render policy serializes");
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    fn validate(&self) -> Result<(), RenderError> {
        for (name, text) in [
            ("leading_whitespace", &self.leading_whitespace),
            ("colon_whitespace", &self.colon_whitespace),
            ("comma_whitespace", &self.comma_whitespace),
            ("trailing_whitespace", &self.trailing_whitespace),
        ] {
            if !text.chars().all(char::is_whitespace) {
                return Err(RenderError::InvalidPolicy {
                    field: name,
                    message: "layout fields may contain whitespace only".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Runtime render or schema-validation failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RenderError {
    #[error("invalid render policy {field}: {message}")]
    InvalidPolicy {
        field: &'static str,
        message: String,
    },
    #[error(transparent)]
    Validation(#[from] SchemaValidationError),
}

/// A frozen-schema definition or semantic-value validation failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SchemaValidationError {
    #[error("invalid frozen schema at {path}: {message}")]
    InvalidSchema { path: String, message: String },
    #[error("invalid sidecar value at {path}: {message}")]
    InvalidValue { path: String, message: String },
}

/// Render a validated top-level object using only the frozen schema and policy.
pub fn render_object(
    schema: &FrozenObjectSchema,
    value: &Value,
    policy: &RenderPolicy,
) -> Result<String, RenderError> {
    policy.validate()?;
    schema.validate_value(value)?;

    let mut rendered = String::new();
    rendered.push_str(&policy.leading_whitespace);
    render_object_contents(&schema.properties, value, policy, &mut rendered);
    rendered.push_str(&policy.trailing_whitespace);
    Ok(rendered)
}

fn render_value(
    schema: &FrozenSchema,
    value: &Value,
    policy: &RenderPolicy,
    rendered: &mut String,
) {
    match schema {
        FrozenSchema::Object { properties } => {
            render_object_contents(properties, value, policy, rendered)
        }
        FrozenSchema::Array { items } => {
            rendered.push('[');
            let values = value.as_array().expect("value was validated as an array");
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    rendered.push(',');
                    rendered.push_str(&policy.comma_whitespace);
                }
                render_value(items, item, policy, rendered);
            }
            rendered.push(']');
        }
        FrozenSchema::Scalar { .. } => {
            // serde_json provides correct JSON escaping and number spelling for a
            // scalar. Objects and arrays are handled above because their order is
            // owned by the frozen schema rather than the model-produced map.
            rendered.push_str(&serde_json::to_string(value).expect("validated scalar serializes"));
        }
    }
}

fn render_object_contents(
    properties: &[FrozenProperty],
    value: &Value,
    policy: &RenderPolicy,
    rendered: &mut String,
) {
    let object = value.as_object().expect("value was validated as an object");
    rendered.push('{');
    let mut first = true;
    for property in properties {
        let Some(child) = object.get(&property.name) else {
            continue;
        };
        if !first {
            rendered.push(',');
            rendered.push_str(&policy.comma_whitespace);
        }
        first = false;
        rendered.push_str(
            &serde_json::to_string(&property.name).expect("schema property name serializes"),
        );
        rendered.push(':');
        rendered.push_str(&policy.colon_whitespace);
        render_value(&property.schema, child, policy, rendered);
    }
    rendered.push('}');
}

fn validate_properties(
    properties: &[FrozenProperty],
    path: &str,
) -> Result<(), SchemaValidationError> {
    let mut names = BTreeSet::new();
    for property in properties {
        if property.name.is_empty() {
            return Err(invalid_schema(path, "property names must not be empty"));
        }
        if !names.insert(&property.name) {
            return Err(invalid_schema(
                path,
                format!("duplicate property '{}'", property.name),
            ));
        }
        validate_schema(&property.schema, &format!("{path}.{}", property.name))?;
    }
    Ok(())
}

fn validate_schema(schema: &FrozenSchema, path: &str) -> Result<(), SchemaValidationError> {
    match schema {
        FrozenSchema::Object { properties } => validate_properties(properties, path),
        FrozenSchema::Array { items } => validate_schema(items, &format!("{path}[]")),
        FrozenSchema::Scalar {
            scalar_type,
            enumeration,
        } => {
            if let Some(values) = enumeration {
                if values.is_empty() {
                    return Err(invalid_schema(path, "scalar enum must not be empty"));
                }
                for value in values {
                    if !matches_scalar_type(*scalar_type, value) {
                        return Err(invalid_schema(
                            path,
                            "scalar enum member does not match its scalar type",
                        ));
                    }
                }
            }
            Ok(())
        }
    }
}

fn validate_object(
    properties: &[FrozenProperty],
    value: &Value,
    path: &str,
) -> Result<(), SchemaValidationError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_value(path, "expected an object"))?;
    for property in properties {
        match object.get(&property.name) {
            Some(child) => validate_node(
                &property.schema,
                child,
                &format!("{path}.{}", property.name),
            )?,
            None if property.required => {
                return Err(invalid_value(
                    path,
                    format!("missing required property '{}'", property.name),
                ))
            }
            None => {}
        }
    }
    for name in object.keys() {
        if !properties.iter().any(|property| property.name == *name) {
            return Err(invalid_value(path, format!("unknown property '{name}'")));
        }
    }
    Ok(())
}

fn validate_node(
    schema: &FrozenSchema,
    value: &Value,
    path: &str,
) -> Result<(), SchemaValidationError> {
    match schema {
        FrozenSchema::Object { properties } => validate_object(properties, value, path),
        FrozenSchema::Array { items } => {
            let values = value
                .as_array()
                .ok_or_else(|| invalid_value(path, "expected an array"))?;
            for (index, item) in values.iter().enumerate() {
                validate_node(items, item, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        FrozenSchema::Scalar {
            scalar_type,
            enumeration,
        } => {
            if !matches_scalar_type(*scalar_type, value) {
                return Err(invalid_value(
                    path,
                    format!("expected {}", scalar_type_name(*scalar_type)),
                ));
            }
            if let Some(values) = enumeration {
                if !values.iter().any(|member| member == value) {
                    return Err(invalid_value(path, "value is not in the schema enum"));
                }
            }
            Ok(())
        }
    }
}

fn matches_scalar_type(scalar_type: FrozenScalarType, value: &Value) -> bool {
    match scalar_type {
        FrozenScalarType::String => value.is_string(),
        FrozenScalarType::Number => value.is_number(),
        FrozenScalarType::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
        FrozenScalarType::Boolean => value.is_boolean(),
        FrozenScalarType::Null => value.is_null(),
    }
}

fn scalar_type_name(scalar_type: FrozenScalarType) -> &'static str {
    match scalar_type {
        FrozenScalarType::String => "string",
        FrozenScalarType::Number => "number",
        FrozenScalarType::Integer => "integer",
        FrozenScalarType::Boolean => "boolean",
        FrozenScalarType::Null => "null",
    }
}

fn invalid_schema(path: &str, message: impl Into<String>) -> SchemaValidationError {
    SchemaValidationError::InvalidSchema {
        path: path.to_string(),
        message: message.into(),
    }
}

fn invalid_value(path: &str, message: impl Into<String>) -> SchemaValidationError {
    SchemaValidationError::InvalidValue {
        path: path.to_string(),
        message: message.into(),
    }
}
