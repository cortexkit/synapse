//! Parser and validator for the `synapse-json-schema-v1` grammar subset.
//!
//! The grammar contract accepts a strict subset of JSON Schema 2020-12:
//! `$schema`, `type`, `properties`, `required`, `additionalProperties`,
//! `items`, and `enum`. `type` is mandatory on every node and is exactly one of
//! `object`, `array`, `string`, `number`, `integer`, `boolean`, or `null`
//! (type arrays are rejected). Object schemas must set `additionalProperties`
//! to `false`; array schemas must carry exactly one `items` schema (tuple form
//! is rejected). Every other keyword — `$ref`, combinators, conditionals,
//! pattern/regex, numeric and string bounds, tuple arrays, and unknown
//! keywords — is rejected rather than ignored.
//!
//! Validation produces a compact [`Schema`] arena that the compiler
//! ([`crate::owned_decode_grammar_scheduler::grammar_compile`]) turns into a
//! token-ID automaton. Two stable error classes are distinguished per the error
//! contract: malformed JSON or malformed schema *structure* is
//! `grammar_parse_failed`; a schema that is well-formed but outside the
//! accepted subset or its checked-in limits is `grammar_feature_unsupported`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::owned_decode_grammar_scheduler::grammar_limits::GrammarLimits;
use crate::owned_decode_routing::error::OwnedDecodeError;

/// The accepted schema subset, compiled into an arena of nodes.
///
/// Node index zero is always the root. Container nodes reference child nodes by
/// index, so the whole schema is a small, cloneable, deterministic value that
/// can be digested for identity purposes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    nodes: Vec<SchemaNode>,
}

impl Schema {
    /// The root node (index zero).
    pub fn root(&self) -> &SchemaNode {
        &self.nodes[0]
    }

    /// Look up a node by index.
    pub fn node(&self, index: usize) -> &SchemaNode {
        &self.nodes[index]
    }

    /// Total number of nodes in the arena. Used by the compiler to enforce the
    /// compiled-state limit and by tests to assert deterministic output.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// The declared JSON type of a schema node. `type` is mandatory in the subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

impl SchemaType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "object" => Some(Self::Object),
            "array" => Some(Self::Array),
            "string" => Some(Self::String),
            "number" => Some(Self::Number),
            "integer" => Some(Self::Integer),
            "boolean" => Some(Self::Boolean),
            "null" => Some(Self::Null),
            _ => None,
        }
    }

    /// Whether this type admits a scalar JSON literal (everything except the two
    /// container types). Enum literals are only meaningful on scalar types.
    pub fn is_scalar(self) -> bool {
        !matches!(self, Self::Object | Self::Array)
    }
}

/// One node in the schema arena.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchemaNode {
    pub ty: SchemaType,
    pub kind: NodeKind,
}

/// The type-specific payload of a schema node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    /// An object schema. `additionalProperties:false` is enforced at parse time,
    /// so `properties` is the complete, closed set of allowed keys.
    Object {
        /// Property name to child-node index, in declaration order.
        properties: Vec<(String, usize)>,
        /// The subset of property names that must appear.
        required: BTreeSet<String>,
    },
    /// An array schema with exactly one item schema (tuple form is rejected).
    Array { items: usize },
    /// A scalar schema, optionally restricted to a finite enum of literals.
    Scalar {
        enumeration: Option<Vec<EnumLiteral>>,
    },
}

impl SchemaNode {
    /// For an object node, the child node index of a named property.
    pub fn property_node(&self, name: &str) -> Option<usize> {
        match &self.kind {
            NodeKind::Object { properties, .. } => properties
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, index)| *index),
            _ => None,
        }
    }
}

/// A finite enum literal, typed to match its declaring node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EnumLiteral {
    Str(String),
    /// A JSON number that is not necessarily integral.
    Number(f64),
    /// An integral JSON number.
    Integer(i64),
    Bool(bool),
    Null,
}

impl EnumLiteral {
    /// The rendered JSON text of this literal, used by the automaton to restrict
    /// string/number/boolean/null output to the enum members.
    pub fn json_text(&self) -> String {
        match self {
            EnumLiteral::Str(text) => {
                // serde_json's string serialization emits exactly the JSON escape
                // sequences the grammar contract requires for string output.
                serde_json::to_string(text).expect("string literal serializes")
            }
            EnumLiteral::Number(value) => format_number(*value),
            EnumLiteral::Integer(value) => value.to_string(),
            EnumLiteral::Bool(value) => value.to_string(),
            EnumLiteral::Null => "null".to_string(),
        }
    }
}

/// Format a finite f64 the way JSON renders numbers, avoiding Rust's default
/// scientific notation for values that JSON would write plainly.
fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        serde_json::Number::from_f64(value)
            .map(|number| number.to_string())
            .unwrap_or_else(|| value.to_string())
    }
}

/// A schema parse or validation failure, tagged with the stable wire error it
/// maps to and a human-readable reason for diagnostics and tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaError {
    /// `grammar_parse_failed` for malformed JSON/structure, or
    /// `grammar_feature_unsupported` for an out-of-subset or over-limit schema.
    pub kind: OwnedDecodeError,
    pub message: String,
}

impl SchemaError {
    fn parse(message: impl Into<String>) -> Self {
        Self {
            kind: OwnedDecodeError::GrammarParseFailed,
            message: message.into(),
        }
    }

    pub(super) fn feature(message: impl Into<String>) -> Self {
        Self {
            kind: OwnedDecodeError::GrammarFeatureUnsupported,
            message: message.into(),
        }
    }

    /// The stable wire error ID for this failure.
    pub fn wire_error(&self) -> OwnedDecodeError {
        self.kind
    }
}

/// The exact keyword set accepted by the subset. Any object key outside this set
/// is rejected as an unsupported feature.
const ACCEPTED_KEYWORDS: &[&str] = &[
    "$schema",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
];

/// Parse a raw grammar string into a validated [`Schema`].
///
/// The string must be valid JSON (else `grammar_parse_failed`) encoding a schema
/// in the accepted subset within `limits` (else `grammar_feature_unsupported`).
pub fn parse_schema(raw: &str, limits: &GrammarLimits) -> Result<Schema, SchemaError> {
    if raw.len() > limits.max_schema_bytes {
        return Err(SchemaError::feature(format!(
            "schema is {} bytes, exceeding the {} byte limit",
            raw.len(),
            limits.max_schema_bytes
        )));
    }
    let value: Value = serde_json::from_str(raw)
        .map_err(|err| SchemaError::parse(format!("invalid JSON: {err}")))?;
    let mut builder = SchemaBuilder {
        nodes: Vec::new(),
        limits,
        property_count: 0,
        enum_count: 0,
    };
    let root = builder.build_node(&value, 1)?;
    debug_assert_eq!(root, 0, "the first built node is the root");
    Ok(Schema {
        nodes: builder.nodes,
    })
}

struct SchemaBuilder<'a> {
    nodes: Vec<SchemaNode>,
    limits: &'a GrammarLimits,
    property_count: usize,
    enum_count: usize,
}

impl<'a> SchemaBuilder<'a> {
    /// Build a node from a JSON schema fragment at the given nesting depth
    /// (root is depth one). Returns the node's arena index.
    fn build_node(&mut self, value: &Value, depth: usize) -> Result<usize, SchemaError> {
        if depth > self.limits.max_nesting_depth {
            return Err(SchemaError::feature(format!(
                "schema nesting depth exceeds the {} limit",
                self.limits.max_nesting_depth
            )));
        }
        let object = value
            .as_object()
            .ok_or_else(|| SchemaError::parse("a schema node must be a JSON object"))?;

        // Reject any keyword outside the accepted subset before interpreting the
        // node, so unknown and reference keywords fail closed rather than being
        // silently ignored.
        for key in object.keys() {
            if !ACCEPTED_KEYWORDS.contains(&key.as_str()) {
                return Err(SchemaError::feature(format!(
                    "keyword '{key}' is not in the synapse-json-schema-v1 subset"
                )));
            }
        }

        if let Some(schema_decl) = object.get("$schema") {
            let dialect = schema_decl
                .as_str()
                .ok_or_else(|| SchemaError::parse("$schema must be a string dialect URI"))?;
            if dialect
                != crate::owned_decode_grammar_scheduler::grammar_limits::JSON_SCHEMA_DIALECT_2020_12
            {
                return Err(SchemaError::feature(format!(
                    "unsupported $schema dialect '{dialect}'"
                )));
            }
        }

        // `type` is mandatory and must be a single string (type arrays rejected).
        let type_value = object
            .get("type")
            .ok_or_else(|| SchemaError::feature("schema node is missing the required 'type'"))?;
        let type_str = type_value.as_str().ok_or_else(|| {
            SchemaError::feature("type must be a single string; type arrays are rejected")
        })?;
        let ty = SchemaType::parse(type_str)
            .ok_or_else(|| SchemaError::feature(format!("unknown type '{type_str}'")))?;

        // Reserve this node's slot before building children so the parent always
        // lands at a lower index than its children and the root is index zero.
        let index = self.nodes.len();
        self.nodes.push(SchemaNode {
            ty: SchemaType::Null,
            kind: NodeKind::Scalar { enumeration: None },
        });
        let kind = match ty {
            SchemaType::Object => self.build_object(object, depth)?,
            SchemaType::Array => self.build_array(object, depth)?,
            scalar => self.build_scalar(scalar, object)?,
        };
        self.nodes[index] = SchemaNode { ty, kind };
        Ok(index)
    }

    fn build_object(
        &mut self,
        object: &serde_json::Map<String, Value>,
        depth: usize,
    ) -> Result<NodeKind, SchemaError> {
        // Keywords that are only meaningful on other node types must not leak
        // onto an object node: the subset rejects them rather than silently
        // ignoring them (a silently dropped `enum` would enforce nothing).
        for key in ["items", "enum"] {
            if object.contains_key(key) {
                return Err(SchemaError::feature(format!(
                    "keyword '{key}' is not valid on an object node"
                )));
            }
        }

        // Objects must explicitly close themselves: additionalProperties must be
        // present and false. Absent or true is rejected.
        match object.get("additionalProperties") {
            Some(Value::Bool(false)) => {}
            Some(_) => {
                return Err(SchemaError::feature(
                    "object schemas must set additionalProperties to false",
                ))
            }
            None => {
                return Err(SchemaError::feature(
                    "object schemas must explicitly set additionalProperties to false",
                ))
            }
        }

        let mut properties = Vec::new();
        if let Some(props_value) = object.get("properties") {
            let props_object = props_value
                .as_object()
                .ok_or_else(|| SchemaError::parse("'properties' must be a JSON object"))?;
            for (name, sub_schema) in props_object {
                self.property_count += 1;
                if self.property_count > self.limits.max_property_count {
                    return Err(SchemaError::feature(format!(
                        "total property count exceeds the {} limit",
                        self.limits.max_property_count
                    )));
                }
                // Build the child first; its returned index is correct because this
                // node's slot is already reserved, so children land at higher indices.
                let child_index = self.build_node(sub_schema, depth + 1)?;
                properties.push((name.clone(), child_index));
            }
        }

        let mut required = BTreeSet::new();
        if let Some(required_value) = object.get("required") {
            let required_array = required_value
                .as_array()
                .ok_or_else(|| SchemaError::parse("'required' must be a JSON array"))?;
            for entry in required_array {
                let name = entry
                    .as_str()
                    .ok_or_else(|| SchemaError::parse("'required' entries must be strings"))?;
                if !properties.iter().any(|(key, _)| key == name) {
                    return Err(SchemaError::feature(format!(
                        "required property '{name}' is not declared in 'properties'"
                    )));
                }
                required.insert(name.to_string());
            }
        }

        Ok(NodeKind::Object {
            properties,
            required,
        })
    }

    fn build_array(
        &mut self,
        object: &serde_json::Map<String, Value>,
        depth: usize,
    ) -> Result<NodeKind, SchemaError> {
        // Keywords that are only meaningful on other node types must not leak
        // onto an array node; the subset rejects them rather than ignoring them.
        for key in ["properties", "required", "additionalProperties", "enum"] {
            if object.contains_key(key) {
                return Err(SchemaError::feature(format!(
                    "keyword '{key}' is not valid on an array node"
                )));
            }
        }

        let items_value = object
            .get("items")
            .ok_or_else(|| SchemaError::feature("array schemas must contain one 'items' schema"))?;
        if items_value.is_array() {
            return Err(SchemaError::feature(
                "tuple-form 'items' arrays are rejected; use a single item schema",
            ));
        }
        let child_index = self.build_node(items_value, depth + 1)?;
        Ok(NodeKind::Array { items: child_index })
    }

    fn build_scalar(
        &mut self,
        ty: SchemaType,
        object: &serde_json::Map<String, Value>,
    ) -> Result<NodeKind, SchemaError> {
        // Container-only keywords must not leak onto scalar nodes.
        for key in ["properties", "required", "additionalProperties", "items"] {
            if object.contains_key(key) {
                return Err(SchemaError::feature(format!(
                    "keyword '{key}' is not valid on a {ty:?} node"
                )));
            }
        }

        let enumeration = match object.get("enum") {
            None => None,
            Some(Value::Array(literals)) => {
                if literals.is_empty() {
                    return Err(SchemaError::feature(
                        "enum must contain at least one literal",
                    ));
                }
                let mut parsed = Vec::with_capacity(literals.len());
                for literal in literals {
                    self.enum_count += 1;
                    if self.enum_count > self.limits.max_enum_count {
                        return Err(SchemaError::feature(format!(
                            "total enum literal count exceeds the {} limit",
                            self.limits.max_enum_count
                        )));
                    }
                    parsed.push(match_literal_to_type(literal, ty)?);
                }
                Some(parsed)
            }
            Some(_) => {
                return Err(SchemaError::parse(
                    "'enum' must be a JSON array of literals",
                ));
            }
        };

        Ok(NodeKind::Scalar { enumeration })
    }
}

/// Coerce a raw enum literal to the declared scalar type, rejecting mismatches.
fn match_literal_to_type(literal: &Value, ty: SchemaType) -> Result<EnumLiteral, SchemaError> {
    match ty {
        SchemaType::String => literal
            .as_str()
            .map(|text| EnumLiteral::Str(text.to_string()))
            .ok_or_else(|| SchemaError::feature("string enum literal is not a JSON string")),
        SchemaType::Number => literal
            .as_f64()
            .map(EnumLiteral::Number)
            .ok_or_else(|| SchemaError::feature("number enum literal is not a JSON number")),
        SchemaType::Integer => literal.as_i64().map(EnumLiteral::Integer).ok_or_else(|| {
            SchemaError::feature("integer enum literal is not an integral JSON number")
        }),
        SchemaType::Boolean => literal
            .as_bool()
            .map(EnumLiteral::Bool)
            .ok_or_else(|| SchemaError::feature("boolean enum literal is not a JSON boolean")),
        SchemaType::Null => {
            if literal.is_null() {
                Ok(EnumLiteral::Null)
            } else {
                Err(SchemaError::feature("null enum literal is not JSON null"))
            }
        }
        SchemaType::Object | SchemaType::Array => Err(SchemaError::feature(
            "enum is supported only on scalar types in this subset",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> GrammarLimits {
        GrammarLimits::default()
    }

    #[test]
    fn accepts_object_with_required_and_closed_properties() {
        let raw = r#"{
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"],
            "additionalProperties": false
        }"#;
        let schema = parse_schema(raw, &limits()).expect("valid object schema parses");
        assert_eq!(schema.root().ty, SchemaType::Object);
        assert_eq!(schema.node_count(), 3);
        let name = schema.root().property_node("name").expect("name property");
        assert_eq!(schema.node(name).ty, SchemaType::String);
    }

    #[test]
    fn accepts_array_with_single_items() {
        let raw = r#"{ "type": "array", "items": { "type": "string" } }"#;
        let schema = parse_schema(raw, &limits()).expect("valid array schema parses");
        assert_eq!(schema.root().ty, SchemaType::Array);
        match &schema.root().kind {
            NodeKind::Array { items } => assert_eq!(schema.node(*items).ty, SchemaType::String),
            other => panic!("expected array node, got {other:?}"),
        }
    }

    #[test]
    fn accepts_string_enum() {
        let raw = r#"{ "type": "string", "enum": ["red", "green", "blue"] }"#;
        let schema = parse_schema(raw, &limits()).expect("string enum parses");
        match &schema.root().kind {
            NodeKind::Scalar { enumeration } => {
                let enumeration = enumeration.as_ref().expect("enum present");
                assert_eq!(enumeration.len(), 3);
                assert_eq!(enumeration[0].json_text(), "\"red\"");
            }
            other => panic!("expected scalar node, got {other:?}"),
        }
    }

    #[test]
    fn rejects_malformed_json_as_parse_failed() {
        let error = parse_schema("{ not json", &limits()).expect_err("malformed JSON rejected");
        assert_eq!(error.wire_error(), OwnedDecodeError::GrammarParseFailed);
    }

    #[test]
    fn rejects_missing_type_as_feature_unsupported() {
        let raw = r#"{ "properties": {} }"#;
        let error = parse_schema(raw, &limits()).expect_err("missing type rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn rejects_type_arrays() {
        let raw = r#"{ "type": ["string", "null"] }"#;
        let error = parse_schema(raw, &limits()).expect_err("type array rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn rejects_object_without_additional_properties_false() {
        let raw = r#"{ "type": "object", "properties": {} }"#;
        let error = parse_schema(raw, &limits()).expect_err("absent additionalProperties rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );

        let raw_true = r#"{ "type": "object", "properties": {}, "additionalProperties": true }"#;
        let error =
            parse_schema(raw_true, &limits()).expect_err("true additionalProperties rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn rejects_tuple_form_items() {
        let raw = r#"{ "type": "array", "items": [{ "type": "string" }, { "type": "integer" }] }"#;
        let error = parse_schema(raw, &limits()).expect_err("tuple items rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn rejects_unknown_and_reference_keywords() {
        for raw in [
            r#"{ "type": "string", "pattern": "^a" }"#,
            r#"{ "type": "string", "minLength": 1 }"#,
            r##"{ "$ref": "#/definitions/x" }"##,
            r#"{ "type": "string", "allOf": [] }"#,
            r#"{ "type": "string", "anyOf": [] }"#,
            r#"{ "type": "string", "oneOf": [] }"#,
            r#"{ "type": "string", "not": {} }"#,
            r#"{ "type": "string", "description": "annotation" }"#,
        ] {
            let error = parse_schema(raw, &limits()).expect_err("unsupported keyword rejected");
            assert_eq!(
                error.wire_error(),
                OwnedDecodeError::GrammarFeatureUnsupported,
                "raw: {raw}"
            );
        }
    }

    #[test]
    fn rejects_enum_literal_type_mismatch() {
        let raw = r#"{ "type": "integer", "enum": [1, "two"] }"#;
        let error = parse_schema(raw, &limits()).expect_err("mismatched enum literal rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn rejects_required_property_not_declared() {
        let raw = r#"{
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": ["b"],
            "additionalProperties": false
        }"#;
        let error = parse_schema(raw, &limits()).expect_err("undeclared required rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn enforces_nesting_depth_limit() {
        // Build a chain of arrays deeper than the limit.
        let mut raw = String::from(r#"{ "type": "string" }"#);
        for _ in 0..40 {
            raw = format!(r#"{{ "type": "array", "items": {raw} }}"#);
        }
        let error = parse_schema(&raw, &limits()).expect_err("over-deep schema rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn enforces_schema_byte_limit() {
        let tiny = GrammarLimits {
            max_schema_bytes: 8,
            ..GrammarLimits::default()
        };
        let raw = r#"{ "type": "string" }"#;
        let error = parse_schema(raw, &tiny).expect_err("oversized schema rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn rejects_cross_node_keywords_on_object_and_array() {
        // In-subset keywords that are not meaningful on the node type are
        // rejected rather than silently dropped (audit probes).
        let probes = [
            // `enum` on an object node: previously compiled and enforced
            // nothing, letting non-member documents through.
            r#"{
                "type": "object",
                "properties": { "a": { "type": "integer" } },
                "required": ["a"],
                "additionalProperties": false,
                "enum": [{"a": 1}]
            }"#,
            // `enum` on an array node.
            r#"{ "type": "array", "items": { "type": "string" }, "enum": [["a"]] }"#,
            // `items` on an object node.
            r#"{
                "type": "object",
                "properties": {},
                "additionalProperties": false,
                "items": { "type": "string" }
            }"#,
            // Object keywords on an array node.
            r#"{
                "type": "array",
                "items": { "type": "string" },
                "properties": {},
                "required": [],
                "additionalProperties": false
            }"#,
        ];
        for raw in probes {
            let error = parse_schema(raw, &limits()).expect_err("cross-node keyword rejected");
            assert_eq!(
                error.wire_error(),
                OwnedDecodeError::GrammarFeatureUnsupported,
                "raw: {raw}"
            );
        }
    }

    #[test]
    fn object_enum_rejection_prevents_non_member_documents() {
        // The object-enum shape the subset cannot enforce is rejected at parse
        // time, so no automaton ever exists that could accept a non-member
        // document like {"a":2} for enum [{"a":1}].
        let raw = r#"{
            "type": "object",
            "properties": { "a": { "type": "integer" } },
            "required": ["a"],
            "additionalProperties": false,
            "enum": [{"a": 1}]
        }"#;
        assert!(
            parse_schema(raw, &limits()).is_err(),
            "object enum must be rejected so it cannot fail open"
        );
    }

    #[test]
    fn rejects_wrong_dialect() {
        let raw = r#"{ "$schema": "http://json-schema.org/draft-07/schema#", "type": "string" }"#;
        let error = parse_schema(raw, &limits()).expect_err("wrong dialect rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }
}
