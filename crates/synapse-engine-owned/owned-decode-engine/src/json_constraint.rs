//! Token-byte constraints for JSON generation in the shared decode controller.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

/// A reusable bit mask over tokenizer token ids.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenMask {
    words: Vec<u64>,
    allowed: usize,
}

impl TokenMask {
    fn empty(vocab_size: usize) -> Self {
        Self {
            words: vec![0; vocab_size.div_ceil(64)],
            allowed: 0,
        }
    }

    fn insert(&mut self, token_id: u32) {
        let token_id = token_id as usize;
        let Some(word) = self.words.get_mut(token_id / 64) else {
            return;
        };
        let bit = 1u64 << (token_id % 64);
        if *word & bit == 0 {
            *word |= bit;
            self.allowed += 1;
        }
    }

    pub(crate) fn contains(&self, token_id: u32) -> bool {
        let token_id = token_id as usize;
        self.words
            .get(token_id / 64)
            .is_some_and(|word| word & (1u64 << (token_id % 64)) != 0)
    }

    pub(crate) fn len(&self) -> usize {
        self.allowed
    }

    /// Applies the mask in place for samplers that consume a full logit vector.
    #[allow(dead_code)]
    pub(crate) fn apply(&self, logits: &mut [f32]) {
        for (token_id, logit) in logits.iter_mut().enumerate() {
            if !self.contains(token_id as u32) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    pub(crate) fn token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, word)| {
                let mut remaining = *word;
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        return None;
                    }
                    let bit = remaining.trailing_zeros();
                    remaining &= remaining - 1;
                    Some((word_index * 64 + bit as usize) as u32)
                })
            })
    }
}

/// Backend-independent constraint interface used by the decode loop.
pub trait DecodeConstraint {
    fn allowed(&mut self) -> Result<Arc<TokenMask>>;
    fn advance(&mut self, token_id: u32) -> Result<()>;
    fn is_complete(&self) -> bool;
    fn describe(&self) -> String;
}

#[derive(Clone, Debug, Default)]
struct TrieNode {
    children: Vec<(u8, usize)>,
    token_ids: Vec<u32>,
}

/// Exact token-id to emitted-byte mapping plus a prefix-sharing token trie.
#[derive(Debug)]
pub struct TokenVocabulary {
    pieces: Vec<Option<Vec<u8>>>,
    trie: Vec<TrieNode>,
}

impl TokenVocabulary {
    /// Build the exact token-id-to-emitted-byte table used by owned decode.
    ///
    /// The supervised decode worker calls this once while loading a model and
    /// caches the result. Grammar requests interpret the module-compiled
    /// automaton against these bytes; no tokenizer logic is re-derived per
    /// generation and raw schemas never cross the worker boundary.
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        let serialized = tokenizer
            .to_string(false)
            .map_err(|error| anyhow::anyhow!("serialize tokenizer: {error}"))?;
        let description: Value = serde_json::from_str(&serialized)
            .context("parse serialized tokenizer while building constraint vocabulary")?;
        ensure!(
            byte_level_decoder(&description["decoder"]),
            "constrained decoding currently requires a ByteLevel decoder"
        );
        let vocab = tokenizer.get_vocab(true);
        let vocab_size = vocab
            .values()
            .copied()
            .max()
            .map_or(0, |token_id| token_id as usize + 1);
        ensure!(vocab_size > 0, "constraint tokenizer vocabulary is empty");

        let added = tokenizer.get_added_vocabulary();
        let reverse = byte_level_reverse_alphabet();
        let mut pieces = vec![None; vocab_size];
        for (token, token_id) in vocab {
            if added.is_special_token(&token) {
                continue;
            }
            let bytes = decode_byte_level_piece_with(&token, &reverse)
                .with_context(|| format!("token {token_id} cannot be mapped to bytes"))?;
            if !bytes.is_empty() {
                pieces[token_id as usize] = Some(bytes);
            }
        }
        Ok(Self::from_pieces(pieces))
    }

    pub(crate) fn from_pieces(pieces: Vec<Option<Vec<u8>>>) -> Self {
        let mut vocabulary = Self {
            pieces,
            trie: vec![TrieNode::default()],
        };
        for token_id in 0..vocabulary.pieces.len() {
            let Some(piece) = vocabulary.pieces[token_id].clone() else {
                continue;
            };
            let mut node = 0;
            for byte in piece {
                let next = vocabulary.trie[node]
                    .children
                    .iter()
                    .find_map(|&(edge, child)| (edge == byte).then_some(child));
                node = if let Some(next) = next {
                    next
                } else {
                    let next = vocabulary.trie.len();
                    vocabulary.trie.push(TrieNode::default());
                    vocabulary.trie[node].children.push((byte, next));
                    next
                };
            }
            vocabulary.trie[node].token_ids.push(token_id as u32);
        }
        for node in &mut vocabulary.trie {
            node.children.sort_unstable_by_key(|&(byte, _)| byte);
        }
        vocabulary
    }

    /// Number of token IDs represented by this vocabulary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether the vocabulary contains no token IDs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Exact bytes emitted by `token_id`, or `None` for a token with no
    /// content-byte representation.
    #[must_use]
    pub fn token_piece(&self, token_id: u32) -> Option<&[u8]> {
        self.pieces
            .get(token_id as usize)
            .and_then(Option::as_deref)
    }
}

fn byte_level_decoder(decoder: &Value) -> bool {
    match decoder.get("type").and_then(Value::as_str) {
        Some("ByteLevel") => true,
        Some("Sequence") => decoder
            .get("decoders")
            .and_then(Value::as_array)
            .is_some_and(|decoders| decoders.len() == 1 && decoders.iter().all(byte_level_decoder)),
        _ => false,
    }
}

#[cfg(test)]
fn decode_byte_level_piece(token: &str) -> Result<Vec<u8>> {
    decode_byte_level_piece_with(token, &byte_level_reverse_alphabet())
}

fn decode_byte_level_piece_with(token: &str, reverse: &HashMap<char, u8>) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(token.len());
    for character in token.chars() {
        let Some(byte) = reverse.get(&character).copied() else {
            return Ok(token.as_bytes().to_vec());
        };
        bytes.push(byte);
    }
    Ok(bytes)
}

fn byte_level_reverse_alphabet() -> HashMap<char, u8> {
    let mut direct = Vec::new();
    direct.extend(33u16..=126);
    direct.extend(161u16..=172);
    direct.extend(174u16..=255);
    let direct_set = direct.iter().copied().collect::<HashSet<_>>();
    let mut next = 256u32;
    let mut reverse = HashMap::with_capacity(256);
    for byte in 0u16..=255 {
        let codepoint = if direct_set.contains(&byte) {
            byte as u32
        } else {
            let codepoint = next;
            next += 1;
            codepoint
        };
        reverse.insert(
            char::from_u32(codepoint).expect("ByteLevel codepoints are valid Unicode"),
            byte as u8,
        );
    }
    reverse
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum SchemaNode {
    Any,
    Object(ObjectSchema),
    Array(SchemaRef),
    String,
    StringEnum(Arc<Vec<String>>),
    Number,
    Boolean,
    Null,
}

type SchemaRef = Arc<SchemaNode>;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ObjectSchema {
    properties: Vec<PropertySchema>,
    additional: Option<SchemaRef>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PropertySchema {
    name: String,
    schema: SchemaRef,
    required: bool,
}

/// Parsed, deliberately small JSON-schema subset used by the JSON recognizer.
#[derive(Clone, Debug)]
pub(crate) struct JsonSchema {
    root: SchemaRef,
}

impl JsonSchema {
    pub(crate) fn from_value(value: &Value) -> Result<Self> {
        Ok(Self {
            root: parse_schema(value, "$schema")?,
        })
    }

    pub(crate) fn validate(&self, value: &Value) -> Result<()> {
        validate_schema(&self.root, value, "$output")
    }
}

fn parse_schema(value: &Value, at: &str) -> Result<SchemaRef> {
    let object = value
        .as_object()
        .with_context(|| format!("{at} must be a JSON object"))?;
    const SUPPORTED: &[&str] = &[
        "$schema",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
    ];
    for key in object.keys() {
        ensure!(
            SUPPORTED.contains(&key.as_str()),
            "{at}.{key} is outside the constrained-decoding schema subset"
        );
    }

    if let Some(values) = object.get("enum") {
        ensure_schema_keys(object, at, &["type", "enum"])?;
        let values = values
            .as_array()
            .with_context(|| format!("{at}.enum must be an array"))?;
        ensure!(!values.is_empty(), "{at}.enum must not be empty");
        let strings = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .with_context(|| format!("{at}.enum[{index}] must be a string"))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            object
                .get("type")
                .and_then(Value::as_str)
                .is_none_or(|kind| kind == "string"),
            "{at}.enum supports only string values"
        );
        return Ok(Arc::new(SchemaNode::StringEnum(Arc::new(strings))));
    }

    let Some(kind) = object.get("type") else {
        ensure!(
            object
                .keys()
                .all(|key| ["$schema", "title", "description"].contains(&key.as_str())),
            "{at} requires type or enum"
        );
        return Ok(Arc::new(SchemaNode::Any));
    };
    let kind = kind
        .as_str()
        .with_context(|| format!("{at}.type must be a string"))?;
    let node = match kind {
        "object" => {
            ensure_schema_keys(
                object,
                at,
                &["type", "properties", "required", "additionalProperties"],
            )?;
            let properties = object
                .get("properties")
                .map(|value| {
                    value
                        .as_object()
                        .with_context(|| format!("{at}.properties must be an object"))
                })
                .transpose()?
                .cloned()
                .unwrap_or_default();
            let required = object
                .get("required")
                .map(|value| {
                    value
                        .as_array()
                        .with_context(|| format!("{at}.required must be an array"))?
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            value
                                .as_str()
                                .map(str::to_owned)
                                .with_context(|| format!("{at}.required[{index}] must be a string"))
                        })
                        .collect::<Result<HashSet<_>>>()
                })
                .transpose()?
                .unwrap_or_default();
            for name in &required {
                ensure!(
                    properties.contains_key(name),
                    "{at}.required contains unknown property {name:?}"
                );
            }
            let mut parsed = Vec::with_capacity(properties.len());
            for (name, schema) in properties {
                parsed.push(PropertySchema {
                    schema: parse_schema(&schema, &format!("{at}.properties.{name}"))?,
                    required: required.contains(&name),
                    name,
                });
            }
            let additional = match object.get("additionalProperties") {
                None | Some(Value::Bool(true)) => Some(Arc::new(SchemaNode::Any)),
                Some(Value::Bool(false)) => None,
                Some(schema @ Value::Object(_)) => {
                    Some(parse_schema(schema, &format!("{at}.additionalProperties"))?)
                }
                Some(_) => bail!("{at}.additionalProperties must be a boolean or schema"),
            };
            SchemaNode::Object(ObjectSchema {
                properties: parsed,
                additional,
            })
        }
        "array" => {
            ensure_schema_keys(object, at, &["type", "items"])?;
            let item = object
                .get("items")
                .with_context(|| format!("{at}.items is required for array schemas"))?;
            SchemaNode::Array(parse_schema(item, &format!("{at}.items"))?)
        }
        "string" => {
            ensure_schema_keys(object, at, &["type"])?;
            SchemaNode::String
        }
        "number" => {
            ensure_schema_keys(object, at, &["type"])?;
            SchemaNode::Number
        }
        "boolean" => {
            ensure_schema_keys(object, at, &["type"])?;
            SchemaNode::Boolean
        }
        "null" => {
            ensure_schema_keys(object, at, &["type"])?;
            SchemaNode::Null
        }
        unsupported => bail!("{at}.type {unsupported:?} is outside the schema subset"),
    };
    Ok(Arc::new(node))
}

fn ensure_schema_keys(
    object: &serde_json::Map<String, Value>,
    at: &str,
    allowed: &[&str],
) -> Result<()> {
    const METADATA: &[&str] = &["$schema", "title", "description"];
    for key in object.keys() {
        ensure!(
            METADATA.contains(&key.as_str()) || allowed.contains(&key.as_str()),
            "{at}.{key} does not apply to this schema type"
        );
    }
    Ok(())
}

fn validate_schema(schema: &SchemaNode, value: &Value, at: &str) -> Result<()> {
    match schema {
        SchemaNode::Any => Ok(()),
        SchemaNode::String => {
            ensure!(value.is_string(), "{at} must be a string");
            Ok(())
        }
        SchemaNode::StringEnum(values) => {
            let value = value
                .as_str()
                .with_context(|| format!("{at} must be a string"))?;
            ensure!(
                values.iter().any(|candidate| candidate == value),
                "{at} is outside enum"
            );
            Ok(())
        }
        SchemaNode::Number => {
            ensure!(value.is_number(), "{at} must be a number");
            Ok(())
        }
        SchemaNode::Boolean => {
            ensure!(value.is_boolean(), "{at} must be a boolean");
            Ok(())
        }
        SchemaNode::Null => {
            ensure!(value.is_null(), "{at} must be null");
            Ok(())
        }
        SchemaNode::Array(item) => {
            let values = value
                .as_array()
                .with_context(|| format!("{at} must be an array"))?;
            for (index, value) in values.iter().enumerate() {
                validate_schema(item, value, &format!("{at}[{index}]"))?;
            }
            Ok(())
        }
        SchemaNode::Object(schema) => {
            let value = value
                .as_object()
                .with_context(|| format!("{at} must be an object"))?;
            for property in &schema.properties {
                if property.required {
                    ensure!(
                        value.contains_key(&property.name),
                        "{at}.{} is required",
                        property.name
                    );
                }
                if let Some(value) = value.get(&property.name) {
                    validate_schema(&property.schema, value, &format!("{at}.{}", property.name))?;
                }
            }
            for (name, value) in value {
                if schema
                    .properties
                    .iter()
                    .any(|property| property.name == *name)
                {
                    continue;
                }
                let additional = schema
                    .additional
                    .as_ref()
                    .with_context(|| format!("{at}.{name} is not an allowed property"))?;
                validate_schema(additional, value, &format!("{at}.{name}"))?;
            }
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct JsonParser {
    state: ParserState,
    stack: Vec<Frame>,
    lexeme: Option<Lexeme>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ParserState {
    Value(SchemaRef),
    ObjectKeyOrEnd,
    ObjectKey,
    ObjectColon,
    ObjectCommaOrEnd,
    ArrayValueOrEnd(SchemaRef),
    ArrayCommaOrEnd,
    End,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Frame {
    Object(ObjectFrame),
    Array(ArrayFrame),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ObjectFrame {
    schema: SchemaRef,
    seen: Vec<bool>,
    pending_schema: Option<SchemaRef>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ArrayFrame {
    item_schema: SchemaRef,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Lexeme {
    String(StringLexeme),
    Number(NumberLexeme),
    Literal(LiteralState),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StringLexeme {
    mode: StringMode,
    pending_utf8: Vec<u8>,
    target: StringTarget,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum StringTarget {
    Key(KeyTarget),
    Value,
    Enum(EnumTarget),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct KeyTarget {
    names: Arc<Vec<String>>,
    captured: Option<Vec<u8>>,
    candidates: Vec<usize>,
    allow_unknown: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct EnumTarget {
    values: Arc<Vec<String>>,
    captured: Vec<u8>,
    candidates: Vec<usize>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum StringMode {
    Normal,
    Escape,
    Unicode { value: u16, digits: u8 },
    LowSurrogateBackslash { high: u16 },
    LowSurrogateU { high: u16 },
    LowSurrogateUnicode { high: u16, value: u16, digits: u8 },
}

const MAX_NUMBER_BYTES: usize = 32;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct NumberLexeme {
    state: NumberState,
    bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum NumberState {
    Sign,
    Zero,
    Integer,
    Dot,
    Fraction,
    ExponentMark,
    ExponentSign,
    Exponent,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum LiteralKind {
    True,
    False,
    Null,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LiteralState {
    kind: LiteralKind,
    matched: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LexemeStep {
    Continue,
    Complete,
    CompleteAndReprocess,
    Invalid,
}

impl JsonParser {
    fn new(schema: SchemaRef) -> Self {
        Self {
            state: ParserState::Value(schema),
            stack: Vec::new(),
            lexeme: None,
        }
    }

    fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().copied().all(|byte| self.advance_byte(byte))
    }

    fn advance_byte(&mut self, byte: u8) -> bool {
        if let Some(lexeme) = self.lexeme.take() {
            return self.advance_lexeme(lexeme, byte);
        }
        self.advance_structural(byte)
    }

    fn advance_lexeme(&mut self, lexeme: Lexeme, byte: u8) -> bool {
        match lexeme {
            Lexeme::String(mut string) => match string.advance(byte) {
                LexemeStep::Continue => {
                    if !string.can_complete() {
                        return false;
                    }
                    self.lexeme = Some(Lexeme::String(string));
                    true
                }
                LexemeStep::Complete => self.finish_string(string.target),
                LexemeStep::CompleteAndReprocess | LexemeStep::Invalid => false,
            },
            Lexeme::Number(mut number) => match number.advance(byte) {
                LexemeStep::Continue => {
                    self.lexeme = Some(Lexeme::Number(number));
                    true
                }
                LexemeStep::CompleteAndReprocess => {
                    self.complete_value() && self.advance_structural(byte)
                }
                LexemeStep::Complete | LexemeStep::Invalid => false,
            },
            Lexeme::Literal(mut literal) => match literal.advance(byte) {
                LexemeStep::Continue => {
                    self.lexeme = Some(Lexeme::Literal(literal));
                    true
                }
                LexemeStep::Complete => self.complete_value(),
                LexemeStep::CompleteAndReprocess | LexemeStep::Invalid => false,
            },
        }
    }

    fn advance_structural(&mut self, byte: u8) -> bool {
        let state = self.state.clone();
        match state {
            ParserState::Value(schema) => {
                is_json_whitespace(byte) || self.start_value(schema, byte)
            }
            ParserState::ObjectKeyOrEnd => {
                if is_json_whitespace(byte) {
                    true
                } else if byte == b'}' {
                    self.close_object()
                } else if byte == b'"' {
                    self.start_key()
                } else {
                    false
                }
            }
            ParserState::ObjectKey => {
                if is_json_whitespace(byte) {
                    true
                } else if byte == b'"' {
                    self.start_key()
                } else {
                    false
                }
            }
            ParserState::ObjectColon => {
                if is_json_whitespace(byte) {
                    true
                } else if byte == b':' {
                    let Some(Frame::Object(frame)) = self.stack.last_mut() else {
                        return false;
                    };
                    let Some(schema) = frame.pending_schema.take() else {
                        return false;
                    };
                    self.state = ParserState::Value(schema);
                    true
                } else {
                    false
                }
            }
            ParserState::ObjectCommaOrEnd => {
                if is_json_whitespace(byte) {
                    true
                } else if byte == b',' {
                    let Some(Frame::Object(frame)) = self.stack.last() else {
                        return false;
                    };
                    let can_add_property = object_schema(&frame.schema).is_none_or(|schema| {
                        schema.additional.is_some() || frame.seen.iter().any(|seen| !seen)
                    });
                    if !can_add_property {
                        return false;
                    }
                    self.state = ParserState::ObjectKey;
                    true
                } else if byte == b'}' {
                    self.close_object()
                } else {
                    false
                }
            }
            ParserState::ArrayValueOrEnd(schema) => {
                if is_json_whitespace(byte) {
                    true
                } else if byte == b']' {
                    self.close_array()
                } else {
                    self.start_value(schema, byte)
                }
            }
            ParserState::ArrayCommaOrEnd => {
                if is_json_whitespace(byte) {
                    true
                } else if byte == b',' {
                    let Some(Frame::Array(frame)) = self.stack.last() else {
                        return false;
                    };
                    self.state = ParserState::Value(frame.item_schema.clone());
                    true
                } else if byte == b']' {
                    self.close_array()
                } else {
                    false
                }
            }
            ParserState::End => is_json_whitespace(byte),
        }
    }

    fn start_value(&mut self, schema: SchemaRef, byte: u8) -> bool {
        match byte {
            b'{' if accepts_object(&schema) => {
                let seen = object_schema(&schema).map_or(0, |object| object.properties.len());
                self.stack.push(Frame::Object(ObjectFrame {
                    schema,
                    seen: vec![false; seen],
                    pending_schema: None,
                }));
                self.state = ParserState::ObjectKeyOrEnd;
                true
            }
            b'[' if accepts_array(&schema) => {
                let item_schema = match schema.as_ref() {
                    SchemaNode::Array(item) => item.clone(),
                    SchemaNode::Any => Arc::new(SchemaNode::Any),
                    _ => return false,
                };
                self.stack.push(Frame::Array(ArrayFrame {
                    item_schema: item_schema.clone(),
                }));
                self.state = ParserState::ArrayValueOrEnd(item_schema);
                true
            }
            b'"' if accepts_string(&schema) => {
                let target = match schema.as_ref() {
                    SchemaNode::StringEnum(values) => StringTarget::Enum(EnumTarget {
                        values: values.clone(),
                        captured: Vec::new(),
                        candidates: (0..values.len()).collect(),
                    }),
                    _ => StringTarget::Value,
                };
                self.lexeme = Some(Lexeme::String(StringLexeme::new(target)));
                true
            }
            b'-' if accepts_number(&schema) => {
                self.lexeme = Some(Lexeme::Number(NumberLexeme {
                    state: NumberState::Sign,
                    bytes: 1,
                }));
                true
            }
            b'0' if accepts_number(&schema) => {
                self.lexeme = Some(Lexeme::Number(NumberLexeme {
                    state: NumberState::Zero,
                    bytes: 1,
                }));
                true
            }
            b'1'..=b'9' if accepts_number(&schema) => {
                self.lexeme = Some(Lexeme::Number(NumberLexeme {
                    state: NumberState::Integer,
                    bytes: 1,
                }));
                true
            }
            b't' if accepts_boolean(&schema) => {
                self.lexeme = Some(Lexeme::Literal(LiteralState {
                    kind: LiteralKind::True,
                    matched: 1,
                }));
                true
            }
            b'f' if accepts_boolean(&schema) => {
                self.lexeme = Some(Lexeme::Literal(LiteralState {
                    kind: LiteralKind::False,
                    matched: 1,
                }));
                true
            }
            b'n' if accepts_null(&schema) => {
                self.lexeme = Some(Lexeme::Literal(LiteralState {
                    kind: LiteralKind::Null,
                    matched: 1,
                }));
                true
            }
            _ => false,
        }
    }

    fn start_key(&mut self) -> bool {
        let Some(Frame::Object(frame)) = self.stack.last() else {
            return false;
        };
        let target = if let Some(schema) = object_schema(&frame.schema) {
            KeyTarget {
                names: Arc::new(
                    schema
                        .properties
                        .iter()
                        .map(|property| property.name.clone())
                        .collect(),
                ),
                captured: Some(Vec::new()),
                candidates: frame
                    .seen
                    .iter()
                    .enumerate()
                    .filter_map(|(index, seen)| (!seen).then_some(index))
                    .collect(),
                allow_unknown: schema.additional.is_some(),
            }
        } else {
            KeyTarget {
                names: Arc::new(Vec::new()),
                captured: None,
                candidates: Vec::new(),
                allow_unknown: true,
            }
        };
        if !target.allow_unknown && target.candidates.is_empty() {
            return false;
        }
        self.lexeme = Some(Lexeme::String(StringLexeme::new(StringTarget::Key(target))));
        true
    }

    fn finish_string(&mut self, target: StringTarget) -> bool {
        match target {
            StringTarget::Value => self.complete_value(),
            StringTarget::Enum(target) => {
                if !target
                    .candidates
                    .iter()
                    .any(|&index| target.values[index].as_bytes() == target.captured.as_slice())
                {
                    return false;
                }
                self.complete_value()
            }
            StringTarget::Key(target) => self.finish_key(target),
        }
    }

    fn finish_key(&mut self, target: KeyTarget) -> bool {
        let Some(Frame::Object(frame)) = self.stack.last_mut() else {
            return false;
        };
        let Some(schema) = object_schema(&frame.schema) else {
            frame.pending_schema = Some(Arc::new(SchemaNode::Any));
            self.state = ParserState::ObjectColon;
            return true;
        };
        let captured = target.captured.unwrap_or_default();
        let known = schema
            .properties
            .iter()
            .position(|property| property.name.as_bytes() == captured);
        frame.pending_schema = if let Some(index) = known {
            if frame.seen[index] {
                return false;
            }
            frame.seen[index] = true;
            Some(schema.properties[index].schema.clone())
        } else {
            schema.additional.clone()
        };
        if frame.pending_schema.is_none() {
            return false;
        }
        self.state = ParserState::ObjectColon;
        true
    }

    fn close_object(&mut self) -> bool {
        let Some(Frame::Object(frame)) = self.stack.last() else {
            return false;
        };
        if let Some(schema) = object_schema(&frame.schema) {
            if schema
                .properties
                .iter()
                .zip(&frame.seen)
                .any(|(property, seen)| property.required && !seen)
            {
                return false;
            }
        }
        self.stack.pop();
        self.complete_value()
    }

    fn close_array(&mut self) -> bool {
        if !matches!(self.stack.last(), Some(Frame::Array(_))) {
            return false;
        }
        self.stack.pop();
        self.complete_value()
    }

    fn complete_value(&mut self) -> bool {
        self.state = match self.stack.last() {
            Some(Frame::Object(_)) => ParserState::ObjectCommaOrEnd,
            Some(Frame::Array(_)) => ParserState::ArrayCommaOrEnd,
            None => ParserState::End,
        };
        true
    }

    fn is_complete(&self) -> bool {
        if self.stack.is_empty() && matches!(self.state, ParserState::End) && self.lexeme.is_none()
        {
            return true;
        }
        self.stack.is_empty()
            && matches!(self.state, ParserState::Value(_))
            && matches!(
                self.lexeme,
                Some(Lexeme::Number(NumberLexeme {
                    state: NumberState::Zero
                        | NumberState::Integer
                        | NumberState::Fraction
                        | NumberState::Exponent,
                    ..
                }))
            )
    }
}

impl StringLexeme {
    fn new(target: StringTarget) -> Self {
        Self {
            mode: StringMode::Normal,
            pending_utf8: Vec::new(),
            target,
        }
    }

    fn advance(&mut self, byte: u8) -> LexemeStep {
        match self.mode.clone() {
            StringMode::Normal => {
                if !self.pending_utf8.is_empty() {
                    return if self.push_utf8_byte(byte) {
                        LexemeStep::Continue
                    } else {
                        LexemeStep::Invalid
                    };
                }
                match byte {
                    b'"' => LexemeStep::Complete,
                    b'\\' => {
                        self.mode = StringMode::Escape;
                        LexemeStep::Continue
                    }
                    0x00..=0x1f => LexemeStep::Invalid,
                    0x20..=0x7f => {
                        if self.append_decoded(&[byte]) {
                            LexemeStep::Continue
                        } else {
                            LexemeStep::Invalid
                        }
                    }
                    _ => {
                        if self.push_utf8_byte(byte) {
                            LexemeStep::Continue
                        } else {
                            LexemeStep::Invalid
                        }
                    }
                }
            }
            StringMode::Escape => match byte {
                b'"' | b'\\' | b'/' => self.append_escape(byte),
                b'b' => self.append_escape(0x08),
                b'f' => self.append_escape(0x0c),
                b'n' => self.append_escape(b'\n'),
                b'r' => self.append_escape(b'\r'),
                b't' => self.append_escape(b'\t'),
                b'u' => {
                    self.mode = StringMode::Unicode {
                        value: 0,
                        digits: 0,
                    };
                    LexemeStep::Continue
                }
                _ => LexemeStep::Invalid,
            },
            StringMode::Unicode { value, digits } => {
                let Some(digit) = hex_digit(byte) else {
                    return LexemeStep::Invalid;
                };
                let value = (value << 4) | digit as u16;
                let digits = digits + 1;
                if digits < 4 {
                    self.mode = StringMode::Unicode { value, digits };
                    return LexemeStep::Continue;
                }
                if (0xd800..=0xdbff).contains(&value) {
                    self.mode = StringMode::LowSurrogateBackslash { high: value };
                    LexemeStep::Continue
                } else if (0xdc00..=0xdfff).contains(&value) {
                    LexemeStep::Invalid
                } else {
                    self.append_scalar(value as u32)
                }
            }
            StringMode::LowSurrogateBackslash { high } => {
                if byte != b'\\' {
                    return LexemeStep::Invalid;
                }
                self.mode = StringMode::LowSurrogateU { high };
                LexemeStep::Continue
            }
            StringMode::LowSurrogateU { high } => {
                if byte != b'u' {
                    return LexemeStep::Invalid;
                }
                self.mode = StringMode::LowSurrogateUnicode {
                    high,
                    value: 0,
                    digits: 0,
                };
                LexemeStep::Continue
            }
            StringMode::LowSurrogateUnicode {
                high,
                value,
                digits,
            } => {
                let Some(digit) = hex_digit(byte) else {
                    return LexemeStep::Invalid;
                };
                let value = (value << 4) | digit as u16;
                let digits = digits + 1;
                if digits < 4 {
                    self.mode = StringMode::LowSurrogateUnicode {
                        high,
                        value,
                        digits,
                    };
                    return LexemeStep::Continue;
                }
                if !(0xdc00..=0xdfff).contains(&value) {
                    return LexemeStep::Invalid;
                }
                let scalar = 0x1_0000 + (((high as u32 - 0xd800) << 10) | (value as u32 - 0xdc00));
                self.append_scalar(scalar)
            }
        }
    }

    fn append_escape(&mut self, byte: u8) -> LexemeStep {
        if !self.append_decoded(&[byte]) {
            return LexemeStep::Invalid;
        }
        self.mode = StringMode::Normal;
        LexemeStep::Continue
    }

    fn append_scalar(&mut self, scalar: u32) -> LexemeStep {
        let Some(character) = char::from_u32(scalar) else {
            return LexemeStep::Invalid;
        };
        let mut encoded = [0; 4];
        if !self.append_decoded(character.encode_utf8(&mut encoded).as_bytes()) {
            return LexemeStep::Invalid;
        }
        self.mode = StringMode::Normal;
        LexemeStep::Continue
    }

    fn push_utf8_byte(&mut self, byte: u8) -> bool {
        self.pending_utf8.push(byte);
        if !self.append_decoded(&[byte]) {
            return false;
        }
        match std::str::from_utf8(&self.pending_utf8) {
            Ok(_) => {
                self.pending_utf8.clear();
                true
            }
            Err(error) => error.error_len().is_none() && self.pending_utf8.len() < 4,
        }
    }

    fn can_complete(&self) -> bool {
        let (values, captured, candidates, allow_unknown) = match &self.target {
            StringTarget::Value => return true,
            StringTarget::Key(target) => {
                let Some(captured) = target.captured.as_deref() else {
                    return true;
                };
                (
                    target.names.as_slice(),
                    captured,
                    target.candidates.as_slice(),
                    target.allow_unknown,
                )
            }
            StringTarget::Enum(target) => (
                target.values.as_slice(),
                target.captured.as_slice(),
                target.candidates.as_slice(),
                false,
            ),
        };
        if allow_unknown {
            return true;
        }
        let next_units = |candidate: usize| {
            values[candidate]
                .as_bytes()
                .get(captured.len()..)
                .and_then(|suffix| std::str::from_utf8(suffix).ok())
                .and_then(|suffix| suffix.chars().next())
                .map(|character| {
                    let mut encoded = [0u16; 2];
                    let units = character.encode_utf16(&mut encoded);
                    (units[0], units.get(1).copied())
                })
        };
        match self.mode {
            StringMode::Normal => !candidates.is_empty(),
            StringMode::Escape => candidates
                .iter()
                .copied()
                .any(|candidate| next_units(candidate).is_some()),
            StringMode::Unicode { value, digits } => candidates.iter().copied().any(|candidate| {
                next_units(candidate).is_some_and(|(high, _)| utf16_hex_prefix(high, value, digits))
            }),
            StringMode::LowSurrogateBackslash { high } | StringMode::LowSurrogateU { high } => {
                candidates.iter().copied().any(|candidate| {
                    next_units(candidate).is_some_and(|(candidate_high, low)| {
                        candidate_high == high && low.is_some()
                    })
                })
            }
            StringMode::LowSurrogateUnicode {
                high,
                value,
                digits,
            } => candidates.iter().copied().any(|candidate| {
                next_units(candidate).is_some_and(|(candidate_high, low)| {
                    candidate_high == high
                        && low.is_some_and(|low| utf16_hex_prefix(low, value, digits))
                })
            }),
        }
    }

    fn append_decoded(&mut self, bytes: &[u8]) -> bool {
        match &mut self.target {
            StringTarget::Value => true,
            StringTarget::Key(target) => {
                let Some(captured) = &mut target.captured else {
                    return true;
                };
                captured.extend_from_slice(bytes);
                target
                    .candidates
                    .retain(|&index| target.names[index].as_bytes().starts_with(captured));
                target.allow_unknown || !target.candidates.is_empty()
            }
            StringTarget::Enum(target) => {
                target.captured.extend_from_slice(bytes);
                target.candidates.retain(|&index| {
                    target.values[index]
                        .as_bytes()
                        .starts_with(&target.captured)
                });
                !target.candidates.is_empty()
            }
        }
    }
}

impl NumberLexeme {
    fn advance(&mut self, byte: u8) -> LexemeStep {
        let step = self.state.advance(byte);
        if step == LexemeStep::Continue {
            self.bytes += 1;
            if self.bytes > MAX_NUMBER_BYTES
                || (self.bytes == MAX_NUMBER_BYTES && !self.state.is_terminal())
            {
                return LexemeStep::Invalid;
            }
        }
        step
    }
}

impl NumberState {
    fn advance(&mut self, byte: u8) -> LexemeStep {
        use NumberState::*;
        match (*self, byte) {
            (Sign, b'0') => *self = Zero,
            (Sign, b'1'..=b'9') => *self = Integer,
            (Zero, b'.') | (Integer, b'.') => *self = Dot,
            (Zero, b'e' | b'E') | (Integer, b'e' | b'E') => *self = ExponentMark,
            (Integer, b'0'..=b'9') => {}
            (Dot, b'0'..=b'9') => *self = Fraction,
            (Fraction, b'0'..=b'9') => {}
            (Fraction, b'e' | b'E') => *self = ExponentMark,
            (ExponentMark, b'+' | b'-') => *self = ExponentSign,
            (ExponentMark, b'0'..=b'9') | (ExponentSign, b'0'..=b'9') => *self = Exponent,
            (Exponent, b'0'..=b'9') => {}
            (Zero | Integer | Fraction | Exponent, _) => {
                return LexemeStep::CompleteAndReprocess;
            }
            _ => return LexemeStep::Invalid,
        }
        LexemeStep::Continue
    }

    fn is_terminal(self) -> bool {
        matches!(
            self,
            NumberState::Zero
                | NumberState::Integer
                | NumberState::Fraction
                | NumberState::Exponent
        )
    }
}

impl LiteralState {
    fn advance(&mut self, byte: u8) -> LexemeStep {
        let expected = match self.kind {
            LiteralKind::True => b"true".as_slice(),
            LiteralKind::False => b"false".as_slice(),
            LiteralKind::Null => b"null".as_slice(),
        };
        if expected.get(self.matched).copied() != Some(byte) {
            return LexemeStep::Invalid;
        }
        self.matched += 1;
        if self.matched == expected.len() {
            LexemeStep::Complete
        } else {
            LexemeStep::Continue
        }
    }
}

fn object_schema(schema: &SchemaNode) -> Option<&ObjectSchema> {
    match schema {
        SchemaNode::Object(schema) => Some(schema),
        _ => None,
    }
}

fn accepts_object(schema: &SchemaNode) -> bool {
    matches!(schema, SchemaNode::Any | SchemaNode::Object(_))
}

fn accepts_array(schema: &SchemaNode) -> bool {
    matches!(schema, SchemaNode::Any | SchemaNode::Array(_))
}

fn accepts_string(schema: &SchemaNode) -> bool {
    matches!(
        schema,
        SchemaNode::Any | SchemaNode::String | SchemaNode::StringEnum(_)
    )
}

fn accepts_number(schema: &SchemaNode) -> bool {
    matches!(schema, SchemaNode::Any | SchemaNode::Number)
}

fn accepts_boolean(schema: &SchemaNode) -> bool {
    matches!(schema, SchemaNode::Any | SchemaNode::Boolean)
}

fn accepts_null(schema: &SchemaNode) -> bool {
    matches!(schema, SchemaNode::Any | SchemaNode::Null)
}

fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n')
}

fn utf16_hex_prefix(unit: u16, prefix: u16, digits: u8) -> bool {
    debug_assert!(digits <= 4);
    digits == 0 || unit >> (4 * (4 - digits)) == prefix
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// JSON constraint with lazy parser-state-to-vocabulary-mask caching.
pub struct JsonConstraint {
    vocabulary: Arc<TokenVocabulary>,
    stop_tokens: HashSet<u32>,
    parser: JsonParser,
    masks: HashMap<JsonParser, Arc<TokenMask>>,
    debug_prefix: Vec<u8>,
    terminated: bool,
}

impl JsonConstraint {
    pub(crate) fn new(
        vocabulary: Arc<TokenVocabulary>,
        schema: Option<&JsonSchema>,
        stop_tokens: &HashSet<u32>,
    ) -> Self {
        Self {
            vocabulary,
            stop_tokens: stop_tokens.clone(),
            parser: JsonParser::new(
                schema
                    .map(|schema| schema.root.clone())
                    .unwrap_or_else(|| Arc::new(SchemaNode::Any)),
            ),
            masks: HashMap::new(),
            debug_prefix: Vec::new(),
            terminated: false,
        }
    }

    fn compute_mask(&self) -> TokenMask {
        let mut mask = TokenMask::empty(self.vocabulary.len());
        if self.parser.is_complete() && !self.stop_tokens.is_empty() {
            for &stop_token in &self.stop_tokens {
                mask.insert(stop_token);
            }
            return mask;
        }
        let reject_optional_whitespace = self.parser.lexeme.is_none();
        let mut pending = Vec::new();

        for &(byte, child) in &self.vocabulary.trie[0].children {
            let mut parser = self.parser.clone();
            if parser.advance_byte(byte) {
                pending.push((child, parser));
            }
        }
        while let Some((node_index, parser)) = pending.pop() {
            let node = &self.vocabulary.trie[node_index];
            for &token_id in &node.token_ids {
                let is_optional_whitespace = reject_optional_whitespace
                    && self
                        .vocabulary
                        .token_piece(token_id)
                        .is_some_and(|piece| piece.iter().copied().all(is_json_whitespace));
                if !is_optional_whitespace {
                    mask.insert(token_id);
                }
            }
            for &(byte, child) in &node.children {
                let mut next = parser.clone();
                if next.advance_byte(byte) {
                    pending.push((child, next));
                }
            }
        }
        if self.parser.is_complete() {
            for &stop_token in &self.stop_tokens {
                mask.insert(stop_token);
            }
        }
        mask
    }
}

impl DecodeConstraint for JsonConstraint {
    fn allowed(&mut self) -> Result<Arc<TokenMask>> {
        ensure!(!self.terminated, "constraint already consumed an EOS token");
        if let Some(mask) = self.masks.get(&self.parser) {
            return Ok(mask.clone());
        }
        let mask = Arc::new(self.compute_mask());
        ensure!(
            mask.len() > 0,
            "JSON constraint has no token that can extend prefix {:?} in state {:?}",
            String::from_utf8_lossy(&self.debug_prefix),
            self.parser
        );
        self.masks.insert(self.parser.clone(), mask.clone());
        Ok(mask)
    }

    fn advance(&mut self, token_id: u32) -> Result<()> {
        if self.stop_tokens.contains(&token_id) {
            ensure!(
                self.parser.is_complete(),
                "EOS cannot terminate an incomplete JSON value"
            );
            self.terminated = true;
            return Ok(());
        }
        let piece = self
            .vocabulary
            .token_piece(token_id)
            .with_context(|| format!("token {token_id} has no JSON-visible byte sequence"))?;
        ensure!(
            self.parser.advance_bytes(piece),
            "token {token_id} violates the JSON constraint"
        );
        self.debug_prefix.extend_from_slice(piece);
        Ok(())
    }

    fn is_complete(&self) -> bool {
        self.terminated || self.parser.is_complete()
    }

    fn describe(&self) -> String {
        format!(
            "prefix {:?} in state {:?}",
            String::from_utf8_lossy(&self.debug_prefix),
            self.parser
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocabulary(pieces: &[&[u8]], stops: &[u32]) -> (Arc<TokenVocabulary>, HashSet<u32>) {
        let mut mapped = pieces
            .iter()
            .map(|piece| Some(piece.to_vec()))
            .collect::<Vec<_>>();
        let required = stops
            .iter()
            .copied()
            .max()
            .map_or(mapped.len(), |token| mapped.len().max(token as usize + 1));
        mapped.resize(required, None);
        for &stop in stops {
            mapped[stop as usize] = None;
        }
        (
            Arc::new(TokenVocabulary::from_pieces(mapped)),
            stops.iter().copied().collect(),
        )
    }

    fn schema(value: Value) -> JsonSchema {
        JsonSchema::from_value(&value).unwrap()
    }

    #[test]
    fn public_vocabulary_view_preserves_exact_emitted_bytes() {
        let vocabulary = TokenVocabulary::from_pieces(vec![
            None,
            Some(vec![0, b'{', 0xff]),
            Some(b"text".to_vec()),
        ]);
        assert_eq!(vocabulary.len(), 3);
        assert!(!vocabulary.is_empty());
        assert_eq!(vocabulary.token_piece(0), None);
        assert_eq!(vocabulary.token_piece(1), Some(&[0, b'{', 0xff][..]));
        assert_eq!(vocabulary.token_piece(2), Some(&b"text"[..]));
    }

    #[test]
    fn token_can_straddle_multiple_json_boundaries() {
        let pieces: &[&[u8]] = &[b"{\"answer\":\"yes\"}", b"{\"answer\":\"no\"}", b" prose"];
        let (vocab, stops) = vocabulary(pieces, &[3]);
        let schema = schema(serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["yes"]}},
            "required": ["answer"],
            "additionalProperties": false
        }));
        let mut constraint = JsonConstraint::new(vocab, Some(&schema), &stops);
        let mask = constraint.allowed().unwrap();
        assert!(mask.contains(0));
        assert!(!mask.contains(1));
        assert!(!mask.contains(2));
        assert!(!mask.contains(3));
        constraint.advance(0).unwrap();
        assert!(constraint.is_complete());
        assert!(constraint.allowed().unwrap().contains(3));
    }

    #[test]
    fn split_utf8_tokens_require_valid_continuations() {
        let pieces: &[&[u8]] = &[b"\"\xe2", b"\x82", b"\xac\"", b"x", b"\xc0", b"\""];
        let (vocab, stops) = vocabulary(pieces, &[6]);
        let mut constraint = JsonConstraint::new(vocab, None, &stops);
        constraint.advance(0).unwrap();
        let first = constraint.allowed().unwrap();
        assert!(first.contains(1));
        assert!(!first.contains(3));
        assert!(!first.contains(5));
        constraint.advance(1).unwrap();
        let second = constraint.allowed().unwrap();
        assert!(second.contains(2));
        assert!(!second.contains(4));
        constraint.advance(2).unwrap();
        assert!(constraint.is_complete());
    }

    #[test]
    fn optional_whitespace_is_suppressed_but_string_whitespace_is_preserved() {
        let pieces: &[&[u8]] = &[b"\"", b" ", b"\""];
        let (vocab, stops) = vocabulary(pieces, &[3]);
        let mut constraint = JsonConstraint::new(vocab, None, &stops);
        assert!(!constraint.allowed().unwrap().contains(1));
        constraint.advance(0).unwrap();
        assert!(constraint.allowed().unwrap().contains(1));
        constraint.advance(1).unwrap();
        constraint.advance(2).unwrap();
        assert!(constraint.is_complete());
    }

    #[test]
    fn schema_key_escape_prefixes_cannot_enter_a_dead_end() {
        let pieces: &[&[u8]] = &[b"{\"re\\ue", b"{\"result\":\"allow\"}"];
        let (vocab, stops) = vocabulary(pieces, &[2]);
        let schema = schema(serde_json::json!({
            "type": "object",
            "properties": {"result": {"type": "string", "enum": ["allow"]}},
            "required": ["result"],
            "additionalProperties": false
        }));
        let mut constraint = JsonConstraint::new(vocab, Some(&schema), &stops);
        let mask = constraint.allowed().unwrap();
        assert!(!mask.contains(0));
        assert!(mask.contains(1));
    }

    #[test]
    fn string_escapes_and_surrogate_pairs_are_incremental() {
        let pieces: &[&[u8]] = &[b"\"\\uD83D", b"\\uDE00\"", b"\\u0041\"", b"\\q"];
        let (vocab, stops) = vocabulary(pieces, &[4]);
        let mut constraint = JsonConstraint::new(vocab, None, &stops);
        constraint.advance(0).unwrap();
        let mask = constraint.allowed().unwrap();
        assert!(mask.contains(1));
        assert!(!mask.contains(2));
        assert!(!mask.contains(3));
        constraint.advance(1).unwrap();
        assert!(constraint.is_complete());
    }

    #[test]
    fn eos_is_gated_until_the_root_value_is_complete() {
        let pieces: &[&[u8]] = &[b"{", b"}", b"null", b" "];
        let (vocab, stops) = vocabulary(pieces, &[4]);
        let mut constraint = JsonConstraint::new(vocab, None, &stops);
        let initial = constraint.allowed().unwrap();
        assert!(!initial.contains(3));
        assert!(!initial.contains(4));
        constraint.advance(0).unwrap();
        assert!(!constraint.allowed().unwrap().contains(4));
        constraint.advance(1).unwrap();
        let complete = constraint.allowed().unwrap();
        assert!(!complete.contains(3));
        assert!(complete.contains(4));
        constraint.advance(4).unwrap();
        assert!(constraint.is_complete());
    }

    #[test]
    fn number_literals_are_bounded_to_keep_generation_progressing() {
        let mut parser = JsonParser::new(Arc::new(SchemaNode::Number));
        assert!(parser.advance_bytes(b"11111111111111111111111111111111"));
        assert!(parser.is_complete());
        assert!(!parser.advance_byte(b'1'));
    }

    #[test]
    fn required_keys_are_order_independent_and_enums_are_exact() {
        let schema = schema(serde_json::json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["safe", "fast"]},
                "score": {"type": "number"}
            },
            "required": ["mode", "score"],
            "additionalProperties": false
        }));
        for value in [
            br#"{"mode":"safe","score":1}"#.as_slice(),
            br#"{"score":-1.25e2,"mode":"fast"}"#.as_slice(),
        ] {
            let mut parser = JsonParser::new(schema.root.clone());
            assert!(parser.advance_bytes(value));
            assert!(parser.is_complete());
        }
        for value in [
            br#"{"mode":"unsafe","score":1}"#.as_slice(),
            br#"{"mode":"safe"}"#.as_slice(),
            br#"{"mode":"safe","score":"1"}"#.as_slice(),
            br#"{"mode":"safe","score":1,"extra":true}"#.as_slice(),
        ] {
            let mut parser = JsonParser::new(schema.root.clone());
            assert!(!parser.advance_bytes(value) || !parser.is_complete());
        }
    }

    #[test]
    fn token_mask_can_set_invalid_logits_to_negative_infinity() {
        let mut mask = TokenMask::empty(4);
        mask.insert(1);
        mask.insert(3);
        let mut logits = [1.0, 2.0, 3.0, 4.0];
        mask.apply(&mut logits);
        assert_eq!(logits, [f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY, 4.0]);
        assert_eq!(mask.token_ids().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn byte_level_reverse_mapping_preserves_all_bytes() {
        let reverse = byte_level_reverse_alphabet();
        assert_eq!(reverse.len(), 256);
        let mut encoded = reverse.iter().collect::<Vec<_>>();
        encoded.sort_unstable_by_key(|(_, byte)| **byte);
        let token = encoded
            .iter()
            .map(|(character, _)| **character)
            .collect::<String>();
        assert_eq!(
            decode_byte_level_piece(&token).unwrap(),
            (0u8..=255).collect::<Vec<_>>()
        );
        assert_eq!(decode_byte_level_piece("文").unwrap(), "文".as_bytes());
    }

    #[test]
    fn schema_rejects_keywords_on_the_wrong_type() {
        assert!(JsonSchema::from_value(&serde_json::json!({
            "type": "string",
            "properties": {}
        }))
        .is_err());
        assert!(JsonSchema::from_value(&serde_json::json!({"type": "integer"})).is_err());
    }

    #[test]
    fn schema_validator_matches_the_parser_subset() {
        let schema = schema(serde_json::json!({
            "type": "object",
            "properties": {
                "result": {"type": "string", "enum": ["ok", "deny"]},
                "confidence": {"type": "number"}
            },
            "required": ["result"],
            "additionalProperties": false
        }));
        schema
            .validate(&serde_json::json!({"confidence": 0.9, "result": "ok"}))
            .unwrap();
        assert!(schema
            .validate(&serde_json::json!({"result": "refuse"}))
            .is_err());
    }
}
