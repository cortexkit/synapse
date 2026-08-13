//! A byte-level JSON grammar automaton constrained by a compiled [`Schema`].
//!
//! This is the worker-side constraint runtime in model form: at each generation
//! step it computes the set of grammar-permitted continuation bytes (and, by
//! extension, permitted token IDs), and it confirms a valid transition before a
//! content token is committed. It implements exactly the `synapse-json-schema-v1`
//! subset over RFC 8259 JSON byte syntax.
//!
//! The automaton is a deterministic pushdown machine:
//! - a stack of open containers (objects and arrays),
//! - an optional active scalar read (string, number, literal, or object key),
//! - and a `complete` flag set once exactly one top-level JSON value closes.
//!
//! States are cheap, cloneable, and comparable so tests can fork a state across
//! candidate continuations and assert on the resulting permitted-byte sets.

use std::collections::{BTreeSet, VecDeque};

use synapse_core::SpanClass;

use crate::owned_decode_grammar_scheduler::grammar_schema::{
    EnumLiteral, NodeKind, Schema, SchemaType,
};

/// JSON structural whitespace (RFC 8259 §2): space, tab, line feed, carriage
/// return. Permitted between any two structural tokens.
const WHITESPACE: &[u8] = b" \t\n\r";

fn is_whitespace(byte: u8) -> bool {
    WHITESPACE.contains(&byte)
}

/// Whether a byte is a JSON hex digit (`0-9`, `a-f`, `A-F`). The four bytes
/// following a `\u` escape must all be hex digits per RFC 8259.
const fn is_hex_digit(byte: u8) -> bool {
    byte.is_ascii_hexdigit()
}

/// The incremental automaton state. Clone to fork a candidate continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// Open containers; empty means we are at the top level.
    stack: Vec<Frame>,
    /// The active scalar read, if we are inside one.
    scalar: Option<Scalar>,
    /// True once exactly one complete top-level JSON value has closed.
    complete: bool,
}

impl State {
    /// Attribute the next token to JSON structure or a semantic value.
    ///
    /// Object keys are structural because they are fixed by the schema. A token
    /// that opens a scalar is classified after its first byte is stepped, which
    /// keeps the opening quote or first literal byte with the value it begins.
    pub fn token_span_class(&self, automaton: &Automaton, token: &[u8]) -> SpanClass {
        let mut state = self.clone();
        let mut value = semantic_scalar(&state.scalar);
        for &byte in token {
            let Ok(next) = automaton.step(&state, byte) else {
                break;
            };
            value |= semantic_scalar(&next.scalar);
            state = next;
        }
        if value {
            SpanClass::Value
        } else {
            SpanClass::Structural
        }
    }
}

fn semantic_scalar(scalar: &Option<Scalar>) -> bool {
    scalar
        .as_ref()
        .is_some_and(|scalar| !matches!(scalar, Scalar::ObjectKey { .. }))
}

/// An open container frame.
#[derive(Clone, Debug, PartialEq)]
enum Frame {
    Object {
        /// Schema node index of this object.
        node: usize,
        phase: ObjPhase,
        /// Property names already emitted (enforces `additionalProperties:false`
        /// and duplicate-key rejection).
        emitted_keys: BTreeSet<String>,
        /// The schema node governing the value we are about to read (set when the
        /// phase is `Value`).
        pending_value_node: Option<usize>,
        /// The key whose value we are reading/just read.
        active_key: String,
    },
    Array {
        /// Schema node index of this array.
        node: usize,
        phase: ArrPhase,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjPhase {
    /// Expecting a property key (opening quote) or the closing brace.
    KeyOrClose,
    /// Expecting the colon after a key.
    Colon,
    /// Expecting the value for the active key.
    Value,
    /// Expecting a comma or the closing brace.
    CommaOrClose,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArrPhase {
    /// Expecting an item value or the closing bracket.
    ItemOrClose,
    /// Expecting a comma or the closing bracket.
    CommaOrClose,
}

/// An active scalar read.
#[derive(Clone, Debug, PartialEq)]
enum Scalar {
    /// A general JSON string (no enum restriction). `unicode_remaining` counts
    /// the hex digits still owed to an in-progress `\u` escape (zero outside
    /// one).
    String {
        buf: Vec<u8>,
        escape: bool,
        unicode_remaining: u8,
    },
    /// A string restricted to a set of enum members. `members` holds the exact
    /// inner-content bytes (between the quotes, JSON-escaped) of each member.
    EnumString {
        buf: Vec<u8>,
        escape: bool,
        unicode_remaining: u8,
        members: Vec<Vec<u8>>,
    },
    /// An object key restricted to the not-yet-emitted property names.
    ObjectKey { buf: Vec<u8>, allowed: Vec<String> },
    /// A JSON number. `integer_only` forbids fraction and exponent parts.
    Number { num: NumPhase, integer_only: bool },
    /// A number restricted to a set of enum members' JSON number texts.
    EnumNumber { buf: Vec<u8>, members: Vec<String> },
    /// A `true`/`false`/`null` literal, optionally restricted to an enum subset.
    Literal { target: Vec<u8>, pos: usize },
}

/// RFC 8259 number sub-states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumPhase {
    /// Seen a leading `-`, awaiting the integer part.
    Minus,
    /// Integer part is exactly `0` (no further leading digits allowed).
    Zero,
    /// Integer part `[1-9][0-9]*` in progress.
    Int,
    /// Seen the decimal point, awaiting the first fraction digit.
    Dot,
    /// Fraction digits in progress.
    Frac,
    /// Seen `e`/`E`, awaiting sign or first exponent digit.
    ExpE,
    /// Seen the exponent sign, awaiting the first exponent digit.
    ExpSign,
    /// Exponent digits in progress.
    ExpDigit,
}

impl NumPhase {
    /// Whether the number is a syntactically complete value in this phase (so a
    /// structural delimiter may legally terminate it).
    fn is_complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigit)
    }
}

/// A transition error. Maps to `grammar_unsatisfiable` when it reflects that no
/// continuation is selectable; the generation layer interprets it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepError {
    pub message: String,
}

/// The automaton, bound to a compiled schema.
#[derive(Clone, Debug)]
pub struct Automaton {
    schema: Schema,
}

impl Automaton {
    /// Build an automaton for a validated schema.
    pub fn new(schema: Schema) -> Self {
        Self { schema }
    }

    /// The schema this automaton enforces.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The initial state: expecting one top-level value governed by the root.
    pub fn initial(&self) -> State {
        State {
            stack: Vec::new(),
            scalar: None,
            complete: false,
        }
    }

    /// Whether exactly one complete top-level value has closed (string, literal,
    /// or container). Top-level numbers report completion through
    /// [`Automaton::has_complete_value`] because they have no closing delimiter.
    pub fn is_complete(&self, state: &State) -> bool {
        state.complete
    }

    /// Whether a complete top-level value exists, including a top-level number in
    /// a syntactically complete phase. This is the signal the generation layer
    /// uses to end with `finish_reason=grammar_complete`.
    pub fn has_complete_value(&self, state: &State) -> bool {
        if state.complete {
            return true;
        }
        if !state.stack.is_empty() {
            return false;
        }
        match &state.scalar {
            Some(Scalar::Number { num, .. }) => num.is_complete(),
            Some(Scalar::EnumNumber { buf, members }) => members
                .iter()
                .any(|member| member.as_bytes() == buf.as_slice()),
            _ => false,
        }
    }

    /// The set of bytes that are valid continuations from `state`. An empty set
    /// means no content token is selectable (`grammar_unsatisfiable`).
    pub fn permitted_bytes(&self, state: &State) -> BTreeSet<u8> {
        let mut out = BTreeSet::new();
        if state.complete {
            // After a complete top-level value only whitespace is legal; the
            // generation layer stops with grammar_complete rather than emitting it.
            out.extend(WHITESPACE.iter().copied());
            return out;
        }
        if let Some(scalar) = &state.scalar {
            self.permitted_scalar_bytes(state, scalar, &mut out);
            return out;
        }
        // Structural position: leading whitespace is always permitted before the
        // next structural token.
        out.extend(WHITESPACE.iter().copied());
        self.permitted_structural_bytes(state, &mut out);
        out
    }

    fn permitted_scalar_bytes(&self, state: &State, scalar: &Scalar, out: &mut BTreeSet<u8>) {
        match scalar {
            Scalar::String {
                escape,
                unicode_remaining,
                ..
            } => {
                if *unicode_remaining > 0 {
                    // Inside a `\u` escape only the owed hex digits are legal.
                    out.extend(b'0'..=b'9');
                    out.extend(b'a'..=b'f');
                    out.extend(b'A'..=b'F');
                } else if *escape {
                    out.extend(b"\"\\/bfnrtu".iter().copied());
                } else {
                    // Any printable byte except quote and backslash continues the
                    // string; backslash starts an escape; quote closes it.
                    for byte in 0x20u8..=0x7e {
                        if byte != b'"' && byte != b'\\' {
                            out.insert(byte);
                        }
                    }
                    out.insert(b'\\');
                    out.insert(b'"');
                }
            }
            Scalar::EnumString {
                buf,
                escape,
                unicode_remaining,
                members,
            } => {
                if *unicode_remaining > 0 {
                    // Hex digits owed to the escape, restricted to those that
                    // keep at least one member reachable.
                    for byte in permitted_restricted_next(buf, members, true) {
                        if is_hex_digit(byte) {
                            out.insert(byte);
                        }
                    }
                } else if *escape {
                    // Within an escape, permit only escape continuations that keep
                    // at least one member reachable.
                    for byte in permitted_restricted_next(buf, members, true) {
                        out.insert(byte);
                    }
                } else {
                    for byte in permitted_restricted_next(buf, members, false) {
                        out.insert(byte);
                    }
                    // The closing quote is permitted only when the buffer exactly
                    // matches a member's inner content.
                    if members.contains(buf) {
                        out.insert(b'"');
                    }
                }
            }
            Scalar::ObjectKey { buf, allowed } => {
                let allowed_bytes: Vec<Vec<u8>> = allowed
                    .iter()
                    .map(|name| name.as_bytes().to_vec())
                    .collect();
                for byte in permitted_restricted_next(buf, &allowed_bytes, false) {
                    out.insert(byte);
                }
                if allowed_bytes.contains(buf) {
                    out.insert(b'"');
                }
            }
            Scalar::Number { num, integer_only } => {
                out.extend(number_extension_bytes(*num, *integer_only));
                if num.is_complete() {
                    // A complete number may be terminated by any structural byte
                    // valid in the enclosing context.
                    self.append_value_terminators(state, out);
                }
            }
            Scalar::EnumNumber { buf, members } => {
                let member_bytes: Vec<Vec<u8>> = members
                    .iter()
                    .map(|member| member.as_bytes().to_vec())
                    .collect();
                for byte in permitted_restricted_next(buf, &member_bytes, false) {
                    out.insert(byte);
                }
                if members
                    .iter()
                    .any(|member| member.as_bytes() == buf.as_slice())
                {
                    self.append_value_terminators(state, out);
                }
            }
            Scalar::Literal { target, pos } => {
                if *pos < target.len() {
                    out.insert(target[*pos]);
                }
            }
        }
    }

    /// Bytes that may legally follow a completed scalar value in the enclosing
    /// context: structural delimiters for containers, whitespace at top level.
    fn append_value_terminators(&self, state: &State, out: &mut BTreeSet<u8>) {
        match state.stack.last() {
            None => out.extend(WHITESPACE.iter().copied()),
            Some(Frame::Object { .. }) => {
                out.insert(b',');
                out.insert(b'}');
            }
            Some(Frame::Array { .. }) => {
                out.insert(b',');
                out.insert(b']');
            }
        }
    }

    fn permitted_structural_bytes(&self, state: &State, out: &mut BTreeSet<u8>) {
        match state.stack.last() {
            None => {
                // Top level: a value governed by the root node.
                self.append_value_start_bytes(0, out);
            }
            Some(Frame::Object {
                node,
                phase,
                emitted_keys,
                pending_value_node,
                ..
            }) => match phase {
                ObjPhase::KeyOrClose => {
                    let allowed = remaining_property_names(&self.schema, *node, emitted_keys);
                    if !allowed.is_empty() {
                        out.insert(b'"');
                    }
                    if required_satisfied(&self.schema, *node, emitted_keys) {
                        out.insert(b'}');
                    }
                }
                ObjPhase::Colon => {
                    out.insert(b':');
                }
                ObjPhase::Value => {
                    if let Some(value_node) = pending_value_node {
                        self.append_value_start_bytes(*value_node, out);
                    }
                }
                ObjPhase::CommaOrClose => {
                    out.insert(b',');
                    if required_satisfied(&self.schema, *node, emitted_keys) {
                        out.insert(b'}');
                    }
                }
            },
            Some(Frame::Array { node, phase }) => {
                let item_node = item_node_of(&self.schema, *node);
                match phase {
                    ArrPhase::ItemOrClose => {
                        self.append_value_start_bytes(item_node, out);
                        out.insert(b']');
                    }
                    ArrPhase::CommaOrClose => {
                        out.insert(b',');
                        out.insert(b']');
                    }
                }
            }
        }
    }

    /// The bytes that may begin a value governed by schema node `node`.
    fn append_value_start_bytes(&self, node: usize, out: &mut BTreeSet<u8>) {
        let schema_node = self.schema.node(node);
        match schema_node.ty {
            SchemaType::Object => {
                out.insert(b'{');
            }
            SchemaType::Array => {
                out.insert(b'[');
            }
            SchemaType::String => {
                out.insert(b'"');
            }
            SchemaType::Number | SchemaType::Integer => {
                out.insert(b'-');
                out.extend(b'0'..=b'9');
            }
            SchemaType::Boolean => {
                out.insert(b't');
                out.insert(b'f');
            }
            SchemaType::Null => {
                out.insert(b'n');
            }
        }
    }

    /// Advance the state by one byte.
    pub fn step(&self, state: &State, byte: u8) -> Result<State, StepError> {
        let mut state = state.clone();
        if state.complete {
            if is_whitespace(byte) {
                return Ok(state);
            }
            return Err(StepError {
                message: "input after a complete top-level value".to_string(),
            });
        }
        loop {
            let scalar = match state.scalar.clone() {
                Some(scalar) => scalar,
                None => return self.step_structural(&mut state, byte),
            };
            match self.step_scalar(&scalar, byte)? {
                ScalarStep::Consumed(next) => {
                    state.scalar = Some(next);
                    return Ok(state);
                }
                ScalarStep::Closed => {
                    state.scalar = None;
                    match scalar {
                        // An object key closing is not a finished value: it records
                        // the key and moves the frame to the colon phase.
                        Scalar::ObjectKey { buf, .. } => {
                            let key = String::from_utf8(buf).map_err(|_| StepError {
                                message: "object key is not valid UTF-8".to_string(),
                            })?;
                            self.adopt_key(&mut state, key)?;
                            return Ok(state);
                        }
                        // A number closes on a delimiter that is not part of the
                        // number; finish the value and reprocess that delimiter as a
                        // structural byte in the updated context.
                        Scalar::Number { .. } | Scalar::EnumNumber { .. } => {
                            self.finish_value(&mut state);
                            continue;
                        }
                        // Strings and literals close on a byte that is part of the
                        // scalar (the quote or the final literal character), so the
                        // value finishes and the byte is fully consumed.
                        _ => {
                            self.finish_value(&mut state);
                            return Ok(state);
                        }
                    }
                }
            }
        }
    }

    /// Record a just-read object key on the top frame: claim the key, resolve the
    /// schema node that governs its value, and advance to the colon phase.
    fn adopt_key(&self, state: &mut State, key: String) -> Result<(), StepError> {
        let node = match state.stack.last() {
            Some(Frame::Object { node, .. }) => *node,
            _ => {
                return Err(StepError {
                    message: "object key read outside an object".to_string(),
                })
            }
        };
        let value_node = self
            .schema
            .node(node)
            .property_node(&key)
            .ok_or_else(|| StepError {
                message: format!("object key '{key}' is not an allowed property"),
            })?;
        match state.stack.last_mut() {
            Some(Frame::Object {
                phase,
                emitted_keys,
                pending_value_node,
                active_key,
                ..
            }) => {
                emitted_keys.insert(key.clone());
                *pending_value_node = Some(value_node);
                *active_key = key;
                *phase = ObjPhase::Colon;
                Ok(())
            }
            _ => unreachable!("top frame checked above"),
        }
    }

    /// Feed a multi-byte token, returning the resulting state. A token is a valid
    /// transition only if every byte is accepted in sequence.
    pub fn commit_token(&self, state: &State, token: &[u8]) -> Result<State, StepError> {
        let mut state = state.clone();
        for &byte in token {
            state = self.step(&state, byte)?;
        }
        Ok(state)
    }

    /// Whether a token is a valid continuation from `state` (every byte accepted).
    pub fn token_is_permitted(&self, state: &State, token: &[u8]) -> bool {
        if token.is_empty() {
            return false;
        }
        self.commit_token(state, token).is_ok()
    }

    /// Whether constrained decode may select `token` from `state`.
    ///
    /// JSON whitespace is optional between structural values. Suppressing tokens
    /// made entirely of optional whitespace keeps generation moving and matches
    /// the worker's established JSON constraint mask. Whitespace remains legal
    /// inside strings and while terminating scalar lexemes.
    pub fn token_is_decode_permitted(&self, state: &State, token: &[u8]) -> bool {
        if state.scalar.is_none() && token.iter().copied().all(is_whitespace) {
            return false;
        }
        self.token_is_permitted(state, token)
    }

    fn step_scalar(&self, scalar: &Scalar, byte: u8) -> Result<ScalarStep, StepError> {
        match scalar {
            Scalar::String {
                buf,
                escape,
                unicode_remaining,
            } => {
                let mut buf = buf.clone();
                if *unicode_remaining > 0 {
                    // A `\u` escape owes exactly four hex digits (RFC 8259).
                    // Anything else is invalid JSON byte syntax. Surrogate
                    // ranges are not rejected: a lone high-surrogate escape is
                    // legal JSON syntax, and this layer validates hex digits
                    // only, not code-point semantics.
                    if !is_hex_digit(byte) {
                        return Err(StepError {
                            message: "\\u escape must be followed by four hex digits".to_string(),
                        });
                    }
                    buf.push(byte);
                    Ok(ScalarStep::Consumed(Scalar::String {
                        buf,
                        escape: false,
                        unicode_remaining: unicode_remaining - 1,
                    }))
                } else if *escape {
                    Self::validate_escape_byte(byte)?;
                    buf.push(byte);
                    Ok(ScalarStep::Consumed(Scalar::String {
                        buf,
                        escape: false,
                        unicode_remaining: if byte == b'u' { 4 } else { 0 },
                    }))
                } else if byte == b'\\' {
                    buf.push(byte);
                    Ok(ScalarStep::Consumed(Scalar::String {
                        buf,
                        escape: true,
                        unicode_remaining: 0,
                    }))
                } else if byte == b'"' {
                    Ok(ScalarStep::Closed)
                } else if byte < 0x20 {
                    Err(StepError {
                        message: "unescaped control character in string".to_string(),
                    })
                } else {
                    buf.push(byte);
                    Ok(ScalarStep::Consumed(Scalar::String {
                        buf,
                        escape: false,
                        unicode_remaining: 0,
                    }))
                }
            }
            Scalar::EnumString {
                buf,
                escape,
                unicode_remaining,
                members,
            } => {
                let mut buf = buf.clone();
                if *unicode_remaining > 0 {
                    // A `\u` escape owes exactly four hex digits (RFC 8259);
                    // hex validation only (see the general string arm for the
                    // surrogate note).
                    if !is_hex_digit(byte) {
                        return Err(StepError {
                            message: "\\u escape must be followed by four hex digits".to_string(),
                        });
                    }
                    buf.push(byte);
                    if !any_member_has_prefix(&buf, members) {
                        return Err(StepError {
                            message: "string enum escape leads to no member".to_string(),
                        });
                    }
                    Ok(ScalarStep::Consumed(Scalar::EnumString {
                        buf,
                        escape: false,
                        unicode_remaining: unicode_remaining - 1,
                        members: members.clone(),
                    }))
                } else if *escape {
                    Self::validate_escape_byte(byte)?;
                    buf.push(byte);
                    if !any_member_has_prefix(&buf, members) {
                        return Err(StepError {
                            message: "string enum escape leads to no member".to_string(),
                        });
                    }
                    Ok(ScalarStep::Consumed(Scalar::EnumString {
                        buf,
                        escape: false,
                        unicode_remaining: if byte == b'u' { 4 } else { 0 },
                        members: members.clone(),
                    }))
                } else if byte == b'\\' {
                    buf.push(byte);
                    if !any_member_has_prefix(&buf, members) {
                        return Err(StepError {
                            message: "string enum has no member with this escape".to_string(),
                        });
                    }
                    Ok(ScalarStep::Consumed(Scalar::EnumString {
                        buf,
                        escape: true,
                        unicode_remaining: 0,
                        members: members.clone(),
                    }))
                } else if byte == b'"' {
                    if members.contains(&buf) {
                        Ok(ScalarStep::Closed)
                    } else {
                        Err(StepError {
                            message: "string is not an enum member".to_string(),
                        })
                    }
                } else {
                    buf.push(byte);
                    if !any_member_has_prefix(&buf, members) {
                        return Err(StepError {
                            message: "string enum has no member with this prefix".to_string(),
                        });
                    }
                    Ok(ScalarStep::Consumed(Scalar::EnumString {
                        buf,
                        escape: false,
                        unicode_remaining: 0,
                        members: members.clone(),
                    }))
                }
            }
            Scalar::ObjectKey { buf, allowed } => {
                let mut buf = buf.clone();
                if byte == b'"' {
                    let key = String::from_utf8(buf.clone()).map_err(|_| StepError {
                        message: "object key is not valid UTF-8".to_string(),
                    })?;
                    if allowed.contains(&key) {
                        Ok(ScalarStep::Closed)
                    } else {
                        Err(StepError {
                            message: "object key is not an allowed property".to_string(),
                        })
                    }
                } else {
                    buf.push(byte);
                    let allowed_bytes: Vec<Vec<u8>> = allowed
                        .iter()
                        .map(|name| name.as_bytes().to_vec())
                        .collect();
                    if !any_member_has_prefix(&buf, &allowed_bytes) {
                        return Err(StepError {
                            message: "object key matches no allowed property".to_string(),
                        });
                    }
                    Ok(ScalarStep::Consumed(Scalar::ObjectKey {
                        buf,
                        allowed: allowed.clone(),
                    }))
                }
            }
            Scalar::Number { num, integer_only } => match number_step(*num, *integer_only, byte) {
                NumberStep::Extend(next) => Ok(ScalarStep::Consumed(Scalar::Number {
                    num: next,
                    integer_only: *integer_only,
                })),
                NumberStep::Close => Ok(ScalarStep::Closed),
                NumberStep::Reject => Err(StepError {
                    message: "invalid number continuation".to_string(),
                }),
            },
            Scalar::EnumNumber { buf, members } => {
                let mut buf = buf.clone();
                buf.push(byte);
                if members
                    .iter()
                    .any(|member| member.as_bytes() == buf.as_slice())
                {
                    // Exactly matches a member; the next byte (a delimiter) closes.
                    // Stay in the scalar; a non-matching extension byte will be
                    // rejected below, and a delimiter triggers Close via the
                    // prefix check failing to extend.
                    return Ok(ScalarStep::Consumed(Scalar::EnumNumber {
                        buf,
                        members: members.clone(),
                    }));
                }
                if any_member_has_prefix_str(&buf, members) {
                    Ok(ScalarStep::Consumed(Scalar::EnumNumber {
                        buf,
                        members: members.clone(),
                    }))
                } else {
                    // The byte did not extend toward any member: if the buffer
                    // before this byte was a complete member, treat this byte as a
                    // delimiter that closes the number.
                    let prior = &buf[..buf.len() - 1];
                    if members.iter().any(|member| member.as_bytes() == prior) {
                        Ok(ScalarStep::Closed)
                    } else {
                        Err(StepError {
                            message: "number is not an enum member".to_string(),
                        })
                    }
                }
            }
            Scalar::Literal { target, pos } => {
                let pos = *pos;
                if pos < target.len() && target[pos] == byte {
                    let pos = pos + 1;
                    if pos == target.len() {
                        Ok(ScalarStep::Closed)
                    } else {
                        Ok(ScalarStep::Consumed(Scalar::Literal {
                            target: target.clone(),
                            pos,
                        }))
                    }
                } else {
                    Err(StepError {
                        message: "literal does not match true/false/null".to_string(),
                    })
                }
            }
        }
    }

    fn validate_escape_byte(byte: u8) -> Result<(), StepError> {
        // `u` is accepted here; the four hex digits that follow are validated
        // by the `unicode_remaining` counter in the string arms (RFC 8259
        // requires exactly four).
        if b"\"\\/bfnrtu".contains(&byte) {
            Ok(())
        } else {
            Err(StepError {
                message: "invalid string escape sequence".to_string(),
            })
        }
    }

    fn step_structural(&self, state: &mut State, byte: u8) -> Result<State, StepError> {
        if is_whitespace(byte) {
            return Ok(state.clone());
        }
        // Determine the governing node and the structural role of this byte.
        match state.stack.last().cloned() {
            None => {
                // Top-level value start.
                self.begin_value(state, 0, byte)?;
                Ok(state.clone())
            }
            Some(Frame::Object {
                node,
                phase,
                emitted_keys,
                pending_value_node,
                active_key: _,
            }) => match phase {
                ObjPhase::KeyOrClose => {
                    if byte == b'}' {
                        if !required_satisfied(&self.schema, node, &emitted_keys) {
                            return Err(StepError {
                                message: "object closed before required properties".to_string(),
                            });
                        }
                        state.stack.pop();
                        self.finish_value(state);
                        return Ok(state.clone());
                    }
                    if byte == b'"' {
                        let allowed = remaining_property_names(&self.schema, node, &emitted_keys);
                        if allowed.is_empty() {
                            return Err(StepError {
                                message: "no further properties allowed".to_string(),
                            });
                        }
                        state.scalar = Some(Scalar::ObjectKey {
                            buf: Vec::new(),
                            allowed,
                        });
                        return Ok(state.clone());
                    }
                    Err(StepError {
                        message: "expected object key or closing brace".to_string(),
                    })
                }
                ObjPhase::Colon => {
                    if byte == b':' {
                        Self::replace_top_phase_obj(state, ObjPhase::Value);
                        Ok(state.clone())
                    } else {
                        Err(StepError {
                            message: "expected colon after object key".to_string(),
                        })
                    }
                }
                ObjPhase::Value => {
                    let value_node = pending_value_node.ok_or_else(|| StepError {
                        message: "object value phase missing its node".to_string(),
                    })?;
                    self.begin_value(state, value_node, byte)?;
                    Ok(state.clone())
                }
                ObjPhase::CommaOrClose => {
                    if byte == b'}' {
                        if !required_satisfied(&self.schema, node, &emitted_keys) {
                            return Err(StepError {
                                message: "object closed before required properties".to_string(),
                            });
                        }
                        state.stack.pop();
                        self.finish_value(state);
                        return Ok(state.clone());
                    }
                    if byte == b',' {
                        Self::replace_top_phase_obj(state, ObjPhase::KeyOrClose);
                        return Ok(state.clone());
                    }
                    Err(StepError {
                        message: "expected comma or closing brace in object".to_string(),
                    })
                }
            },
            Some(Frame::Array { node, phase }) => match phase {
                ArrPhase::ItemOrClose => {
                    if byte == b']' {
                        state.stack.pop();
                        self.finish_value(state);
                        return Ok(state.clone());
                    }
                    let item_node = item_node_of(&self.schema, node);
                    self.begin_value(state, item_node, byte)?;
                    Ok(state.clone())
                }
                ArrPhase::CommaOrClose => {
                    if byte == b']' {
                        state.stack.pop();
                        self.finish_value(state);
                        return Ok(state.clone());
                    }
                    if byte == b',' {
                        Self::replace_top_phase_arr(state, ArrPhase::ItemOrClose);
                        return Ok(state.clone());
                    }
                    Err(StepError {
                        message: "expected comma or closing bracket in array".to_string(),
                    })
                }
            },
        }
    }

    /// Begin reading a value governed by schema node `node`, given its first byte.
    fn begin_value(&self, state: &mut State, node: usize, byte: u8) -> Result<(), StepError> {
        let schema_node = self.schema.node(node);
        match schema_node.ty {
            SchemaType::Object => {
                if byte != b'{' {
                    return Err(StepError {
                        message: "expected an object value".to_string(),
                    });
                }
                state.stack.push(Frame::Object {
                    node,
                    phase: ObjPhase::KeyOrClose,
                    emitted_keys: BTreeSet::new(),
                    pending_value_node: None,
                    active_key: String::new(),
                });
                Ok(())
            }
            SchemaType::Array => {
                if byte != b'[' {
                    return Err(StepError {
                        message: "expected an array value".to_string(),
                    });
                }
                state.stack.push(Frame::Array {
                    node,
                    phase: ArrPhase::ItemOrClose,
                });
                Ok(())
            }
            SchemaType::String => {
                if byte != b'"' {
                    return Err(StepError {
                        message: "expected a string value".to_string(),
                    });
                }
                state.scalar = Some(self.begin_string_scalar(node));
                Ok(())
            }
            SchemaType::Number | SchemaType::Integer => {
                let integer_only = schema_node.ty == SchemaType::Integer;
                state.scalar = Some(self.begin_number_scalar(node, integer_only, byte)?);
                Ok(())
            }
            SchemaType::Boolean => {
                let target = match byte {
                    b't' => b"true".to_vec(),
                    b'f' => b"false".to_vec(),
                    _ => {
                        return Err(StepError {
                            message: "expected a boolean value".to_string(),
                        })
                    }
                };
                self.check_literal_allowed(node, &target)?;
                state.scalar = Some(Scalar::Literal { target, pos: 1 });
                Ok(())
            }
            SchemaType::Null => {
                if byte != b'n' {
                    return Err(StepError {
                        message: "expected a null value".to_string(),
                    });
                }
                let target = b"null".to_vec();
                self.check_literal_allowed(node, &target)?;
                state.scalar = Some(Scalar::Literal { target, pos: 1 });
                Ok(())
            }
        }
    }

    fn begin_string_scalar(&self, node: usize) -> Scalar {
        match &self.schema.node(node).kind {
            NodeKind::Scalar {
                enumeration: Some(enumeration),
            } => {
                let members = enumeration
                    .iter()
                    .filter_map(|literal| match literal {
                        EnumLiteral::Str(text) => Some(enum_string_inner(text)),
                        _ => None,
                    })
                    .collect();
                Scalar::EnumString {
                    buf: Vec::new(),
                    escape: false,
                    unicode_remaining: 0,
                    members,
                }
            }
            _ => Scalar::String {
                buf: Vec::new(),
                escape: false,
                unicode_remaining: 0,
            },
        }
    }

    fn begin_number_scalar(
        &self,
        node: usize,
        integer_only: bool,
        byte: u8,
    ) -> Result<Scalar, StepError> {
        match &self.schema.node(node).kind {
            NodeKind::Scalar {
                enumeration: Some(enumeration),
            } => {
                let members: Vec<String> = enumeration
                    .iter()
                    .filter_map(|literal| match literal {
                        EnumLiteral::Number(value) => Some(enum_number_text(*value)),
                        EnumLiteral::Integer(value) => Some(value.to_string()),
                        _ => None,
                    })
                    .collect();
                let buf = vec![byte];
                if !any_member_has_prefix_str(&buf, &members) {
                    return Err(StepError {
                        message: "number enum has no member with this prefix".to_string(),
                    });
                }
                Ok(Scalar::EnumNumber { buf, members })
            }
            _ => {
                let num = match byte {
                    b'-' => NumPhase::Minus,
                    b'0' => NumPhase::Zero,
                    b'1'..=b'9' => NumPhase::Int,
                    _ => {
                        return Err(StepError {
                            message: "invalid number start".to_string(),
                        })
                    }
                };
                Ok(Scalar::Number { num, integer_only })
            }
        }
    }

    fn check_literal_allowed(&self, node: usize, target: &[u8]) -> Result<(), StepError> {
        if let NodeKind::Scalar {
            enumeration: Some(enumeration),
        } = &self.schema.node(node).kind
        {
            let text = std::str::from_utf8(target).unwrap_or("");
            let allowed = enumeration.iter().any(|literal| match literal {
                EnumLiteral::Bool(value) => value.to_string() == text,
                EnumLiteral::Null => text == "null",
                _ => false,
            });
            if !allowed {
                return Err(StepError {
                    message: "literal is not an enum member".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Record that a scalar/container value just finished, advancing the enclosing
    /// context (or marking top-level completion).
    fn finish_value(&self, state: &mut State) {
        match state.stack.last_mut() {
            None => {
                state.complete = true;
            }
            Some(Frame::Object {
                phase, active_key, ..
            }) => {
                *phase = ObjPhase::CommaOrClose;
                active_key.clear();
            }
            Some(Frame::Array { phase, .. }) => {
                *phase = ArrPhase::CommaOrClose;
            }
        }
    }

    fn replace_top_phase_obj(state: &mut State, phase: ObjPhase) {
        if let Some(Frame::Object { phase: current, .. }) = state.stack.last_mut() {
            *current = phase;
        }
    }

    fn replace_top_phase_arr(state: &mut State, phase: ArrPhase) {
        if let Some(Frame::Array { phase: current, .. }) = state.stack.last_mut() {
            *current = phase;
        }
    }
}

/// The outcome of feeding one byte to an active scalar.
enum ScalarStep {
    /// The byte was consumed and the scalar continues in this new form.
    Consumed(Scalar),
    /// The scalar closed; the byte may need structural reprocessing (numbers).
    Closed,
}

enum NumberStep {
    Extend(NumPhase),
    Close,
    Reject,
}

/// Advance a number sub-state by one byte. A structural delimiter closes a
/// complete number; digits and number punctuation extend it.
fn number_step(num: NumPhase, integer_only: bool, byte: u8) -> NumberStep {
    let is_delimiter = byte == b',' || byte == b'}' || byte == b']' || is_whitespace(byte);
    match num {
        NumPhase::Minus => match byte {
            b'0' => NumberStep::Extend(NumPhase::Zero),
            b'1'..=b'9' => NumberStep::Extend(NumPhase::Int),
            _ => NumberStep::Reject,
        },
        NumPhase::Zero => match byte {
            b'.' if !integer_only => NumberStep::Extend(NumPhase::Dot),
            b'e' | b'E' if !integer_only => NumberStep::Extend(NumPhase::ExpE),
            _ if is_delimiter => NumberStep::Close,
            _ => NumberStep::Reject,
        },
        NumPhase::Int => match byte {
            b'0'..=b'9' => NumberStep::Extend(NumPhase::Int),
            b'.' if !integer_only => NumberStep::Extend(NumPhase::Dot),
            b'e' | b'E' if !integer_only => NumberStep::Extend(NumPhase::ExpE),
            _ if is_delimiter => NumberStep::Close,
            _ => NumberStep::Reject,
        },
        NumPhase::Dot => match byte {
            b'0'..=b'9' => NumberStep::Extend(NumPhase::Frac),
            _ => NumberStep::Reject,
        },
        NumPhase::Frac => match byte {
            b'0'..=b'9' => NumberStep::Extend(NumPhase::Frac),
            b'e' | b'E' => NumberStep::Extend(NumPhase::ExpE),
            _ if is_delimiter => NumberStep::Close,
            _ => NumberStep::Reject,
        },
        NumPhase::ExpE => match byte {
            b'+' | b'-' => NumberStep::Extend(NumPhase::ExpSign),
            b'0'..=b'9' => NumberStep::Extend(NumPhase::ExpDigit),
            _ => NumberStep::Reject,
        },
        NumPhase::ExpSign => match byte {
            b'0'..=b'9' => NumberStep::Extend(NumPhase::ExpDigit),
            _ => NumberStep::Reject,
        },
        NumPhase::ExpDigit => match byte {
            b'0'..=b'9' => NumberStep::Extend(NumPhase::ExpDigit),
            _ if is_delimiter => NumberStep::Close,
            _ => NumberStep::Reject,
        },
    }
}

/// The bytes that may extend a number in the given phase (excluding delimiters).
fn number_extension_bytes(num: NumPhase, integer_only: bool) -> BTreeSet<u8> {
    let mut out = BTreeSet::new();
    match num {
        NumPhase::Minus => {
            out.extend(b'0'..=b'9');
        }
        NumPhase::Zero => {
            if !integer_only {
                out.insert(b'.');
                out.insert(b'e');
                out.insert(b'E');
            }
        }
        NumPhase::Int => {
            out.extend(b'0'..=b'9');
            if !integer_only {
                out.insert(b'.');
                out.insert(b'e');
                out.insert(b'E');
            }
        }
        NumPhase::Dot => {
            out.extend(b'0'..=b'9');
        }
        NumPhase::Frac => {
            out.extend(b'0'..=b'9');
            out.insert(b'e');
            out.insert(b'E');
        }
        NumPhase::ExpE => {
            out.insert(b'+');
            out.insert(b'-');
            out.extend(b'0'..=b'9');
        }
        NumPhase::ExpSign => {
            out.extend(b'0'..=b'9');
        }
        NumPhase::ExpDigit => {
            out.extend(b'0'..=b'9');
        }
    }
    out
}

/// The inner-content bytes of an enum string member: the JSON serialization with
/// the surrounding quotes stripped, leaving exactly the bytes that appear between
/// the quotes on the wire (including any JSON escape sequences).
fn enum_string_inner(text: &str) -> Vec<u8> {
    let serialized = serde_json::to_string(text).expect("string literal serializes");
    let bytes = serialized.as_bytes();
    // Strip the leading and trailing quote characters.
    bytes[1..bytes.len() - 1].to_vec()
}

fn enum_number_text(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        serde_json::Number::from_f64(value)
            .map(|number| number.to_string())
            .unwrap_or_else(|| value.to_string())
    }
}

/// Whether any member has `buf` as a byte prefix (or equals it).
fn any_member_has_prefix(buf: &[u8], members: &[Vec<u8>]) -> bool {
    members
        .iter()
        .any(|member| member.len() >= buf.len() && &member[..buf.len()] == buf)
}

fn any_member_has_prefix_str(buf: &[u8], members: &[String]) -> bool {
    members
        .iter()
        .any(|member| member.len() >= buf.len() && &member.as_bytes()[..buf.len()] == buf)
}

/// The set of bytes that keep at least one member reachable as a continuation of
/// `buf`. When `in_escape` is set, the backslash already in `buf` is treated as
/// part of the prefix comparison directly.
fn permitted_restricted_next(buf: &[u8], members: &[Vec<u8>], _in_escape: bool) -> BTreeSet<u8> {
    let mut out = BTreeSet::new();
    for member in members {
        if member.len() > buf.len() && &member[..buf.len()] == buf {
            out.insert(member[buf.len()]);
        }
    }
    out
}

/// The not-yet-emitted property names of an object node, in declaration order.
fn remaining_property_names(
    schema: &Schema,
    node: usize,
    emitted: &BTreeSet<String>,
) -> Vec<String> {
    match &schema.node(node).kind {
        NodeKind::Object { properties, .. } => properties
            .iter()
            .map(|(name, _)| name.clone())
            .filter(|name| !emitted.contains(name))
            .collect(),
        _ => Vec::new(),
    }
}

/// Whether every required property of an object node has been emitted.
fn required_satisfied(schema: &Schema, node: usize, emitted: &BTreeSet<String>) -> bool {
    match &schema.node(node).kind {
        NodeKind::Object { required, .. } => required.is_subset(emitted),
        _ => true,
    }
}

fn item_node_of(schema: &Schema, node: usize) -> usize {
    match &schema.node(node).kind {
        NodeKind::Array { items } => *items,
        _ => node,
    }
}

/// A small helper used by the generation layer to scan a queue of candidate
/// tokens and keep only those that are valid continuations. Exposed so the
/// compiler/generation code and tests share one masking implementation.
pub fn mask_tokens(automaton: &Automaton, state: &State, vocabulary: &[String]) -> Vec<usize> {
    let mut permitted = Vec::new();
    for (index, token) in vocabulary.iter().enumerate() {
        if automaton.token_is_permitted(state, token.as_bytes()) {
            permitted.push(index);
        }
    }
    permitted
}

/// Drain a queue of bytes through the automaton, returning the final state. Used
/// by tests to assert that a whole document is accepted.
pub fn drain_bytes(automaton: &Automaton, bytes: &[u8]) -> Result<State, StepError> {
    let mut state = automaton.initial();
    let mut queue: VecDeque<u8> = bytes.iter().copied().collect();
    while let Some(byte) = queue.pop_front() {
        state = automaton.step(&state, byte)?;
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_grammar_scheduler::grammar_limits::GrammarLimits;
    use crate::owned_decode_grammar_scheduler::grammar_schema::parse_schema;

    fn automaton(raw: &str) -> Automaton {
        let schema = parse_schema(raw, &GrammarLimits::default()).expect("schema parses");
        Automaton::new(schema)
    }

    fn accept(raw_schema: &str, document: &str) {
        let automaton = automaton(raw_schema);
        let state = drain_bytes(&automaton, document.as_bytes())
            .unwrap_or_else(|err| panic!("document '{document}' rejected: {}", err.message));
        assert!(
            automaton.has_complete_value(&state),
            "document '{document}' did not complete a value"
        );
    }

    fn reject(raw_schema: &str, document: &str) {
        let automaton = automaton(raw_schema);
        let result = drain_bytes(&automaton, document.as_bytes());
        let state = match result {
            Err(_) => return, // rejected mid-stream, as expected
            Ok(state) => state,
        };
        assert!(
            !automaton.has_complete_value(&state),
            "document '{document}' unexpectedly completed a value"
        );
    }

    const OBJECT_SCHEMA: &str = r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    }"#;

    #[test]
    fn accepts_a_valid_object() {
        accept(OBJECT_SCHEMA, r#"{"name":"ada","age":36}"#);
    }

    #[test]
    fn accepts_object_with_whitespace() {
        accept(
            OBJECT_SCHEMA,
            "{\n  \"name\" : \"ada\" ,\n  \"age\" : 36\n}",
        );
    }

    #[test]
    fn rejects_missing_required_property() {
        reject(OBJECT_SCHEMA, r#"{"name":"ada"}"#);
    }

    #[test]
    fn rejects_unknown_property() {
        reject(OBJECT_SCHEMA, r#"{"name":"ada","age":36,"extra":1}"#);
    }

    #[test]
    fn rejects_wrong_value_type() {
        reject(OBJECT_SCHEMA, r#"{"name":"ada","age":"old"}"#);
    }

    #[test]
    fn accepts_array_of_strings() {
        let schema = r#"{ "type": "array", "items": { "type": "string" } }"#;
        accept(schema, r#"["a","b","c"]"#);
        accept(schema, "[]");
    }

    #[test]
    fn rejects_array_with_wrong_item_type() {
        let schema = r#"{ "type": "array", "items": { "type": "integer" } }"#;
        reject(schema, r#"[1,"two"]"#);
    }

    #[test]
    fn accepts_nested_object() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                    "additionalProperties": false
                }
            },
            "required": ["address"],
            "additionalProperties": false
        }"#;
        accept(schema, r#"{"address":{"city":"paris"}}"#);
        reject(schema, r#"{"address":{}}"#);
    }

    #[test]
    fn string_enum_restricts_output() {
        let schema = r#"{ "type": "string", "enum": ["red", "green", "blue"] }"#;
        accept(schema, r#""green""#);
        reject(schema, r#""yellow""#);

        let automaton = automaton(schema);
        let state = automaton.initial();
        // After the opening quote, only member-first-bytes are permitted.
        let after_quote = automaton.step(&state, b'"').expect("open quote");
        let permitted = automaton.permitted_bytes(&after_quote);
        // All members start with distinct letters r/g/b.
        assert!(permitted.contains(&b'r'));
        assert!(permitted.contains(&b'g'));
        assert!(permitted.contains(&b'b'));
        assert!(!permitted.contains(&b'y'));
    }

    #[test]
    fn accepts_boolean_and_null() {
        accept(r#"{ "type": "boolean" }"#, "true");
        accept(r#"{ "type": "boolean" }"#, "false");
        accept(r#"{ "type": "null" }"#, "null");
        reject(r#"{ "type": "boolean" }"#, "null");
    }

    #[test]
    fn accepts_numbers_and_integers() {
        accept(r#"{ "type": "integer" }"#, "42");
        accept(r#"{ "type": "integer" }"#, "-7");
        accept(r#"{ "type": "number" }"#, "3.14");
        accept(r#"{ "type": "number" }"#, "-0.5e2");
        // An integer node rejects a fractional value.
        reject(r#"{ "type": "integer" }"#, "3.14");
        // Leading zero is not valid JSON.
        reject(r#"{ "type": "integer" }"#, "007");
    }

    #[test]
    fn integer_in_object_closes_on_brace() {
        // The number must finalize on the `}` delimiter and then close the object.
        accept(OBJECT_SCHEMA, r#"{"name":"ada","age":36}"#);
    }

    #[test]
    fn permitted_bytes_empty_means_unsatisfiable() {
        // A boolean node after consuming "true" is complete; a fresh `x` is not a
        // valid start, so an object schema with no remaining valid key yields no
        // permitted structural bytes once closed.
        let schema = r#"{
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": ["a"],
            "additionalProperties": false
        }"#;
        let automaton = automaton(schema);
        // After emitting the only property, only `}` is permitted (no more keys).
        let state = drain_bytes(&automaton, br#"{"a":"x""#).expect("prefix accepted");
        let permitted = automaton.permitted_bytes(&state);
        assert!(permitted.contains(&b'}'));
        assert!(!permitted.contains(&b'"'), "no further keys allowed");
    }

    #[test]
    fn token_masking_keeps_only_valid_continuations() {
        let automaton = automaton(r#"{ "type": "string" }"#);
        let state = automaton.initial();
        let vocabulary = vec![
            "\"".to_string(),  // valid: opens the string
            "{".to_string(),   // invalid: object open, not a string
            "abc".to_string(), // invalid: bare letters at top level
        ];
        let permitted = mask_tokens(&automaton, &state, &vocabulary);
        assert_eq!(permitted, vec![0]);
    }

    #[test]
    fn decode_mask_suppresses_only_structural_whitespace() {
        let automaton = automaton(r#"{ "type": "string" }"#);
        let initial = automaton.initial();
        assert!(!automaton.token_is_decode_permitted(&initial, b" "));
        assert!(automaton.token_is_decode_permitted(&initial, b"\""));

        let inside_string = automaton
            .commit_token(&initial, b"\"")
            .expect("string opener is permitted");
        assert!(automaton.token_is_decode_permitted(&inside_string, b" "));
    }

    #[test]
    fn string_escapes_are_accepted() {
        accept(r#"{ "type": "string" }"#, r#""a\"b\\c""#);
        accept(r#"{ "type": "string" }"#, r#""line\nbreak""#);
    }

    #[test]
    fn unicode_escape_requires_four_hex_digits() {
        // RFC 8259: `\u` must be followed by exactly four hex digits.
        // Audit probes: non-hex digits and a short sequence are rejected.
        reject(r#"{ "type": "string" }"#, r#""a\uZZZZb""#);
        reject(r#"{ "type": "string" }"#, r#""\u41""#);
        reject(r#"{ "type": "string" }"#, r#""a\u00ZZ""#);
        // A quote where a hex digit is owed is rejected too.
        reject(r#"{ "type": "string" }"#, r#""\u""#);
    }

    #[test]
    fn valid_unicode_escapes_are_accepted_including_lone_surrogates() {
        // Exactly four hex digits is valid JSON escape syntax...
        accept(r#"{ "type": "string" }"#, r#""a\u0041b""#);
        accept(r#"{ "type": "string" }"#, r#""\u00e9""#);
        // ...and a lone high-surrogate escape is legal JSON syntax as well:
        // the automaton validates hex digits only, never code-point semantics,
        // so it must not over-reject surrogate ranges.
        accept(r#"{ "type": "string" }"#, r#""\ud800""#);
        accept(r#"{ "type": "string" }"#, r#""\uD83D\uDE00""#);
    }

    #[test]
    fn masking_permits_only_hex_digits_inside_unicode_escape() {
        // After `\u`, the permitted set is exactly the hex digits; a wrong
        // implementation that kept the general string continuations would
        // permit quotes and letters like `z` and fail here.
        let automaton = automaton(r#"{ "type": "string" }"#);
        let state = drain_bytes(&automaton, br#""\u"#).expect("escape start accepted");
        let permitted = automaton.permitted_bytes(&state);
        assert!(!permitted.is_empty());
        for byte in &permitted {
            assert!(
                byte.is_ascii_hexdigit(),
                "only hex digits are permitted inside a \\u escape, got {:?}",
                *byte as char
            );
        }
        assert_eq!(permitted.len(), 22, "ten digits plus twelve a-f/A-F");
    }
}
