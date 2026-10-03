//! Template oracle: the Qwen3-Reranker template literals as the upstream model
//! card states them.
//!
//! The oracle is a committed copy of the model card at the pinned revision
//! (`oracles/qwen3-reranker-0.6b-README.md`). The card's Transformers usage
//! example defines the template as Python literals; this module reads them
//! from the first such example:
//!
//! - `instruction = '...'`, the default instruction inside `format_instruction`;
//! - the `"<Instruct>: ...".format(...)` body string;
//! - `prefix = "..."` and `suffix = "..."`.

use crate::manifest::Template;
use crate::{perr, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleTemplate {
    pub prefix: String,
    pub instruction: String,
    pub body_format: String,
    pub suffix: String,
}

/// Decode one Python string literal (single or double quoted, no prefix)
/// starting at the beginning of `text`. Returns the value and the rest.
fn python_literal(text: &str) -> Result<(String, &str)> {
    let mut chars = text.char_indices();
    let quote = match chars.next() {
        Some((_, q @ ('"' | '\''))) => q,
        _ => {
            return Err(perr!(
                "expected a Python string literal at `{}`",
                preview(text)
            ))
        }
    };
    let mut value = String::new();
    while let Some((index, ch)) = chars.next() {
        match ch {
            '\\' => {
                let (_, escaped) = chars
                    .next()
                    .ok_or_else(|| perr!("unterminated escape in Python literal"))?;
                value.push(match escaped {
                    'n' => '\n',
                    't' => '\t',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    other => return Err(perr!("unsupported escape `\\{other}` in Python literal")),
                });
            }
            c if c == quote => return Ok((value, &text[index + 1..])),
            '\n' => return Err(perr!("newline inside a single-line Python literal")),
            c => value.push(c),
        }
    }
    Err(perr!("unterminated Python literal at `{}`", preview(text)))
}

fn preview(text: &str) -> String {
    text.chars().take(40).collect()
}

/// The literal assigned by the first line whose stripped form starts with
/// `<name> = ` and continues with a string literal.
fn assigned_literal(readme: &str, name: &str) -> Result<String> {
    let lead = format!("{name} = ");
    for line in readme.lines() {
        let stripped = line.trim_start();
        if let Some(rest) = stripped.strip_prefix(&lead) {
            if rest.starts_with('"') || rest.starts_with('\'') {
                return Ok(python_literal(rest)?.0);
            }
        }
    }
    Err(perr!("the oracle has no `{name} = \"...\"` assignment"))
}

/// The first string literal that begins with `marker`, e.g. the template body.
fn literal_starting_with(readme: &str, marker: &str) -> Result<String> {
    for quote in ['"', '\''] {
        let needle = format!("{quote}{marker}");
        if let Some(start) = readme.find(&needle) {
            return Ok(python_literal(&readme[start..])?.0);
        }
    }
    Err(perr!(
        "the oracle has no string literal starting with `{marker}`"
    ))
}

pub fn parse_oracle(readme: &str) -> Result<OracleTemplate> {
    Ok(OracleTemplate {
        prefix: assigned_literal(readme, "prefix")?,
        instruction: assigned_literal(readme, "instruction")?,
        body_format: literal_starting_with(readme, "<Instruct>: ")?,
        suffix: assigned_literal(readme, "suffix")?,
    })
}

/// Fail unless every manifest template literal equals the oracle's.
pub fn check_template(template: &Template, readme: &str) -> Result<()> {
    let oracle = parse_oracle(readme)?;
    let pairs = [
        ("prefix", &template.prefix, &oracle.prefix),
        ("instruction", &template.instruction, &oracle.instruction),
        ("body_format", &template.body_format, &oracle.body_format),
        ("suffix", &template.suffix, &oracle.suffix),
    ];
    for (name, manifest, oracle) in pairs {
        if manifest != oracle {
            return Err(perr!(
                "qwen_template_mismatch: template `{name}` {manifest:?} differs from the oracle {oracle:?}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_literals_decode_escapes() {
        let (value, rest) = python_literal(r#""a\nb \"c\"" tail"#).unwrap();
        assert_eq!(value, "a\nb \"c\"");
        assert_eq!(rest, " tail");
        assert_eq!(python_literal("'it\\'s'").unwrap().0, "it's");
        assert!(python_literal("\"open").is_err());
    }
}
