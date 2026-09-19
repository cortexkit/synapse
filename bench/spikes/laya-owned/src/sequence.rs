use anyhow::{anyhow, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

pub fn python_json(v: &Value) -> String {
    match v {
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), python_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => v.to_string(),
    }
}
fn text(v: &Value) -> String {
    v.as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| python_json(v))
}

pub fn render_options(q: &Value) -> Vec<String> {
    let crit = &q["criteria"];
    match q["type"].as_str().unwrap() {
        "choice" => {
            if let Some(list) = crit.as_array() {
                return list.iter().map(text).collect();
            }
            crit.as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| {
                    if v.is_null() || v.as_str() == Some("") {
                        k.clone()
                    } else {
                        format!("{k}: {}", text(v))
                    }
                })
                .collect()
        }
        "score" => crit
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| format!("level {i}: {}", text(v)))
            .collect(),
        "noul" => [
            ("false", "no, the statement does not hold"),
            ("true", "yes, the statement holds"),
        ]
        .into_iter()
        .map(|(key, default)| {
            let v = &crit[key];
            format!(
                "{key}: {}",
                if v.is_null() || v.as_str() == Some("") {
                    default.into()
                } else {
                    text(v)
                }
            )
        })
        .collect(),
        _ => panic!("unknown question type"),
    }
}

pub fn build_sequence(
    tok: &Tokenizer,
    state: &Value,
    q: &Value,
    max_len: usize,
    head_max_len: usize,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let mask = "[MASK]";
    let id = |s: &str| {
        tok.token_to_id(s)
            .ok_or_else(|| anyhow!("missing token {s}"))
    };
    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tok
            .encode(s.replace(mask, " "), false)
            .map_err(|e| anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    };
    let mut head = encode(&format!(
        "{} question: {}",
        q["type"].as_str().unwrap(),
        text(&q["instructions"])
    ))?;
    let mut options = Vec::new();
    for opt in render_options(q) {
        let mut tokens = encode(&format!(" {opt}"))?;
        tokens.truncate(48);
        tokens.insert(0, id(mask)?);
        options.push(tokens);
    }
    let mut budget = head_max_len as isize - options.iter().map(Vec::len).sum::<usize>() as isize;
    if budget < 16 {
        let per = ((head_max_len as isize - 16) / options.len().max(1) as isize).max(4) as usize;
        for opt in &mut options {
            opt.truncate(per);
        }
        budget = head_max_len as isize - options.iter().map(Vec::len).sum::<usize>() as isize;
    }
    head.truncate(budget.max(8) as usize);
    let mut ids = vec![id("[CLS]")?];
    ids.extend(head);
    ids.push(id("[SEP]")?);
    let mut markers = Vec::new();
    for opt in options {
        markers.push(ids.len());
        ids.extend(opt);
    }
    ids.push(id("[SEP]")?);
    let room = max_len.saturating_sub(ids.len() + 1);
    ids.extend(encode(&text(state))?.into_iter().take(room));
    ids.push(id("[SEP]")?);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    Ok((ids, markers))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn json_spacing_and_unicode_match_python() {
        let state: Value = serde_json::from_str(r#"{"é":[true,null,8080],"x":"☕"}"#).unwrap();
        assert_eq!(
            python_json(&state),
            "{\"é\": [true, null, 8080], \"x\": \"☕\"}"
        );
    }
}
