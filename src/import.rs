//! JSONL import parsing (records, fields, vectors).

use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::{self, Read};
use std::path::Path;

use crate::config::DEFAULT_TEXT_FIELD;
use crate::filter::validate_meta_field_name;

#[derive(Debug, Clone)]
pub(crate) struct ImportRecord {
    pub(crate) external_id: Option<String>,
    pub(crate) vector_field: String,
    pub(crate) vector_text: String,
    pub(crate) vector: Option<Vec<f32>>,
    pub(crate) meta: serde_json::Value,
}

pub(crate) fn load_import_records(
    input: Option<&Path>,
    fallback_vector_field: Option<&str>,
    fallback_text_field: Option<&str>,
) -> Result<Vec<ImportRecord>> {
    let (source, content) = match input {
        Some(path) => (
            path.display().to_string(),
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
        ),
        None => {
            let mut content = String::new();
            io::stdin()
                .read_to_string(&mut content)
                .context("reading JSONL from stdin")?;
            ("stdin".to_string(), content)
        }
    };
    let mut records = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        records.push(
            parse_import_record(line, fallback_vector_field, fallback_text_field)
                .with_context(|| format!("parsing {source} line {}", line_idx + 1))?,
        );
    }
    if records.is_empty() {
        bail!("no JSONL records found in {source}");
    }
    Ok(records)
}

pub(crate) fn parse_import_record(
    input: &str,
    fallback_vector_field: Option<&str>,
    fallback_text_field: Option<&str>,
) -> Result<ImportRecord> {
    let value: serde_json::Value = serde_json::from_str(input)?;
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow!("JSONL record must be an object"))?;

    let external_id = obj
        .get("id")
        .or_else(|| obj.get("pk"))
        .map(json_scalar_to_string)
        .transpose()
        .context("record `id`/`pk` must be a scalar value")?;

    let fields = normalized_fields(obj)?;
    let vector_field =
        resolve_vector_field(obj, &fields, fallback_vector_field, fallback_text_field)?;
    let vector_value = fields
        .get(&vector_field)
        .ok_or_else(|| anyhow!("vector field `{vector_field}` is missing from fields"))?;
    let vector_text = vector_value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow!("vector field `{vector_field}` must be a string"))?;
    if vector_text.trim().is_empty() {
        bail!("vector field `{vector_field}` is empty");
    }
    let vector = resolve_record_vector(obj, &vector_field)?;

    let mut meta = serde_json::Map::new();
    if let Some(id) = external_id.as_ref() {
        meta.insert(
            "external_id".to_string(),
            serde_json::Value::String(id.clone()),
        );
    }
    for (field, value) in fields {
        if field != vector_field {
            meta.insert(field, value);
        }
    }

    Ok(ImportRecord {
        external_id,
        vector_field,
        vector_text,
        vector,
        meta: serde_json::Value::Object(meta),
    })
}

fn normalized_fields(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let raw_fields = if let Some(fields) = obj.get("fields") {
        fields
            .as_object()
            .ok_or_else(|| anyhow!("record `fields` must be an object"))?
            .clone()
    } else {
        obj.iter()
            .filter(|(key, _)| !is_reserved_record_key(key))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    };

    let mut out = serde_json::Map::new();
    for (field, value) in raw_fields {
        validate_meta_field_name(&field)?;
        out.insert(field, field_value(value)?);
    }
    if out.is_empty() {
        bail!("record has no fields to import");
    }
    Ok(out)
}

fn field_value(value: serde_json::Value) -> Result<serde_json::Value> {
    if let Some(obj) = value.as_object() {
        if obj.contains_key("index") || obj.contains_key("value") {
            return obj
                .get("value")
                .cloned()
                .ok_or_else(|| anyhow!("field descriptor must include `value`"));
        }
    }
    Ok(value)
}

fn resolve_vector_field(
    obj: &serde_json::Map<String, serde_json::Value>,
    fields: &serde_json::Map<String, serde_json::Value>,
    fallback: Option<&str>,
    text_fallback: Option<&str>,
) -> Result<String> {
    let mut candidates = Vec::new();

    if let Some(value) = obj.get("vector_field") {
        let field = value
            .as_str()
            .ok_or_else(|| anyhow!("record `vector_field` must be a string"))?;
        candidates.push(field.to_string());
    }

    if let Some(value) = obj.get("vector_fields") {
        let items = value
            .as_array()
            .ok_or_else(|| anyhow!("record `vector_fields` must be an array"))?;
        for item in items {
            let field = item
                .as_str()
                .ok_or_else(|| anyhow!("record `vector_fields` items must be strings"))?;
            candidates.push(field.to_string());
        }
    }

    candidates.extend(vector_fields_from_descriptors(obj)?);
    candidates.extend(vector_fields_from_vectors(obj)?);

    if candidates.is_empty() {
        if let Some(field) = fallback {
            candidates.push(field.to_string());
        } else if let Some(field) = text_fallback {
            candidates.push(field.to_string());
        } else if fields.contains_key(DEFAULT_TEXT_FIELD) {
            candidates.push(DEFAULT_TEXT_FIELD.to_string());
        } else if fields.contains_key("content") {
            candidates.push("content".to_string());
        }
    }

    candidates.sort();
    candidates.dedup();
    if candidates.len() != 1 {
        bail!(
            "expected exactly one vector field, found {} ({:?})",
            candidates.len(),
            candidates
        );
    }

    let field = candidates.remove(0);
    validate_meta_field_name(&field)?;
    if !fields.contains_key(&field) {
        bail!("vector field `{field}` is not present in fields");
    }
    Ok(field)
}

fn vector_fields_from_descriptors(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<String>> {
    let Some(fields) = obj.get("fields").and_then(|v| v.as_object()) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for (field, value) in fields {
        let Some(desc) = value.as_object() else {
            continue;
        };
        let Some(index) = desc.get("index").and_then(|v| v.as_array()) else {
            continue;
        };
        if index.iter().any(|v| v.as_str() == Some("vector")) {
            out.push(field.clone());
        }
    }
    Ok(out)
}

fn vector_fields_from_vectors(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<String>> {
    let Some(vectors) = obj.get("vectors") else {
        return Ok(Vec::new());
    };
    let vectors = vectors
        .as_object()
        .ok_or_else(|| anyhow!("record `vectors` must be an object keyed by vector field"))?;
    Ok(vectors.keys().cloned().collect())
}

fn resolve_record_vector(
    obj: &serde_json::Map<String, serde_json::Value>,
    vector_field: &str,
) -> Result<Option<Vec<f32>>> {
    let direct = obj.get("vector");
    let keyed = obj
        .get("vectors")
        .and_then(|vectors| vectors.as_object())
        .and_then(|vectors| vectors.get(vector_field));

    match (direct, keyed) {
        (Some(_), Some(_)) => {
            bail!("record cannot contain both `vector` and `vectors.{vector_field}`")
        }
        (Some(value), None) | (None, Some(value)) => parse_vector_value(value).map(Some),
        (None, None) => Ok(None),
    }
}

fn is_reserved_record_key(key: &str) -> bool {
    matches!(
        key,
        "id" | "pk" | "fields" | "vector_field" | "vector_fields" | "vector" | "vectors"
    )
}

pub(crate) fn parse_vector_value(value: &serde_json::Value) -> Result<Vec<f32>> {
    let items = value
        .as_array()
        .ok_or_else(|| anyhow!("vector must be a JSON array"))?;
    if items.is_empty() {
        bail!("vector cannot be empty");
    }

    items
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let number = value
                .as_f64()
                .ok_or_else(|| anyhow!("vector item {idx} must be a number"))?;
            if !number.is_finite() || number < f32::MIN as f64 || number > f32::MAX as f64 {
                bail!("vector item {idx} is outside finite f32 range");
            }
            Ok(number as f32)
        })
        .collect()
}

pub(crate) fn load_vector_arg(
    vector: Option<&str>,
    vector_file: Option<&Path>,
) -> Result<Option<Vec<f32>>> {
    match (vector, vector_file) {
        (Some(_), Some(_)) => bail!("pass only one of --vector or --vector-file"),
        (Some(vector), None) => {
            let value: serde_json::Value =
                serde_json::from_str(vector).context("parsing --vector JSON array")?;
            parse_vector_value(&value).map(Some)
        }
        (None, Some(path)) => {
            let content = fs::read_to_string(path)
                .with_context(|| format!("reading vector file {}", path.display()))?;
            let value: serde_json::Value =
                serde_json::from_str(&content).context("parsing --vector-file JSON array")?;
            parse_vector_value(&value).map(Some)
        }
        (None, None) => Ok(None),
    }
}

pub(crate) fn json_scalar_to_string(value: &serde_json::Value) -> Result<String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        _ => bail!("expected string, number, or boolean"),
    }
}

#[cfg(test)]
#[path = "import_test.rs"]
mod tests;
