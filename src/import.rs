//! JSONL import parsing (records, fields, vectors).

use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use crate::config::DEFAULT_TEXT_FIELD;
use crate::filter::validate_meta_field_name;

/// Upper bound for a single JSONL line. Longer lines fail with a clear error
/// instead of ballooning memory.
pub(crate) const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Capacity of the streaming reader's internal buffer; input larger than this
/// is consumed incrementally, one bounded line at a time.
const READER_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct ImportRecord {
    pub(crate) external_id: Option<String>,
    pub(crate) vector_field: String,
    pub(crate) vector_text: String,
    pub(crate) vector: Option<Vec<f32>>,
    pub(crate) meta: serde_json::Value,
}

/// Bounded JSONL reader: yields parsed [`ImportRecord`]s one line at a time
/// (or in fixed-size batches) without loading the whole file/stdin into
/// memory. Peak memory is the reader buffer plus one line plus one batch.
pub(crate) struct ImportRecordReader {
    source: String,
    fallback_vector_field: Option<String>,
    fallback_text_field: Option<String>,
    line_no: usize,
    reader: BufReader<Box<dyn Read>>,
}

impl ImportRecordReader {
    pub(crate) fn open(
        input: Option<&Path>,
        fallback_vector_field: Option<&str>,
        fallback_text_field: Option<&str>,
    ) -> Result<Self> {
        let (source, inner): (String, Box<dyn Read>) = match input {
            Some(path) => (
                path.display().to_string(),
                Box::new(
                    fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
                ),
            ),
            None => ("stdin".to_string(), Box::new(io::stdin())),
        };
        Ok(Self {
            source,
            fallback_vector_field: fallback_vector_field.map(str::to_string),
            fallback_text_field: fallback_text_field.map(str::to_string),
            line_no: 0,
            reader: BufReader::with_capacity(READER_BUFFER_BYTES, inner),
        })
    }

    /// Human-readable input label ("stdin" or the file path).
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    /// Next parsed record; `Ok(None)` at end of input. Errors carry 1-based
    /// line numbers counting every physical line (blank lines included).
    pub(crate) fn next_record(&mut self) -> Result<Option<ImportRecord>> {
        loop {
            let line = match self.next_line()? {
                Some(line) => line,
                None => return Ok(None),
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record = parse_import_record(
                line,
                self.fallback_vector_field.as_deref(),
                self.fallback_text_field.as_deref(),
            )
            .with_context(|| format!("parsing {} line {}", self.source, self.line_no))?;
            return Ok(Some(record));
        }
    }

    /// Next up to `batch_size` records as one bounded batch; `Ok(None)` at
    /// end of input. Never returns an empty batch.
    pub(crate) fn next_batch(&mut self, batch_size: usize) -> Result<Option<Vec<ImportRecord>>> {
        let batch_size = batch_size.max(1);
        let mut batch = Vec::new();
        while batch.len() < batch_size {
            match self.next_record()? {
                Some(record) => batch.push(record),
                None => break,
            }
        }
        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }

    /// Read the next physical line (newline included), enforcing the
    /// per-line byte cap so a pathologically long line cannot balloon memory.
    /// Returns `Ok(None)` at end of input.
    fn next_line(&mut self) -> Result<Option<String>> {
        let mut raw: Vec<u8> = Vec::new();
        let n = (&mut self.reader)
            .take(MAX_LINE_BYTES as u64 + 1)
            .read_until(b'\n', &mut raw)
            .with_context(|| format!("reading {}", self.source))?;
        if n == 0 {
            return Ok(None);
        }
        self.line_no += 1;
        if raw.len() > MAX_LINE_BYTES {
            bail!(
                "line {} of {} exceeds the {}-byte JSONL line limit",
                self.line_no,
                self.source,
                MAX_LINE_BYTES
            );
        }
        let line = String::from_utf8(raw).with_context(|| {
            format!(
                "line {} of {} is not valid UTF-8",
                self.line_no, self.source
            )
        })?;
        Ok(Some(line))
    }
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
#[path = "import_tests.rs"]
mod tests;
