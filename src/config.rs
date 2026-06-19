//! Configuration parsing and resolution helpers for the CLI.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

pub(crate) const DEFAULT_MODEL: &str = "bge-m3";
pub(crate) const DEFAULT_TEXT_FIELD: &str = "text";
pub(crate) const DEFAULT_VECTOR_FIELD: &str = "embedding";

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AppConfig {
    #[serde(
        default,
        alias = "storage_path",
        alias = "db",
        alias = "db_path",
        alias = "index",
        alias = "index_path"
    )]
    pub(crate) data_path: Option<PathBuf>,
    #[serde(
        default,
        alias = "model",
        alias = "default_model",
        alias = "embedding_model"
    )]
    pub(crate) default_vector_model: Option<String>,
    #[serde(default)]
    pub(crate) provider: Option<String>,
    #[serde(default)]
    pub(crate) base_url: Option<String>,
    #[serde(default)]
    pub(crate) embedding: Option<String>,
}

pub(crate) fn load_config(config: Option<&str>) -> Result<AppConfig> {
    let Some(config) = config else {
        return Ok(AppConfig::default());
    };
    let trimmed = config.trim();
    if trimmed.is_empty() {
        bail!("config cannot be empty");
    }

    let json = if trimmed.starts_with('{') {
        trimmed.to_string()
    } else {
        fs::read_to_string(trimmed).with_context(|| format!("reading config file {}", trimmed))?
    };
    serde_json::from_str(&json).context("parsing config JSON")
}

pub(crate) fn load_json_arg(input: &str) -> Result<String> {
    if let Some(path) = input.strip_prefix('@') {
        fs::read_to_string(path).with_context(|| format!("reading JSON file {path}"))
    } else if input.trim_start().starts_with('{') {
        Ok(input.to_string())
    } else {
        fs::read_to_string(input).with_context(|| format!("reading JSON file {input}"))
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct EmbeddingConfig {
    pub(crate) model: Option<String>,
    pub(crate) provider: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) dimensions: Option<usize>,
    pub(crate) text_field: Option<String>,
    pub(crate) vector_field: Option<String>,
}

pub(crate) fn parse_embedding_config(input: Option<&str>) -> Result<EmbeddingConfig> {
    let Some(input) = input else {
        return Ok(EmbeddingConfig::default());
    };
    let raw = load_json_arg(input)?;
    let value: serde_json::Value = serde_json::from_str(&raw).context("parsing embedding JSON")?;
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow!("embedding config must be a JSON object"))?;
    Ok(EmbeddingConfig {
        model: obj
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        provider: obj
            .get("provider")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        base_url: obj
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        dimensions: obj
            .get("dimensions")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        text_field: obj
            .get("text_field")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        vector_field: obj
            .get("vector_field")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    })
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SchemaDefaults {
    pub(crate) text_field: Option<String>,
    pub(crate) vector_field: Option<String>,
    pub(crate) dim: Option<usize>,
}

pub(crate) fn parse_schema_defaults(input: Option<&str>) -> Result<SchemaDefaults> {
    let Some(input) = input else {
        return Ok(SchemaDefaults::default());
    };
    let raw = load_json_arg(input)?;
    let value: serde_json::Value = serde_json::from_str(&raw).context("parsing schema JSON")?;
    let fields = value
        .get("fields")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("schema JSON must contain fields array"))?;

    let mut text_fields = Vec::new();
    let mut vector_fields = Vec::new();
    for field in fields {
        let Some(obj) = field.as_object() else {
            continue;
        };
        let Some(name) = obj.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(kind) = obj.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        match kind {
            "string" => text_fields.push(name.to_string()),
            "vector_fp32" => vector_fields.push((
                name.to_string(),
                obj.get("dimension")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize),
            )),
            _ => {}
        }
    }

    let text_field = if text_fields.iter().any(|field| field == DEFAULT_TEXT_FIELD) {
        Some(DEFAULT_TEXT_FIELD.to_string())
    } else if text_fields.len() == 1 {
        text_fields.into_iter().next()
    } else {
        None
    };
    let vector_field = if let Some((name, dim)) = vector_fields
        .iter()
        .find(|(name, _)| name == DEFAULT_VECTOR_FIELD)
        .cloned()
    {
        Some((name, dim))
    } else if vector_fields.len() == 1 {
        vector_fields.into_iter().next()
    } else {
        None
    };

    Ok(SchemaDefaults {
        text_field,
        vector_field: vector_field.as_ref().map(|(name, _)| name.clone()),
        dim: vector_field.and_then(|(_, dim)| dim),
    })
}

pub(crate) fn resolve_db_path(db: Option<PathBuf>, config: &AppConfig) -> Result<PathBuf> {
    db.or_else(|| config.data_path.clone())
        .ok_or_else(|| anyhow!("missing db path: pass --db or config data_path"))
}

pub(crate) fn resolve_model(model: Option<String>, config: &AppConfig) -> String {
    model
        .or_else(|| config.default_vector_model.clone())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

pub(crate) fn resolve_provider(provider: Option<String>, config: &AppConfig) -> Option<String> {
    provider.or_else(|| config.provider.clone())
}

pub(crate) fn resolve_base_url(base_url: Option<String>, config: &AppConfig) -> Option<String> {
    base_url.or_else(|| config.base_url.clone())
}

pub(crate) fn normalize_provider_model(
    model: &str,
    provider: Option<&str>,
) -> Result<(String, Option<String>)> {
    if let Some(provider) = provider {
        if let Some((model_provider, model_name)) = model.split_once('/') {
            if model_provider != provider {
                bail!(
                    "model `{model}` conflicts with provider `{provider}`; use `{model_name}` or provider `{model_provider}`"
                );
            }
            return Ok((model_name.to_string(), Some(provider.to_string())));
        }
        return Ok((model.to_string(), Some(provider.to_string())));
    }

    Ok((model.to_string(), None))
}

pub(crate) fn path_arg_to_optional(path: Option<PathBuf>) -> Option<PathBuf> {
    path.filter(|path| path.as_os_str() != "-")
}

pub(crate) fn merge_embedding_arg(cli: Option<String>, config: &AppConfig) -> Option<String> {
    cli.or_else(|| config.embedding.clone())
}

#[cfg(test)]
#[path = "config_test.rs"]
mod tests;
