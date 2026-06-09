//! turbovec-rs — Persistent vector index CLI with semantic search.
//!
//! Uses turbovec for 2-4bit quantized vector storage and the embeddings crate
//! for generating embeddings via Ollama / BGE-M3.
//!
//! # Examples
//!
//! ```bash
//! # Init an index
//! turbovec-rs init --db /tmp/docs.tvim
//!
//! # Or use a config JSON string / file
//! turbovec-rs -c '{"data_path":"/tmp/docs.tvim","default_vector_model":"ollama/bge-m3"}' stats
//!
//! # Import JSONL records
//! turbovec-rs import --db /tmp/docs.tvim --input docs.jsonl --provider ollama
//!
//! # Search
//! turbovec-rs search --db /tmp/docs.tvim --query "什么是编程" --provider ollama
//! turbovec-rs search --db /tmp/docs.tvim --vector '[0.1,0.2,...]'
//!
//! # Show index stats
//! turbovec-rs stats --db /tmp/docs.tvim
//! ```

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use embeddings::{resolve_api_key_for_provider, EmbedClient};
use filterql::validate::Policy;
use filterql::{CmpOp, Compile, Value as FilterValue};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use turbovec::IdMapIndex;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "turbovec-rs",
    version = "0.1.0",
    about = "Persistent vector index with semantic search (turbovec + embeddings)"
)]
struct Cli {
    /// Configuration as a JSON string or path to a JSON config file
    #[arg(short = 'c', long, global = true)]
    config: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new empty index
    Init {
        /// Path to the database/index file (.tvim)
        #[arg(long)]
        db: Option<PathBuf>,
        /// Vector dimensionality [default: 1024 for bge-m3]
        #[arg(long, default_value_t = 1024)]
        dim: usize,
        /// Quantization bit width (2, 3, or 4) [default: 4]
        #[arg(long, default_value_t = 4)]
        bits: usize,
    },
    /// Import JSONL records into the index
    Import {
        /// Path to the database/index file (.tvim)
        #[arg(long)]
        db: Option<PathBuf>,
        /// Optional zvec-style schema JSON string or @file. Used for defaults only.
        #[arg(long)]
        schema: Option<String>,
        /// Optional zvec-style embedding JSON string or @file
        #[arg(long)]
        embedding: Option<String>,
        /// JSONL file to import. Omit or pass "-" to read stdin.
        #[arg(long)]
        input: Option<PathBuf>,
        /// Embedding model (overrides config; default: bge-m3)
        #[arg(long)]
        model: Option<String>,
        /// Provider (auto-detected if omitted)
        #[arg(long)]
        provider: Option<String>,
        /// Custom base URL
        #[arg(long)]
        base_url: Option<String>,
        /// Batch size for embedding API calls
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// Fallback vector field when a record omits vector_field/vector_fields
        #[arg(long)]
        vector_field: Option<String>,
        /// Text field to embed when importing zvec-style {"pk","fields"} records
        #[arg(long)]
        text_field: Option<String>,
        /// Vector dimensionality used when creating a missing db
        #[arg(long)]
        dim: Option<usize>,
        /// Quantization bit width used when creating a missing db
        #[arg(long, default_value_t = 4)]
        bits: usize,
        /// Accepted for zvec CLI parity. Existing primary keys cannot be overwritten yet.
        #[arg(long)]
        upsert: bool,
    },
    /// Search the index with a text query
    Search {
        /// Path to the database/index file (.tvim)
        #[arg(long)]
        db: Option<PathBuf>,
        /// Query text
        #[arg(long)]
        query: Option<String>,
        /// Query vector as a JSON array, e.g. '[0.1,0.2,0.3]'
        #[arg(long)]
        vector: Option<String>,
        /// Path to a JSON file containing the query vector array
        #[arg(long)]
        vector_file: Option<PathBuf>,
        /// Number of results
        #[arg(long, short = 'k', default_value_t = 10)]
        top_k: usize,
        /// Embedding model (must match import; overrides config; default: bge-m3)
        #[arg(long)]
        model: Option<String>,
        /// Provider (auto-detected if omitted)
        #[arg(long)]
        provider: Option<String>,
        /// Custom base URL
        #[arg(long)]
        base_url: Option<String>,
        /// SQL-like metadata filter, e.g. "source = 'docs' AND lang = 'zh'"
        #[arg(long)]
        filter: Option<String>,
    },
    /// Run a zvec-style SQL metadata query
    Query {
        /// Path to the database/index file (.tvim)
        #[arg(long = "db-path")]
        db_path: Option<PathBuf>,
        /// SQL query: SELECT ... FROM <collection> WHERE ... [LIMIT n]
        #[arg(long)]
        sql: String,
    },
    /// Export JSONL records from the index
    Export {
        /// Path to the database/index file (.tvim)
        #[arg(long)]
        db: Option<PathBuf>,
        /// Optional zvec-style schema JSON string or @file. Accepted for parity.
        #[arg(long)]
        schema: Option<String>,
        /// Output JSONL file. Omit or pass "-" to write stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        /// SQL-like metadata filter
        #[arg(long)]
        filter: Option<String>,
        /// Accepted for zvec parity, but raw vectors are not recoverable from turbovec.
        #[arg(long)]
        include_vectors: bool,
    },
    /// Return internal vector IDs matching a metadata filter
    #[command(hide = true)]
    FilterIds {
        /// Path to the database/index file (.tvim)
        #[arg(long)]
        db: Option<PathBuf>,
        /// SQL-like metadata filter, e.g. "source = 'docs' AND lang = 'zh'"
        #[arg(long)]
        filter: String,
    },
    /// Show index metadata
    Stats {
        /// Path to the database/index file (.tvim)
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Run a REST API over a directory of turbovec indexes
    #[command(hide = true)]
    Serve {
        /// Directory containing db files
        #[arg(long = "db-root")]
        db_root: PathBuf,
        /// Bind address
        #[arg(long, default_value = "127.0.0.1:8787")]
        bind: String,
    },
    /// Run a stdio MCP server over one fixed db
    #[command(hide = true)]
    Mcp {
        /// Path to the database/index file (.tvim)
        #[arg(long)]
        db: Option<PathBuf>,
        /// Optional zvec-style schema JSON string or @file. Accepted for parity.
        #[arg(long)]
        schema: Option<String>,
        /// Optional zvec-style embedding JSON string or @file
        #[arg(long)]
        embedding: Option<String>,
    },
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AppConfig {
    #[serde(
        default,
        alias = "storage_path",
        alias = "db",
        alias = "db_path",
        alias = "index",
        alias = "index_path"
    )]
    data_path: Option<PathBuf>,
    #[serde(
        default,
        alias = "model",
        alias = "default_model",
        alias = "embedding_model"
    )]
    default_vector_model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    embedding: Option<String>,
}

const DEFAULT_MODEL: &str = "bge-m3";
const DEFAULT_TEXT_FIELD: &str = "text";
const DEFAULT_VECTOR_FIELD: &str = "embedding";

fn load_config(config: Option<&str>) -> Result<AppConfig> {
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

fn load_json_arg(input: &str) -> Result<String> {
    if let Some(path) = input.strip_prefix('@') {
        fs::read_to_string(path).with_context(|| format!("reading JSON file {path}"))
    } else if input.trim_start().starts_with('{') {
        Ok(input.to_string())
    } else {
        fs::read_to_string(input).with_context(|| format!("reading JSON file {input}"))
    }
}

#[derive(Debug, Clone, Default)]
struct EmbeddingConfig {
    model: Option<String>,
    provider: Option<String>,
    base_url: Option<String>,
    dimensions: Option<usize>,
    text_field: Option<String>,
    vector_field: Option<String>,
}

fn parse_embedding_config(input: Option<&str>) -> Result<EmbeddingConfig> {
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
struct SchemaDefaults {
    text_field: Option<String>,
    vector_field: Option<String>,
    dim: Option<usize>,
}

fn parse_schema_defaults(input: Option<&str>) -> Result<SchemaDefaults> {
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

fn resolve_db_path(db: Option<PathBuf>, config: &AppConfig) -> Result<PathBuf> {
    db.or_else(|| config.data_path.clone())
        .ok_or_else(|| anyhow!("missing db path: pass --db or config data_path"))
}

fn resolve_model(model: Option<String>, config: &AppConfig) -> String {
    model
        .or_else(|| config.default_vector_model.clone())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

fn resolve_provider(provider: Option<String>, config: &AppConfig) -> Option<String> {
    provider.or_else(|| config.provider.clone())
}

fn resolve_base_url(base_url: Option<String>, config: &AppConfig) -> Option<String> {
    base_url.or_else(|| config.base_url.clone())
}

fn normalize_provider_model(
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

// ---------------------------------------------------------------------------
// Sidecar helpers
// ---------------------------------------------------------------------------

/// Index metadata persisted alongside the .tvim file.
#[derive(Serialize, Deserialize, Default)]
struct IndexMeta {
    next_id: u64,
    dim: usize,
    bits: usize,
    model: String,
}

fn sqlite_path(index: &Path) -> PathBuf {
    let mut p = index.to_path_buf();
    p.set_extension("tvim.sqlite");
    p
}

fn meta_path(index: &Path) -> PathBuf {
    let mut p = index.to_path_buf();
    p.set_extension("tvim.meta.json");
    p
}

fn load_meta(index: &Path) -> Result<IndexMeta> {
    let path = meta_path(index);
    let data = fs::read_to_string(&path).context("reading meta file")?;
    Ok(serde_json::from_str(&data)?)
}

fn save_meta(index: &Path, meta: &IndexMeta) -> Result<()> {
    let path = meta_path(index);
    let data = serde_json::to_string_pretty(meta)?;
    fs::write(&path, data).context("writing meta file")?;
    Ok(())
}

#[derive(Debug, Clone)]
struct DocRow {
    external_id: Option<String>,
    vector_field: String,
    text: String,
    meta: serde_json::Value,
}

fn open_sidecar(index: &Path) -> Result<Connection> {
    let path = sqlite_path(index);
    let conn = Connection::open(&path)
        .with_context(|| format!("opening SQLite sidecar {}", path.display()))?;
    init_sidecar_schema(&conn)?;
    Ok(conn)
}

fn init_sidecar_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS docs (
          id INTEGER PRIMARY KEY,
          external_id TEXT,
          vector_field TEXT NOT NULL DEFAULT 'content',
          text TEXT NOT NULL,
          meta TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS docs_external_id ON docs(external_id);
        "#,
    )
    .context("initializing SQLite sidecar schema")?;
    let _ = conn.execute("ALTER TABLE docs ADD COLUMN external_id TEXT", []);
    let _ = conn.execute(
        "ALTER TABLE docs ADD COLUMN vector_field TEXT NOT NULL DEFAULT 'content'",
        [],
    );
    let _ = conn.execute(
        "CREATE INDEX IF NOT EXISTS docs_external_id ON docs(external_id)",
        [],
    );
    Ok(())
}

fn insert_doc(
    conn: &Connection,
    id: u64,
    external_id: Option<&str>,
    vector_field: &str,
    text: &str,
    meta: &serde_json::Value,
) -> Result<()> {
    let id = i64::try_from(id).context("document id does not fit SQLite INTEGER")?;
    let meta_json = serde_json::to_string(meta)?;
    conn.execute(
        "INSERT OR REPLACE INTO docs (id, external_id, vector_field, text, meta) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![id, external_id, vector_field, text, meta_json],
    )
    .context("inserting document metadata into SQLite sidecar")?;
    Ok(())
}

fn sqlite_doc_count(conn: &Connection) -> Result<usize> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM docs", [], |row| row.get(0))?;
    usize::try_from(count).context("SQLite doc count is negative or too large")
}

fn external_id_exists(conn: &Connection, external_id: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM docs WHERE external_id = ?1",
        params![external_id],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn query_doc_ids(conn: &Connection, filter: Option<&str>) -> Result<Vec<u64>> {
    let (sql, params) = if let Some(filter) = filter {
        let compiled = compile_filter(filter)?;
        let sql = if compiled.clause.is_empty() {
            "SELECT id FROM docs ORDER BY id".to_string()
        } else {
            format!("SELECT id FROM docs WHERE {} ORDER BY id", compiled.clause)
        };
        (sql, compiled.params)
    } else {
        ("SELECT id FROM docs ORDER BY id".to_string(), Vec::new())
    };

    let mut stmt = conn.prepare(&sql).context("preparing document id query")?;
    let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
        let id: i64 = row.get(0)?;
        Ok(id)
    })?;

    let mut ids = Vec::new();
    for row in rows {
        let id = row.context("reading document id")?;
        ids.push(u64::try_from(id).context("SQLite document id is negative")?);
    }
    Ok(ids)
}

fn load_docs_by_ids(index: &Path, ids: &[u64]) -> Result<HashMap<u64, DocRow>> {
    let conn = open_sidecar(index)?;
    load_docs_by_ids_sqlite(&conn, ids)
}

fn load_docs_by_ids_sqlite(conn: &Connection, ids: &[u64]) -> Result<HashMap<u64, DocRow>> {
    let mut docs = HashMap::with_capacity(ids.len());
    let mut stmt =
        conn.prepare("SELECT external_id, vector_field, text, meta FROM docs WHERE id = ?1")?;
    for &id in ids {
        let sql_id = i64::try_from(id).context("document id does not fit SQLite INTEGER")?;
        let row = stmt.query_row(params![sql_id], |row| {
            let external_id: Option<String> = row.get(0)?;
            let vector_field: String = row.get(1)?;
            let text: String = row.get(2)?;
            let meta_json: String = row.get(3)?;
            Ok((external_id, vector_field, text, meta_json))
        });
        if let Ok((external_id, vector_field, text, meta_json)) = row {
            let meta = serde_json::from_str(&meta_json).unwrap_or_else(|_| serde_json::json!({}));
            docs.insert(
                id,
                DocRow {
                    external_id,
                    vector_field,
                    text,
                    meta,
                },
            );
        }
    }
    Ok(docs)
}

// ---------------------------------------------------------------------------
// JSONL import helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ImportRecord {
    external_id: Option<String>,
    vector_field: String,
    vector_text: String,
    vector: Option<Vec<f32>>,
    meta: serde_json::Value,
}

fn load_import_records(
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

fn parse_import_record(
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

fn parse_vector_value(value: &serde_json::Value) -> Result<Vec<f32>> {
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

fn load_vector_arg(vector: Option<&str>, vector_file: Option<&Path>) -> Result<Option<Vec<f32>>> {
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

fn json_scalar_to_string(value: &serde_json::Value) -> Result<String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        _ => bail!("expected string, number, or boolean"),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn build_client(
    model: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> Result<EmbedClient> {
    let (model, provider) = normalize_provider_model(model, provider)?;
    let mut client = if let Some(p) = provider.as_deref() {
        let api_key = resolve_api_key_for_provider(p)?;
        EmbedClient::new(p, &model, api_key)?
    } else {
        EmbedClient::from_env(&model)?
    };
    if let Some(url) = base_url {
        client = client.with_base_url(url);
    }
    Ok(client)
}

fn flatten_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(embeddings.len() * embeddings.first().map_or(0, |v| v.len()));
    for emb in embeddings {
        flat.extend_from_slice(emb);
    }
    flat
}

fn validate_vectors_dim(vectors: &[Vec<f32>], dim: usize) -> Result<()> {
    for (idx, vector) in vectors.iter().enumerate() {
        if vector.len() != dim {
            bail!(
                "vector dimension mismatch at batch item {}: index expects {}, got {}",
                idx,
                dim,
                vector.len()
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Metadata filter helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SqliteWhere {
    clause: String,
    params: Vec<SqlValue>,
}

#[derive(Debug, Default)]
struct SqliteFilterCompiler;

fn filter_policy() -> Policy {
    Policy::new()
        .max_depth(8)
        .max_comparisons(32)
        .max_in_list(256)
}

fn compile_filter(filter: &str) -> Result<SqliteWhere> {
    let expr = filterql::sql::parse(filter).context("parsing metadata filter")?;
    filter_policy()
        .validate(&expr)
        .map_err(|report| anyhow!("invalid metadata filter:\n{}", report.render()))?;
    filterql::compile(&expr, &mut SqliteFilterCompiler)
        .map_err(|err| anyhow!("compiling metadata filter: {err}"))
}

fn filter_ids(index: &Path, filter: &str) -> Result<Vec<u64>> {
    let path = sqlite_path(index);
    if !path.exists() {
        bail!("metadata filter requires SQLite sidecar {}", path.display());
    }

    let compiled = compile_filter(filter)?;
    let conn = open_sidecar(index)?;
    let sql = if compiled.clause.is_empty() {
        "SELECT id FROM docs".to_string()
    } else {
        format!("SELECT id FROM docs WHERE {}", compiled.clause)
    };
    let mut stmt = conn
        .prepare(&sql)
        .context("preparing metadata filter query")?;
    let rows = stmt.query_map(params_from_iter(compiled.params.iter()), |row| {
        let id: i64 = row.get(0)?;
        Ok(id)
    })?;

    let mut ids = Vec::new();
    for row in rows {
        let id = row.context("reading filtered document id")?;
        ids.push(u64::try_from(id).context("SQLite document id is negative")?);
    }
    Ok(ids)
}

fn validate_meta_field_name(field: &str) -> Result<()> {
    if field.is_empty() {
        bail!("metadata field name cannot be empty");
    }
    for part in field.split('.') {
        let mut chars = part.chars();
        let Some(first) = chars.next() else {
            bail!("metadata field `{field}` contains an empty path segment");
        };
        if !(first.is_ascii_alphabetic() || first == '_') {
            bail!("metadata field `{field}` must start each path segment with ASCII letter or `_`");
        }
        if chars.any(|c| !(c.is_ascii_alphanumeric() || c == '_')) {
            bail!("metadata field `{field}` contains unsupported characters");
        }
    }
    Ok(())
}

fn json_path_param(field: &str) -> Result<SqlValue> {
    validate_meta_field_name(field)?;
    Ok(SqlValue::Text(format!("$.{field}")))
}

fn sqlite_value(value: &FilterValue) -> Result<SqlValue> {
    Ok(match value {
        FilterValue::Str(s) => SqlValue::Text(s.clone()),
        FilterValue::Int(n) | FilterValue::Date(n) => SqlValue::Integer(*n),
        FilterValue::Float(n) => SqlValue::Real(*n),
        FilterValue::Bool(b) => SqlValue::Integer(i64::from(*b)),
        FilterValue::Null => SqlValue::Null,
        FilterValue::List(_) => bail!("list value cannot be bound as a scalar"),
    })
}

fn merge_params(mut parts: Vec<SqliteWhere>, separator: &str) -> SqliteWhere {
    let mut params = Vec::new();
    let clauses = parts
        .drain(..)
        .filter(|part| !part.clause.is_empty())
        .map(|part| {
            params.extend(part.params);
            part.clause
        })
        .collect::<Vec<_>>();

    SqliteWhere {
        clause: if clauses.is_empty() {
            String::new()
        } else {
            format!("({})", clauses.join(separator))
        },
        params,
    }
}

impl Compile for SqliteFilterCompiler {
    type Output = SqliteWhere;
    type Error = anyhow::Error;

    fn and(&mut self, parts: Vec<SqliteWhere>) -> Result<SqliteWhere> {
        Ok(merge_params(parts, " AND "))
    }

    fn or(&mut self, parts: Vec<SqliteWhere>) -> Result<SqliteWhere> {
        if parts.is_empty() {
            return Ok(SqliteWhere {
                clause: "(0 = 1)".to_string(),
                params: Vec::new(),
            });
        }
        Ok(merge_params(parts, " OR "))
    }

    fn not(&mut self, part: SqliteWhere) -> Result<SqliteWhere> {
        Ok(SqliteWhere {
            clause: format!("NOT ({})", part.clause),
            params: part.params,
        })
    }

    fn compare(&mut self, field: &str, op: CmpOp, value: &FilterValue) -> Result<SqliteWhere> {
        let path = json_path_param(field)?;
        match op {
            CmpOp::Exists => {
                let present = !matches!(value, FilterValue::Bool(false));
                Ok(SqliteWhere {
                    clause: format!(
                        "json_type(meta, ?) IS {}NULL",
                        if present { "NOT " } else { "" }
                    ),
                    params: vec![path],
                })
            }
            CmpOp::Eq if matches!(value, FilterValue::Null) => Ok(SqliteWhere {
                clause: "json_type(meta, ?) IS NULL".to_string(),
                params: vec![path],
            }),
            CmpOp::Ne if matches!(value, FilterValue::Null) => Ok(SqliteWhere {
                clause: "json_type(meta, ?) IS NOT NULL".to_string(),
                params: vec![path],
            }),
            CmpOp::Eq
            | CmpOp::Ne
            | CmpOp::Lt
            | CmpOp::Le
            | CmpOp::Gt
            | CmpOp::Ge
            | CmpOp::Like
            | CmpOp::NotLike => {
                let sql_op = match op {
                    CmpOp::Eq => "=",
                    CmpOp::Ne => "!=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                    CmpOp::Like => "LIKE",
                    CmpOp::NotLike => "NOT LIKE",
                    _ => unreachable!(),
                };
                Ok(SqliteWhere {
                    clause: format!("json_extract(meta, ?) {sql_op} ?"),
                    params: vec![path, sqlite_value(value)?],
                })
            }
            CmpOp::In | CmpOp::NotIn => {
                let FilterValue::List(items) = value else {
                    bail!("{} requires a list value", op.sql());
                };
                if items.is_empty() {
                    return Ok(SqliteWhere {
                        clause: if op == CmpOp::In {
                            "(0 = 1)".to_string()
                        } else {
                            "(1 = 1)".to_string()
                        },
                        params: Vec::new(),
                    });
                }

                let placeholders = vec!["?"; items.len()].join(", ");
                let sql_op = if op == CmpOp::In { "IN" } else { "NOT IN" };
                let params = items.iter().map(sqlite_value).collect::<Result<Vec<_>>>()?;
                let mut all_params = Vec::with_capacity(params.len() + 1);
                all_params.push(path);
                all_params.extend(params);
                Ok(SqliteWhere {
                    clause: format!("json_extract(meta, ?) {sql_op} ({placeholders})"),
                    params: all_params,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

fn cmd_init(index: &Path, dim: usize, bits: usize) -> Result<()> {
    create_index(index, dim, bits)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "db": index.display().to_string(),
            "dimension": dim,
            "bits": bits,
            "created": true
        }))?
    );
    Ok(())
}

fn create_index(index: &Path, dim: usize, bits: usize) -> Result<()> {
    if dim == 0 || !dim.is_multiple_of(8) {
        bail!("dim must be a positive multiple of 8, got {dim}");
    }
    if ![2, 3, 4].contains(&bits) {
        bail!("bits must be 2, 3, or 4, got {bits}");
    }

    let idx = IdMapIndex::new(dim, bits).context("creating IdMapIndex")?;
    idx.write(index).context("writing .tvim file")?;

    let conn = open_sidecar(index)?;
    init_sidecar_schema(&conn)?;
    save_meta(
        index,
        &IndexMeta {
            next_id: 1,
            dim,
            bits,
            model: String::new(),
        },
    )?;

    Ok(())
}

struct AddOptions<'a> {
    db: &'a Path,
    input: Option<&'a Path>,
    model: Option<&'a str>,
    provider: Option<&'a str>,
    base_url: Option<&'a str>,
    batch_size: usize,
    vector_field: Option<&'a str>,
    text_field: Option<&'a str>,
    dim: Option<usize>,
    bits: usize,
    upsert: bool,
}

async fn cmd_add(opts: AddOptions<'_>) -> Result<()> {
    let AddOptions {
        db,
        input,
        model,
        provider,
        base_url,
        batch_size,
        vector_field,
        text_field,
        dim,
        bits,
        upsert,
    } = opts;

    if let Some(input) = input {
        if !input.exists() {
            bail!("input file not found: {}", input.display());
        }
    }
    if upsert {
        eprintln!(
            "warning: --upsert currently behaves like insert unless the primary key already exists"
        );
    }

    let records = load_import_records(input, vector_field, text_field)?;

    if !db.exists() {
        let inferred_dim = dim
            .or_else(|| {
                records
                    .iter()
                    .find_map(|record| record.vector.as_ref().map(Vec::len))
            })
            .unwrap_or(1024);
        create_index(db, inferred_dim, bits)?;
    }

    let mut meta = load_meta(db)?;
    let mut idx = IdMapIndex::load(db).context("loading .tvim index")?;
    let conn = open_sidecar(db)?;
    let batch_size = batch_size.max(1);
    let needs_embedding = records.iter().any(|record| record.vector.is_none());
    let embedding_model = needs_embedding.then(|| model.unwrap_or(DEFAULT_MODEL));
    let mut used_embedding = false;

    eprintln!(
        "{} JSONL records to import (batch_size={})",
        records.len(),
        batch_size
    );

    let client = if let Some(model) = embedding_model {
        Some(build_client(model, provider, base_url)?)
    } else {
        None
    };

    let mut added = 0usize;
    for batch in records.chunks(batch_size) {
        let mut batch_vectors = batch
            .iter()
            .map(|record| record.vector.clone())
            .collect::<Vec<_>>();
        let embed_indices = batch_vectors
            .iter()
            .enumerate()
            .filter_map(|(idx, vector)| vector.is_none().then_some(idx))
            .collect::<Vec<_>>();

        if !embed_indices.is_empty() {
            let client = client
                .as_ref()
                .ok_or_else(|| anyhow!("records without vectors require an embedding model"))?;
            let texts = embed_indices
                .iter()
                .map(|&idx| batch[idx].vector_text.clone())
                .collect::<Vec<_>>();
            let output = client
                .embed(texts)
                .await
                .context("embedding import records")?;
            if output.embeddings.len() != embed_indices.len() {
                bail!(
                    "embedding count mismatch: sent {}, received {}",
                    embed_indices.len(),
                    output.embeddings.len()
                );
            }
            for (&idx, vector) in embed_indices.iter().zip(output.embeddings) {
                batch_vectors[idx] = Some(vector);
            }
            used_embedding = true;
        }

        let vectors = batch_vectors
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow!("missing vector after embedding import batch"))?;
        validate_vectors_dim(&vectors, meta.dim)?;

        for record in batch {
            if let Some(external_id) = record.external_id.as_deref() {
                if external_id_exists(&conn, external_id)? {
                    bail!(
                        "primary key `{external_id}` already exists; turbovec-rs cannot overwrite vectors in-place yet"
                    );
                }
            }
        }

        let ids: Vec<u64> = (meta.next_id..meta.next_id + batch.len() as u64).collect();
        let flat = flatten_embeddings(&vectors);

        idx.add_with_ids_2d(&flat, meta.dim, &ids)
            .context("adding vectors to index")?;

        for (&id, record) in ids.iter().zip(batch.iter()) {
            insert_doc(
                &conn,
                id,
                record.external_id.as_deref(),
                &record.vector_field,
                &record.vector_text,
                &record.meta,
            )?;
        }

        meta.next_id += batch.len() as u64;
        added += batch.len();

        eprintln!("+{}/{} imported", added, records.len());
    }

    // Persist index and meta
    idx.write(db).context("writing index")?;
    if used_embedding {
        meta.model = embedding_model.unwrap_or(DEFAULT_MODEL).to_string();
    } else if let Some(model) = model {
        meta.model = model.to_string();
    }
    save_meta(db, &meta)?;

    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "success": added,
            "errors": 0,
            "total": idx.len()
        }))?
    );
    Ok(())
}

struct SearchOptions<'a> {
    index: &'a Path,
    query: Option<&'a str>,
    vector: Option<Vec<f32>>,
    top_k: usize,
    model: &'a str,
    provider: Option<&'a str>,
    base_url: Option<&'a str>,
    filter: Option<&'a str>,
}

async fn cmd_search(opts: SearchOptions<'_>) -> Result<()> {
    let SearchOptions {
        index,
        query,
        vector,
        top_k,
        model,
        provider,
        base_url,
        filter,
    } = opts;

    if !index.exists() {
        bail!("index not found: {}", index.display());
    }

    let idx = IdMapIndex::load(index).context("loading .tvim index")?;
    if idx.is_empty() {
        bail!("index is empty (import documents first)");
    }

    let allowlist = if let Some(filter) = filter {
        let ids = filter_ids(index, filter)?;
        if ids.is_empty() {
            println!("[]");
            return Ok(());
        }
        Some(ids)
    } else {
        None
    };

    let query_vec = match (query, vector) {
        (Some(_), Some(_)) => bail!("pass only one of --query, --vector, or --vector-file"),
        (Some(query), None) => {
            let client = build_client(model, provider, base_url)?;
            client.embed_one(query).await.context("embedding query")?
        }
        (None, Some(vector)) => {
            if vector.len() != idx.dim() {
                bail!(
                    "query vector dimension mismatch: index expects {}, got {}",
                    idx.dim(),
                    vector.len()
                );
            }
            vector
        }
        (None, None) => bail!("search requires one of --query, --vector, or --vector-file"),
    };

    let (scores, ids) = if let Some(allowlist) = allowlist.as_deref() {
        idx.search_with_allowlist(&query_vec, top_k, Some(allowlist))
    } else {
        idx.search(&query_vec, top_k)
    };

    // Build JSON output
    let docs = load_docs_by_ids(index, &ids)?;
    let mut results = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let score = scores[i];
        let doc = docs.get(&id);
        let text = doc
            .map(|doc| doc.text.clone())
            .unwrap_or_else(|| format!("<id {} text missing>", id));
        let external_id = doc.and_then(|doc| doc.external_id.clone());
        let vector_field = doc
            .map(|doc| doc.vector_field.clone())
            .unwrap_or_else(|| "content".to_string());
        let meta = doc
            .map(|doc| doc.meta.clone())
            .unwrap_or_else(|| serde_json::json!({}));
        results.push(serde_json::json!({
            "id": id,
            "external_id": external_id,
            "vector_field": vector_field,
            "score": score,
            "text": text,
            "meta": meta,
        }));
    }

    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}

fn cmd_filter_ids(index: &Path, filter: &str) -> Result<()> {
    if !index.exists() {
        bail!("index not found: {}", index.display());
    }

    let ids = filter_ids(index, filter)?;
    println!("{}", serde_json::to_string_pretty(&ids)?);
    Ok(())
}

fn doc_to_export_json(id: u64, doc: &DocRow) -> serde_json::Value {
    let mut fields = match doc.meta.as_object() {
        Some(meta) => meta.clone(),
        None => serde_json::Map::new(),
    };
    fields.insert(
        doc.vector_field.clone(),
        serde_json::Value::String(doc.text.clone()),
    );
    serde_json::json!({
        "pk": doc.external_id.clone().unwrap_or_else(|| id.to_string()),
        "fields": fields
    })
}

fn cmd_export(
    db: &Path,
    output: Option<&Path>,
    filter: Option<&str>,
    include_vectors: bool,
) -> Result<()> {
    if include_vectors {
        bail!("--include-vectors is not supported: turbovec-rs cannot reconstruct raw vectors from the quantized index");
    }
    if !db.exists() {
        bail!("db not found: {}", db.display());
    }

    let conn = open_sidecar(db)?;
    let ids = query_doc_ids(&conn, filter)?;
    let docs = load_docs_by_ids_sqlite(&conn, &ids)?;

    let writer: Box<dyn Write> = match output {
        Some(path) => Box::new(
            fs::File::create(path)
                .with_context(|| format!("creating export output {}", path.display()))?,
        ),
        None => Box::new(io::stdout()),
    };
    let mut writer = io::BufWriter::new(writer);
    for id in ids {
        if let Some(doc) = docs.get(&id) {
            serde_json::to_writer(&mut writer, &doc_to_export_json(id, doc))?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SqlMetadataQuery {
    select: Vec<String>,
    filter: Option<String>,
    limit: Option<usize>,
}

fn find_keyword_ci(haystack: &str, keyword: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&keyword.to_ascii_lowercase())
}

fn parse_sql_metadata_query(sql: &str) -> Result<SqlMetadataQuery> {
    let trimmed = sql.trim().trim_end_matches(';').trim();
    let from_pos = find_keyword_ci(trimmed, " from ").ok_or_else(|| {
        anyhow!("query must use SELECT ... FROM <collection> [WHERE ...] [LIMIT n]")
    })?;
    if !trimmed[..from_pos]
        .trim_start()
        .to_ascii_lowercase()
        .starts_with("select ")
    {
        bail!("query must start with SELECT");
    }

    let select_part = trimmed[6..from_pos].trim();
    let after_from = trimmed[from_pos + " from ".len()..].trim();
    let where_pos = find_keyword_ci(after_from, " where ");
    let limit_pos = find_keyword_ci(after_from, " limit ");

    let tail_start = match (where_pos, limit_pos) {
        (Some(w), Some(l)) => w.min(l),
        (Some(w), None) => w,
        (None, Some(l)) => l,
        (None, None) => after_from.len(),
    };
    let collection = after_from[..tail_start].trim();
    if collection.is_empty() {
        bail!("query must name a collection after FROM");
    }

    let filter = where_pos
        .map(|where_pos| {
            let start = where_pos + " where ".len();
            let end = limit_pos
                .filter(|limit_pos| *limit_pos > where_pos)
                .unwrap_or(after_from.len());
            after_from[start..end].trim().to_string()
        })
        .filter(|filter| !filter.is_empty());

    let limit = match limit_pos {
        Some(limit_pos) => {
            let start = limit_pos + " limit ".len();
            let value = after_from[start..].trim();
            Some(
                value
                    .parse::<usize>()
                    .with_context(|| format!("invalid LIMIT value `{value}`"))?,
            )
        }
        None => None,
    };

    let select = if select_part == "*" {
        Vec::new()
    } else {
        select_part
            .split(',')
            .map(|field| field.trim().to_string())
            .filter(|field| !field.is_empty())
            .collect()
    };

    Ok(SqlMetadataQuery {
        select,
        filter,
        limit,
    })
}

fn doc_to_query_record(id: u64, doc: Option<&DocRow>, select: &[String]) -> serde_json::Value {
    let pk = doc
        .and_then(|doc| doc.external_id.clone())
        .unwrap_or_else(|| id.to_string());
    let mut full = serde_json::Map::new();
    full.insert("id".to_string(), serde_json::json!(id));
    full.insert("pk".to_string(), serde_json::json!(pk));
    full.insert(
        "doc_id".to_string(),
        doc.and_then(|doc| doc.external_id.clone())
            .map(serde_json::Value::String)
            .unwrap_or(serde_json::Value::Null),
    );
    full.insert("score".to_string(), serde_json::Value::Null);

    if let Some(doc) = doc {
        full.insert(
            "external_id".to_string(),
            doc.external_id
                .clone()
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null),
        );
        full.insert(
            "vector_field".to_string(),
            serde_json::Value::String(doc.vector_field.clone()),
        );
        full.insert(
            "text".to_string(),
            serde_json::Value::String(doc.text.clone()),
        );
        full.insert("meta".to_string(), doc.meta.clone());
    }

    if select.is_empty() {
        return serde_json::Value::Object(full);
    }

    let mut projected = serde_json::Map::new();
    for field in select {
        if let Some(value) = full.get(field).cloned() {
            projected.insert(field.clone(), value);
        } else if let Some(doc) = doc {
            let value = doc
                .meta
                .get(field)
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            projected.insert(field.clone(), value);
        } else {
            projected.insert(field.clone(), serde_json::Value::Null);
        }
    }
    serde_json::Value::Object(projected)
}

fn cmd_query(db: &Path, sql: &str) -> Result<()> {
    if !db.exists() {
        bail!("db not found: {}", db.display());
    }
    let query = parse_sql_metadata_query(sql)?;
    let conn = open_sidecar(db)?;
    let mut ids = query_doc_ids(&conn, query.filter.as_deref())?;
    if let Some(limit) = query.limit {
        ids.truncate(limit);
    }
    let docs = load_docs_by_ids_sqlite(&conn, &ids)?;
    let records = ids
        .into_iter()
        .map(|id| doc_to_query_record(id, docs.get(&id), &query.select))
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "records": records }))?
    );
    Ok(())
}

fn cmd_info(index: &Path) -> Result<()> {
    if !index.exists() {
        bail!("db not found: {}", index.display());
    }

    let idx = IdMapIndex::load(index).context("loading .tvim index")?;
    let meta = load_meta(index).ok();

    let file_size = fs::metadata(index)?.len();
    let texts_count = if sqlite_path(index).exists() {
        let conn = open_sidecar(index)?;
        sqlite_doc_count(&conn).unwrap_or(0)
    } else {
        0
    };

    let meta_json = meta
        .map(|m| {
            serde_json::json!({
                "bits": m.bits,
                "model": m.model,
                "next_id": m.next_id
            })
        })
        .unwrap_or_else(|| serde_json::json!({}));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "db": index.display().to_string(),
            "dimension": idx.dim(),
            "vectors": idx.len(),
            "texts": texts_count,
            "index_size_bytes": file_size,
            "meta": meta_json
        }))?
    );
    Ok(())
}

fn path_arg_to_optional(path: Option<PathBuf>) -> Option<PathBuf> {
    path.filter(|path| path.as_os_str() != "-")
}

fn merge_embedding_arg(cli: Option<String>, config: &AppConfig) -> Option<String> {
    cli.or_else(|| config.embedding.clone())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(cli.config.as_deref())?;

    match cli.command {
        Commands::Init { db, dim, bits } => {
            let db = resolve_db_path(db, &config)?;
            cmd_init(&db, dim, bits)
        }
        Commands::Import {
            db,
            schema,
            embedding,
            input,
            model,
            provider,
            base_url,
            batch_size,
            vector_field,
            text_field,
            dim,
            bits,
            upsert,
        } => {
            let db = resolve_db_path(db, &config)?;
            let embedding_arg = merge_embedding_arg(embedding, &config);
            let embedding = parse_embedding_config(embedding_arg.as_deref())?;
            let schema = parse_schema_defaults(schema.as_deref())?;
            let model = model
                .or(embedding.model)
                .or_else(|| config.default_vector_model.clone());
            let provider = provider
                .or(embedding.provider)
                .or_else(|| config.provider.clone());
            let base_url = base_url
                .or(embedding.base_url)
                .or_else(|| config.base_url.clone());
            let vector_field = vector_field
                .or(embedding.vector_field)
                .or(schema.vector_field);
            let text_field = text_field.or(embedding.text_field).or(schema.text_field);
            let dim = dim.or(embedding.dimensions).or(schema.dim);
            let input = path_arg_to_optional(input);
            cmd_add(AddOptions {
                db: &db,
                input: input.as_deref(),
                model: model.as_deref(),
                provider: provider.as_deref(),
                base_url: base_url.as_deref(),
                batch_size,
                vector_field: vector_field.as_deref(),
                text_field: text_field.as_deref(),
                dim,
                bits,
                upsert,
            })
            .await
        }
        Commands::Search {
            db,
            query,
            vector,
            vector_file,
            top_k,
            model,
            provider,
            base_url,
            filter,
        } => {
            let db = resolve_db_path(db, &config)?;
            let model = resolve_model(model, &config);
            let provider = resolve_provider(provider, &config);
            let base_url = resolve_base_url(base_url, &config);
            let query_vector = load_vector_arg(vector.as_deref(), vector_file.as_deref())?;
            cmd_search(SearchOptions {
                index: &db,
                query: query.as_deref(),
                vector: query_vector,
                top_k,
                model: &model,
                provider: provider.as_deref(),
                base_url: base_url.as_deref(),
                filter: filter.as_deref(),
            })
            .await
        }
        Commands::Query { db_path, sql } => {
            let db = resolve_db_path(db_path, &config)?;
            cmd_query(&db, &sql)
        }
        Commands::Export {
            db,
            schema,
            output,
            filter,
            include_vectors,
        } => {
            let db = resolve_db_path(db, &config)?;
            let _ = parse_schema_defaults(schema.as_deref())?;
            let output = path_arg_to_optional(output);
            cmd_export(&db, output.as_deref(), filter.as_deref(), include_vectors)
        }
        Commands::FilterIds { db, filter } => {
            let db = resolve_db_path(db, &config)?;
            cmd_filter_ids(&db, &filter)
        }
        Commands::Stats { db } => {
            let db = resolve_db_path(db, &config)?;
            cmd_info(&db)
        }
        Commands::Serve { .. } => {
            bail!(
                "serve is not implemented for turbovec-rs yet; use CLI import/export/query/search"
            )
        }
        Commands::Mcp { .. } => {
            bail!("mcp is not implemented for turbovec-rs yet; use CLI import/export/query/search")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filterql::Expr;

    #[test]
    fn sqlite_path_preserves_tvim_stem() {
        assert_eq!(
            sqlite_path(Path::new("/tmp/docs.tvim")),
            PathBuf::from("/tmp/docs.tvim.sqlite")
        );
    }

    #[test]
    fn cli_parses_filter_ids_subcommand() {
        let cli = Cli::parse_from([
            "turbovec-rs",
            "filter-ids",
            "--db",
            "/tmp/docs.tvim",
            "--filter",
            "lang = 'zh'",
        ]);

        match cli.command {
            Commands::FilterIds { db, filter } => {
                assert_eq!(db, Some(PathBuf::from("/tmp/docs.tvim")));
                assert_eq!(filter, "lang = 'zh'");
            }
            _ => panic!("expected filter-ids subcommand"),
        }
    }

    #[test]
    fn cli_parses_search_vector_without_query() {
        let cli = Cli::parse_from([
            "turbovec-rs",
            "search",
            "--db",
            "/tmp/docs.tvim",
            "--vector",
            "[0.1,0.2]",
        ]);

        match cli.command {
            Commands::Search { query, vector, .. } => {
                assert!(query.is_none());
                assert_eq!(vector.as_deref(), Some("[0.1,0.2]"));
            }
            _ => panic!("expected search subcommand"),
        }
    }

    #[test]
    fn loads_config_from_json_string_with_aliases() {
        let config = load_config(Some(
            r#"{"storage_path":"/tmp/docs.tvim","model":"ollama/bge-m3","base_url":"http://localhost:11434"}"#,
        ))
        .unwrap();

        assert_eq!(config.data_path, Some(PathBuf::from("/tmp/docs.tvim")));
        assert_eq!(
            config.default_vector_model.as_deref(),
            Some("ollama/bge-m3")
        );
        assert_eq!(config.base_url.as_deref(), Some("http://localhost:11434"));
    }

    #[test]
    fn loads_config_from_file_path() {
        let path = std::env::temp_dir().join(format!(
            "turbovec-rs-config-test-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"{"data_path":"/tmp/from-file.tvim","default_vector_model":"bge-m3"}"#,
        )
        .unwrap();

        let config = load_config(Some(path.to_str().unwrap())).unwrap();

        assert_eq!(config.data_path, Some(PathBuf::from("/tmp/from-file.tvim")));
        assert_eq!(config.default_vector_model.as_deref(), Some("bge-m3"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn resolves_index_and_model_with_cli_precedence() {
        let config = AppConfig {
            data_path: Some(PathBuf::from("/tmp/config.tvim")),
            default_vector_model: Some("ollama/bge-m3".to_string()),
            provider: Some("ollama".to_string()),
            base_url: Some("http://example.test".to_string()),
            embedding: None,
        };

        assert_eq!(
            resolve_db_path(Some(PathBuf::from("/tmp/cli.tvim")), &config).unwrap(),
            PathBuf::from("/tmp/cli.tvim")
        );
        assert_eq!(
            resolve_db_path(None, &config).unwrap(),
            PathBuf::from("/tmp/config.tvim")
        );
        assert_eq!(
            resolve_model(Some("cli-model".to_string()), &config),
            "cli-model"
        );
        assert_eq!(resolve_model(None, &config), "ollama/bge-m3");
        assert_eq!(
            resolve_provider(Some("yxt".to_string()), &config).as_deref(),
            Some("yxt")
        );
        assert_eq!(
            resolve_base_url(Some("http://cli.test".to_string()), &config).as_deref(),
            Some("http://cli.test")
        );
    }

    #[test]
    fn normalizes_provider_prefixed_model_when_provider_is_explicit() {
        let (model, provider) = normalize_provider_model("ollama/bge-m3", Some("ollama")).unwrap();
        assert_eq!(model, "bge-m3");
        assert_eq!(provider.as_deref(), Some("ollama"));

        let err = normalize_provider_model("ollama/bge-m3", Some("yxt"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("conflicts"));
    }

    #[test]
    fn sqlite_schema_initializes_and_counts_docs() {
        let conn = Connection::open_in_memory().unwrap();
        init_sidecar_schema(&conn).unwrap();
        insert_doc(
            &conn,
            42,
            Some("external-42"),
            "content",
            "hello",
            &serde_json::json!({"source":"docs","lang":"zh"}),
        )
        .unwrap();
        assert_eq!(sqlite_doc_count(&conn).unwrap(), 1);
    }

    #[test]
    fn filter_compiler_uses_placeholders_and_params() {
        let compiled =
            compile_filter("source = 'docs' AND lang = 'zh' AND kind IN ('guide','api')").unwrap();
        assert!(compiled.clause.contains("json_extract(meta, ?) = ?"));
        assert!(compiled.clause.contains("json_extract(meta, ?) IN (?, ?)"));
        assert_eq!(compiled.params.len(), 7);
    }

    #[test]
    fn filter_compiler_rejects_invalid_field_names() {
        let err = compile_filter("bad-field = 'x'").unwrap_err().to_string();
        assert!(err.contains("unsupported characters") || err.contains("parsing metadata filter"));
    }

    #[test]
    fn filter_ids_queries_sqlite_sidecar() {
        let index = std::env::temp_dir().join(format!(
            "turbovec-rs-filter-test-{}-{}.tvim",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let sqlite = sqlite_path(&index);
        let _ = fs::remove_file(&sqlite);

        let conn = open_sidecar(&index).unwrap();
        insert_doc(
            &conn,
            1,
            Some("a"),
            "content",
            "a",
            &serde_json::json!({"source":"docs","lang":"zh","kind":"guide","created_at":1700000000}),
        )
        .unwrap();
        insert_doc(
            &conn,
            2,
            Some("b"),
            "content",
            "b",
            &serde_json::json!({"source":"docs","lang":"en","kind":"api","created_at":1700000001}),
        )
        .unwrap();
        drop(conn);

        let ids = filter_ids(
            &index,
            "source = 'docs' AND lang = 'zh' AND created_at >= 1700000000",
        )
        .unwrap();
        assert_eq!(ids, vec![1]);

        let _ = fs::remove_file(sqlite);
    }

    #[test]
    fn empty_in_lists_have_total_boolean_semantics() {
        let in_clause = filterql::compile(
            &Expr::cmp("source", CmpOp::In, FilterValue::List(Vec::new())),
            &mut SqliteFilterCompiler,
        )
        .unwrap();
        assert_eq!(in_clause.clause, "(0 = 1)");

        let not_in_clause = filterql::compile(
            &Expr::cmp("source", CmpOp::NotIn, FilterValue::List(Vec::new())),
            &mut SqliteFilterCompiler,
        )
        .unwrap();
        assert_eq!(not_in_clause.clause, "(1 = 1)");
    }

    #[test]
    fn parses_rag_style_jsonl_record() {
        let record = parse_import_record(
            r#"{"id":"doc-1","vector_field":"content","fields":{"content":"hello vector","doc":"guide","lang":"zh"}}"#,
            None,
            None,
        )
        .unwrap();

        assert_eq!(record.external_id.as_deref(), Some("doc-1"));
        assert_eq!(record.vector_field, "content");
        assert_eq!(record.vector_text, "hello vector");
        assert_eq!(record.meta["external_id"], "doc-1");
        assert_eq!(record.meta["doc"], "guide");
        assert!(record.meta.get("content").is_none());
        assert!(record.vector.is_none());
    }

    #[test]
    fn parses_descriptor_jsonl_record_with_vector_marker() {
        let record = parse_import_record(
            r#"{"id":7,"fields":{"content":{"value":"semantic text","index":["vector"]},"kind":{"value":"note","index":["filter"]}}}"#,
            None,
            None,
        )
        .unwrap();

        assert_eq!(record.external_id.as_deref(), Some("7"));
        assert_eq!(record.vector_field, "content");
        assert_eq!(record.vector_text, "semantic text");
        assert_eq!(record.meta["kind"], "note");
    }

    #[test]
    fn parses_direct_precomputed_vector() {
        let record = parse_import_record(
            r#"{"id":"doc-1","vector_field":"content","fields":{"content":"kept text","lang":"zh"},"vector":[0.1,0.2,-0.3]}"#,
            None,
            None,
        )
        .unwrap();

        assert_eq!(record.vector_field, "content");
        assert_eq!(record.vector.as_deref(), Some(&[0.1, 0.2, -0.3][..]));
        assert_eq!(record.vector_text, "kept text");
    }

    #[test]
    fn parses_keyed_precomputed_vector_and_infers_field() {
        let record = parse_import_record(
            r#"{"id":"doc-1","fields":{"content":"kept text","lang":"zh"},"vectors":{"content":[0.1,0.2]}}"#,
            None,
            None,
        )
        .unwrap();

        assert_eq!(record.vector_field, "content");
        assert_eq!(record.vector.as_deref(), Some(&[0.1, 0.2][..]));
    }

    #[test]
    fn parses_query_vector_from_arg_and_file() {
        let vector = load_vector_arg(Some("[0.1,0.2]"), None).unwrap();
        assert_eq!(vector.as_deref(), Some(&[0.1, 0.2][..]));

        let path = std::env::temp_dir().join(format!(
            "turbovec-rs-vector-test-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&path, "[0.3,0.4]").unwrap();
        let vector = load_vector_arg(None, Some(&path)).unwrap();
        assert_eq!(vector.as_deref(), Some(&[0.3, 0.4][..]));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_multiple_vector_fields() {
        let err = parse_import_record(
            r#"{"vector_fields":["title","body"],"fields":{"title":"a","body":"b"}}"#,
            None,
            None,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("expected exactly one vector field"));
    }

    #[test]
    fn parses_zvec_style_jsonl_record() {
        let record = parse_import_record(
            r#"{"pk":"doc-1","fields":{"text":"semantic text","category":"tech"}}"#,
            None,
            Some("text"),
        )
        .unwrap();

        assert_eq!(record.external_id.as_deref(), Some("doc-1"));
        assert_eq!(record.vector_field, "text");
        assert_eq!(record.vector_text, "semantic text");
        assert_eq!(record.meta["category"], "tech");
        assert!(record.meta.get("text").is_none());
    }

    #[test]
    fn parses_sql_metadata_query() {
        let query = parse_sql_metadata_query(
            "SELECT pk, category FROM articles WHERE category = 'tech' LIMIT 20",
        )
        .unwrap();
        assert_eq!(query.select, vec!["pk", "category"]);
        assert_eq!(query.filter.as_deref(), Some("category = 'tech'"));
        assert_eq!(query.limit, Some(20));
    }
}
