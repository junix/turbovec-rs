//! turbovec-rs — Persistent vector index CLI with semantic search.
//!
//! Uses turbovec for 2-4bit quantized vector storage, chonkie for chunking,
//! and the embeddings crate for generating embeddings via Ollama / BGE-M3.
//!
//! # Examples
//!
//! ```bash
//! # Init an index
//! turbovec-rs init --index /tmp/docs.tvim
//!
//! # Add documents (one per line)
//! turbovec-rs add --index /tmp/docs.tvim --file docs.txt --provider ollama
//!
//! # Add with chunking (split long texts into ~500 char chunks)
//! turbovec-rs add --index /tmp/docs.tvim --file article.md --provider ollama --chunk-size 500
//!
//! # Search
//! turbovec-rs search --index /tmp/docs.tvim --query "什么是编程" --provider ollama
//!
//! # Show index info
//! turbovec-rs info --index /tmp/docs.tvim
//! ```

use anyhow::{anyhow, bail, Context, Result};
use chonkie::{
    CharChunker, Chunker, RecursiveChunker, RecursiveRules, SentenceChunker, TiktokenTokenizer,
};
use clap::{Parser, Subcommand};
use embeddings::{resolve_api_key_for_provider, EmbedClient};
use filterql::validate::{Policy, Schema, ValueType};
use filterql::{CmpOp, Compile, Value as FilterValue};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use turbovec::IdMapIndex;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "turbovec-rs",
    version = "0.1.0",
    about = "Persistent vector index with semantic search (turbovec + embeddings + chonkie)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum ChunkStrategy {
    /// One document per line (no chunking, default)
    Line,
    /// Fixed-size character chunks
    Char,
    /// Sentence-boundary chunks
    Sentence,
    /// Recursive: paragraph → sentence → word
    Recursive,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new empty index
    Init {
        /// Path to the index file (.tvim)
        #[arg(long)]
        index: PathBuf,
        /// Vector dimensionality [default: 1024 for bge-m3]
        #[arg(long, default_value_t = 1024)]
        dim: usize,
        /// Quantization bit width (2, 3, or 4) [default: 4]
        #[arg(long, default_value_t = 4)]
        bits: usize,
    },
    /// Add documents from a file to the index
    Add {
        /// Path to the index file (.tvim)
        #[arg(long)]
        index: PathBuf,
        /// Text file to ingest
        #[arg(long)]
        file: PathBuf,
        /// Embedding model [default: bge-m3]
        #[arg(long, default_value = "bge-m3")]
        model: String,
        /// Provider (auto-detected if omitted)
        #[arg(long)]
        provider: Option<String>,
        /// Custom base URL
        #[arg(long)]
        base_url: Option<String>,
        /// Batch size for embedding API calls
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// Chunking strategy
        #[arg(long, default_value = "line")]
        chunk: ChunkStrategy,
        /// Chunk size in characters (for char/recursive strategies)
        #[arg(long, default_value_t = 500)]
        chunk_size: usize,
        /// Overlap between chunks in characters
        #[arg(long, default_value_t = 50)]
        chunk_overlap: usize,
        /// JSON object metadata applied to every inserted chunk
        #[arg(long, default_value = "{}")]
        meta: String,
    },
    /// Search the index with a text query
    Search {
        /// Path to the index file (.tvim)
        #[arg(long)]
        index: PathBuf,
        /// Query text
        #[arg(long)]
        query: String,
        /// Number of results
        #[arg(long, short = 'k', default_value_t = 10)]
        top_k: usize,
        /// Embedding model (must match the model used for add)
        #[arg(long, default_value = "bge-m3")]
        model: String,
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
    /// Show index metadata
    Info {
        /// Path to the index file (.tvim)
        #[arg(long)]
        index: PathBuf,
    },
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

fn texts_path(index: &Path) -> PathBuf {
    let mut p = index.to_path_buf();
    p.set_extension("tvim.texts.jsonl");
    p
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

fn load_texts(index: &Path) -> Result<HashMap<u64, String>> {
    let path = texts_path(index);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let mut map = HashMap::new();
    for line in fs::read_to_string(&path)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: serde_json::Value = serde_json::from_str(line)?;
        if let (Some(id), Some(text)) = (entry.get("id"), entry.get("text")) {
            map.insert(
                id.as_u64().unwrap_or(0),
                text.as_str().unwrap_or("").to_string(),
            );
        }
    }
    Ok(map)
}

fn append_texts(index: &Path, entries: &[(u64, String)]) -> Result<()> {
    let path = texts_path(index);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    for (id, text) in entries {
        serde_json::to_writer(&mut file, &serde_json::json!({"id": id, "text": text}))?;
        use std::io::Write;
        writeln!(&mut file)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct DocRow {
    text: String,
    meta: serde_json::Value,
}

fn parse_meta_json(input: &str) -> Result<serde_json::Value> {
    let value: serde_json::Value =
        serde_json::from_str(input).with_context(|| format!("parsing metadata JSON: {input}"))?;
    if !value.is_object() {
        bail!("metadata must be a JSON object");
    }
    Ok(value)
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
          text TEXT NOT NULL,
          meta TEXT NOT NULL DEFAULT '{}',

          source TEXT GENERATED ALWAYS AS (json_extract(meta, '$.source')) VIRTUAL,
          lang TEXT GENERATED ALWAYS AS (json_extract(meta, '$.lang')) VIRTUAL,
          kind TEXT GENERATED ALWAYS AS (json_extract(meta, '$.kind')) VIRTUAL,
          created_at INTEGER GENERATED ALWAYS AS (json_extract(meta, '$.created_at')) VIRTUAL
        );

        CREATE INDEX IF NOT EXISTS docs_source_lang ON docs(source, lang);
        CREATE INDEX IF NOT EXISTS docs_kind ON docs(kind);
        CREATE INDEX IF NOT EXISTS docs_created_at ON docs(created_at);
        "#,
    )
    .context("initializing SQLite sidecar schema")?;
    Ok(())
}

fn insert_doc(conn: &Connection, id: u64, text: &str, meta: &serde_json::Value) -> Result<()> {
    let id = i64::try_from(id).context("document id does not fit SQLite INTEGER")?;
    let meta_json = serde_json::to_string(meta)?;
    conn.execute(
        "INSERT OR REPLACE INTO docs (id, text, meta) VALUES (?1, ?2, ?3)",
        params![id, text, meta_json],
    )
    .context("inserting document metadata into SQLite sidecar")?;
    Ok(())
}

fn sqlite_doc_count(conn: &Connection) -> Result<usize> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM docs", [], |row| row.get(0))?;
    usize::try_from(count).context("SQLite doc count is negative or too large")
}

fn load_docs_by_ids(index: &Path, ids: &[u64]) -> Result<HashMap<u64, DocRow>> {
    let path = sqlite_path(index);
    if path.exists() {
        let conn = open_sidecar(index)?;
        return load_docs_by_ids_sqlite(&conn, ids);
    }

    let texts = load_texts(index)?;
    Ok(texts
        .into_iter()
        .map(|(id, text)| {
            (
                id,
                DocRow {
                    text,
                    meta: serde_json::json!({}),
                },
            )
        })
        .collect())
}

fn load_docs_by_ids_sqlite(conn: &Connection, ids: &[u64]) -> Result<HashMap<u64, DocRow>> {
    let mut docs = HashMap::with_capacity(ids.len());
    let mut stmt = conn.prepare("SELECT text, meta FROM docs WHERE id = ?1")?;
    for &id in ids {
        let sql_id = i64::try_from(id).context("document id does not fit SQLite INTEGER")?;
        let row = stmt.query_row(params![sql_id], |row| {
            let text: String = row.get(0)?;
            let meta_json: String = row.get(1)?;
            Ok((text, meta_json))
        });
        if let Ok((text, meta_json)) = row {
            let meta = serde_json::from_str(&meta_json).unwrap_or_else(|_| serde_json::json!({}));
            docs.insert(id, DocRow { text, meta });
        }
    }
    Ok(docs)
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

/// Turn the input file into a list of text chunks based on the chosen strategy.
fn chunk_file(
    file: &Path,
    strategy: &ChunkStrategy,
    size: usize,
    overlap: usize,
) -> Result<Vec<String>> {
    let content =
        fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;

    if content.trim().is_empty() {
        bail!("file is empty: {}", file.display());
    }

    let texts = match strategy {
        ChunkStrategy::Line => content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),

        ChunkStrategy::Char => {
            let chunker = CharChunker::new(size, overlap);
            chunker
                .chunk(&content)
                .into_iter()
                .map(|c| c.text)
                .collect()
        }

        ChunkStrategy::Sentence => {
            let chunker = SentenceChunker::default();
            chunker
                .chunk(&content)
                .into_iter()
                .map(|c| c.text)
                .collect()
        }

        ChunkStrategy::Recursive => {
            let rules = RecursiveRules::default();
            let tokenizer = TiktokenTokenizer::new("gpt-4");
            let chunker = RecursiveChunker::new(size, overlap, rules, tokenizer);
            chunker
                .chunk(&content)
                .into_iter()
                .map(|c| c.text)
                .collect()
        }
    };

    Ok(texts)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn build_client(
    model: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> Result<EmbedClient> {
    let mut client = if let Some(p) = provider {
        let api_key = resolve_api_key_for_provider(p)?;
        EmbedClient::new(p, model, api_key)?
    } else {
        EmbedClient::from_env(model)?
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

fn filter_schema() -> Schema {
    Schema::new()
        .field("source", ValueType::Str)
        .field("lang", ValueType::Str)
        .field("kind", ValueType::Str)
        .field("created_at", ValueType::Int)
        .strict()
}

fn filter_policy() -> Policy {
    Policy::new()
        .schema(filter_schema())
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

fn field_column(field: &str) -> Result<&'static str> {
    match field {
        "source" => Ok("source"),
        "lang" => Ok("lang"),
        "kind" => Ok("kind"),
        "created_at" => Ok("created_at"),
        _ => bail!("unsupported metadata field `{field}`"),
    }
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
        let field = field_column(field)?;
        match op {
            CmpOp::Exists => {
                let present = !matches!(value, FilterValue::Bool(false));
                Ok(SqliteWhere {
                    clause: format!("{field} IS {}NULL", if present { "NOT " } else { "" }),
                    params: Vec::new(),
                })
            }
            CmpOp::Eq if matches!(value, FilterValue::Null) => Ok(SqliteWhere {
                clause: format!("{field} IS NULL"),
                params: Vec::new(),
            }),
            CmpOp::Ne if matches!(value, FilterValue::Null) => Ok(SqliteWhere {
                clause: format!("{field} IS NOT NULL"),
                params: Vec::new(),
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
                    clause: format!("{field} {sql_op} ?"),
                    params: vec![sqlite_value(value)?],
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
                Ok(SqliteWhere {
                    clause: format!("{field} {sql_op} ({placeholders})"),
                    params,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

fn cmd_init(index: &Path, dim: usize, bits: usize) -> Result<()> {
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

    eprintln!(
        "✅ Index created: {} (dim={}, bits={})",
        index.display(),
        dim,
        bits
    );
    Ok(())
}

struct AddOptions<'a> {
    index: &'a Path,
    file: &'a Path,
    model: &'a str,
    provider: Option<&'a str>,
    base_url: Option<&'a str>,
    batch_size: usize,
    chunk_strategy: &'a ChunkStrategy,
    chunk_size: usize,
    chunk_overlap: usize,
    meta_json: &'a str,
}

async fn cmd_add(opts: AddOptions<'_>) -> Result<()> {
    let AddOptions {
        index,
        file,
        model,
        provider,
        base_url,
        batch_size,
        chunk_strategy,
        chunk_size,
        chunk_overlap,
        meta_json,
    } = opts;

    if !file.exists() {
        bail!("file not found: {}", file.display());
    }
    if !index.exists() {
        bail!("index not found: {} (run `init` first)", index.display());
    }

    let doc_meta = parse_meta_json(meta_json)?;
    let texts = chunk_file(file, chunk_strategy, chunk_size, chunk_overlap)?;

    if texts.is_empty() {
        bail!("no text produced from {}", file.display());
    }

    let mut meta = load_meta(index)?;
    let mut idx = IdMapIndex::load(index).context("loading .tvim index")?;
    let conn = open_sidecar(index)?;

    eprintln!(
        "📄 {} chunks to add (strategy={:?}, batch_size={})",
        texts.len(),
        chunk_strategy,
        batch_size
    );

    let client = build_client(model, provider, base_url)?;

    let mut added = 0usize;
    for batch in texts.chunks(batch_size) {
        let output = client
            .embed(batch.to_vec())
            .await
            .context("embedding texts")?;

        // Validate dimension
        if let Some(first) = output.embeddings.first() {
            if first.len() != meta.dim {
                bail!(
                    "embedding dimension mismatch: index expects {}, model produced {}",
                    meta.dim,
                    first.len()
                );
            }
        }

        let ids: Vec<u64> = (meta.next_id..meta.next_id + batch.len() as u64).collect();
        let flat = flatten_embeddings(&output.embeddings);

        idx.add_with_ids_2d(&flat, meta.dim, &ids)
            .context("adding vectors to index")?;

        // Append texts to sidecar
        let entries: Vec<(u64, String)> = ids
            .iter()
            .zip(batch.iter())
            .map(|(&id, t)| (id, t.clone()))
            .collect();
        for (&id, text) in ids.iter().zip(batch.iter()) {
            insert_doc(&conn, id, text, &doc_meta)?;
        }
        append_texts(index, &entries)?;

        meta.next_id += batch.len() as u64;
        added += batch.len();

        eprintln!("   +{}/{} embedded", added, texts.len());
    }

    // Persist index and meta
    idx.write(index).context("writing index")?;
    meta.model = model.to_string();
    save_meta(index, &meta)?;

    eprintln!("✅ Added {added} chunks (total: {})", idx.len());
    Ok(())
}

async fn cmd_search(
    index: &Path,
    query: &str,
    top_k: usize,
    model: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
    filter: Option<&str>,
) -> Result<()> {
    if !index.exists() {
        bail!("index not found: {}", index.display());
    }

    let idx = IdMapIndex::load(index).context("loading .tvim index")?;
    if idx.is_empty() {
        bail!("index is empty (add documents first)");
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

    let texts = load_texts(index)?;
    let client = build_client(model, provider, base_url)?;
    let query_vec = client.embed_one(query).await.context("embedding query")?;

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
            .or_else(|| texts.get(&id).cloned())
            .unwrap_or_else(|| format!("<id {} text missing>", id));
        let meta = doc
            .map(|doc| doc.meta.clone())
            .unwrap_or_else(|| serde_json::json!({}));
        results.push(serde_json::json!({
            "id": id,
            "score": score,
            "text": text,
            "meta": meta,
        }));
    }

    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}

fn cmd_info(index: &Path) -> Result<()> {
    if !index.exists() {
        bail!("index not found: {}", index.display());
    }

    let idx = IdMapIndex::load(index).context("loading .tvim index")?;
    let meta = load_meta(index).ok();

    let file_size = fs::metadata(index)?.len();
    let texts_count = if sqlite_path(index).exists() {
        let conn = open_sidecar(index)?;
        sqlite_doc_count(&conn).unwrap_or(0)
    } else {
        load_texts(index).map(|t| t.len()).unwrap_or(0)
    };

    println!("📊 Index: {}", index.display());
    println!("   Dimension:   {}", idx.dim());
    println!("   Vectors:     {}", idx.len());
    println!("   Texts:       {texts_count}");
    println!(
        "   Index size:  {} bytes ({:.1} KB)",
        file_size,
        file_size as f64 / 1024.0
    );
    if let Some(m) = meta {
        println!("   Bit width:   {}", m.bits);
        println!("   Model:       {}", m.model);
        println!("   Next ID:     {}", m.next_id);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { index, dim, bits } => cmd_init(&index, dim, bits),
        Commands::Add {
            index,
            file,
            model,
            provider,
            base_url,
            batch_size,
            chunk,
            chunk_size,
            chunk_overlap,
            meta,
        } => {
            cmd_add(AddOptions {
                index: &index,
                file: &file,
                model: &model,
                provider: provider.as_deref(),
                base_url: base_url.as_deref(),
                batch_size,
                chunk_strategy: &chunk,
                chunk_size,
                chunk_overlap,
                meta_json: &meta,
            })
            .await
        }
        Commands::Search {
            index,
            query,
            top_k,
            model,
            provider,
            base_url,
            filter,
        } => {
            cmd_search(
                &index,
                &query,
                top_k,
                &model,
                provider.as_deref(),
                base_url.as_deref(),
                filter.as_deref(),
            )
            .await
        }
        Commands::Info { index } => cmd_info(&index),
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
    fn metadata_must_be_json_object() {
        assert!(parse_meta_json(r#"{"source":"docs"}"#).is_ok());
        assert!(parse_meta_json(r#"["docs"]"#).is_err());
        assert!(parse_meta_json("not json").is_err());
    }

    #[test]
    fn sqlite_schema_initializes_and_counts_docs() {
        let conn = Connection::open_in_memory().unwrap();
        init_sidecar_schema(&conn).unwrap();
        insert_doc(
            &conn,
            42,
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
        assert!(compiled.clause.contains("source = ?"));
        assert!(compiled.clause.contains("lang = ?"));
        assert!(compiled.clause.contains("kind IN (?, ?)"));
        assert_eq!(compiled.params.len(), 4);
    }

    #[test]
    fn filter_compiler_rejects_unknown_fields() {
        let err = compile_filter("secret = 'x'").unwrap_err().to_string();
        assert!(err.contains("invalid metadata filter"));
        assert!(err.contains("secret"));
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
            "a",
            &serde_json::json!({"source":"docs","lang":"zh","kind":"guide","created_at":1700000000}),
        )
        .unwrap();
        insert_doc(
            &conn,
            2,
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
}
