//! turbovec-rs — Persistent vector index CLI with semantic search.
//!
//! Uses turbovec for 2-4bit quantized vector storage and the embeddings crate
//! for generating embeddings via Ollama / BGE-M3 (or any supported provider).
//!
//! # Examples
//!
//! ```bash
//! # Init an index
//! turbovec-rs init --index /tmp/docs.tvim
//!
//! # Add documents (one per line)
//! turbovec-rs add --index /tmp/docs.tvim --file docs.txt
//!
//! # Search
//! turbovec-rs search --index /tmp/docs.tvim --query "什么是编程"
//!
//! # Show index info
//! turbovec-rs info --index /tmp/docs.tvim
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use embeddings::{resolve_api_key_for_provider, EmbedClient};
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
    about = "Persistent vector index with semantic search (turbovec + embeddings)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
    /// Add documents from a file (one per line) to the index
    Add {
        /// Path to the index file (.tvim)
        #[arg(long)]
        index: PathBuf,
        /// Text file to ingest (one document per line)
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
            map.insert(id.as_u64().unwrap_or(0), text.as_str().unwrap_or("").to_string());
        }
    }
    Ok(map)
}

fn append_texts(index: &Path, entries: &[(u64, String)]) -> Result<()> {
    let path = texts_path(index);
    let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
    for (id, text) in entries {
        serde_json::to_writer(&mut file, &serde_json::json!({"id": id, "text": text}))?;
        use std::io::Write;
        writeln!(&mut file)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn build_client(model: &str, provider: Option<&str>, base_url: Option<&str>) -> Result<EmbedClient> {
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
// Subcommands
// ---------------------------------------------------------------------------

fn cmd_init(index: &Path, dim: usize, bits: usize) -> Result<()> {
    if dim == 0 || dim % 8 != 0 {
        bail!("dim must be a positive multiple of 8, got {dim}");
    }
    if ![2, 3, 4].contains(&bits) {
        bail!("bits must be 2, 3, or 4, got {bits}");
    }

    let idx = IdMapIndex::new(dim, bits).context("creating IdMapIndex")?;
    idx.write(index).context("writing .tvim file")?;

    // Create empty sidecar files
    fs::File::create(texts_path(index))?;
    save_meta(index, &IndexMeta { next_id: 1, dim, bits, model: String::new() })?;

    eprintln!("✅ Index created: {} (dim={}, bits={})", index.display(), dim, bits);
    Ok(())
}

async fn cmd_add(
    index: &Path,
    file: &Path,
    model: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
    batch_size: usize,
) -> Result<()> {
    if !file.exists() {
        bail!("file not found: {}", file.display());
    }
    if !index.exists() {
        bail!("index not found: {} (run `init` first)", index.display());
    }

    let texts: Vec<String> = fs::read_to_string(file)?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    if texts.is_empty() {
        bail!("no non-empty lines in {}", file.display());
    }

    let mut meta = load_meta(index)?;
    let mut idx = IdMapIndex::load(index).context("loading .tvim index")?;

    eprintln!("📄 {} documents to add (batch_size={})", texts.len(), batch_size);

    let client = build_client(model, provider, base_url)?;

    let mut added = 0usize;
    for chunk in texts.chunks(batch_size) {
        let output = client
            .embed(chunk.to_vec())
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

        let ids: Vec<u64> = (meta.next_id..meta.next_id + chunk.len() as u64).collect();
        let flat = flatten_embeddings(&output.embeddings);

        idx.add_with_ids_2d(&flat, meta.dim, &ids)
            .context("adding vectors to index")?;

        // Append texts to sidecar
        let entries: Vec<(u64, String)> = ids.iter().zip(chunk.iter()).map(|(&id, t)| (id, t.clone())).collect();
        append_texts(index, &entries)?;

        meta.next_id += chunk.len() as u64;
        added += chunk.len();

        eprintln!("   +{}/{} embedded", added, texts.len());
    }

    // Persist index and meta
    idx.write(index).context("writing index")?;
    meta.model = model.to_string();
    save_meta(index, &meta)?;

    eprintln!("✅ Added {added} documents (total: {})", idx.len());
    Ok(())
}

async fn cmd_search(
    index: &Path,
    query: &str,
    top_k: usize,
    model: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> Result<()> {
    if !index.exists() {
        bail!("index not found: {}", index.display());
    }

    let idx = IdMapIndex::load(index).context("loading .tvim index")?;
    if idx.is_empty() {
        bail!("index is empty (add documents first)");
    }

    let texts = load_texts(index)?;
    let client = build_client(model, provider, base_url)?;

    let query_vec = client.embed_one(query).await.context("embedding query")?;
    let query_flat = query_vec;

    let (scores, ids) = idx.search(&query_flat, top_k);

    // Build JSON output
    let mut results = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let score = scores[i];
        let text = texts.get(&id).cloned().unwrap_or_else(|| format!("<id {} text missing>", id));
        results.push(serde_json::json!({
            "id": id,
            "score": score,
            "text": text,
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
    let texts_count = load_texts(index).map(|t| t.len()).unwrap_or(0);

    println!("📊 Index: {}", index.display());
    println!("   Dimension:   {}", idx.dim());
    println!("   Vectors:     {}", idx.len());
    println!("   Texts:       {texts_count}");
    println!("   Index size:  {} bytes ({:.1} KB)", file_size, file_size as f64 / 1024.0);
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
        Commands::Add { index, file, model, provider, base_url, batch_size } => {
            cmd_add(&index, &file, &model, provider.as_deref(), base_url.as_deref(), batch_size).await
        }
        Commands::Search { index, query, top_k, model, provider, base_url } => {
            cmd_search(&index, &query, top_k, &model, provider.as_deref(), base_url.as_deref()).await
        }
        Commands::Info { index } => cmd_info(&index),
    }
}
