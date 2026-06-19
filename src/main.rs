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

mod commands;
mod config;
mod embed;
mod filter;
mod import;
mod sidecar;
mod sql_query;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use commands::{
    cmd_add, cmd_export, cmd_filter_ids, cmd_info, cmd_init, cmd_search, AddOptions, SearchOptions,
};
use config::{
    load_config, merge_embedding_arg, parse_embedding_config, parse_schema_defaults,
    path_arg_to_optional, resolve_base_url, resolve_db_path, resolve_model, resolve_provider,
};
use import::load_vector_arg;
use sql_query::cmd_query;

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
    use std::path::PathBuf;

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
}
