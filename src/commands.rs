//! Subcommand implementations (init/add/search/export/filter-ids/info).

use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use turbovec::IdMapIndex;

use crate::config::DEFAULT_MODEL;
use crate::embed::{build_client, flatten_embeddings, validate_vectors_dim};
use crate::filter::compile_filter;
use crate::import::load_import_records;
use crate::sidecar::{
    external_id_exists, filter_ids_via_sidecar, init_sidecar_schema, insert_doc, load_docs_by_ids,
    load_docs_by_ids_sqlite, open_sidecar, query_doc_ids, save_meta, DocRow, IndexMeta,
};

pub(crate) fn cmd_init(index: &Path, dim: usize, bits: usize) -> Result<()> {
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

pub(crate) fn create_index(index: &Path, dim: usize, bits: usize) -> Result<()> {
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

pub(crate) struct AddOptions<'a> {
    pub(crate) db: &'a Path,
    pub(crate) input: Option<&'a Path>,
    pub(crate) model: Option<&'a str>,
    pub(crate) provider: Option<&'a str>,
    pub(crate) base_url: Option<&'a str>,
    pub(crate) batch_size: usize,
    pub(crate) vector_field: Option<&'a str>,
    pub(crate) text_field: Option<&'a str>,
    pub(crate) dim: Option<usize>,
    pub(crate) bits: usize,
    pub(crate) upsert: bool,
}

pub(crate) async fn cmd_add(opts: AddOptions<'_>) -> Result<()> {
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

    let mut meta = crate::sidecar::load_meta(db)?;
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

pub(crate) struct SearchOptions<'a> {
    pub(crate) index: &'a Path,
    pub(crate) query: Option<&'a str>,
    pub(crate) vector: Option<Vec<f32>>,
    pub(crate) top_k: usize,
    pub(crate) model: &'a str,
    pub(crate) provider: Option<&'a str>,
    pub(crate) base_url: Option<&'a str>,
    pub(crate) filter: Option<&'a str>,
}

pub(crate) async fn cmd_search(opts: SearchOptions<'_>) -> Result<()> {
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

pub(crate) fn cmd_filter_ids(index: &Path, filter: &str) -> Result<()> {
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

pub(crate) fn cmd_export(
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

pub(crate) fn cmd_info(index: &Path) -> Result<()> {
    if !index.exists() {
        bail!("db not found: {}", index.display());
    }

    let idx = IdMapIndex::load(index).context("loading .tvim index")?;
    let meta = crate::sidecar::load_meta(index).ok();

    let file_size = fs::metadata(index)?.len();
    let texts_count = if crate::sidecar::sqlite_path(index).exists() {
        let conn = open_sidecar(index)?;
        crate::sidecar::sqlite_doc_count(&conn).unwrap_or(0)
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

pub(crate) fn filter_ids(index: &Path, filter: &str) -> Result<Vec<u64>> {
    let path = crate::sidecar::sqlite_path(index);
    if !path.exists() {
        bail!("metadata filter requires SQLite sidecar {}", path.display());
    }

    let compiled = compile_filter(filter)?;
    filter_ids_via_sidecar(index, &compiled)
}
