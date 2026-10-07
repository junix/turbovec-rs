//! Subcommand implementations (init/add/search/export/filter-ids/info).

use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use turbovec::IdMapIndex;

use crate::config::DEFAULT_MODEL;
use crate::embed::{build_client, flatten_embeddings, validate_vectors_dim};
use crate::filter::compile_filter;
use crate::import::ImportRecordReader;
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

/// When `db` is missing, infer the dimension (explicit flag, else the first
/// record of the leading batch with a vector, else 1024) and create a fresh
/// index + sidecar schema.
fn bootstrap_missing_index(
    db: &Path,
    dim: Option<usize>,
    records: &[crate::import::ImportRecord],
    bits: usize,
) -> Result<()> {
    let inferred_dim = dim
        .or_else(|| {
            records
                .iter()
                .find_map(|record| record.vector.as_ref().map(Vec::len))
        })
        .unwrap_or(1024);
    create_index(db, inferred_dim, bits)
}

/// Embed text for every record in `batch_vectors` whose entry is `None`.
/// Writes the resulting vectors back in place and returns `true` if at least
/// one embedding call was made (used later to set `meta.model`).
async fn embed_missing_vectors(
    client: Option<&embeddings::EmbedClient>,
    batch: &[crate::import::ImportRecord],
    batch_vectors: &mut [Option<Vec<f32>>],
) -> Result<bool> {
    let embed_indices = batch_vectors
        .iter()
        .enumerate()
        .filter_map(|(idx, vector)| vector.is_none().then_some(idx))
        .collect::<Vec<_>>();
    if embed_indices.is_empty() {
        return Ok(false);
    }
    let client =
        client.ok_or_else(|| anyhow!("records without vectors require an embedding model"))?;
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
    Ok(true)
}

/// Reject any record whose external_id is already present in the sidecar.
/// turbovec-rs cannot overwrite vectors in-place yet.
fn ensure_no_external_id_duplicates(
    conn: &rusqlite::Connection,
    batch: &[crate::import::ImportRecord],
) -> Result<()> {
    for record in batch {
        if let Some(external_id) = record.external_id.as_deref() {
            if external_id_exists(conn, external_id)? {
                bail!(
                    "primary key `{external_id}` already exists; turbovec-rs cannot overwrite vectors in-place yet"
                );
            }
        }
    }
    Ok(())
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

    let batch_size = batch_size.max(1);
    let mut reader = ImportRecordReader::open(input, vector_field, text_field)?;
    // Prime the first (bounded) batch so index bootstrap can infer dim from
    // the leading records without loading the whole input.
    let first_batch = reader
        .next_batch(batch_size)?
        .ok_or_else(|| anyhow!("no JSONL records found in {}", reader.source()))?;

    if !db.exists() {
        bootstrap_missing_index(db, dim, &first_batch, bits)?;
    }

    let mut meta = crate::sidecar::load_meta(db)?;
    let mut idx = IdMapIndex::load(db).context("loading .tvim index")?;
    let conn = open_sidecar(db)?;
    let embed_model = model.unwrap_or(DEFAULT_MODEL);
    let mut client: Option<embeddings::EmbedClient> = None;
    let mut used_embedding = false;

    eprintln!(
        "importing JSONL from {} (batch_size={})",
        reader.source(),
        batch_size
    );

    // Failure policy: an import is atomic per run. Doc rows written by this
    // run are tracked in `inserted_ids` and rolled back when any batch fails,
    // so the sidecar never references ids missing from the on-disk index
    // (which is only rewritten on success). Re-running the same input after a
    // failure resumes from a clean state.
    let mut inserted_ids: Vec<u64> = Vec::new();
    let mut added = 0usize;
    let mut failure: Option<anyhow::Error> = None;
    let mut pending = Some(first_batch);
    while let Some(batch) = pending.take() {
        match import_one_batch(
            &mut client,
            embed_model,
            provider,
            base_url,
            &batch,
            &mut idx,
            &conn,
            &mut meta,
            &mut inserted_ids,
        )
        .await
        {
            Ok(embedded) => {
                used_embedding |= embedded;
                added += batch.len();
                eprintln!("+{added} imported");
            }
            Err(err) => {
                failure = Some(err);
                break;
            }
        }
        match reader.next_batch(batch_size) {
            Ok(next) => pending = next,
            Err(err) => {
                failure = Some(err);
                break;
            }
        }
    }
    if let Some(err) = failure {
        if !inserted_ids.is_empty() {
            let rolled_back = inserted_ids.len();
            match crate::sidecar::delete_docs(&conn, &inserted_ids) {
                Ok(_) => eprintln!("import failed; rolled back {rolled_back} sidecar docs"),
                Err(rollback_err) => eprintln!(
                    "warning: rollback of {rolled_back} imported docs failed: {rollback_err}"
                ),
            }
        }
        return Err(err);
    }

    // Persist index and meta
    idx.write(db).context("writing index")?;
    if used_embedding {
        meta.model = embed_model.to_string();
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

/// Import one batch: build the embedding client lazily on first need, embed
/// records without vectors, validate dimensions, reject duplicate external
/// ids, append vectors to the in-memory index, and write sidecar rows. Every
/// inserted doc id is appended to `inserted_ids` so a failure in a later
/// batch can roll this run's sidecar writes back. Returns whether an
/// embedding call was made for this batch.
async fn import_one_batch(
    client: &mut Option<embeddings::EmbedClient>,
    embed_model: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
    batch: &[crate::import::ImportRecord],
    idx: &mut IdMapIndex,
    conn: &rusqlite::Connection,
    meta: &mut IndexMeta,
    inserted_ids: &mut Vec<u64>,
) -> Result<bool> {
    let mut batch_vectors = batch
        .iter()
        .map(|record| record.vector.clone())
        .collect::<Vec<_>>();

    if batch.iter().any(|record| record.vector.is_none()) && client.is_none() {
        *client = Some(build_client(embed_model, provider, base_url)?);
    }
    let used_embedding = embed_missing_vectors(client.as_ref(), batch, &mut batch_vectors).await?;

    let vectors = batch_vectors
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow!("missing vector after embedding import batch"))?;
    validate_vectors_dim(&vectors, meta.dim)?;

    ensure_no_external_id_duplicates(conn, batch)?;

    let ids: Vec<u64> = (meta.next_id..meta.next_id + batch.len() as u64).collect();
    let flat = flatten_embeddings(&vectors);

    idx.add_with_ids_2d(&flat, meta.dim, &ids)
        .context("adding vectors to index")?;

    for (&id, record) in ids.iter().zip(batch.iter()) {
        insert_doc(
            conn,
            id,
            record.external_id.as_deref(),
            &record.vector_field,
            &record.vector_text,
            &record.meta,
        )?;
        inserted_ids.push(id);
    }

    meta.next_id += batch.len() as u64;
    Ok(used_embedding)
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

#[cfg(test)]
#[path = "commands_tests.rs"]
mod tests;
