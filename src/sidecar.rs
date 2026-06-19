//! SQLite sidecar storing document metadata alongside the quantized index.

use anyhow::{Context, Result};
use rusqlite::{params, params_from_iter, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::filter::{compile_filter, SqliteWhere};

/// Index metadata persisted alongside the .tvim file.
#[derive(Serialize, Deserialize, Default)]
pub(crate) struct IndexMeta {
    pub(crate) next_id: u64,
    pub(crate) dim: usize,
    pub(crate) bits: usize,
    pub(crate) model: String,
}

pub(crate) fn sqlite_path(index: &Path) -> PathBuf {
    let mut p = index.to_path_buf();
    p.set_extension("tvim.sqlite");
    p
}

pub(crate) fn meta_path(index: &Path) -> PathBuf {
    let mut p = index.to_path_buf();
    p.set_extension("tvim.meta.json");
    p
}

pub(crate) fn load_meta(index: &Path) -> Result<IndexMeta> {
    let path = meta_path(index);
    let data = fs::read_to_string(&path).context("reading meta file")?;
    Ok(serde_json::from_str(&data)?)
}

pub(crate) fn save_meta(index: &Path, meta: &IndexMeta) -> Result<()> {
    let path = meta_path(index);
    let data = serde_json::to_string_pretty(meta)?;
    fs::write(&path, data).context("writing meta file")?;
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct DocRow {
    pub(crate) external_id: Option<String>,
    pub(crate) vector_field: String,
    pub(crate) text: String,
    pub(crate) meta: serde_json::Value,
}

pub(crate) fn open_sidecar(index: &Path) -> Result<Connection> {
    let path = sqlite_path(index);
    let conn = Connection::open(&path)
        .with_context(|| format!("opening SQLite sidecar {}", path.display()))?;
    init_sidecar_schema(&conn)?;
    Ok(conn)
}

pub(crate) fn init_sidecar_schema(conn: &Connection) -> Result<()> {
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

pub(crate) fn insert_doc(
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

pub(crate) fn sqlite_doc_count(conn: &Connection) -> Result<usize> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM docs", [], |row| row.get(0))?;
    usize::try_from(count).context("SQLite doc count is negative or too large")
}

pub(crate) fn external_id_exists(conn: &Connection, external_id: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM docs WHERE external_id = ?1",
        params![external_id],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub(crate) fn query_doc_ids(conn: &Connection, filter: Option<&str>) -> Result<Vec<u64>> {
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

pub(crate) fn load_docs_by_ids(index: &Path, ids: &[u64]) -> Result<HashMap<u64, DocRow>> {
    let conn = open_sidecar(index)?;
    load_docs_by_ids_sqlite(&conn, ids)
}

pub(crate) fn load_docs_by_ids_sqlite(
    conn: &Connection,
    ids: &[u64],
) -> Result<HashMap<u64, DocRow>> {
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

/// Resolve ids matching the filter against the SQLite sidecar (used by `filter_ids`).
pub(crate) fn filter_ids_via_sidecar(
    index: &Path,
    compiled: &SqliteWhere,
) -> Result<Vec<u64>> {
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

