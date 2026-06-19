//! zvec-style SQL metadata query parsing and execution.

use anyhow::{anyhow, bail, Context, Result};
use std::path::Path;

use crate::sidecar::{load_docs_by_ids_sqlite, open_sidecar, query_doc_ids, DocRow};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SqlMetadataQuery {
    select: Vec<String>,
    filter: Option<String>,
    limit: Option<usize>,
}

fn find_keyword_ci(haystack: &str, keyword: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&keyword.to_ascii_lowercase())
}

pub(crate) fn parse_sql_metadata_query(sql: &str) -> Result<SqlMetadataQuery> {
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

pub(crate) fn cmd_query(db: &Path, sql: &str) -> Result<()> {
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

#[cfg(test)]
#[path = "sql_query_test.rs"]
mod tests;
