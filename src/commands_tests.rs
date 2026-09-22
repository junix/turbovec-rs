use super::*;
use crate::sidecar::{load_meta, sqlite_path};
use std::fs;
use std::path::PathBuf;

/// Unique temp path for an index (.tvim) and the cleanup guard that removes
/// the .tvim, .tvim.sqlite, and .tvim.meta.json artifacts together.
struct TempIndex {
    path: PathBuf,
}

impl TempIndex {
    fn unique(tag: &str) -> TempIndex {
        let path = std::env::temp_dir().join(format!(
            "turbovec-rs-{}-{}-{}.tvim",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        TempIndex { path }
    }

    fn sqlite(&self) -> PathBuf {
        sqlite_path(&self.path)
    }

    fn meta(&self) -> PathBuf {
        let mut p = self.path.clone();
        p.set_extension("tvim.meta.json");
        p
    }

    fn write_jsonl<I, S>(&self, rows: I) -> PathBuf
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let input = self.path.with_extension("jsonl");
        let body = rows
            .into_iter()
            .map(|r| r.as_ref().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&input, body).unwrap();
        input
    }
}

impl Drop for TempIndex {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_file(self.sqlite());
        let _ = fs::remove_file(self.meta());
        let _ = fs::remove_file(self.path.with_extension("jsonl"));
    }
}

#[tokio::test]
async fn cmd_add_bails_when_input_file_missing() {
    let index = TempIndex::unique("add-missing-input");
    let missing = index.path.with_extension("missing.jsonl");

    let err = cmd_add(AddOptions {
        db: &index.path,
        input: Some(&missing),
        model: None,
        provider: None,
        base_url: None,
        batch_size: 8,
        vector_field: None,
        text_field: None,
        dim: None,
        bits: 4,
        upsert: false,
    })
    .await
    .unwrap_err()
    .to_string();

    assert!(err.contains("input file not found"));
}

#[tokio::test]
async fn cmd_add_imports_precomputed_vectors_and_creates_index() {
    let index = TempIndex::unique("add-import");
    let input = index.write_jsonl([
        r#"{"id":"doc-1","vector_field":"content","fields":{"content":"hello world","lang":"zh"},"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}"#,
        r#"{"id":"doc-2","vector_field":"content","fields":{"content":"second doc","lang":"en"},"vector":[0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]}"#,
    ]);

    cmd_add(AddOptions {
        db: &index.path,
        input: Some(&input),
        model: Some("bge-m3"),
        provider: None,
        base_url: None,
        batch_size: 8,
        vector_field: None,
        text_field: None,
        dim: None,
        bits: 4,
        upsert: false,
    })
    .await
    .unwrap();

    let meta = load_meta(&index.path).unwrap();
    assert_eq!(meta.dim, 8);
    assert_eq!(meta.bits, 4);
    assert_eq!(meta.next_id, 3);
    // Precomputed vectors: model is taken from the explicit --model flag.
    assert_eq!(meta.model, "bge-m3");

    let idx = turbovec::IdMapIndex::load(&index.path).unwrap();
    assert_eq!(idx.len(), 2);
}

#[tokio::test]
async fn cmd_add_bails_on_dim_mismatch() {
    let index = TempIndex::unique("add-dim-mismatch");
    let input = index.write_jsonl([
        // First batch establishes dim=8.
        r#"{"id":"a","vector_field":"content","fields":{"content":"a"},"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}"#,
        // Second batch has the wrong length.
        r#"{"id":"b","vector_field":"content","fields":{"content":"b"},"vector":[0.1,0.2,0.3]}"#,
    ]);

    let err = cmd_add(AddOptions {
        db: &index.path,
        input: Some(&input),
        model: None,
        provider: None,
        base_url: None,
        batch_size: 1, // force two batches so the mismatch surfaces after first persists
        vector_field: None,
        text_field: None,
        dim: None,
        bits: 4,
        upsert: false,
    })
    .await
    .unwrap_err()
    .to_string();

    assert!(err.contains("vector dimension mismatch"));
}

#[tokio::test]
async fn cmd_add_bails_on_duplicate_primary_key() {
    let index = TempIndex::unique("add-dup-pk");

    // Seed the index by importing doc-1 first.
    let first = index.write_jsonl([
        r#"{"id":"doc-1","vector_field":"content","fields":{"content":"first"},"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}"#,
    ]);
    cmd_add(AddOptions {
        db: &index.path,
        input: Some(&first),
        model: None,
        provider: None,
        base_url: None,
        batch_size: 8,
        vector_field: None,
        text_field: None,
        dim: None,
        bits: 4,
        upsert: false,
    })
    .await
    .unwrap();

    // Re-import with the same pk in a single batch.
    let second = index.path.with_extension("2.jsonl");
    fs::write(
        &second,
        r#"{"id":"doc-1","vector_field":"content","fields":{"content":"dup"},"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}"#,
    )
    .unwrap();

    let err = cmd_add(AddOptions {
        db: &index.path,
        input: Some(&second),
        model: None,
        provider: None,
        base_url: None,
        batch_size: 8,
        vector_field: None,
        text_field: None,
        dim: None,
        bits: 4,
        upsert: true,
    })
    .await
    .unwrap_err()
    .to_string();

    assert!(err.contains("primary key `doc-1` already exists"));
}

#[test]
fn cmd_export_helper_projects_meta_into_fields() {
    let doc = DocRow {
        external_id: Some("doc-1".to_string()),
        vector_field: "content".to_string(),
        text: "hello".to_string(),
        meta: serde_json::json!({"lang":"zh"}),
    };
    let value = doc_to_export_json(7, &doc);
    assert_eq!(value["pk"], "doc-1");
    assert_eq!(value["fields"]["lang"], "zh");
    assert_eq!(value["fields"]["content"], "hello");

    // No external_id -> pk falls back to numeric id.
    let doc_no_pk = DocRow {
        external_id: None,
        vector_field: "content".to_string(),
        text: "x".to_string(),
        meta: serde_json::Value::Null,
    };
    let value = doc_to_export_json(42, &doc_no_pk);
    assert_eq!(value["pk"], "42");
    // meta=Null contributes no keys; only the vector_field text is projected.
    assert_eq!(value["fields"], serde_json::json!({"content": "x"}));
}

// ---- cmd_search ----

/// Seed an index with two precomputed 8-dim vectors so cmd_search can run
/// without a live embedding server.
async fn seed_index_for_search(tag: &str) -> TempIndex {
    let index = TempIndex::unique(tag);
    let input = index.write_jsonl([
        r#"{"id":"doc-1","vector_field":"content","fields":{"content":"alpha","lang":"zh"},"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}"#,
        r#"{"id":"doc-2","vector_field":"content","fields":{"content":"beta","lang":"en"},"vector":[0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]}"#,
    ]);
    cmd_add(AddOptions {
        db: &index.path,
        input: Some(&input),
        model: None,
        provider: None,
        base_url: None,
        batch_size: 8,
        vector_field: None,
        text_field: None,
        dim: None,
        bits: 4,
        upsert: false,
    })
    .await
    .unwrap();
    index
}

#[tokio::test]
async fn cmd_search_bails_when_index_missing() {
    let index = TempIndex::unique("search-missing");
    let err = cmd_search(SearchOptions {
        index: &index.path,
        query: None,
        vector: Some(vec![0.1; 8]),
        top_k: 1,
        model: "bge-m3",
        provider: None,
        base_url: None,
        filter: None,
    })
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("index not found"));
}

#[tokio::test]
async fn cmd_search_bails_when_query_and_vector_both_passed() {
    let index = seed_index_for_search("search-both").await;
    let err = cmd_search(SearchOptions {
        index: &index.path,
        query: Some("alpha"),
        vector: Some(vec![0.1; 8]),
        top_k: 1,
        model: "bge-m3",
        provider: None,
        base_url: None,
        filter: None,
    })
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("pass only one of"));
}

#[tokio::test]
async fn cmd_search_bails_when_neither_query_nor_vector() {
    let index = seed_index_for_search("search-none").await;
    let err = cmd_search(SearchOptions {
        index: &index.path,
        query: None,
        vector: None,
        top_k: 1,
        model: "bge-m3",
        provider: None,
        base_url: None,
        filter: None,
    })
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("search requires one of"));
}

#[tokio::test]
async fn cmd_search_bails_on_query_vector_dim_mismatch() {
    let index = seed_index_for_search("search-dim").await;
    let err = cmd_search(SearchOptions {
        index: &index.path,
        query: None,
        vector: Some(vec![0.1; 4]),
        top_k: 1,
        model: "bge-m3",
        provider: None,
        base_url: None,
        filter: None,
    })
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("query vector dimension mismatch"));
    assert!(err.contains("index expects 8"));
}

#[tokio::test]
async fn cmd_search_returns_empty_for_non_matching_filter() {
    // Build an index, then drop a fresh one whose filter matches nothing.
    let index = seed_index_for_search("search-empty-filter").await;
    // filter matches no rows -> early return printing "[]".
    let res = cmd_search(SearchOptions {
        index: &index.path,
        query: None,
        vector: Some(vec![0.1; 8]),
        top_k: 5,
        model: "bge-m3",
        provider: None,
        base_url: None,
        filter: Some("lang = 'fr'"),
    })
    .await;
    assert!(res.is_ok(), "expected early-return Ok, got: {:?}", res);
}

#[tokio::test]
async fn cmd_search_runs_with_precomputed_vector_and_filter() {
    let index = seed_index_for_search("search-ok").await;
    // Successful search restricted to the zh record via allowlist.
    let res = cmd_search(SearchOptions {
        index: &index.path,
        query: None,
        vector: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        top_k: 2,
        model: "bge-m3",
        provider: None,
        base_url: None,
        filter: Some("lang = 'zh'"),
    })
    .await;
    assert!(res.is_ok(), "expected Ok, got: {:?}", res);
}

// ---- cmd_export ----

#[tokio::test]
async fn cmd_export_bails_on_include_vectors() {
    let index = seed_index_for_search("export-inc-vec").await;
    let err = cmd_export(&index.path, None, None, true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("--include-vectors is not supported"));
}

#[tokio::test]
async fn cmd_export_bails_when_db_missing() {
    let index = TempIndex::unique("export-missing");
    let err = cmd_export(&index.path, None, None, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("db not found"));
}

#[tokio::test]
async fn cmd_export_writes_jsonl_file_with_pk_and_fields() {
    let index = seed_index_for_search("export-file").await;
    let out = index.path.with_extension("export.jsonl");

    cmd_export(&index.path, Some(&out), None, false).unwrap();

    let body = fs::read_to_string(&out).unwrap();
    let lines: Vec<&str> = body.trim().lines().collect();
    assert_eq!(lines.len(), 2, "expected one record per imported doc");

    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["pk"], "doc-1");
    // vector_field "content" carries the text; meta carries lang.
    assert_eq!(first["fields"]["content"], "alpha");
    assert_eq!(first["fields"]["lang"], "zh");
    // No raw vector leaked.
    assert!(first.get("vector").is_none() && first["fields"].get("vector").is_none());
    let _ = fs::remove_file(out);
}

#[tokio::test]
async fn cmd_export_respects_filter() {
    let index = seed_index_for_search("export-filter").await;
    let out = index.path.with_extension("export.jsonl");

    cmd_export(&index.path, Some(&out), Some("lang = 'en'"), false).unwrap();

    let body = fs::read_to_string(&out).unwrap();
    let lines: Vec<&str> = body.trim().lines().collect();
    assert_eq!(lines.len(), 1);
    let rec: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(rec["pk"], "doc-2");
    assert_eq!(rec["fields"]["content"], "beta");
    let _ = fs::remove_file(out);
}

#[tokio::test]
async fn cmd_export_to_stdout_succeeds() {
    let index = seed_index_for_search("export-stdout").await;
    // output=None -> stdout writer path; just assert Ok.
    let res = cmd_export(&index.path, None, None, false);
    assert!(res.is_ok(), "expected Ok, got: {:?}", res);
}
