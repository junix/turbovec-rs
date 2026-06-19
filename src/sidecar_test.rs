use super::*;
use crate::commands::filter_ids;
use rusqlite::Connection;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn sqlite_path_preserves_tvim_stem() {
    assert_eq!(
        sqlite_path(Path::new("/tmp/docs.tvim")),
        PathBuf::from("/tmp/docs.tvim.sqlite")
    );
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
