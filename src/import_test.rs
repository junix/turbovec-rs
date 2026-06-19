use super::*;

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
    std::fs::write(&path, "[0.3,0.4]").unwrap();
    let vector = load_vector_arg(None, Some(&path)).unwrap();
    assert_eq!(vector.as_deref(), Some(&[0.3, 0.4][..]));
    let _ = std::fs::remove_file(path);
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
