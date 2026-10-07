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

// ---- ImportRecordReader (bounded streaming) ----

fn write_temp_jsonl(tag: &str, body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "turbovec-rs-import-reader-{}-{}-{}.jsonl",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn reader_streams_records_in_bounded_batches() {
    let path = write_temp_jsonl(
        "stream",
        "{\"id\":\"a\",\"fields\":{\"content\":\"one\"}}\n\n{\"id\":\"b\",\"fields\":{\"content\":\"two\"}}\n",
    );
    let mut reader = ImportRecordReader::open(Some(&path), None, Some("content")).unwrap();
    assert_eq!(reader.source(), path.display().to_string());

    let batch = reader.next_batch(1).unwrap().unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].external_id.as_deref(), Some("a"));
    let batch = reader.next_batch(1).unwrap().unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].external_id.as_deref(), Some("b"));
    assert!(reader.next_batch(1).unwrap().is_none());
    let _ = std::fs::remove_file(path);
}

#[test]
fn reader_groups_records_into_batches_and_skips_blank_lines() {
    let path = write_temp_jsonl(
        "group",
        concat!(
            "{\"id\":\"a\",\"fields\":{\"content\":\"one\"}}\n",
            "\n",
            "{\"id\":\"b\",\"fields\":{\"content\":\"two\"}}\n",
            "{\"id\":\"c\",\"fields\":{\"content\":\"three\"}}\n",
        ),
    );
    let mut reader = ImportRecordReader::open(Some(&path), None, Some("content")).unwrap();

    let batch = reader.next_batch(2).unwrap().unwrap();
    assert_eq!(
        batch
            .iter()
            .map(|r| r.external_id.as_deref().unwrap())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    let batch = reader.next_batch(2).unwrap().unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].external_id.as_deref(), Some("c"));
    assert!(reader.next_batch(2).unwrap().is_none());
    let _ = std::fs::remove_file(path);
}

#[test]
fn reader_reports_parse_errors_with_line_numbers() {
    // Line 1 parses, line 2 is blank, line 3 is broken JSON.
    let path = write_temp_jsonl(
        "line-err",
        "{\"id\":\"a\",\"fields\":{\"content\":\"x\"}}\n\n{not json}\n",
    );
    let mut reader = ImportRecordReader::open(Some(&path), None, Some("content")).unwrap();
    assert!(reader.next_record().unwrap().is_some());
    let err = reader.next_record().unwrap_err().to_string();
    assert!(err.contains("line 3"), "expected line number in: {err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn reader_fails_clearly_on_oversized_line() {
    let mut huge = String::from("{\"id\":\"big\",\"fields\":{\"content\":\"");
    huge.push_str(&"x".repeat(MAX_LINE_BYTES + 1));
    huge.push_str("\"}}\n");
    let path = write_temp_jsonl("oversize", &huge);
    let mut reader = ImportRecordReader::open(Some(&path), None, Some("content")).unwrap();
    let err = reader.next_record().unwrap_err().to_string();
    assert!(err.contains("line 1"), "expected line number in: {err}");
    assert!(err.contains("JSONL line limit"), "expected cap in: {err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn reader_handles_input_larger_than_the_line_cap_in_total() {
    // Total payload (3 x 6 MiB) exceeds MAX_LINE_BYTES; per-line budget holds
    // and the reader never needs the whole file in memory.
    let mut body = String::new();
    for id in ["a", "b", "c"] {
        body.push_str(&format!(
            "{{\"id\":\"{id}\",\"fields\":{{\"content\":\"{}\"}}}}\n",
            "x".repeat(6 * 1024 * 1024)
        ));
    }
    let path = write_temp_jsonl("big-total", &body);
    let mut reader = ImportRecordReader::open(Some(&path), None, Some("content")).unwrap();
    let batch = reader.next_batch(2).unwrap().unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].vector_text.len(), 6 * 1024 * 1024);
    let batch = reader.next_batch(2).unwrap().unwrap();
    assert_eq!(batch.len(), 1);
    assert!(reader.next_batch(2).unwrap().is_none());
    let _ = std::fs::remove_file(path);
}

#[test]
fn reader_returns_none_for_blank_and_missing_input() {
    let path = write_temp_jsonl("blank", "\n  \n");
    let mut reader = ImportRecordReader::open(Some(&path), None, Some("content")).unwrap();
    assert!(reader.next_batch(8).unwrap().is_none());
    let _ = std::fs::remove_file(path);

    let missing = std::env::temp_dir().join("turbovec-rs-import-reader-missing.jsonl");
    let err = ImportRecordReader::open(Some(&missing), None, None)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("opening"), "expected open error in: {err}");
}
