use super::*;
use std::path::PathBuf;

#[test]
fn loads_config_from_json_string_with_aliases() {
    let config = load_config(Some(
        r#"{"storage_path":"/tmp/docs.tvim","model":"ollama/bge-m3","base_url":"http://localhost:11434"}"#,
    ))
    .unwrap();

    assert_eq!(config.data_path, Some(PathBuf::from("/tmp/docs.tvim")));
    assert_eq!(
        config.default_vector_model.as_deref(),
        Some("ollama/bge-m3")
    );
    assert_eq!(config.base_url.as_deref(), Some("http://localhost:11434"));
}

#[test]
fn loads_config_from_file_path() {
    let path = std::env::temp_dir().join(format!(
        "turbovec-rs-config-test-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &path,
        r#"{"data_path":"/tmp/from-file.tvim","default_vector_model":"bge-m3"}"#,
    )
    .unwrap();

    let config = load_config(Some(path.to_str().unwrap())).unwrap();

    assert_eq!(config.data_path, Some(PathBuf::from("/tmp/from-file.tvim")));
    assert_eq!(config.default_vector_model.as_deref(), Some("bge-m3"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn resolves_index_and_model_with_cli_precedence() {
    let config = AppConfig {
        data_path: Some(PathBuf::from("/tmp/config.tvim")),
        default_vector_model: Some("ollama/bge-m3".to_string()),
        provider: Some("ollama".to_string()),
        base_url: Some("http://example.test".to_string()),
        embedding: None,
    };

    assert_eq!(
        resolve_db_path(Some(PathBuf::from("/tmp/cli.tvim")), &config).unwrap(),
        PathBuf::from("/tmp/cli.tvim")
    );
    assert_eq!(
        resolve_db_path(None, &config).unwrap(),
        PathBuf::from("/tmp/config.tvim")
    );
    assert_eq!(
        resolve_model(Some("cli-model".to_string()), &config),
        "cli-model"
    );
    assert_eq!(resolve_model(None, &config), "ollama/bge-m3");
    assert_eq!(
        resolve_provider(Some("yxt".to_string()), &config).as_deref(),
        Some("yxt")
    );
    assert_eq!(
        resolve_base_url(Some("http://cli.test".to_string()), &config).as_deref(),
        Some("http://cli.test")
    );
}

#[test]
fn normalizes_provider_prefixed_model_when_provider_is_explicit() {
    let (model, provider) = normalize_provider_model("ollama/bge-m3", Some("ollama")).unwrap();
    assert_eq!(model, "bge-m3");
    assert_eq!(provider.as_deref(), Some("ollama"));

    let err = normalize_provider_model("ollama/bge-m3", Some("yxt"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("conflicts"));
}

// ---- parse_schema_defaults ----

#[test]
fn schema_none_returns_empty_defaults() {
    let defaults = parse_schema_defaults(None).unwrap();
    assert!(defaults.text_field.is_none());
    assert!(defaults.vector_field.is_none());
    assert!(defaults.dim.is_none());
}

#[test]
fn schema_requires_fields_array() {
    let err = parse_schema_defaults(Some(r#"{"foo":"bar"}"#))
        .unwrap_err()
        .to_string();
    assert!(err.contains("schema JSON must contain fields array"));
}

#[test]
fn schema_picks_default_text_field_when_present() {
    // text field named "text" wins even when other string fields exist.
    let defaults = parse_schema_defaults(Some(
        r#"{"fields":[{"name":"text","type":"string"},{"name":"title","type":"string"}]}"#,
    ))
    .unwrap();
    assert_eq!(defaults.text_field.as_deref(), Some("text"));
}

#[test]
fn schema_picks_lone_text_field() {
    let defaults =
        parse_schema_defaults(Some(r#"{"fields":[{"name":"title","type":"string"}]}"#)).unwrap();
    assert_eq!(defaults.text_field.as_deref(), Some("title"));
}

#[test]
fn schema_returns_none_text_field_when_multiple_ambiguous() {
    let defaults = parse_schema_defaults(Some(
        r#"{"fields":[{"name":"title","type":"string"},{"name":"abstract","type":"string"}]}"#,
    ))
    .unwrap();
    assert!(
        defaults.text_field.is_none(),
        "got {:?}",
        defaults.text_field
    );
}

#[test]
fn schema_picks_default_vector_field_with_dim() {
    let defaults = parse_schema_defaults(Some(
        r#"{"fields":[
            {"name":"embedding","type":"vector_fp32","dimension":1024},
            {"name":"content","type":"string"}
        ]}"#,
    ))
    .unwrap();
    assert_eq!(defaults.vector_field.as_deref(), Some("embedding"));
    assert_eq!(defaults.dim, Some(1024));
    assert_eq!(defaults.text_field.as_deref(), Some("content"));
}

#[test]
fn schema_picks_lone_vector_field_and_carries_dim() {
    let defaults = parse_schema_defaults(Some(
        r#"{"fields":[{"name":"vec","type":"vector_fp32","dimension":768}]}"#,
    ))
    .unwrap();
    assert_eq!(defaults.vector_field.as_deref(), Some("vec"));
    assert_eq!(defaults.dim, Some(768));
}

#[test]
fn schema_returns_none_vector_field_when_multiple_vector_fp32() {
    let defaults = parse_schema_defaults(Some(
        r#"{"fields":[
            {"name":"a","type":"vector_fp32","dimension":8},
            {"name":"b","type":"vector_fp32","dimension":16}
        ]}"#,
    ))
    .unwrap();
    assert!(defaults.vector_field.is_none());
    assert!(defaults.dim.is_none());
}

#[test]
fn schema_skips_fields_missing_name_or_type() {
    // Field missing type and field missing name are silently skipped; the only
    // valid string field becomes the lone text field.
    let defaults = parse_schema_defaults(Some(
        r#"{"fields":[
            {"name":"untyped"},
            {"type":"string"},
            {"name":"content","type":"string"}
        ]}"#,
    ))
    .unwrap();
    assert_eq!(defaults.text_field.as_deref(), Some("content"));
    assert!(defaults.vector_field.is_none());
}

#[test]
fn schema_loads_from_at_file() {
    let path = std::env::temp_dir().join(format!(
        "turbovec-rs-schema-test-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &path,
        r#"{"fields":[{"name":"text","type":"string"},{"name":"embedding","type":"vector_fp32","dimension":512}]}"#,
    )
    .unwrap();
    let defaults = parse_schema_defaults(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(defaults.text_field.as_deref(), Some("text"));
    assert_eq!(defaults.vector_field.as_deref(), Some("embedding"));
    assert_eq!(defaults.dim, Some(512));
    let _ = std::fs::remove_file(path);
}
