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

// ---- resolve_import_overrides ----

fn empty_config() -> AppConfig {
    AppConfig::default()
}

#[test]
fn import_overrides_defaults_to_empty_when_nothing_provided() {
    let overrides = resolve_import_overrides(
        None, None, None, None, None, None, None, None, &empty_config(),
    )
    .unwrap();
    assert!(overrides.model.is_none());
    assert!(overrides.provider.is_none());
    assert!(overrides.base_url.is_none());
    assert!(overrides.vector_field.is_none());
    assert!(overrides.text_field.is_none());
    assert!(overrides.dim.is_none());
}

#[test]
fn import_overrides_cli_flag_beats_embedding_schema_and_config() {
    let config = AppConfig {
        default_vector_model: Some("cfg-model".to_string()),
        provider: Some("cfg-provider".to_string()),
        base_url: Some("http://cfg.test".to_string()),
        ..AppConfig::default()
    };
    let overrides = resolve_import_overrides(
        // schema supplies vector_field/text_field/dim
        Some(r#"{"fields":[{"name":"content","type":"string"},{"name":"emb","type":"vector_fp32","dimension":768}]}"#.to_string()),
        // embedding JSON supplies lower-precedence values
        Some(r#"{"model":"emb-model","provider":"emb-provider","base_url":"http://emb.test","vector_field":"emb-vf","text_field":"emb-tf","dimensions":512}"#.to_string()),
        Some("cli-model".to_string()),
        Some("cli-provider".to_string()),
        Some("http://cli.test".to_string()),
        Some("cli-vf".to_string()),
        Some("cli-tf".to_string()),
        Some(1024),
        &config,
    )
    .unwrap();
    assert_eq!(overrides.model.as_deref(), Some("cli-model"));
    assert_eq!(overrides.provider.as_deref(), Some("cli-provider"));
    assert_eq!(overrides.base_url.as_deref(), Some("http://cli.test"));
    assert_eq!(overrides.vector_field.as_deref(), Some("cli-vf"));
    assert_eq!(overrides.text_field.as_deref(), Some("cli-tf"));
    assert_eq!(overrides.dim, Some(1024));
}

#[test]
fn import_overrides_embedding_json_beats_schema_and_config() {
    let config = AppConfig {
        default_vector_model: Some("cfg-model".to_string()),
        provider: Some("cfg-provider".to_string()),
        base_url: Some("http://cfg.test".to_string()),
        ..AppConfig::default()
    };
    // No CLI flags; schema would also supply a dim, embedding JSON must win.
    let overrides = resolve_import_overrides(
        Some(r#"{"fields":[{"name":"content","type":"string"},{"name":"emb","type":"vector_fp32","dimension":768}]}"#.to_string()),
        Some(r#"{"model":"emb-model","provider":"emb-provider","base_url":"http://emb.test","vector_field":"emb-vf","text_field":"emb-tf","dimensions":512}"#.to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        &config,
    )
    .unwrap();
    assert_eq!(overrides.model.as_deref(), Some("emb-model"));
    assert_eq!(overrides.provider.as_deref(), Some("emb-provider"));
    assert_eq!(overrides.base_url.as_deref(), Some("http://emb.test"));
    assert_eq!(overrides.vector_field.as_deref(), Some("emb-vf"));
    assert_eq!(overrides.text_field.as_deref(), Some("emb-tf"));
    assert_eq!(overrides.dim, Some(512));
}

#[test]
fn import_overrides_schema_supplies_vector_field_text_field_and_dim() {
    // model/provider/base_url come from config; vector_field/text_field/dim
    // come from schema (no embedding JSON). This is the one precedence tier
    // where schema contributes (embedding JSON does not for these when absent).
    let config = AppConfig {
        default_vector_model: Some("cfg-model".to_string()),
        provider: Some("cfg-provider".to_string()),
        base_url: Some("http://cfg.test".to_string()),
        ..AppConfig::default()
    };
    let overrides = resolve_import_overrides(
        Some(r#"{"fields":[{"name":"title","type":"string"},{"name":"vec","type":"vector_fp32","dimension":256}]}"#.to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &config,
    )
    .unwrap();
    assert_eq!(overrides.model.as_deref(), Some("cfg-model"));
    assert_eq!(overrides.provider.as_deref(), Some("cfg-provider"));
    assert_eq!(overrides.base_url.as_deref(), Some("http://cfg.test"));
    assert_eq!(overrides.vector_field.as_deref(), Some("vec"));
    assert_eq!(overrides.text_field.as_deref(), Some("title"));
    assert_eq!(overrides.dim, Some(256));
}

#[test]
fn import_overrides_config_embedding_string_is_merged_via_merge_embedding_arg() {
    // The `embedding` arg is optional; when only AppConfig.embedding is set,
    // merge_embedding_arg picks it up and parse_embedding_config parses it.
    let config = AppConfig {
        embedding: Some(r#"{"model":"from-config-emb"}"#.to_string()),
        ..AppConfig::default()
    };
    let overrides = resolve_import_overrides(
        None, None, None, None, None, None, None, None, &config,
    )
    .unwrap();
    assert_eq!(overrides.model.as_deref(), Some("from-config-emb"));
}

#[test]
fn import_overrides_propagates_invalid_embedding_json_error() {
    let err = resolve_import_overrides(
        None,
        Some("{not valid json".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        &empty_config(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("parsing embedding JSON"));
}

#[test]
fn import_overrides_propagates_invalid_schema_json_error() {
    let err = resolve_import_overrides(
        Some(r#"{"no-fields":true}"#.to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &empty_config(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("schema JSON must contain fields array"));
}
