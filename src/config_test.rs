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
