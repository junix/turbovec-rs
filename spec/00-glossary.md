# 00 — Glossary

| Term | Canonical surface form | Citation | Used in |
|------|------------------------|----------|---------|
| index | index | test:main_init_creates_index_and_emits_json, plan.md §Storage, README §Usage | 01, 02, 03 |
| sidecar | sidecar | code:sidecar.rs (crate doc), plan.md §Storage, README §Dependencies | 02, 03 |
| meta file | meta file | test:main_init_creates_index_and_emits_json (.tvim.meta.json assertion), plan.md §Storage | 02, 03 |
| external id | external id | test:parses_rag_style_jsonl_record (meta.external_id), test:cmd_export_helper_projects_meta_into_fields (pk), plan.md §Import Shape | 02, 03, 04 |
| primary key | external id | code:ensure_no_external_id_duplicates ("primary key `<id>` already exists"), test:cmd_add_bails_on_duplicate_primary_key | 02, 03 |
| pk | external id | test:parses_zvec_style_jsonl_record (`pk` JSON key → external_id), plan.md §Import Shape | 02, 04 |
| internal id | internal id | test:filter_ids_queries_sqlite_sidecar (returned u64 ids), plan.md §Query Flow | 02, 03 |
| vector field | vector field | test:parses_rag_style_jsonl_record, plan.md §Import Shape, code:resolve_vector_field | 02, 03, 04 |
| vector text | vector text | code:parse_import_record (vector_text), test:parses_descriptor_jsonl_record_with_vector_marker | 02, 03 |
| embedding model | embedding model | code:DEFAULT_MODEL, test:cmd_add_imports_precomputed_vectors_and_creates_index (meta.model="bge-m3"), plan.md §CLI UX | 03, 04 |
| provider | provider | config_test:resolves_index_and_model_with_cli_precedence, plan.md §CLI UX | 03, 04 |
| query vector | query vector | test:cmd_search_bails_on_query_vector_dim_mismatch, test:parses_query_vector_from_arg_and_file | 03, 04 |
| metadata filter | metadata filter | code:compile_filter, test:filter_compiler_uses_placeholders_and_params, plan.md §Query Flow | 03, 04 |
| config | config | config_test:loads_config_from_json_string_with_aliases, plan.md §CLI UX | 04 |

Notes on synonym drift (resolved by this table):

- `external id` is the canonical domain noun. The JSON record key `id` and the zvec-style key `pk` are both surface forms of **external id**; the SQLite column `external_id` and the `pk` column in export/query output are also **external id**. The string `primary key` appears only in a diagnostic message and denotes the same concept.
- `index` is the canonical name for the quantized vector store artifact. The CLI flag `--db`, `--db-path` (for the `query` subcommand), and the config keys `data_path`/`storage_path`/`db`/`db_path`/`index`/`index_path` are all surface forms of **index**.
