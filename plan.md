# turbovec-rs metadata filter plan

## Goal

Add metadata filtering to `turbovec-rs` without putting metadata into the
`.tvim` vector index.

The intended design is:

```text
filterql parses and validates the user filter
SQLite stores text and metadata, then returns matching ids
turbovec searches vectors restricted by those ids
```

This gives the CLI a practical MetaFilter equivalent while keeping turbovec
focused on vector top-k search.

## Current State

- `src/main.rs` is a single binary CLI with `init`, `add`, `search`, and `info`.
- The `.tvim` file stores the vector index.
- `*.tvim.meta.json` stores index-level metadata such as `dim`, `bits`, `model`,
  and `next_id`.
- `*.tvim.texts.jsonl` stores only `{ "id": ..., "text": ... }`.
- `search` currently calls `IdMapIndex::search(&query_vec, top_k)`, so every
  vector is eligible.
- The underlying `turbovec` crate already exposes
  `IdMapIndex::search_with_allowlist`, which is the right hook for filtered
  vector search.

## Proposed Components

### SQLite sidecar

Replace or supersede `*.tvim.texts.jsonl` with a SQLite sidecar:

```text
<index>.tvim
<index>.tvim.meta.json
<index>.tvim.sqlite
```

Initial schema:

```sql
CREATE TABLE docs (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}',

  source TEXT GENERATED ALWAYS AS (json_extract(meta, '$.source')) VIRTUAL,
  lang TEXT GENERATED ALWAYS AS (json_extract(meta, '$.lang')) VIRTUAL,
  kind TEXT GENERATED ALWAYS AS (json_extract(meta, '$.kind')) VIRTUAL,
  created_at INTEGER GENERATED ALWAYS AS (json_extract(meta, '$.created_at')) VIRTUAL
);

CREATE INDEX docs_source_lang ON docs(source, lang);
CREATE INDEX docs_kind ON docs(kind);
CREATE INDEX docs_created_at ON docs(created_at);
```

Optional later:

```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(text, content='docs', content_rowid='id');
```

### filterql

Use the local crate at `../filterql` as the filter language layer:

```toml
filterql = { path = "../filterql", default-features = false, features = ["sql", "json", "validate"] }
rusqlite = { version = "0.32", features = ["bundled"] }
```

`filterql` should be used for:

- parsing SQL-like filters, e.g. `source = 'docs' AND lang = 'zh'`
- optionally parsing JSON filters, e.g. `{ "source": "docs" }`
- validating allowed fields, value types, nesting depth, and list sizes
- compiling the AST into a parameterized SQLite `WHERE` clause

Do not use `filterql::render::to_filter_string` directly for SQLite SQL. It is
not a general SQL-injection-safe emitter. Implement a small SQLite compiler that
produces placeholders plus bound parameters.

### SQLite compiler shape

Implement a backend around `filterql::Compile`:

```rust
struct SqliteWhere {
    clause: String,
    params: Vec<SqliteValue>,
}
```

Rules:

- Field names must come from a whitelist map, never directly from user input.
- Values must be bound as query parameters, not interpolated into SQL.
- `AND`, `OR`, and `NOT` should always parenthesize child clauses.
- `IN` and `NOT IN` should expand to the right number of placeholders.
- Empty filter means no `WHERE` clause.
- Unsupported fields or operators should return a clear CLI error.

Suggested allowed fields:

```text
source: text
lang: text
kind: text
created_at: integer epoch seconds
```

## CLI UX

Extend `add`:

```bash
turbovec-rs add \
  --index /tmp/docs.tvim \
  --file docs.txt \
  --provider ollama \
  --meta '{"source":"docs","lang":"zh","kind":"guide"}'
```

For line-based ingest, the same metadata applies to every line/chunk.

Optional later:

```bash
--jsonl
```

Where each input line can be:

```json
{"text":"...", "meta":{"source":"docs","lang":"zh"}}
```

Extend `search`:

```bash
turbovec-rs search \
  --index /tmp/docs.tvim \
  --query "vector search" \
  --provider ollama \
  --filter "source = 'docs' AND lang = 'zh'"
```

Optional later:

```bash
--filter-json '{"source":"docs","lang":"zh"}'
```

## Query Flow

Filtered search:

1. Parse the filter with `filterql::sql::parse`.
2. Validate the parsed AST against the supported metadata schema.
3. Compile the AST into a parameterized SQLite `WHERE` clause.
4. Run:

   ```sql
   SELECT id FROM docs WHERE <compiled-filter>
   ```

5. If no IDs match, return `[]` without calling turbovec.
6. If IDs match, call:

   ```rust
   idx.search_with_allowlist(&query_vec, top_k, Some(&ids))
   ```

7. Fetch result text and metadata from SQLite by returned IDs.
8. Return JSON results containing `id`, `score`, `text`, and `meta`.

Unfiltered search:

1. Call `idx.search(&query_vec, top_k)`.
2. Fetch text and metadata from SQLite by returned IDs.

Important: never pass an empty allowlist to `search_with_allowlist`; the
underlying API panics on empty allowlists.

## Migration Strategy

Phase 1 can keep backwards compatibility with existing JSONL sidecars:

- On `search`, prefer SQLite sidecar if present.
- If SQLite sidecar is absent but `*.tvim.texts.jsonl` exists, read the legacy
  JSONL file as today.
- On `add`, write to SQLite for new indexes.

Optional migration command later:

```bash
turbovec-rs migrate-sidecar --index /tmp/docs.tvim
```

This imports `*.tvim.texts.jsonl` into SQLite with empty metadata.

## Implementation Phases

### Phase 1: SQLite sidecar foundation

- Add `rusqlite` dependency.
- Add helper functions:
  - `sqlite_path(index: &Path) -> PathBuf`
  - `open_sidecar(index: &Path) -> Result<Connection>`
  - `init_sidecar_schema(conn: &Connection) -> Result<()>`
  - `insert_doc(conn, id, text, meta)`
  - `load_docs_by_ids(conn, ids)`
- Update `cmd_init` to create the SQLite sidecar.
- Update `cmd_add` to write `{id, text, meta}` to SQLite.
- Keep legacy `append_texts` only if needed for compatibility.
- Update `cmd_info` to report SQLite doc count.

Acceptance:

- `cargo test` passes.
- `cargo run -- init ...` creates `.tvim`, `.tvim.meta.json`, and
  `.tvim.sqlite`.
- `cargo run -- add ... --meta '{}'` inserts docs into SQLite.
- `info` reports matching vector/doc counts.

### Phase 2: filterql integration

- Add `filterql` dependency from `../filterql`.
- Add `--filter` to `search`.
- Implement filter parsing and validation.
- Implement parameterized SQLite compiler.
- Run SQLite filter query to produce `Vec<u64>`.
- Call `search_with_allowlist` when the filter result is non-empty.
- Return empty JSON array when the filter matches no docs.

Acceptance:

- `search --filter "source = 'docs'"` only returns docs with that metadata.
- `search --filter "lang = 'zh' AND kind IN ('guide','api')"` works.
- Unknown fields fail with a clear error.
- Empty filter results return `[]`, not a panic.

### Phase 3: metadata input formats

- Add `--meta <JSON>` to `add`.
- Validate that `--meta` is a JSON object.
- Apply the metadata to every chunk generated from the file.
- Add optional `--jsonl` ingest mode if needed.

Acceptance:

- `add --meta '{"source":"docs","lang":"zh"}'` persists metadata.
- Search results include `meta`.
- Invalid metadata JSON fails before embedding starts.

### Phase 4: tests and hardening

- Add unit tests for:
  - SQLite path derivation
  - schema initialization
  - filterql-to-SQLite compilation
  - empty allowlist behavior
  - metadata JSON validation
- Add CLI integration tests if the repo adopts test fixtures.
- Cap risky filter shapes with `filterql::validate::Policy`:
  - max depth
  - max comparisons
  - max `IN` list length
  - allowed fields
  - allowed operators per field

Acceptance:

- `cargo test` passes.
- `cargo clippy --all-targets --all-features` passes if clippy is in scope.
- Bad filters produce diagnostics that identify the offending field/operator.

## Later Options

- Add FTS5 keyword filtering over `text`.
- Add `--filter-json`.
- Add `migrate-sidecar`.
- Add configurable metadata indexed fields.
- Add a Tantivy backend if full-text query language becomes more important
  than SQL-style metadata filtering.

## Non-goals

- Do not store metadata inside `.tvim`.
- Do not implement a new filter language in `turbovec-rs`.
- Do not make SQLite perform vector search.
- Do not add a service dependency such as Elasticsearch or Quickwit for the
  local CLI path.
