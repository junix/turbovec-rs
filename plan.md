# turbovec-rs JSONL import and MetaFilter plan

## Goal

Replace file chunking with JSONL record import.

Each input line is one record. The record declares exactly one field to embed as
the vector text. All remaining fields become metadata fields queryable through
`--filter`.

This keeps `turbovec` focused on vector top-k search while using SQLite as the
sidecar store for text, external ids, vector field names, and metadata.

## Import Shape

Supported record shapes:

```json
{"id":"doc-1","vector_field":"content","fields":{"content":"text to embed","doc":"guide","lang":"zh"}}
```

```json
{"id":"doc-2","fields":{"content":{"value":"text to embed","index":["vector"]},"doc":{"value":"guide","index":["filter"]}}}
```

```json
{"id":"doc-3","vector_fields":["content"],"content":"text to embed","doc":"guide"}
```

```json
{"id":"doc-4","vector_field":"content","fields":{"content":"text kept for display","doc":"guide"},"vector":[0.1,0.2,0.3]}
```

```json
{"id":"doc-5","fields":{"content":"text kept for display","doc":"guide"},"vectors":{"content":[0.1,0.2,0.3]}}
```

Rules:

- Exactly one vector field is allowed.
- The vector field must be present and must be a non-empty string.
- `vector` or `vectors.<field>` can provide a precomputed vector and skip
  embedding for that record.
- Precomputed vector dimensions must match the initialized index dimension.
- If the record does not declare the vector field, `import --vector-field <field>`
  can supply a fallback.
- Fields other than the vector field are stored as metadata.
- Top-level `id` is stored as `external_id` and also copied into metadata as
  `external_id`.

## Storage

Files:

```text
<index>.tvim
<index>.tvim.meta.json
<index>.tvim.sqlite
```

SQLite schema:

```sql
CREATE TABLE docs (
  id INTEGER PRIMARY KEY,
  external_id TEXT,
  vector_field TEXT NOT NULL DEFAULT 'content',
  text TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX docs_external_id ON docs(external_id);
```

The metadata JSON is intentionally schemaless. Filter fields are compiled to
`json_extract(meta, ?)` with the JSON path bound as a SQL parameter.

## CLI UX

Config:

```json
{
  "data_path": "/tmp/docs.tvim",
  "default_vector_model": "ollama/bge-m3"
}
```

The config can be passed as a JSON string or as a path to a JSON file:

```bash
turbovec-rs -c '{"data_path":"/tmp/docs.tvim","default_vector_model":"ollama/bge-m3"}' stats
```

```bash
turbovec-rs --config turbovec.json filter-ids --filter "lang = 'zh'"
```

Supported aliases:

- `data_path`, `storage_path`, `index`, or `index_path`
- `default_vector_model`, `model`, `default_model`, or `embedding_model`

Optional config fields:

- `provider`
- `base_url`

Precedence:

```text
explicit CLI flag > config value > built-in default
```

Initialize:

```bash
turbovec-rs init --db /tmp/docs.tvim --dim 1024
```

Import:

```bash
turbovec-rs import \
  --db /tmp/docs.tvim \
  --input docs.jsonl \
  --provider ollama \
  --model bge-m3
```

Import with fallback vector field:

```bash
turbovec-rs import \
  --db /tmp/docs.tvim \
  --input docs.jsonl \
  --provider ollama \
  --vector-field content
```

Search:

```bash
turbovec-rs search \
  --db /tmp/docs.tvim \
  --query "vector search" \
  --provider ollama \
  --filter "lang = 'zh' AND doc = 'guide'"
```

Search with a caller-provided query vector:

```bash
turbovec-rs search \
  --db /tmp/docs.tvim \
  --vector '[0.1,0.2,0.3]' \
  --filter "lang = 'zh'"
```

```bash
turbovec-rs search \
  --db /tmp/docs.tvim \
  --vector-file query-vector.json
```

Query metadata ids without vector search:

```bash
turbovec-rs filter-ids \
  --db /tmp/docs.tvim \
  --filter "lang = 'zh' AND doc = 'guide'"
```

Result rows include:

```json
{
  "id": 1,
  "external_id": "doc-1",
  "vector_field": "content",
  "score": 0.91,
  "text": "text to embed",
  "meta": {"external_id":"doc-1","doc":"guide","lang":"zh"}
}
```

## Query Flow

Filtered search:

1. Parse the filter with `filterql::sql::parse`.
2. Validate expression depth, comparison count, and `IN` list size.
3. Validate metadata field path segments before building JSON paths.
4. Compile to a parameterized SQLite `WHERE` clause.
5. Query SQLite for matching internal ids.
6. If no ids match, return `[]` without calling the vector index.
7. Search vectors with `IdMapIndex::search_with_allowlist`.
8. Fetch text and metadata from SQLite for returned ids.

Standalone filter query:

1. Parse, validate, and compile the filter with the same filter path.
2. Query SQLite for matching internal ids.
3. Return the ids as JSON without embedding the query or touching the vector
   index search path.

Unfiltered search:

1. Embed `--query`, or use `--vector` / `--vector-file` directly.
2. Call `IdMapIndex::search`.
3. Fetch text and metadata from SQLite for returned ids.

## Non-goals

- No chunking in `turbovec-rs import`.
- No BM25 or keyword search in this CLI.
- No multiple vector fields per record.
- No metadata storage inside `.tvim`.
