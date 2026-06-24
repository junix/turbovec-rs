# 04 — Interfaces

The sole contract surface is the command-line interface (subcommands, flags, stdin/stdout/stderr, exit codes) and the on-disk artifact family already defined in `02`. There is no network protocol, no wire format, and no public library API in scope.

## Architecture / layering

Three responsibility layers, named by capability:

1. **CLI dispatch layer** — parses global config + subcommand, resolves db path / model / provider / base URL by precedence (`03`), and routes to exactly one operation.
2. **Import / query pipeline layer** — JSONL parsing, vector-field resolution, precomputed-vector vs embed decision, schema-default extraction, SQL-metadata-query parsing, metadata-filter compilation.
3. **Persistence layer** — the index library (quantized vectors), the SQLite sidecar (docs + meta JSON), and the meta file (IndexMeta).

The dispatch layer **MUST NOT** perform persistence; the persistence layer **MUST NOT** parse CLI input.

## CLI grammar

Global option, before the subcommand:

```abnf
config-opt = "-c" SP config-value / "--config" SP config-value
config-value = json-object-literal / config-file-path
```

`config-value` is interpreted as a JSON object literal when it begins with `{` after trimming; otherwise it is treated as a path to a JSON config file. An empty/whitespace-only value **MUST** fail ("config cannot be empty"). The `--config` option is global and **MAY** appear before any subcommand.

### Subcommand summary

| Subcommand | Hidden? | Implemented? | Stdout | Exit 0 on success |
|------------|---------|--------------|--------|-------------------|
| `init` | no | yes | compact JSON | yes |
| `import` | no | yes | compact JSON | yes |
| `search` | no | yes | pretty JSON array | yes |
| `query` | no | yes | pretty JSON `{records}` | yes |
| `export` | no | yes | JSONL (stdout or file) | yes |
| `filter-ids` | yes | yes | pretty JSON array of ids | yes |
| `stats` | no | yes | pretty JSON | yes |
| `serve` | yes | **no** (fails "not implemented") | — | non-zero |
| `mcp` | yes | **no** (fails "not implemented") | — | non-zero |

### `init`

```abnf
init-cmd = "init" *(SP init-flag)
init-flag = "--db" SP path / "--dim" SP uint / "--bits" SP bits-val
bits-val = "2" / "3" / "4"
```

Defaults: `--dim` = 1024, `--bits` = 4. `--db` is optional here only if supplied via config (resolved by precedence); otherwise **MUST** be supplied.

### `import`

```abnf
import-cmd = "import" *(SP import-flag)
import-flag = "--db" SP path
             / "--schema" SP json-arg
             / "--embedding" SP json-arg
             / "--input" SP input-arg
             / "--model" SP token
             / "--provider" SP token
             / "--base-url" SP url
             / "--batch-size" SP uint
             / "--vector-field" SP field-name
             / "--text-field" SP field-name
             / "--dim" SP uint
             / "--bits" SP bits-val
             / "--upsert"
input-arg = path / "-"         ; "-" or omitted ⇒ read JSONL from stdin
json-arg  = json-object-literal / "@" path / path-to-json-file
```

`--schema` and `--embedding` accept a JSON object literal, an `@`-prefixed file path, or a bare file path to a JSON file. `--schema` is used only to derive defaults (text field, vector field, dimension) via `parse_schema_defaults`; it does not create a schema. `--upsert` is accepted but, per Non-Goals, does not enable overwriting an existing external id.

### `search`

```abnf
search-cmd = "search" *(SP search-flag)
search-flag = "--db" SP path
             / "--query" SP text
             / "--vector" SP json-array-literal
             / "--vector-file" SP path
             / "--top-k" / "-k" SP uint
             / "--model" SP token
             / "--provider" SP token
             / "--base-url" SP url
             / "--filter" SP metadata-filter
```

Exactly one of `--query`, `--vector`, `--vector-file` **MUST** be supplied (`--vector` and `--vector-file` together is also rejected). `--top-k` default = 10.

### `query`

```abnf
query-cmd = "query" *(SP query-flag)
query-flag = "--db-path" SP path / "--sql" SP sql-text
sql-text   = "SELECT" projection "FROM" collection ["WHERE" metadata-filter] ["LIMIT" uint]
projection = "*" / field-name *(*("," SP) field-name)
```

Note the `query` subcommand uses `--db-path` (not `--db`). The `collection` token is parsed but not used to select a store — there is exactly one `docs` collection; it only **MUST** be non-empty. `--sql` is required.

### `export`

```abnf
export-cmd = "export" *(SP export-flag)
export-flag = "--db" SP path
             / "--schema" SP json-arg      ; accepted for parity; defaults unused by export
             / "--output" SP output-arg
             / "--filter" SP metadata-filter
             / "--include-vectors"
output-arg  = path / "-"                   ; "-" or omitted ⇒ write JSONL to stdout
```

`--include-vectors` **MUST** fail (vectors are not recoverable).

### `filter-ids` (hidden)

```abnf
filter-ids-cmd = "filter-ids" *(SP fi-flag)
fi-flag = "--db" SP path / "--filter" SP metadata-filter
```

`--filter` is required.

### `stats`

```abnf
stats-cmd = "stats" *(SP "--db" SP path)
```

### `serve` / `mcp` (hidden, not implemented)

Both accept their flags for parsing parity but **MUST** fail at dispatch with a "not implemented" diagnostic.

## JSONL record shapes (import input)

One JSON object per non-blank line. The grammar is the union of the shapes below; a record **MUST** resolve to exactly one vector field (Algorithm 9).

```abnf
jsonl-line = bare-record / fields-record / descriptor-record / direct-vector-record / keyed-vector-record

bare-record      = "{" id-key "," vector-field-decl "," *bare-field "}"
fields-record    = "{" id-key "," "\"fields\"" ":" fields-object ["," vector-field-decl] "}"
descriptor-record= "{" id-key "," "\"fields\"" ":" descriptor-object "}"
direct-vector-record = "{" id-key "," "\"fields\"" ":" fields-object "," "\"vector\"" ":" json-array "}"
keyed-vector-record  = "{" id-key "," "\"fields\"" ":" fields-object "," "\"vectors\"" ":" keyed-vectors "}"

id-key              = "\"id\"" ":" scalar / "\"pk\"" ":" scalar
vector-field-decl   = "\"vector_field\"" ":" string / "\"vector_fields\"" ":" string-array
fields-object       = "{" *(field-name ":" json-value) "}"
descriptor-object   = "{" *(field-name ":" descriptor) "}"
descriptor          = "{" "\"value\"" ":" json-value ["," "\"index\"" ":" string-array] "}"
keyed-vectors       = "{" *(field-name ":" json-array) "}"
```

Reserved top-level keys (excluded from metadata): `id`, `pk`, `fields`, `vector_field`, `vector_fields`, `vector`, `vectors`.

### Schema-default extraction (`--schema`)

```abnf
schema-json = "{" "\"fields\"" ":" "[" *schema-field] "}"
schema-field = "{" "\"name\"" ":" string "," "\"type\"" ":" schema-type ["," "\"dimension\"" ":" uint] "}"
schema-type = "\"string\"" / "\"vector_fp32\"" / other-type   ; other types skipped
```

Resolution: among `string`-typed fields, a field literally named `text` wins; otherwise a lone string field wins; otherwise the text-field default is None. Among `vector_fp32` fields, a field literally named `embedding` wins; otherwise a lone `vector_fp32` field wins (carrying its `dimension`); otherwise None. Fields missing `name` or `type` are silently skipped.

## Metadata filter DSL (`--filter`)

The filter is a boolean expression over JSON-typed document fields, parsed by the filterql SQL parser. Field references become `json_extract(meta, "$.<field>")` lookups; the supported comparison operators are `=`, `!=`/`<>`, `<`, `<=`, `>`, `>=`, `LIKE`, `NOT LIKE`, `IN (...)`, `NOT IN (...)`, and an existence test. Literals are SQL-quoted by the parser, not interpolated. A filter is validated against a policy before compilation:

| Policy bound | Value | Effect on exceeding |
|--------------|-------|---------------------|
| max depth | 8 | fail "invalid metadata filter" |
| max comparisons | 32 | fail "invalid metadata filter" |
| max IN-list length | 256 | fail "invalid metadata filter" |

These numeric bounds are implementation-defined limits of the companion filterql policy; they are not promoted to observable contracts (a caller cannot distinguish depth=8 from depth=9 through any documented threshold), and an implementation **MAY** choose different limits provided the failure shape remains "invalid metadata filter: \<report\>".

## Config file schema

```abnf
config-json = "{" *(config-key) "}"
config-key  = data-path-key / model-key / "\"provider\"" ":" string / "\"base_url\"" ":" url / "\"embedding\"" ":" json-arg
data-path-key = ("\"data_path\"" / "\"storage_path\"" / "\"db\"" / "\"db_path\"" / "\"index\"" / "\"index_path\"") ":" path
model-key    = ("\"default_vector_model\"" / "\"model\"" / "\"default_model\"" / "\"embedding_model\"") ":" string
```

Unknown keys are ignored (lenient deserialization).

## Extension points

There are no plugin, hook, or user-extensible extension points. The configurable extension surface is limited to:

- the embedding model / provider / base URL (resolved via precedence),
- the import record shapes (the union grammar above),
- the metadata filter DSL (delegated to filterql).

A re-implementation **MUST** preserve the artifact extension mapping (`02`), the result JSON shapes (`02`), and the diagnostic substrings (`03`); it is otherwise free in its choice of quantization library, SQL engine, and embedding client, provided their capabilities match `01` §Dependencies.
