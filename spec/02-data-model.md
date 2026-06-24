# 02 — Data Model

## On-disk artifact family

All persistent state for one index is rooted at a single path `<index>` and fans out into three sibling files that **MUST** share the path stem and differ only by extension:

```
<index>                 quantized vector index          (opaque; owned by the vector index library)
<index>.meta.json       index meta file                  (JSON)
<index>.sqlite          SQLite sidecar                   (document text + metadata)
```

Extension derivation (where `<index>` = `<stem>.tvim`):

| Artifact | Path | Provenance |
|----------|------|------------|
| index | `<stem>.tvim` | the `--db` / `--db-path` / config `data_path` value |
| meta file | `<stem>.tvim.meta.json` | replace `.tvim` extension with `.tvim.meta.json` |
| sidecar | `<stem>.tvim.sqlite` | replace `.tvim` extension with `.tvim.sqlite` |

The sidecar path **MUST** be derivable from the index path by taking the stem and appending `.sqlite`; equivalently the meta file path is the index path with extension replaced by `.meta.json`. An implementation **MUST NOT** place these artifacts in different directories. (observed: the `main_init_creates_index_and_emits_json` smoke test asserts all three files appear alongside the `.tvim` path.)

## Meta file shape

The meta file is a JSON object:

```
record IndexMeta =
  next_id : non-negative integer   ▷ next internal id to assign on import
  dim     : positive integer       ▷ vector dimension, a positive multiple of 8
  bits    : 2 | 3 | 4              ▷ quantization bit width
  model   : string                 ▷ embedding model recorded at import time ("" if never set)
```

`next_id` is monotonically increasing and never reset; it is the source of internal ids assigned during import.

## Sidecar `docs` table

```
record DocRow =
  external_id : optional string    ▷ the record's external id; absent when none was supplied
  vector_field : string            ▷ name of the field whose text was indexed
  text         : string            ▷ the vector text (the indexed string)
  meta         : JSON value        ▷ schemaless metadata object for the document
```

The `docs` table schema **MUST** expose at minimum:

| Column | Type | Constraint |
|--------|------|-----------|
| id | INTEGER | PRIMARY KEY (the internal id) |
| external_id | TEXT | nullable |
| vector_field | TEXT | NOT NULL, default `content` |
| text | TEXT | NOT NULL |
| meta | TEXT | NOT NULL, default `{}` (JSON text) |

An index on `external_id` **MUST** exist to support duplicate-external-id checks. The `meta` column holds a JSON object; metadata filters compile to JSON-path extraction against this column (see `03`). The metadata object is otherwise schemaless — additional keys are stored verbatim and are never promoted to columns.

## Internal id vs external id

Two distinct id namespaces **MUST** be maintained:

- **internal id**: a non-negative integer assigned by the tool (sourced from `next_id`, then incremented per imported record). It is the primary key of `docs` and the id used by the vector index. Internal ids are stable: once assigned they are not reused, even after a failed import.
- **external id**: the caller-supplied identity, taken from the record's `id` or `pk` JSON key. It is stored in `external_id` and is also copied into the document's metadata object under the key `external_id`. It is unique-checked at import time (see `03`). When no external id was supplied, `external_id` is absent and downstream surfaces (export `pk`, query `pk`) fall back to the internal id rendered as a decimal string.

The round-trip invariant:

```
exportRecord(importRecord(R)) . pk  =  (external id of R)  when R has an external id
exportRecord(importRecord(R)) . pk  =  stringify(internal id of R)  otherwise
```

## Vector field

Every imported record declares exactly one **vector field** — the field whose string value is the text to embed (or whose associated value supplies a precomputed vector). The vector field name **MUST** match the metadata field-name grammar (see `04`); the other fields of the record become metadata and **MUST NOT** include the vector field's value in metadata (it is stored in `text` instead).

## Import record (in-memory shape)

```
record ImportRecord =
  external_id  : optional string
  vector_field : string
  vector_text  : string              ▷ non-empty
  vector       : optional list<float32>   ▷ present iff a precomputed vector was supplied
  meta         : JSON object
```

`meta` always contains `external_id` when an external id exists, plus every non-vector-field field of the record, with descriptor `{value, index}` wrappers unwrapped to their `value`.

## Result shapes

### search result list

A search returns a JSON array (pretty-printed, one element per returned id, ordered by descending similarity). Each element:

```
record SearchResult =
  id           : integer            ▷ internal id
  external_id  : string | null
  vector_field : string             ▷ "content" when the document row is missing
  score        : number             ▷ similarity score from the vector index
  text         : string             ▷ "<id <n> text missing>" when the document row is missing
  meta         : JSON object        ▷ {} when the document row is missing
```

### filter-ids result

A JSON array of internal ids (pretty-printed), in the order returned by the sidecar.

### query result

```
record QueryResult = { records : list<QueryRecord> }
```

Each `QueryRecord` is a projection (see `04`); a full (unprojected) record always includes `id`, `pk`, `doc_id` (external id or null), `score` (always null under the `query` path), and, when the document row exists, `external_id`, `vector_field`, `text`, `meta`.

### export record

One JSON object per line, JSONL:

```
record ExportRecord =
  pk     : string                   ▷ external id, or stringified internal id
  fields : JSON object              ▷ metadata fields with the vector field added under its name → text
```

The raw vector **MUST NOT** appear anywhere in an export record.

### stats result

```
record StatsResult =
  db                : string         ▷ index path
  dimension         : integer        ▷ index dimension
  vectors           : integer        ▷ number of vectors in the index
  texts             : integer        ▷ number of rows in the sidecar docs table (0 if no sidecar)
  index_size_bytes  : integer        ▷ byte size of the .tvim file
  meta              : { bits, model, next_id } | {}
```

## Lifecycle

```
state IndexLifecycle =
  Absent        ▷ no .tvim at the path
  Initialized   ▷ .tvim + sidecar + meta exist, zero vectors
  Populated     ▷ .tvim + sidecar + meta exist, ≥ 1 vector
```

| (state, event) | (state', action, output) |
|----------------|--------------------------|
| (Absent, init) | (Initialized, create all three artifacts with next_id=1, model=""; emit `{created:true}` JSON) |
| (Absent, import) | (Populated, bootstrap an index first using inferred/flag dimension and bits, then import) |
| (Initialized, import) | (Populated, assign internal ids from next_id, add vectors, persist) |
| (Populated, import) | (Populated, append; reject any record whose external id already exists) |
| (Initialized|Populated, search) | unchanged, emit result list |
| (Populated, search with empty filter result) | unchanged, emit `[]` without searching vectors |
| (Initialized, search) | unchanged, fail "index is empty" |
| (*, stats) | unchanged, emit StatsResult |
