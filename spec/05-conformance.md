# 05 — Conformance

## Examples

### E.1 init creates the artifact family (happy path)

```
$ turbovec-rs init --db /tmp/docs.tvim --dim 8 --bits 4
{"db":"/tmp/docs.tvim","dimension":8,"bits":4,"created":true}
```

Post-condition: `/tmp/docs.tvim`, `/tmp/docs.tvim.meta.json`, `/tmp/docs.tvim.sqlite` all exist.

### E.2 import with precomputed vectors bootstraps an absent index

Input JSONL (two records, 8-dim vectors):

```json
{"id":"doc-1","vector_field":"content","fields":{"content":"hello world","lang":"zh"},"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}
{"id":"doc-2","vector_field":"content","fields":{"content":"second doc","lang":"en"},"vector":[0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]}
```

After import: IndexMeta.dim=8, bits=4, next_id=3, model=`bge-m3` (from the explicit `--model`); index length=2.

### E.3 dimension mismatch is rejected after the first batch persists

With `--batch-size 1`, the first 8-dim record establishes dim=8; a second record with a 3-dim vector fails: diagnostic contains `vector dimension mismatch`. (The already-imported first record remains persisted.)

### E.4 duplicate external id is rejected

Importing a second record whose `id` equals an already-stored external id fails: diagnostic contains ``primary key `doc-1` already exists``. `--upsert` does not bypass this.

### E.5 filtered search returns `[]` without touching vectors

`search --filter "lang = 'fr'"` against an index with no `fr` documents emits a single-line `[]` and returns exit 0; the vector index is not searched.

### E.6 export round-trips metadata into `fields`

For a document with vector_field `content`, text `alpha`, meta `{"lang":"zh"}`, external id `doc-1`:

```json
{"pk":"doc-1","fields":{"content":"alpha","lang":"zh"}}
```

No `vector` key appears at any level. When no external id exists, `pk` is the stringified internal id.

### E.7 query projects selected fields

`query --sql "SELECT pk, category FROM articles WHERE category = 'tech' LIMIT 20"` returns `{records:[{pk:…, category:…}, …]}`, truncated to 20, ordered by ascending internal id; `score` is null under this path.

## Definition of Done

Grouped by behavior cluster. `[T]` = an existing test asserts this behavior; `[U]` = currently unobserved/untested, must be added for acceptance.

### F. Index lifecycle

- F.1 `init` with valid dim (positive multiple of 8) and bits ∈ {2,3,4} creates index + sidecar + meta and emits `{db, dimension, bits, created:true}`. `[T]`
- F.2 `init` with dim = 0 or not a multiple of 8 fails with `dim must be a positive multiple of 8`. `[U]` (code path exists, not directly asserted)
- F.3 `init` with bits ∉ {2,3,4} fails with `bits must be 2, 3, or 4`. `[U]`
- F.4 Absent index + `import` bootstraps an index using the explicit `--dim`, else the first record's vector length, else 1024. `[T]`
- F.5 The three artifacts always share the path stem and differ only by extension (`.tvim`, `.tvim.meta.json`, `.tvim.sqlite`). `[T]`

### G. Import

- G.1 Importing precomputed vectors increments IndexMeta.next_id by the record count and sets IndexMeta.model from the explicit `--model` when no embedding was used. `[T]`
- G.2 Importing records without vectors invokes the embedding client and, on success, sets IndexMeta.model to the embedding model. `[U]` (requires a live/embedded embedding client; not exercised by current tests)
- G.3 A vector whose length ≠ IndexMeta.dim fails with `vector dimension mismatch`. `[T]`
- G.4 A record whose external id already exists fails with ``primary key `<id>` already exists``; `--upsert` does not override this. `[T]`
- G.5 A missing input file fails with `input file not found`. `[T]`
- G.6 An input with no non-blank lines fails with `no JSONL records found`. `[U]`
- G.7 A record resolving to zero or more than one vector field fails with `expected exactly one vector field`. `[T]`
- G.8 A record containing both `vector` and `vectors.<f>` fails with `record cannot contain both`. `[U]`
- G.9 A record with no fields fails with `record has no fields to import`. `[U]`
- G.10 Records needing embedding with no model/client available fail with `records without vectors require an embedding model`. `[U]`
- G.11 The vector field's value is stored in `text` and is excluded from `meta`; the external id is copied into `meta.external_id`. `[T]`
- G.12 `import` stdout is `{success, errors:0, total}` where `total` is the post-import index length. `[T]` (structure asserted via `idx.len()`; the JSON object shape is observable from code)

### H. Search

- H.1 Search against a missing index fails with `index not found`. `[T]`
- H.2 Search against an empty index fails with `index is empty`. `[U]` (the empty-index branch exists; no test seeds an empty populated index)
- H.3 Supplying both `--query` and a query vector fails with `pass only one of`. `[T]`
- H.4 Supplying neither fails with `search requires one of`. `[T]`
- H.5 A query vector whose length ≠ index dim fails with `query vector dimension mismatch` and `index expects <dim>`. `[T]`
- H.6 A filter matching no documents emits `[]` and returns exit 0 without searching vectors. `[T]`
- H.7 A filtered search with at least one match returns a pretty JSON array of SearchResult, each with `id, external_id, vector_field, score, text, meta`. `[T]` (Ok asserted; field shape observable from code)
- H.8 `--top-k` bounds the result length. `[U]`

### I. Query (SQL metadata)

- I.1 `SELECT <fields> FROM <coll> WHERE <filter> LIMIT n` parses select-list, filter, and limit; returns `{records:[…]}` truncated to LIMIT, ordered by ascending internal id. `[T]` (parser asserted; end-to-end execution `[U]`)
- I.2 A query without `FROM` fails with `query must use SELECT ... FROM`. `[U]`
- I.3 A query not starting with `SELECT` fails with `query must start with SELECT`. `[U]`
- I.4 A query with an empty collection fails with `query must name a collection after FROM`. `[U]`
- I.5 An invalid LIMIT value fails with `invalid LIMIT value`. `[U]`
- I.6 `SELECT *` returns the full record (id, pk, doc_id, score=null, and document fields). `[U]`
- I.7 The `query` path never touches the vector index and `score` is always null. `[U]`

### J. Export

- J.1 `export --include-vectors` fails with `--include-vectors is not supported`. `[T]`
- J.2 `export` against a missing index fails with `db not found`. `[T]`
- J.3 Export writes one JSONL record per document with `pk` (external id or stringified internal id) and `fields` (metadata ∪ {vector field → text}); no raw vector appears. `[T]`
- J.4 `--filter` restricts exported records. `[T]`
- J.5 Omitting `--output` (or passing `-`) writes JSONL to stdout. `[T]`

### K. filter-ids

- K.1 `filter-ids` returns internal ids whose documents match the filter, as a pretty JSON array. `[T]`
- K.2 `filter-ids` against an index with no sidecar fails with `metadata filter requires SQLite sidecar`. `[U]`
- K.3 Compound filters (`AND`, `>=`, IN) resolve correctly. `[T]`

### L. Filter compilation

- L.1 Filters compile to parameterized `json_extract(meta, ?) <op> ?` clauses with all paths and literals bound as parameters (no string interpolation of values). `[T]`
- L.2 `x IN ()` compiles to `(0 = 1)`; `x NOT IN ()` compiles to `(1 = 1)`. `[T]`
- L.3 An invalid field name fails with `unsupported characters` / `must start each path segment` / `cannot be empty`. `[T]` (the `unsupported characters` / `parsing metadata filter` branches are asserted)
- L.4 A filter violating the depth/comparison/IN-list policy fails with `invalid metadata filter`. `[U]`

### M. Config resolution

- M.1 Config is loaded from a JSON object literal (detected by leading `{`) or from a file path; aliases for `data_path` and `default_vector_model` are accepted. `[T]`
- M.2 An empty config string fails with `config cannot be empty`. `[U]`
- M.3 Resolution precedence is CLI flag > config > default, for db path, model, provider, base URL. `[T]`
- M.4 A provider-prefixed model whose prefix ≠ the explicit provider fails with `conflicts`; on match the bare name is used. `[T]`
- M.5 `--schema` extracts text/vector-field defaults and dimension; a schema without `fields` fails with `schema JSON must contain fields array`. `[T]`

### N. stats

- N.1 `stats` reports `dimension`, `vectors`, `texts`, `index_size_bytes`, and `meta.{bits,model,next_id}`. `[T]` (dimension/vectors/meta.bits asserted; full shape observable from code)
- N.2 `stats` against a missing index fails with `db not found`. `[U]`

### O. Not-implemented subcommands

- O.1 `serve` fails with `not implemented for turbovec-rs yet`. `[U]` (visible in code; no test asserts it)
- O.2 `mcp` fails with `not implemented for turbovec-rs yet`. `[U]`

### P. Diagnostic reachability (Sweep A confirmation)

Every diagnostic substring in `03` §Validation is reachable from a public CLI entry (the corresponding subcommand) and is not gated solely behind a private helper. The two parser-layer guards (`init` dim/bits validation, and the search query/vector mutual-exclusion) are reached directly from their subcommands.

## Appendices

### A.1 Observed but not promoted to contract

- The internal `--batch-size` default of 32 and the `max(batch_size, 1)` clamp: the exact default is observable only via flag absence and is incidental; the clamp (treating 0 as 1) is an implementation detail not promoted.
- The `bge-m3` / `content` / `text` / `embedding` built-in default names: these are defaults, not invariants; a re-implementation **MAY** choose different defaults provided the resolution precedence and the "lone field wins" schema rules hold.
- Import progress lines on stderr (`+N/M imported`, `<N> JSONL records to import (batch_size=…)`, the `--upsert` warning): these are human-facing progress diagnostics, not stable contracts; an implementation **MAY** omit or rephrase them.
- The `ALTER TABLE … ADD COLUMN` idempotency statements in sidecar schema init: an internal migration detail, not a contract.
