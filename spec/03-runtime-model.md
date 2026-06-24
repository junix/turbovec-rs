# 03 — Runtime Model

## Execution semantics

Each externally observable operation is one `Algorithm` block. Inputs/outputs use the abstract nouns from `02`.

### Algorithm 1  Init

```
Require: index path P, dimension D, bits B
Ensure:  artifacts P, P.meta.json, P.sqlite all exist; IndexMeta{next_id=1, dim=D, bits=B, model=""} persisted;
         stdout = compact JSON {db, dimension, bits, created:true}; exit 0
 1: if D = 0 OR D mod 8 ≠ 0 then fail "dim must be a positive multiple of 8, got <D>"
 2: if B ∉ {2,3,4} then fail "bits must be 2, 3, or 4, got <B>"
 3: CreateIndex(D, B) → write to P
 4: OpenSidecar(P); InitSidecarSchema()
 5: SaveMeta(P, {next_id←1, dim←D, bits←B, model←""})
 6: emit {db: stringify(P), dimension: D, bits: B, created: true}
```

### Algorithm 2  Import

```
Require: index path P (may be Absent); input source S (file path or stdin); optional embedding client C;
         fallback vector field Vf; fallback text field Tf; optional dim Df; bits B (default 4); upsert flag
Ensure:  every record in S is either persisted as (internal id, vector, doc row) or the whole import fails;
         IndexMeta.next_id advanced by the number of persisted records; IndexMeta.model set if embedding was used
         or an explicit model was given; stdout = compact JSON {success, errors:0, total=<index length>}
 1: if S is a path AND ¬exists(S) then fail "input file not found: <S>"
 2: if upsert then warn (stderr) "--upsert currently behaves like insert unless the primary key already exists"
 3: R ← ParseImportRecords(S, Vf, Tf)               ▷ Algorithm 7
 4: if ¬exists(P) then BootstrapIndex(P, Df, first-record-vector-len-or-1024, B)
 5: M ← LoadMeta(P); Idx ← LoadIndex(P); Conn ← OpenSidecar(P)
 6: bs ← max(batch_size, 1)
 7: for each batch rb of R chunked by bs:
 8:    vb ← for each r in rb: r.vector
 9:    if any vb[i] is None then C MUST be present; embed the missing texts; vb[i] ← embedding
10:    if any vb[i] is None after embedding then fail "missing vector after embedding import batch"
11:    ValidateDim(vb, M.dim)                         ▷ fail "vector dimension mismatch at batch item <i> ..."
12:    for each r in rb with r.external_id ≠ None:
13:       if ExternalIdExists(Conn, r.external_id) then fail "primary key `<id>` already exists; ..."
14:    ids ← [M.next_id .. M.next_id + |rb| - 1]
15:    Idx.AddWithIds(flatten(vb), M.dim, ids)
16:    for each (id, r) in zip(ids, rb): InsertDoc(Conn, id, r)
17:    M.next_id ← M.next_id + |rb|
18: Idx.write(P)
19: if embedding was used then M.model ← used-embedding-model else if explicit model given then M.model ← model
20: SaveMeta(P, M)
21: emit {success: |R|, errors: 0, total: Idx.length()}
```

Stop condition: the loop in step 7 terminates when the chunked batches are exhausted; there is no retry.

### Algorithm 3  Search

```
Require: index path P (Populated); exactly one of: query text Q, query vector Vq, or a vector file;
         top_k K (default 10); embedding model m, provider p, base URL u; optional metadata filter F
Ensure:  stdout = pretty JSON array of SearchResult ordered by descending similarity, length ≤ K;
         OR `[]` (single line) when a filter yields no matching internal ids, without touching the vector index
 1: if ¬exists(P) then fail "index not found: <P>"
 2: Idx ← LoadIndex(P); if Idx.isEmpty then fail "index is empty (import documents first)"
 3: if F ≠ None:
 4:    A ← FilterIds(P, F)                            ▷ Algorithm 5
 5:    if A = ∅ then emit "[]"; return
 6: else A ← None
 7: if (Q ≠ None) AND (Vq ≠ None) then fail "pass only one of --query, --vector, or --vector-file"
 8: if Q ≠ None then v ← Embed(m, p, u, Q)
 9: else if Vq ≠ None then
10:    if |Vq| ≠ Idx.dim then fail "query vector dimension mismatch: index expects <dim>, got <|Vq|>"
11:    v ← Vq
12: else fail "search requires one of --query, --vector, or --vector-file"
13: if A ≠ None then (scores, ids) ← Idx.SearchWithAllowlist(v, K, A)
14: else (scores, ids) ← Idx.Search(v, K)
15: docs ← LoadDocsByIds(P, ids)
16: emit pretty JSON [ for each (id, score): SearchResult{id, score, ...doc-or-defaults} ]
```

### Algorithm 4  Query (zvec-style SQL metadata)

```
Require: index path P; SQL text of form SELECT <proj> FROM <collection> [WHERE <filter>] [LIMIT n]
Ensure:  stdout = pretty JSON {records: [QueryRecord...]}; records ordered by ascending internal id,
         truncated to LIMIT when present; vector index is not touched
 1: if ¬exists(P) then fail "db not found: <P>"
 2: q ← ParseSqlMetadataQuery(sql)                    ▷ Algorithm 8
 3: Conn ← OpenSidecar(P); ids ← QueryDocIds(Conn, q.filter)
 4: if q.limit ≠ None then ids ← truncate(ids, q.limit)
 5: docs ← LoadDocsByIds(Conn, ids)
 6: emit {records: [ Project(id, docs[id], q.select) for id in ids ]}
```

### Algorithm 5  FilterIds

```
Require: index path P; metadata filter F
Ensure:  returns the list of internal ids whose documents match F; FAILS if the sidecar is absent
 1: if ¬exists(P.sqlite) then fail "metadata filter requires SQLite sidecar <path>"
 2: W ← CompileFilter(F)                              ▷ Algorithm 6
 3: ids ← SELECT id FROM docs WHERE W.clause ORDER BY id, bound with W.params
 4: return ids
```

### Algorithm 6  CompileFilter

```
Require: metadata filter F (SQL-ish boolean expression over JSON-typed fields)
Ensure:  returns (clause, params) — a parameterized SQLite WHERE fragment; every field access compiles to
         json_extract/json_type with the JSON path bound as a parameter; every literal bound as a parameter
 1: expr ← FilterQL.sql.parse(F)
 2: Policy{max_depth=8, max_comparisons=32, max_in_list=256}.validate(expr)   ▷ fail "invalid metadata filter: <report>"
 3: return Compile(expr) where:
    - field f → JSON path "$.<f>" (bound), validated by the field-name grammar
    - =,≠,<,≤,>,≥, LIKE, NOT LIKE → "json_extract(meta, ?) <op> ?"
    - field = null → "json_type(meta, ?) IS NULL"; field ≠ null → "json_type(meta, ?) IS NOT NULL"
    - EXISTS f → "json_type(meta, ?) IS [NOT] NULL" (present iff value ≠ false)
    - x IN (...) → "json_extract(meta, ?) IN (?, …)"; empty list → "(0 = 1)"
    - x NOT IN (...) → "json_extract(meta, ?) NOT IN (?, …)"; empty list → "(1 = 1)"
    - AND(a…) → "(a AND b AND …)"; OR(a…) → "(a OR b OR …)" (∅ OR → "(0 = 1)"); NOT a → "NOT (a)"
```

The empty-list semantics are total: `x IN ()` is always false and `x NOT IN ()` is always true, independent of whether the field exists.

### Algorithm 7  ParseImportRecords

```
Require: source S (path or stdin); fallback vector field Vf; fallback text field Tf
Ensure:  returns a non-empty ordered list of ImportRecord; FAILS on the first unparseable/invalid record
 1: text ← read S (file contents, or stdin to EOF)
 2: for each non-blank line L (1-based index i):
 3:    parse L as a JSON object; else fail "parsing <S> line <i>: …"
 4:    external id ← scalar value of the `id` or `pk` key (stringified); else None
 5:    fields ← normalized fields (unwrap {value,index} descriptors to value)
 6:    vector field ← ResolveVectorField(...)         ▷ Algorithm 9
 7:    vector text ← fields[vector field] as non-empty string; else fail
 8:    vector ← ResolveRecordVector(...)              ▷ Algorithm 10
 9:    meta ← {external_id if present} ∪ {fields except vector field}
10:    append ImportRecord{external id, vector field, vector text, vector, meta}
11: if list is empty then fail "no JSONL records found in <S>"
```

### Algorithm 8  ParseSqlMetadataQuery

```
Require: SQL text (case-insensitive keywords)
Ensure:  returns {select, filter, limit}; FAILS unless the text matches SELECT … FROM <collection> [WHERE …] [LIMIT n]
 1: strip leading/trailing whitespace and trailing ';'
 2: require a case-insensitive " from " separator; else fail "query must use SELECT ... FROM ..."
 3: require the head to start with "select " (case-insensitive); else fail "query must start with SELECT"
 4: collection ← token(s) between FROM and WHERE/LIMIT; must be non-empty; else fail "query must name a collection after FROM"
 5: filter ← text between WHERE and LIMIT (if WHERE precedes LIMIT); None if absent or empty
 6: limit ← parse the integer after LIMIT; else None (invalid integer → fail "invalid LIMIT value `<v>`")
 7: select ← "*" → empty list (meaning "full record"); else comma-separated trimmed field names (empties dropped)
```

### Algorithm 9  ResolveVectorField

```
Require: record object; normalized fields; fallback vector field Vf; fallback text field Tf
Ensure:  returns exactly one field name present in `fields`; FAILS if zero or >1 candidates survive
 1: candidates ← []
 2: if record has `vector_field` (string) then add it
 3: if record has `vector_fields` (string array) then add each
 4: add every field whose descriptor has index containing "vector"
 5: add every key under `vectors` (object keyed by field name)
 6: if candidates = ∅:
 7:    if Vf ≠ None then add Vf
 8:    else if Tf ≠ None then add Tf
 9:    else if fields has key "text" then add "text"
10:    else if fields has key "content" then add "content"
11: sort + dedup candidates
12: if |candidates| ≠ 1 then fail "expected exactly one vector field, found <n> (<candidates>)"
13: f ← the sole candidate; validate f by the field-name grammar; if f ∉ fields then fail
14: return f
```

### Algorithm 10  ResolveRecordVector

```
Require: record object; resolved vector field f
Ensure:  returns Some(vector) if a precomputed vector is supplied, else None; FAILS if both forms are present
 1: direct ← record.`vector`; keyed ← record.`vectors`.`f`
 2: if direct ≠ None AND keyed ≠ None then fail "record cannot contain both `vector` and `vectors.<f>`"
 3: if direct ≠ None then return ParseVectorValue(direct)
 4: if keyed  ≠ None then return ParseVectorValue(keyed)
 5: return None
```

`ParseVectorValue` requires a non-empty JSON array of finite numbers within the f32 range, else fails with a per-item diagnostic.

## Config resolution precedence

For db path, model, provider, base URL, and embedding-config/string:

```
explicit CLI flag  >  config value  >  built-in default
```

| Resolved value | CLI flag | Config key (and aliases) | Default |
|----------------|----------|--------------------------|---------|
| index path | `--db` / `--db-path` (query only) | `data_path` (aliases: `storage_path`, `db`, `db_path`, `index`, `index_path`) | (none — MUST be supplied) |
| embedding model | `--model` | `default_vector_model` (aliases: `model`, `default_model`, `embedding_model`) | `bge-m3` |
| provider | `--provider` | `provider` | (auto-detect from model) |
| base URL | `--base-url` | `base_url` | (provider default) |

When an explicit `--provider` is given alongside a model of the form `<provider>/<name>`, the prefix **MUST** match the provider or the tool fails ("model `<m>` conflicts with provider `<p>`"); on match the bare model name is used.

## Validation / Diagnostics / Error Model

All failures propagate as a non-zero process exit with a human-readable diagnostic on stderr. The tool does not define machine-readable error codes. Stable diagnostic substrings (asserted by tests, hence observable contracts an implementation **MUST** preserve as substrings of the diagnostic):

| Trigger | Diagnostic MUST contain |
|---------|-------------------------|
| `init` dim not a positive multiple of 8 | `dim must be a positive multiple of 8` |
| `init` bits not in {2,3,4} | `bits must be 2, 3, or 4` |
| `import` input file missing | `input file not found` |
| `import` no non-blank lines | `no JSONL records found` |
| `import` a vector's length ≠ index dim | `vector dimension mismatch` |
| `import` external id already present | ``primary key `<id>` already exists`` |
| `import` >1 vector field candidate | `expected exactly one vector field` |
| `import` record has both `vector` and `vectors.<f>` | `record cannot contain both \`vector\` and \`vectors.<f>\`` |
| `import` record with no fields | `record has no fields to import` |
| `import` records need embedding but no model/client | `records without vectors require an embedding model` |
| `search`/`export`/`query`/`filter-ids` index absent | `index not found` (search) / `db not found` (export, query) |
| `search` on an empty index | `index is empty` |
| `search` both `--query` and `--vector`/`--vector-file` | `pass only one of` |
| `search` neither query nor vector | `search requires one of` |
| `search` query vector length ≠ index dim | `query vector dimension mismatch` and `index expects <dim>` |
| `export --include-vectors` | `--include-vectors is not supported` |
| `filter-ids`/filtered search with no sidecar | `metadata filter requires SQLite sidecar` |
| invalid metadata filter (policy/parse) | `invalid metadata filter` or `parsing metadata filter` |
| invalid metadata field name | `unsupported characters` / `must start each path segment` / `cannot be empty` |
| `serve` / `mcp` subcommand | `not implemented for turbovec-rs yet` |
| empty config string (`-c ""`) | `config cannot be empty` |
| schema JSON without `fields` array | `schema JSON must contain fields array` |
| SQL query without `FROM` | `query must use SELECT ... FROM` |
| SQL query not starting with `SELECT` | `query must start with SELECT` |
| SQL query with empty collection | `query must name a collection after FROM` |
| invalid LIMIT value | `invalid LIMIT value` |

### Metadata field-name grammar

A metadata field name (and every dot-separated path segment) **MUST** satisfy: non-empty; each segment starts with an ASCII letter or `_`; remaining characters are ASCII alphanumeric or `_`. Dot (`.`) is the only permitted separator and denotes a JSON path. A failing name produces one of the diagnostics above.

## Retries / timeouts / cancellation

There is no application-level retry, timeout, or cancellation logic defined by this tool. Embedding calls and sidecar queries are delegated to the companion interfaces; their failure propagates as a normal diagnostic. An implementation **MUST NOT** invent retry/timeout behavior not present in the companion interfaces.

## Concurrency

Import batches are processed sequentially in record order; internal ids are assigned in order. There is no concurrent mutation of the index. The tool is a single-process CLI; concurrent invocation against the same index is unspecified.
