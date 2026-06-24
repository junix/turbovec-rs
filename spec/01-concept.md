# 01 — Concept

## Scope

This specification covers the command-line tool **turbovec-rs**: a persistent, locally-stored quantized vector index paired with a SQLite sidecar that stores document text and schemaless metadata, together exposing init, import, search, metadata-query, export, and stats subcommands. The scope is the externally observable behavior of the CLI (exit codes, stdout/stderr, on-disk artifacts). The REST `serve` and stdio `mcp` subcommands are out of scope — they are declared but not implemented (see §Non-Goals).

## Problem Statement

A local collection of documents needs to be indexed once and queried many times by semantic similarity, while also supporting metadata-based filtering and projection. Full-precision vectors are large; quantization to 2/3/4 bits dramatically shrinks storage while remaining searchable. Document text and arbitrary metadata must be stored alongside the quantized vectors so results can be hydrated without a separate store.

## Goals

- Persist a quantized vector index plus a schemaless metadata sidecar on local disk, addressed by a single index path.
- Import JSONL records (one record per line), embedding the text of a single declared vector field when no precomputed vector is supplied, or ingesting a supplied precomputed vector.
- Retrieve the top-k most similar documents for a query text (embedded on demand) or a caller-supplied query vector, optionally restricted to documents matching a metadata filter.
- Query document metadata by a zvec-style `SELECT ... FROM ... [WHERE ...] [LIMIT n]` statement, and export documents back to JSONL.
- Report index health (dimension, vector count, document count, file size, quantization bits).

## Non-Goals

- The hidden `serve` and `mcp` subcommands are declared for CLI parity but **MUST NOT** be considered implemented: invoking either **MUST** fail with a "not implemented" diagnostic (observed, not a forward contract on future behavior).
- Raw vectors cannot be reconstructed from the quantized index; therefore `export --include-vectors` **MUST** fail (no vector recovery is provided).
- No in-place vector overwrite: an external id that already exists in the sidecar **MUST** cause import to fail; `--upsert` is accepted for CLI parity but does not enable overwriting.
- No BM25, keyword, or hybrid search.
- No chunking or document splitting during import; exactly one record per JSONL line.
- No multiple vector fields per record; exactly one vector field is permitted.

## Notational conventions

1. **Normative keywords** follow RFC 2119 / RFC 8174: **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY**. All other strength adverbs ("usually", "generally", "typically") do not appear in normative statements.
2. **Input syntax** (CLI argument grammar, JSONL record shapes, SQL-ish query, metadata filter DSL) is expressed in ABNF (RFC 5234 + RFC 7405), referencing the RFC 5234 §B.1 core rules.
3. **Binary layout** of the on-disk `.tvim` index file is owned by the companion `turbovec` index library and is out of scope; only the *coexistence* of sibling artifacts (extension mapping) is specified here. A bit-box is therefore not required.
4. **State machines** for import batching and filtered search are given as `(state, event) → (state', action, output)` transition tables.
5. **Execution semantics** for each externally observable operation are given as paper-style `Algorithm` blocks with explicit `Require:` / `Ensure:` lines.
6. Domain nouns use the canonical surface forms in `00-glossary.md`; the synonym table there resolves `id`/`pk` → external id and `--db`/`data_path` → index.

## Design Principles

- **One index path, three sibling artifacts.** All state for a given index hangs off a single path; the index, sidecar, and meta file share the same stem and differ only by extension. A new implementation **MUST** preserve this extension mapping (see `02`).
- **Quantized vectors are write-only.** Once a vector is quantized and added, it cannot be read back at full precision. Every operation that would require raw vectors **MUST** fail explicitly rather than silently degrade.
- **Metadata is schemaless JSON.** Document metadata is stored as a single JSON object per document; metadata filters are compiled to JSON-path extraction against that object, never to fixed columns (beyond the small fixed `docs` table).
- **Filtering precedes vector search.** When a metadata filter is supplied, the matching internal ids are computed first; if that set is empty, vector search is never invoked.
- **CLI flags override config which overrides built-in defaults.** Resolution precedence is uniform across db path, model, provider, and base URL.

## Dependencies / Companion interfaces

The tool composes four external interface capabilities (named by capability, not vendor):

1. A **quantized vector index library** providing: create with (dimension, bits); load/write to a path; add a flat list of floats for N ids of fixed dimension; top-k search of a query vector over the whole index or restricted to an allowlist of ids; report dimension and length.
2. An **embedding client** providing: resolve an API credential for a named provider; construct a client from (provider, model, credential) or from ambient environment; override a base URL; embed a list of texts and embed a single text, returning fixed-length float vectors.
3. A **metadata filter language** ("filterql") providing: parse an SQL-ish boolean expression; validate against a policy (depth / comparison / IN-list bounds); compile to a target backend's parameterized WHERE clause; expose comparison operators (=, ≠, <, ≤, >, ≥, LIKE, NOT LIKE, IN, NOT IN, EXISTS) over JSON values.
4. An **embedded SQL engine** providing: a `docs` table with id / external id / vector field / text / meta columns; JSON-path extraction (`json_extract`, `json_type`); parameter binding; count and ordered id queries.
