# spec/ CHANGELOG

## 2026-06-24T04:18 — CREATED

Full normative spec authored from code + tests + plan.md + README (no prior `spec/`).

- Files: 00, 01, 02, 03, 04, 05
- Code basis: 35db9c1
- Glossary (`00`) added: core nouns ≥ 5 plus synonym drift (`id`/`pk` → external id; `--db`/`data_path` → index).
- feature matrix: omitted (single-entry CLI subsystem, observable features < 15 per skill threshold).
- DoD coverage tags reflect actual test evidence; embedding-client and several parser-error branches marked `[U]` (not invented).
- Notable non-goals locked: `serve`/`mcp` not implemented, `export --include-vectors` not supported, no in-place vector overwrite (external-id uniqueness).
