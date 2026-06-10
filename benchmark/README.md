# Benchmark

Synthetic benchmark scripts for `turbovec-rs`.

The benchmark uses precomputed random vectors so it does not call any embedding
provider. This keeps the test focused on JSONL import, `turbovec` search,
SQLite metadata lookup, and result hydration.

## Scripts

Generate a dataset:

```bash
benchmark/generate_100k_jsonl.sh /tmp/turbovec-perf 100000 128
```

Run the full benchmark:

```bash
benchmark/run_search_benchmark.sh /tmp/turbovec-perf 100000 128
```

The full benchmark performs:

1. release build if needed
2. synthetic JSONL/query-vector generation
3. index initialization
4. JSONL import with precomputed vectors
5. warm-up searches
6. 5 no-filter searches
7. 5 filtered searches
8. 5 `filter-ids` runs

Filter condition:

```text
tenant = 'tenant_042'
```

With the default data generator this matches 1000 records out of 100000.

## Recorded Result

This result was run on 2026-06-10 after commit:

```text
3ff8143 feat: import jsonl metadata and precomputed vectors
```

Command shape:

```bash
cargo build --release
benchmark/generate_100k_jsonl.sh /tmp/turbovec-perf.dIHibv 100000 128
target/release/turbovec-rs init --db /tmp/turbovec-perf.dIHibv/perf.tvim --dim 128
target/release/turbovec-rs import --db /tmp/turbovec-perf.dIHibv/perf.tvim --input /tmp/turbovec-perf.dIHibv/docs-100k.jsonl --batch-size 4096
```

Dataset:

```text
records: 100000
dim: 128
bits: 4
vectors: random precomputed vectors
jsonl size: 144M
.tvim size: 8.1M
sqlite sidecar: 19M
filter: tenant = 'tenant_042'
filter matches: 1000 ids
```

Import:

```text
real 46.31
user 2.09
sys 23.26
```

Search timings use `target/release/turbovec-rs`, query vector from
`query.json`, top-k default `10`, one warm-up run before measurement, and
`/usr/bin/time -p`.

No filter:

```text
real 0.15
real 0.15
real 0.15
real 0.15
real 0.15
avg  0.1500
```

With filter:

```text
real 0.18
real 0.18
real 0.18
real 0.18
real 0.18
avg  0.1800
```

Standalone `filter-ids`:

```text
real 0.04
real 0.04
real 0.04
real 0.04
real 0.04
avg  0.0400
```

## Recorded 1024-D Compression Result

This result was run on 2026-06-10 after commit:

```text
4dbf9c7 feat(cli): align interface with zvec
```

Command shape:

```bash
cargo build --release
benchmark/generate_100k_jsonl.sh /tmp/turbovec-100k-1024.tF4GJD 100000 1024
target/release/turbovec-rs init --db /tmp/turbovec-100k-1024.tF4GJD/perf.tvim --dim 1024
target/release/turbovec-rs import --db /tmp/turbovec-100k-1024.tF4GJD/perf.tvim --input /tmp/turbovec-100k-1024.tF4GJD/docs-100000.jsonl --batch-size 4096
```

Dataset:

```text
records: 100000
dim: 1024
bits: 4
vectors: random precomputed vectors
jsonl size: 945.40 MiB (991321235 bytes)
raw fp32 vector size: 390.62 MiB (409600000 bytes)
.tvim size: 49.98 MiB (52408210 bytes)
sqlite sidecar: 18.20 MiB (19087360 bytes)
meta json: 66 bytes
total persisted size: 68.18 MiB
filter: tenant = 'tenant_042'
```

Compression:

```text
vector index vs raw fp32: 7.82x smaller, 12.79% of raw
index + sqlite sidecar vs raw fp32: 5.73x smaller, 17.45% of raw
index + sqlite sidecar vs jsonl: 13.87x smaller, 7.21% of jsonl
```

Import:

```text
success: 100000
errors: 0
total: 100000
real 49.24
user 5.72
sys 28.64
```

Search timings use `target/release/turbovec-rs`, query vector from
`query.json`, and top-k `10`.

No filter:

```text
real 0.51
user 0.65
sys 0.40
```

With filter:

```text
real 0.55
user 0.68
sys 0.39
```
