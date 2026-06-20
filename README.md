# turbovec-rs

> Persistent vector index CLI with semantic search — `turbovec` for 2–4 bit quantized storage, `embeddings` for query vectorization (Ollama / BGE-M3).

`turbovec-rs` 是一个面向本地文档集合的"持久化向量索引 + 语义检索"命令行。
底层用 `turbovec` 的 `IdMapIndex` 做 2/3/4 bit 量化存储，配一个 SQLite
sidecar 存元数据与文档原文；查询时通过 `embeddings` crate（默认
Ollama 上的 `bge-m3`，1024 维）把 query 文本向量化，再在量化空间里
找 top-k。

子命令：`init` / `import` / `search` / `query`（zvec 风格 SQL 元数据查询）/
`export` / `stats` / `filter-ids`（隐藏）/ `serve` & `mcp`（隐藏，暂未实现）。

## Build

```bash
cargo build --release
```

## Test

```bash
cargo test
```

## Install

```bash
just install    # installs to ~/sync/bin_<arch>/
```

`just install` 走 `crates/CLAUDE.md` 的 `install_bin` 约定：
ARM64 写到 `~/sync/bin_arm64/turbovec-rs`，x86_64 写到
`~/sync/bin_x86/turbovec-rs`。

## Usage

最常见的工作流是 init → import → search。索引文件后缀 `.tvim`，
sidecar SQLite 跟它同目录同主名。

```bash
# 1. 建索引
turbovec-rs init --db /tmp/docs.tvim

# 2. 灌数据：JSONL，每行 {"id": "...", "text": "...", ...}
turbovec-rs import --db /tmp/docs.tvim --input docs.jsonl --provider ollama

# 3. 语义检索
turbovec-rs search --db /tmp/docs.tvim --query "什么是编程" --provider ollama

# 4. 用原始向量搜（绕过 embedding 调用）
turbovec-rs search --db /tmp/docs.tvim --vector '[0.1,0.2,0.3]'

# 5. 元数据 SQL 查询（zvec 风格）
turbovec-rs query --db-path /tmp/docs.tvim \
    --sql "SELECT id, text FROM docs WHERE source = 'wiki' LIMIT 10"

# 6. 看一眼索引健康
turbovec-rs stats --db /tmp/docs.tvim
```

配置走 `-c` / `--config` 接收 JSON 字符串或 `@file`：

```bash
turbovec-rs -c '{"data_path":"/tmp/docs.tvim","default_vector_model":"ollama/bge-m3"}' stats
```

> ⚠️ `serve` / `mcp` 子命令目前 `bail!` 报 not implemented——CLI
> import / export / query / search 才是当前主路径。

## Dependencies

| Crate | Role |
|-------|------|
| `turbovec` 0.8 | 2–4 bit 量化向量索引（`IdMapIndex`） |
| `embeddings` (path) | Embedding 客户端（Ollama / BGE-M3 等） |
| `filterql` (path) | 元数据 filter 编译为 SQL（sql/json/validate features） |
| `rusqlite` 0.32 (bundled) | sidecar 元数据存储 |
| `clap` 4 (derive) | CLI |
| `anyhow` / `serde` / `serde_json` / `tokio` | errors, IO, async |

## License

GPL-3.0-or-later
