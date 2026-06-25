# turbovec-rs Agent Guide

`turbovec-rs` 是面向本地文档集合的「持久化向量索引 + 语义检索」CLI。底层用
`turbovec` 的 `IdMapIndex` 做 2/3/4 bit 量化存储，查询时通过 `embeddings` crate
（默认 Ollama 上的 `bge-m3`，1024 维）把 query 文本向量化，再在量化空间里找 top-k。
文档原文、external id、metadata 不进量化库，而是落到同目录的 SQLite sidecar——
`turbovec` 只负责向量 top-k，SQLite 负责「向量之外的一切」。

> 本仓库没有自己的 justfile，跟随 `crates/CLAUDE.md` 的根级约定。
> License: GPL-3.0-or-later。

## Build / Test / Install

```bash
cargo build --release          # 唯一构建方式（无 justfile）
cargo test                     # 单测 + tests/main_smoke.rs 集成冒烟
cargo test --test main_smoke   # 仅跑二进制端到端冒烟（init/import/search/query/export）

# 安装二进制到架构目录（按 crates/CLAUDE.md 约定）：
arch_suffix=$(arch)            # arm64 / x86_64 → 按实际判断
cp target/release/turbovec-rs ~/sync/bin_${arch_suffix}/turbovec-rs
```

`benchmark/` 下是独立的 shell 脚本（生成 100k JSONL + 跑 import/search 计时），
不依赖 embedding provider，用预生成随机向量。详见 `benchmark/README.md`（含
1024 维下 ~7.8x 压缩、单次 search ~0.5s 的记录值）。

## 子命令 / 主路径

`src/main.rs` 里 `Cli` 用 clap derive 定义子命令。**只有 CLI 路径是活的**，
`serve` / `mcp` 被 `#[command(hide = true)]` 标注且 `main()` 里直接
`bail!("... not implemented ...")`——改它们之前先确认是否真要做。

| 子命令 | 作用 | 入口 |
|--------|------|------|
| `init` | 建空索引 + sidecar schema | `cmd_init` |
| `import` | 灌 JSONL（按 batch 调 embedding 或吃预计算向量） | `cmd_add` |
| `search` | 文本/向量 top-k，可选 `--filter` 元数据过滤 | `cmd_search` |
| `query` | zvec 风格 SQL 元数据查询（不经向量） | `cmd_query` |
| `export` | 导出 JSONL（**不支持 `--include-vectors`**，量化不可逆） | `cmd_export` |
| `stats` | 打印索引健康信息 | `cmd_info` |
| `filter-ids` (隐藏) | 仅返回匹配 filter 的内部 id | `cmd_filter_ids` |

## 架构 / 模块边界

源码 `src/` 按职责严格分层，**测试物理分离**（`*_test.rs`，源文件末尾三行
`#[cfg(test)] #[path = "..."] mod tests;`，遵循 `crates/CLAUDE.md` 约定）：

```
main.rs        CLI 定义 + 子命令派发 + 配置解析入口
commands.rs    cmd_*：业务编排（init/add/search/export/filter-ids/info）
import.rs      JSONL 解析：记录形状归一化、vector_field 解析、预计算向量提取
embed.rs       embeddings crate 客户端构造 + 向量展平/维度校验
sidecar.rs     SQLite sidecar：docs 表 schema、meta 读写、按 id 装载文档
filter.rs      filterql → SQLite WHERE 编译器（SqliteFilterCompiler impl Compile）
sql_query.rs   zvec 风格 SQL 的手写解析（SELECT/FROM/WHERE/LIMIT），非真 SQL 引擎
config.rs      AppConfig/EmbeddingConfig/SchemaDefaults 解析与优先级合并
```

### 数据流（import）

```
JSONL → load_import_records → ImportRecord{external_id, vector_field, vector_text,
                                            vector?, meta}
      → 若 vector 缺失：build_client → embed_missing_vectors（按 batch_size）
      → validate_vectors_dim（必须等于索引 dim）
      → ensure_no_external_id_duplicates（external_id 唯一，无 upsert）
      → IdMapIndex::add_with_ids_2d 写量化库
      → insert_doc 写 SQLite docs 表（text + meta JSON）
      → meta.next_id 推进；最后 idx.write + save_meta 落盘
```

### 数据流（search）

```
query/vector → （query 时）build_client → embed_one
            → 可选 compile_filter → filter_ids_via_sidecar 得到 allowlist
            → IdMapIndex::search 或 search_with_allowlist
            → load_docs_by_ids（从 SQLite 取 text/meta）→ 拼装 JSON 输出
```

## 关键约定与坑

- **索引文件三件套**（spec/02 锁定）：`<stem>.tvim`（量化索引）+
  `<stem>.tvim.meta.json`（IndexMeta：next_id/dim/bits/model）+
  `<stem>.tvim.sqlite`（docs 表）。三者必须同目录、同主名，扩展名由 `--db`
  路径推导。`sidecar::sqlite_path` / `meta_path` 是唯一派生入口，不要手拼。
- **dim 必须是 8 的倍数**，`bits` 只能取 2/3/4（见 `commands::create_index`）。
- **量化不可逆**：`export --include-vectors` 直接 `bail!`；raw 向量无法还原。
- **external_id 不可覆盖**：`import` 遇到已存在的 external_id 会 `bail!`（无
  in-place overwrite）。`--upsert` 仅是 zvec CLI 平移占位，行为同 insert。
- **filter 用 filterql**：`--filter` 和 `--sql` 的 WHERE 都走 `filterql`
  编译，metadata 字段名走 `validate_meta_field_name`（每段以字母/`_` 开头，
  仅允许字母数字/`_`），值通过 `json_extract(meta, ?)` 查 JSON。`filter.rs`
  里的 `Policy` 限制了 depth/comparisons/in-list 上限。
- **配置优先级**：CLI flag > `--embedding` JSON > `--schema` JSON > `-c` config。
  `merge_embedding_arg` / `parse_embedding_config` / `parse_schema_defaults`
  是三层合并的入口，改合并顺序先读 `main.rs` 里 Search/Import 分支。
- **依赖是 path 依赖**：`embeddings` / `filterql` 指向 `../embeddings`、
  `../filterql`。在 `crates/` workspace 外单独 clone 本仓库会编译失败，需先
  拉这两个 sibling crate（或按 ADR-751 改 git URL）。

## 测试分层

- `src/*_test.rs`：每个模块的纯逻辑单测（filter 编译、import 解析、SQL 解析等）。
- `tests/main_smoke.rs`：**端到端二进制冒烟**，用 `CARGO_BIN_EXE_turbovec-rs`
  起真实子进程，锁 CLI 退出码/stdout/stderr 契约。注意：`cargo llvm-cov` 不会
  把覆盖率归到 `main`（子进程不被插桩）——这些是行为锁，不是覆盖率驱动。
