# turbovec-rs 约定

本项目用量化索引保存向量，并用同名 SQLite sidecar 保存文本、external ID 与 metadata。当前已有 justfile，`embeddings` / `filterql` 使用 Git 依赖；不要恢复旧 workspace path 依赖。

- 索引三件套 `<stem>.tvim`、`<stem>.tvim.meta.json`、`<stem>.tvim.sqlite` 必须同目录同主名，路径只经 sidecar helper 派生。
- dimension 必须满足底层量化约束；bits 只接受实现支持值。
- 量化不可逆，export 不得声称可恢复 raw vectors。
- external ID 唯一且当前无覆盖更新；兼容 flag 不得伪装成真正 upsert。
- filter 与 SQL-WHERE 都通过 filterql 编译，并限制字段名、深度、比较数和 IN-list。
- 配置合并顺序保持单一实现；不要在各 subcommand 重复。
- serve/MCP 等隐藏未实现入口不得文档化为可用能力。
- 单元测试覆盖解析/编译，二进制 smoke 锁定 stdout/stderr/退出码和三件套行为。

修改后运行 `just test`；持久化变化再跑 init→import→reopen→search/query/export E2E，并验证 `just install`。
