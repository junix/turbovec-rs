# turbovec-rs 架构信息图

这是一张独立于 `docs/architecture.html` 的长幅信息图。旧版
`docs/architecture-infographic.*` 只重复了架构 HTML 中的泛化框图，不能解释本项目，
因此已删除并由当前 SVG 规范源取代。

## 视觉合同

- 受众：新贡献者、架构评审者与下游集成方
- 论点：把仓库的真实入口、实现分区、真实依赖与验证证据放在一条可回查的阅读路径上。
- 读者问题：turbovec-rs 的代码从哪里进入、由哪些分区构成、怎样运行并如何验证？
- 语言：简体中文；代码、路径与命令保留原文
- 媒介：2000 × 3000 静态矢量长图（高度按内容自适应）
- 规范源：`architecture.svg`（原生文字、稳定语义 ID，可直接编辑）
- 交付：SVG + 3000px 宽 PNG + PDF + 语义/证据 JSON + SVG lint 编号图

## 叙事与证据

阅读顺序：项目指纹 → 架构地形 → 使用路径 → 规模与依赖 → 验证与溯源。

- README：`README.md`
- manifests：`Cargo.toml`, `justfile`
- 源码规模：17 个源码文件、3389 个非空文本行
- 测试边界：14 个路径命中测试规则的已跟踪文件
- 最近提交：2026-08-29 docs: 更新架构文档; 2026-08-29 docs: 架构文档英文内容翻译为中文; 2026-08-26 docs: relocate architecture infographic to docs/infographics
- 计算口径：逐个已跟踪源码文件统计非空文本行；不等同于语句数或复杂度
- 推断边界：文件、计数、命令、依赖与最近提交是仓库事实；职责名称仅按路径命名归类，均显式标注为推断。

`architecture.source.json` 保存上述事实、公式、入口、命令、依赖、分区与静态引用证据。
组件名称如需人工修正，写 `overrides.json`（`component_labels` / `component_roles` /
`commands`）后重新生成，不要手改 SVG 中的事实文本。

## 构建

`architecture.svg` 是规范源；编辑后重新发布派生产物（本次使用 `google-chrome`）：

```bash
# 渲染器二选一：rsvg-convert（若安装），或 google-chrome headless + 包装 HTML（命令见 render.json）
google-chrome --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size=3000,4500 --virtual-time-budget=10000 --screenshot=architecture.png _render-png.html
google-chrome --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=10000 --print-to-pdf=architecture.pdf _render-pdf.html
```

SVG 检查命令、选择器、覆盖率、哈希和 finding 计数见
`architecture.render.json`。`architecture.lint.png` 是编号候选图；
`architecture.png` 是不带标注的最终视觉证明。
