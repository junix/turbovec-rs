# turbovec-rs 架构信息图

这是一张独立于 `docs/architecture.html` 的长幅信息图。旧版
`docs/architecture-infographic.*` 只重复了架构 HTML 中的泛化框图，不能解释本项目，
因此已删除并由当前 SVG 规范源取代。

## 视觉合同

- 受众：新贡献者、架构评审者与下游集成方
- 论点：把仓库的真实入口、实现分区、可执行用法与验证证据放在一条可回查的阅读路径上。
- 读者问题：turbovec-rs 的代码从哪里进入、由哪些分区构成、怎样运行并如何验证？
- 语言：简体中文；代码、路径与命令保留原文
- 媒介：2000 × 2920 静态矢量长图
- 规范源：`architecture.svg`（原生文字、稳定语义 ID，可直接编辑）
- 交付：SVG + 3000px 宽 PNG + PDF + 语义/证据 JSON + SVG lint 编号图

## 叙事与证据

阅读顺序：项目指纹 → 架构地形 → 使用路径 → 规模与依赖 → 验证与溯源。

- README：`README.md`
- manifests：`Cargo.toml`, `justfile`
- 源码规模：17 个源码文件、3389 个非空文本行
- 测试边界：14 个路径命中测试规则的已跟踪文件
- 计算口径：逐个已跟踪源码文件统计非空文本行；不等同于语句数或复杂度
- 推断边界：文件、计数、命令与依赖是仓库事实；职责名称仅按路径命名归类，均显式标注为推断。

`architecture.source.json` 保存上述事实、公式、入口、命令、依赖、分区与静态引用证据。

## 构建

`architecture.svg` 是规范源；编辑后重新发布派生产物：

```bash
rsvg-convert -w 3000 architecture.svg -o architecture.png
rsvg-convert -f pdf -o architecture.pdf architecture.svg
```

SVG 检查命令、选择器、覆盖率、哈希和 finding 计数见
`architecture.render.json`。`architecture.lint.png` 是编号候选图；
`architecture.png` 是不带标注的最终视觉证明。
