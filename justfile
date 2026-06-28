# turbovec-rs — Rust CLI 工具

set shell := ["bash", "-euo", "pipefail", "-c"]

# 架构后缀：ARM64 → arm64，x86_64 → x86
arch_suffix := if arch() == "aarch64" { "arm64" } else { "x86" }

# 编译型二进制安装目录（按架构隔离，ADR-749）
install_bin := home_directory() / "sync" / ("bin_" + arch_suffix)

# target 目录：按 cargo metadata 解析实际 target 目录
target_dir := `cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])'`

# 二进制名称
bin_name := "turbovec-rs"

default: build

# 构建（Release 模式）
build:
    cargo build --release

# 运行测试
test:
    cargo test

# 安装到 ~/sync/bin_<arch>/
install: build
    mkdir -p {{ install_bin }}
    cp {{ target_dir }}/release/{{ bin_name }} {{ install_bin }}/{{ bin_name }}
    codesign -f -s - {{ install_bin }}/{{ bin_name }} 2>/dev/null || true
