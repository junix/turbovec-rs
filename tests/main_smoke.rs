//! Integration smoke tests for the `turbovec-rs` binary.
//!
//! These exercise `main()`'s subcommand dispatch end-to-end against the real
//! binary built by Cargo. They lock user-facing behavior (CLI exit codes and
//! stdout/stderr contracts) that unit tests on `cmd_*` cannot reach.
//!
//! NOTE: coverage attribution: `cargo llvm-cov` instruments the test binary but
//! not a separately-spawned release/debug binary subprocess, so these tests do
//! NOT move the lcov number for `main`; they are behavior locks, not coverage
//! drivers.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    env!("CARGO_BIN_EXE_turbovec-rs").into()
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> TempDir {
        let path = std::env::temp_dir().join(format!(
            "turbovec-rs-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn index(&self) -> PathBuf {
        self.0.join("index.tvim")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn main_init_creates_index_and_emits_json() {
    let dir = TempDir::new();
    let index = dir.index();

    let output = Command::new(bin())
        .args([
            "init",
            "--db",
            index.to_str().unwrap(),
            "--dim",
            "8",
            "--bits",
            "4",
        ])
        .output()
        .expect("spawn turbovec-rs");
    assert!(output.status.success(), "init failed: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(json["dimension"], 8);
    assert_eq!(json["bits"], 4);
    assert_eq!(json["created"], true);

    assert!(index.exists(), ".tvim not created");
    assert!(index.with_extension("tvim.meta.json").exists());
    assert!(index.with_extension("tvim.sqlite").exists());
}

#[test]
fn main_stats_reports_dimension_after_init() {
    let dir = TempDir::new();
    let index = dir.index();

    let init = Command::new(bin())
        .args([
            "init",
            "--db",
            index.to_str().unwrap(),
            "--dim",
            "8",
            "--bits",
            "3",
        ])
        .output()
        .unwrap();
    assert!(init.status.success(), "init failed: {:?}", init);

    let stats = Command::new(bin())
        .args(["stats", "--db", index.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(stats.status.success(), "stats failed: {:?}", stats);

    let json: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&stats.stdout).trim()).unwrap();
    assert_eq!(json["dimension"], 8);
    assert_eq!(json["vectors"], 0);
    assert_eq!(json["meta"]["bits"], 3);
}

#[test]
fn main_search_errors_on_missing_index() {
    let dir = TempDir::new();
    let missing = dir.0.join("nope.tvim");

    let output = Command::new(bin())
        .args([
            "search",
            "--db",
            missing.to_str().unwrap(),
            "--vector",
            "[0.1,0.2]",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("index not found"), "got stderr: {stderr}");
}

#[test]
fn main_export_errors_on_missing_db() {
    let dir = TempDir::new();
    let missing = dir.0.join("nope.tvim");

    let output = Command::new(bin())
        .args(["export", "--db", missing.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("db not found"), "got stderr: {stderr}");
}
