#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Generate a synthetic JSONL corpus with precomputed vectors.

Usage:
  benchmark/generate_100k_jsonl.sh OUT_DIR [RECORDS] [DIM]

Defaults:
  RECORDS=100000
  DIM=128

Outputs:
  OUT_DIR/docs-<records>.jsonl
  OUT_DIR/query.json
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

out_dir="${1:?missing OUT_DIR}"
records="${2:-100000}"
dim="${3:-128}"

mkdir -p "$out_dir"

docs_path="$out_dir/docs-${records}.jsonl"
query_path="$out_dir/query.json"

awk -v records="$records" -v dim="$dim" '
BEGIN {
  srand(42);
  for (i = 0; i < records; i++) {
    tenant = sprintf("tenant_%03d", i % 100);
    group = sprintf("group_%02d", i % 20);
    lang = (i % 2 == 0 ? "zh" : "en");
    kind = (i % 5 == 0 ? "guide" : "note");
    printf "{\"id\":\"doc-%06d\",\"vector_field\":\"content\",\"fields\":{\"content\":\"synthetic document %06d\",\"tenant\":\"%s\",\"group\":\"%s\",\"lang\":\"%s\",\"kind\":\"%s\",\"bucket\":%d},\"vector\":[", i, i, tenant, group, lang, kind, i % 1000;
    for (j = 0; j < dim; j++) {
      v = (rand() * 2) - 1;
      printf "%s%.6f", (j == 0 ? "" : ","), v;
    }
    print "]}";
  }
}
' > "$docs_path"

awk -v dim="$dim" '
BEGIN {
  srand(7);
  printf "[";
  for (j = 0; j < dim; j++) {
    v = (rand() * 2) - 1;
    printf "%s%.6f", (j == 0 ? "" : ","), v;
  }
  print "]";
}
' > "$query_path"

printf 'docs=%s\n' "$docs_path"
printf 'query=%s\n' "$query_path"
