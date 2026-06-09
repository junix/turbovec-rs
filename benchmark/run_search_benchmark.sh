#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run turbovec-rs synthetic search benchmark.

Usage:
  benchmark/run_search_benchmark.sh OUT_DIR [RECORDS] [DIM]

Defaults:
  RECORDS=100000
  DIM=128

Environment:
  TURBOVEC_BIN  Path to turbovec-rs binary. Defaults to target/release/turbovec-rs.

Outputs are written under OUT_DIR.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

out_dir="${1:?missing OUT_DIR}"
records="${2:-100000}"
dim="${3:-128}"
bin="${TURBOVEC_BIN:-target/release/turbovec-rs}"

mkdir -p "$out_dir"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

if [[ ! -x "$bin" ]]; then
  cargo build --release
fi

"$script_dir/generate_100k_jsonl.sh" "$out_dir" "$records" "$dim"

index="$out_dir/perf.tvim"
docs="$out_dir/docs-${records}.jsonl"
query="$out_dir/query.json"
filter_expr="tenant = 'tenant_042'"

rm -f "$index" "$out_dir/perf.tvim.meta.json" "$out_dir/perf.tvim.sqlite"

/usr/bin/time -p "$bin" init --db "$index" --dim "$dim" 2> "$out_dir/init.time"
/usr/bin/time -p "$bin" import --db "$index" --input "$docs" --batch-size 4096 2> "$out_dir/import.time"

"$bin" stats --db "$index" > "$out_dir/stats.json"

"$bin" search --db "$index" --vector-file "$query" > "$out_dir/warm-nofilter.json"
"$bin" search --db "$index" --vector-file "$query" --filter "$filter_expr" > "$out_dir/warm-filter.json"

: > "$out_dir/no-filter.time"
for i in 1 2 3 4 5; do
  /usr/bin/time -p "$bin" search --db "$index" --vector-file "$query" \
    > "$out_dir/no-filter-$i.json" 2>> "$out_dir/no-filter.time"
done

: > "$out_dir/with-filter.time"
for i in 1 2 3 4 5; do
  /usr/bin/time -p "$bin" search --db "$index" --vector-file "$query" --filter "$filter_expr" \
    > "$out_dir/with-filter-$i.json" 2>> "$out_dir/with-filter.time"
done

: > "$out_dir/filter-ids.time"
for i in 1 2 3 4 5; do
  /usr/bin/time -p "$bin" filter-ids --db "$index" --filter "$filter_expr" \
    > "$out_dir/filter-ids-$i.json" 2>> "$out_dir/filter-ids.time"
done

summarize() {
  awk '/real/ {
    n++;
    real += $2;
    if (min == 0 || $2 < min) min = $2;
    if ($2 > max) max = $2;
  }
  END {
    if (n == 0) {
      print "n=0 avg=0.0000 min=0.0000 max=0.0000";
    } else {
      printf "n=%d avg=%.4f min=%.4f max=%.4f\n", n, real / n, min, max;
    }
  }' "$1"
}

{
  printf 'records=%s\n' "$records"
  printf 'dim=%s\n' "$dim"
  printf 'filter=%s\n' "$filter_expr"
  printf 'docs_size='
  du -h "$docs" | awk '{print $1}'
  printf 'index_size='
  du -h "$index" | awk '{print $1}'
  printf 'sqlite_size='
  du -h "$out_dir/perf.tvim.sqlite" | awk '{print $1}'
  printf 'filter_ids_count='
  rg -c '^[[:space:]]*[0-9]+' "$out_dir/filter-ids-1.json"
  printf 'no_filter='
  summarize "$out_dir/no-filter.time"
  printf 'with_filter='
  summarize "$out_dir/with-filter.time"
  printf 'filter_ids='
  summarize "$out_dir/filter-ids.time"
} | tee "$out_dir/summary.txt"
