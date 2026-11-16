#!/usr/bin/env bash
# Runs tinyinfer and llama.cpp (CPU only) on the same model and precision,
# then writes bench/results/compare.csv, a markdown table, and charts.
#
#   brew install llama.cpp
#   scripts/bench_vs_llamacpp.sh [threads] [reps]
set -euo pipefail
cd "$(dirname "$0")/.."

THREADS=${1:-5,8,10,12}
REPS=${2:-5}
MODEL_DIR=models/qwen2.5-0.5b
GGUF=models/gguf/qwen2.5-0.5b-instruct-f32.gguf
OUT=bench/results
mkdir -p "$OUT"

if [ ! -f "$GGUF" ]; then
  .venv/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-fp16.gguf', local_dir='models/gguf')"
  llama-quantize models/gguf/qwen2.5-0.5b-instruct-fp16.gguf "$GGUF" F32
fi

echo "engine,threads,test,tps,sd" > "$OUT/compare.csv"

echo "== tinyinfer"
./build/bench_model "$MODEL_DIR" -t "$THREADS" -p 128 -n 64 -r "$REPS" | tee "$OUT/tinyinfer.md"
grep -E '^\| llama' "$OUT/tinyinfer.md" | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$5); split($6,a,"\\+-"); \
  gsub(/ /,"",a[1]); gsub(/ /,"",a[2]); print "tinyinfer," $4 "," $5 "," a[1] "," a[2]}' >> "$OUT/compare.csv"

echo "== llama.cpp (CPU only)"
llama-bench -m "$GGUF" -dev none -ngl 0 -t "$THREADS" -p 128 -n 64 -r "$REPS" -o csv 2>/dev/null > "$OUT/llamacpp_raw.csv"
.venv/bin/python - "$OUT" <<'EOF'
import csv, sys
out = sys.argv[1]
rows = list(csv.DictReader(open(f"{out}/llamacpp_raw.csv")))
with open(f"{out}/compare.csv", "a") as f:
    for r in rows:
        test = f"pp{r['n_prompt']}" if int(r["n_prompt"]) > 0 else f"tg{r['n_gen']}"
        f.write(f"llama.cpp,{r['n_threads']},{test},{float(r['avg_ts']):.2f},{float(r['stddev_ts']):.2f}\n")
EOF

.venv/bin/python scripts/plot_bench.py "$OUT/compare.csv" "$OUT"
