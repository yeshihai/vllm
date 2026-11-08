#!/bin/bash
# P/D separated serving benchmarks with cold (unique-prompt) prefill.
# Topology: prefill :8100 -> toy proxy :8192 -> decode :8200.
#
# The kernel warmup uses seed 9999 and is not part of the measured set.
# Measured points use --num-warmups 0 and a distinct SEED_BASE per
# configuration, so prefix caches stay cold and the ON/OFF runs never share
# prompts. (--num-warmups 1 reuses the measured prompt and turns TTFT into a
# prefix-cache hit; do not use it for cold numbers.)
#
# Usage: SEED_BASE=10000 bash bench_pd_dsv41.sh <tag> [prefill|longdecode|throughput|all]
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_dev
TAG="${1:?usage: bench_pd_dsv41.sh <tag> [suite]}"
SUITE="${2:-all}"
SEED_BASE="${SEED_BASE:-10000}"
BASE="${BASE:-http://127.0.0.1:8192}"
METRICS_BASE="${METRICS_BASE:-http://127.0.0.1:8200}"
MODEL="${MODEL:-DeepSeek-V4.1-Flash-1017-H20-Perf-noxzhao}"
TOK="${TOK:-/dockerdata/models/DeepSeek-V4.1-Flash}"
OUT="${OUT:-/root/bench_results/${TAG}}"
mkdir -p "$OUT"

snap() {
  curl -s --max-time 5 "$METRICS_BASE/metrics" \
    | awk "/^vllm:hisparse_(cache_hits_total|cache_misses_total|host_to_device_bytes_total)/ {print \"  METRIC \" \$0}" \
    || true
}

run_bench() {
  local isl=$1 osl=$2 conc=$3 prompts=$4 seed=$5
  no_proxy=127.0.0.1,localhost vllm bench serve --backend openai --base-url "$BASE" \
    --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$isl" --random-output-len "$osl" \
    --random-range-ratio 0 --num-prompts "$prompts" --max-concurrency "$conc" \
    --ignore-eos --seed "$seed" --temperature=0 --num-warmups 0 \
    --metric-percentiles 50,90,99 --save-result --result-dir "$OUT"
}

bench() {
  local name=$1 isl=$2 osl=$3 conc=$4 prompts=$5 seed=$6
  echo "### ${name}: isl=${isl} osl=${osl} concurrency=${conc} prompts=${prompts} $(date -Is)"
  snap
  run_bench "$isl" "$osl" "$conc" "$prompts" "$seed"
  snap
  echo "### ${name} done $(date -Is)"
}

warmup() {
  echo "### warmup $(date -Is)"
  run_bench 8192 8 1 1 9999 >/dev/null
  echo "### warmup done $(date -Is)"
}

warmup

if [[ "$SUITE" == all || "$SUITE" == prefill ]]; then
  for isl in ${PREFILL_ISL:-8192 32768 65536 131072 262144}; do
    bench "pd_prefill_${isl}" "$isl" 8 1 1 "$((SEED_BASE + isl))"
  done
fi

if [[ "$SUITE" == all || "$SUITE" == longdecode ]]; then
  for conc in 1 2 4; do
    bench "pd_longdecode_c${conc}" 131072 64 "$conc" 4 "$((SEED_BASE + 200 + conc))"
  done
fi

if [[ "$SUITE" == all || "$SUITE" == throughput ]]; then
  bench "pd_throughput" 4096 128 32 64 "$((SEED_BASE + 300))"
fi

echo "results in $OUT"
