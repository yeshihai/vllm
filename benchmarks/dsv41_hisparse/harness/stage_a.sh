#!/bin/bash
# Stage A: ON/OFF x {4096,8192,16384,32768} at three representative points.
set -u
ROOT=/dockerdata/yeshihai_dsv41_buffer
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/vllm_dev
export PATH="/root/miniconda3/envs/vllm_dev/bin:$PATH"
export no_proxy=127.0.0.1,localhost
OUT="$ROOT/results/stage_a_20261107_r1"
LOGS="$ROOT/logs/stage_a_20261107_r1"
mkdir -p "$OUT" "$LOGS"
PIDFILE="$OUT/current.pid"
PROGRESS="$OUT/progress.log"

BASE=http://127.0.0.1:8021
MODEL=DeepSeek-V4.1-Flash-1017-H20-Perf-noxzhao
TOK=/dockerdata/models/DeepSeek-V4.1-Flash

log() { echo "[$(date -Is)] $*" | tee -a "$PROGRESS"; }

snap() {
  local mode=$1 mbt=$2 work=$3 phase=$4
  local line
  line=$(curl -s --max-time 10 "$BASE/metrics" | grep -E '^vllm:hisparse_(cache_hits_total|cache_misses_total|host_to_device_bytes_total)' | tr '\n' ' ')
  echo "[$(date -Is)] $mode mbt=$mbt $work $phase $line" >> "$OUT/metrics.log"
}

wait_health() {
  local i code
  for i in $(seq 1 75); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$BASE/health" || true)
    if [ "$code" = 200 ]; then return 0; fi
    sleep 10
  done
  return 1
}

stop_service() {
  local pid=""
  [ -f "$PIDFILE" ] && pid=$(cat "$PIDFILE")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    log "stopping service pid=$pid"
    kill -TERM "$pid"
    for i in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$pid" 2>/dev/null; then
      log "WARN pid=$pid still alive after 120s; sending second TERM"
      kill -TERM "$pid"
      for i in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
      done
    fi
  fi
  rm -f "$PIDFILE"
  local used
  for i in $(seq 1 48); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 1000 ]; then
      log "gpu0 free after $((i*5))s wait"
      return 0
    fi
    sleep 5
  done
  log "ERROR gpu0 still busy: $used MiB"
  return 1
}

start_service() {
  local mode=$1 mbt=$2
  local script logf
  if [ "$mode" = on ]; then
    script="$ROOT/scripts/start_on_mbt.sh"
  else
    script="$ROOT/scripts/start_off_mbt.sh"
  fi
  logf="$LOGS/serve_${mode}_${mbt}.log"
  log "starting $mode mbt=$mbt -> $logf"
  setsid nohup bash "$script" "$mbt" > "$logf" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 5
  if ! wait_health; then
    log "ERROR health timeout for $mode mbt=$mbt pid=$(cat "$PIDFILE")"
    tail -40 "$logf" >> "$PROGRESS"
    stop_service
    return 1
  fi
  log "health=200 $mode mbt=$mbt pid=$(cat "$PIDFILE")"
  grep -E 'GPU \+ host KV cache size|GPU KV cache size|Maximum concurrency|steady-state maximum concurrency' "$logf" \
    > "$OUT/capacity_${mode}_${mbt}.txt" || true
  cat "$OUT/capacity_${mode}_${mbt}.txt" | tee -a "$PROGRESS"
  return 0
}

bench() {
  local mode=$1 mbt=$2 name=$3 isl=$4 osl=$5 conc=$6 prompts=$7 seed=$8
  local outdir="$OUT/$mode/$mbt/$name"
  local logf="$LOGS/bench_${mode}_${mbt}_${name}.log"
  mkdir -p "$outdir"
  log "bench $mode mbt=$mbt $name"
  snap "$mode" "$mbt" "$name" before
  vllm bench serve --backend openai --base-url "$BASE" \
    --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$isl" --random-output-len "$osl" \
    --random-range-ratio 0 --num-prompts "$prompts" --max-concurrency "$conc" \
    --ignore-eos --seed "$seed" --temperature=0 --num-warmups 0 \
    --metric-percentiles 50,90,99 --save-result --result-dir "$outdir" \
    2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  snap "$mode" "$mbt" "$name" after
  log "bench $mode mbt=$mbt $name rc=$rc"
  return 0
}

run_mode_config() {
  local mode=$1 mbt=$2
  if ! start_service "$mode" "$mbt"; then
    echo "FAILED_START $mode $mbt" >> "$OUT/failures.txt"
    return 0
  fi
  bench "$mode" "$mbt" a_1m_c1 1048000 64 1 1 701
  bench "$mode" "$mbt" a_131k_c16 131072 1 16 16 402
  bench "$mode" "$mbt" a_131k_c32 131072 1 32 32 403
  stop_service
}

stop_service

for MBT in 4096 8192 16384 32768; do
  for MODE in on off; do
    log "===== stage A $MODE mbt=$MBT ====="
    run_mode_config "$MODE" "$MBT"
  done
done

log "stage A complete"
