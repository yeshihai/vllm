#!/bin/bash
# Full GSM8K-1319 at 4096, ON then OFF.
set -u
ROOT=/dockerdata/yeshihai_dsv41_buffer
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/vllm_dev
export PATH="/root/miniconda3/envs/vllm_dev/bin:$PATH"
export no_proxy=127.0.0.1,localhost
OUT="$ROOT/results/stage_b_20261107_r1"
LOGS="$ROOT/logs/stage_b_20261107_r1"
PIDFILE="$OUT/accuracy.pid"
BASE=http://127.0.0.1:8021

log() { echo "[$(date -Is)] $*" | tee -a "$OUT/accuracy_full_progress.log"; }

stop_service() {
  local pid=""
  [ -f "$PIDFILE" ] && pid=$(cat "$PIDFILE")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    log "stopping pid=$pid"
    kill -TERM "$pid"
    for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; sleep 10; fi
  fi
  rm -f "$PIDFILE"
  local used
  for i in $(seq 1 48); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 1000 ] && { log "gpu free"; return 0; }
    sleep 5
  done
  log "ERROR gpu busy $used"
  return 1
}

start_service() {
  local mode=$1
  local script="$ROOT/scripts/start_on_mbt.sh"
  [ "$mode" = off ] && script="$ROOT/scripts/start_off_mbt.sh"
  local logf="$LOGS/accuracy_serve_${mode}_4096.log"
  log "starting $mode"
  setsid nohup bash "$script" 4096 > "$logf" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  local code
  for i in $(seq 1 75); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$BASE/health" || true)
    [ "$code" = 200 ] && { log "health=200 $mode"; return 0; }
    sleep 10
  done
  log "ERROR health timeout $mode"
  tail -20 "$logf" >> "$OUT/accuracy_full_progress.log"
  stop_service
  return 1
}

stop_service
for MODE in on off; do
  start_service "$MODE" || continue
  log "GSM8K full $MODE start"
  python scripts/dsv41_gsm8k.py --limit 1319 --concurrency 16 --tag "stage_b_${MODE}_4096_full1319" 2>&1 | tee "$LOGS/gsm8k_${MODE}_full1319.log"
  log "GSM8K full $MODE rc=${PIPESTATUS[0]}"
  stop_service
done
log "accuracy full complete"
