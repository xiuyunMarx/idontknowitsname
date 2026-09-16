#!/bin/bash
# The four concurrency sweeps, one server per arm: start the arm, sweep coding,
# fact_check, finance and BFCL on it, stop it, next arm.
#   ./benchmark/run_sweeps.bash [arm ...]       default: ours relayout vanilla
# Progress: benchmark/mixed_results/sweeps/progress.log (one line per level, SWEEPS-DONE at the end).
# Environment: PYTHON (interpreter), HOST_GB (host KV tier; default 8).
set -uo pipefail
cd "$(dirname "$0")/.."
P=${PYTHON:-/home/xiaoyu/miniconda3/envs/sglang/bin/python}
HOST_GB=${HOST_GB:-8}
ARMS=${*:-"ours relayout vanilla"}
OUT=benchmark/mixed_results/sweeps
LOG=$OUT/progress.log
mkdir -p $OUT/logs

stop_server() {
  pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server)|[s]glang::' 2>/dev/null || true
}
on_signal() {
  stop_server
  exit 130
}
trap stop_server EXIT
trap on_signal INT TERM

for ARM in $ARMS; do
  SLOG=$OUT/logs/sweep_${ARM}_server.log
  echo "$(date +%T) $ARM server start" | tee -a $LOG
  HOST=$HOST_GB LOG=$SLOG ./start_server.bash $ARM || { echo "$(date +%T) $ARM server failed" | tee -a $LOG; continue; }
  for SWEEP in coding fact_check finance BFCL; do
    echo "$(date +%T) $ARM $SWEEP start" | tee -a $LOG
    $P -m benchmark.sweep_$SWEEP $ARM --server-log $SLOG > $OUT/logs/sweep_${SWEEP}_${ARM}.out 2>&1
    RC=$?
    grep "^\[sweep\]" $OUT/logs/sweep_${SWEEP}_${ARM}.out | tee -a $LOG
    echo "$(date +%T) $ARM $SWEEP rc=$RC" | tee -a $LOG
  done
  stop_server
  sleep 5
done
echo "$(date +%T) SWEEPS-DONE" | tee -a $LOG
