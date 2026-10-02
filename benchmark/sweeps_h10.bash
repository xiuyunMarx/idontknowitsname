#!/bin/bash
# The four per-agent concurrency sweeps of one arm on one server, in the configuration of the paper's sweeps
# (benchmark/mixed_results/sweeps/*_sweep_<arm>.csv, 2026-09-19 six-arm rerun): HOST=10, CLIP=4096, KV=31830,
# Qwen3-14B-AWQ; fact c=4,6,8,10,12 k=6; coding c=24,32,40,48,52 k=3; BFCL c=4,8,16,24,32 k=3; finance
# c=4,8,12,16,20 k=6 (k = sessions per lane). One server per arm, workloads back to back, no flush in between,
# like benchmark/run_sweeps.bash.
#   ./benchmark/sweeps_h10.bash <arm> [arm ...]
# Progress: benchmark/mixed_results/sweeps/progress.log; per-level driver output sweeps/logs/<tag>.out.
set -uo pipefail
cd "$(dirname "$0")/.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
OUT=benchmark/mixed_results/sweeps; LOG=$OUT/progress.log
export CLIP=4096 MODEL=Qwen/Qwen3-14B-AWQ KV=31830
mkdir -p $OUT/logs
stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server|KVFlow_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
trap stop_server EXIT
trap on_signal INT TERM
for ARM in "$@"; do
  SLOG=$OUT/logs/sweep_${ARM}_server.log
  echo "$(date '+%m-%d %T') $ARM sweeps server start (HOST=10 CLIP=$CLIP KV=$KV)" | tee -a $LOG
  HOST=10 LOG=$SLOG ./start_server.bash $ARM --engine-log info || { echo "$(date '+%m-%d %T') $ARM server failed" | tee -a $LOG; continue; }
  while read -r SWEEP C K; do
    echo "$(date '+%m-%d %T') $SWEEP $ARM start (HOST=10 c=$C k=$K)" | tee -a $LOG
    $P -m benchmark.sweep_$SWEEP $ARM --c $C --sessions $K --server-log $SLOG > $OUT/logs/sweep_${SWEEP}_${ARM}.out 2>&1
    RC=$?
    grep "^\[sweep\]" $OUT/logs/sweep_${SWEEP}_${ARM}.out | tee -a $LOG
    echo "$(date '+%m-%d %T') $SWEEP $ARM rc=$RC" | tee -a $LOG
  done <<'LEVELS'
coding 24,32,40,48,52 3
fact_check 4,6,8,10,12 6
BFCL 4,8,16,24,32 3
finance 4,8,12,16,20 6
LEVELS
  stop_server; sleep 5
done
echo "$(date '+%m-%d %T') SWEEPS-DONE $*" | tee -a $LOG
