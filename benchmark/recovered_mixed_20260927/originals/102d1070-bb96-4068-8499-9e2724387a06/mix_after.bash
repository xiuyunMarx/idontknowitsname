#!/bin/bash
# Wait for the sweep driver (sweep_all.bash) to finish, then run the mixed workload on every arm.
#   lanes fact_check=4, BFCL_agent=4, coding_agent=12; coding lanes fixed 6 sessions, fact/BFCL follow;
#   HOST=8, CLIP=4096. Output: benchmark/mixed_results/mix/ (mix.log, mix20_<arm>.out, server logs).
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
S=/tmp/claude-1007/-home-xiaoyu-idontknowitsname/102d1070-bb96-4068-8499-9e2724387a06/scratchpad
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
OUT=benchmark/mixed_results/mix; LOG=$OUT/mix.log
LANES=fact_check=4,BFCL_agent=4,coding_agent=12
SESS=coding_agent=6,fact_check=0,BFCL_agent=0
ARMS=${ARMS:-"ours vanilla relayout kvonly cachescout continuum"}
export CLIP=4096
mkdir -p $OUT

stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
trap stop_server EXIT
trap on_signal INT TERM

SWEEP_PID=$(cat $S/sweep_all.pid)
while kill -0 "$SWEEP_PID" 2>/dev/null; do sleep 60; done
echo "$(date +%T) sweeps finished (driver pid $SWEEP_PID gone); starting mix" | tee -a $LOG
sleep 10

for ARM in $ARMS; do
  TAG=mix20_${ARM}; SLOG=$OUT/${TAG}_server.log
  echo "$(date +%T) $TAG start lanes=$LANES sessions=$SESS (CLIP=$CLIP HOST=8)" | tee -a $LOG
  HOST=8 LOG=$SLOG ./start_server.bash $ARM --engine-log info || { echo "$(date +%T) $TAG server failed" | tee -a $LOG; stop_server; sleep 5; continue; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --server-log $SLOG > $OUT/$TAG.out 2>&1
  RC=$?
  grep "^\[$TAG" $OUT/$TAG.out | tee -a $LOG
  echo "$(date +%T) $TAG rc=$RC" | tee -a $LOG
  stop_server; sleep 5
done
echo "$(date +%T) MIX-DONE" | tee -a $LOG
