#!/bin/bash
# Mixed workload on Qwen3-14B-AWQ: lanes fact_check=4, BFCL_agent=4, coding_agent=12; coding lanes fixed 6 sessions,
# HOST=16 (97.7k host tokens on 14B), KV=31830, xgrammar, CLIP=4096. Arms: ours then relayout.
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
OUT=benchmark/mixed_results/mix_h16; LOG=$OUT/mix.log
LANES=fact_check=4,BFCL_agent=4,coding_agent=12
SESS=coding_agent=6,fact_check=0,BFCL_agent=0
ARMS=${ARMS:-"ours relayout"}
export CLIP=4096 MODEL=Qwen/Qwen3-14B-AWQ KV=31830
mkdir -p $OUT
stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
trap stop_server EXIT
trap on_signal INT TERM
for ARM in $ARMS; do
  TAG=mix20h16_${ARM}; SLOG=$OUT/${TAG}_server.log
  echo "$(date +%T) $TAG start lanes=$LANES sessions=$SESS ($MODEL xgrammar CLIP=$CLIP HOST=16 KV=$KV)" | tee -a $LOG
  HOST=16 LOG=$SLOG ./start_server.bash $ARM --engine-log info || { echo "$(date +%T) $TAG server failed" | tee -a $LOG; stop_server; sleep 5; continue; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --server-log $SLOG > $OUT/$TAG.out 2>&1
  RC=$?
  grep "^\[$TAG" $OUT/$TAG.out | tee -a $LOG
  echo "$(date +%T) $TAG rc=$RC" | tee -a $LOG
  stop_server; sleep 5
done
echo "$(date +%T) MIX-DONE" | tee -a $LOG
