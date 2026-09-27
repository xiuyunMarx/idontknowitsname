#!/bin/bash
# After the sweep6_h10 driver ends (ALL-DONE): mix HOST=10 on continuum WITH program-level scheduling (--program-fcfs),
# tag mix20h10_continuum_sched, same lanes/rules as the other mix arms; driver dir moved into mixed_results/mix/.
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
S=/tmp/claude-1007/-home-xiaoyu-idontknowitsname/102d1070-bb96-4068-8499-9e2724387a06/scratchpad
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
OUT=benchmark/mixed_results/mix; LOG=$OUT/mix.log
LANES=fact_check=4,BFCL_agent=4,coding_agent=12
SESS=coding_agent=6,fact_check=0,BFCL_agent=0
export CLIP=4096 MODEL=Qwen/Qwen3-14B-AWQ KV=31830
stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
trap stop_server EXIT
trap on_signal INT TERM
while kill -0 "$(cat $S/sweep6_h10.pid)" 2>/dev/null; do sleep 60; done
sleep 10
TAG=mix20h10_continuum_sched; SLOG=$OUT/${TAG}_server.log
echo "$(date +%T) $TAG start lanes=$LANES sessions=$SESS ($MODEL xgrammar CLIP=$CLIP HOST=10 KV=$KV, continuum --program-fcfs)" | tee -a $LOG
HOST=10 LOG=$SLOG ./start_server.bash continuum --engine-log info --program-fcfs || { echo "$(date +%T) $TAG server failed" | tee -a $LOG; stop_server; exit 1; }
$P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --server-log $SLOG > $OUT/$TAG.out 2>&1
RC=$?
grep "^\[$TAG" $OUT/$TAG.out | tee -a $LOG
echo "$(date +%T) $TAG rc=$RC" | tee -a $LOG
[[ -d benchmark/mixed_results/cache/$TAG ]] && mv benchmark/mixed_results/cache/$TAG $OUT/
stop_server; sleep 5
echo "$(date +%T) CSCHED-DONE" | tee -a $LOG
