#!/bin/bash
# Ablation (user 2026-09-22): ours without prefetch (server flag --no-promote: bands + protection + admission only)
# on the HOST=10 mix, tag mix20h10_ours_noprefetch. Waits for the bfcl_c4 driver (end of the current queue).
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
S=/tmp/claude-1007/-home-xiaoyu-idontknowitsname/102d1070-bb96-4068-8499-9e2724387a06/scratchpad
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MIX=benchmark/mixed_results/mix; MLOG=$MIX/mix.log
LANES=fact_check=4,BFCL_agent=4,coding_agent=12
SESS=coding_agent=6,fact_check=0,BFCL_agent=0
export CLIP=4096 MODEL=Qwen/Qwen3-14B-AWQ KV=31830
stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
trap stop_server EXIT
trap on_signal INT TERM
while kill -0 "$(cat $S/bfcl_c4.pid)" 2>/dev/null; do sleep 60; done
sleep 10
TAG=mix20h10_ours_noprefetch; SLOG=$MIX/${TAG}_server.log
echo "$(date '+%m-%d %T') $TAG start lanes=$LANES sessions=$SESS ($MODEL xgrammar CLIP=$CLIP HOST=10 KV=$KV, ours --no-promote)" | tee -a $MLOG
HOST=10 LOG=$SLOG ./start_server.bash ours --engine-log info --no-promote || { echo "$(date '+%m-%d %T') $TAG server failed" | tee -a $MLOG; exit 1; }
$P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --server-log $SLOG > $MIX/$TAG.out 2>&1
RC=$?
grep "^\[$TAG" $MIX/$TAG.out | tee -a $MLOG
echo "$(date '+%m-%d %T') $TAG rc=$RC" | tee -a $MLOG
[[ -d benchmark/mixed_results/cache/$TAG ]] && mv benchmark/mixed_results/cache/$TAG $MIX/
stop_server; sleep 5
echo "$(date '+%m-%d %T') NOPF-DONE" | tee -a $MLOG
