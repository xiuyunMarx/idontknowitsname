#!/bin/bash
# 4x4 mix at one host size, one server per arm:
#   ./benchmark/mix4x4_hN.bash <HOST_GB> <arm>...      e.g.  ./benchmark/mix4x4_hN.bash 12 ours relayout vanilla
# lanes fact_check=4,BFCL_agent=4,coding_agent=4,doc_analysis=4; coding fixed at 6 sessions per lane, the rest
# follow; CLIP=4096, KV=31830, Qwen3-14B-AWQ. Output mixed_results/mix4x4_h<HOST_GB>/<arm>/ (server.log,
# driver.out, sessions.jsonl, session_events.jsonl, requests.jsonl, summary.json), mix.log and
# mix4x4_h<HOST_GB>.csv at the top; then `python -m benchmark.mix_steady_csv --dir <that dir>` for the steady table.
# Host sweep 2026-09-28/29 = HOST 10/12/14/16 x (ours relayout vanilla), ~3.7 h per level. Queue levels with
#   nohup bash -c "for H in 18 20; do ./benchmark/mix4x4_hN.bash \$H ours relayout vanilla; done" &
# Cancel with kill -9 while it waits: its TERM trap kills every server.
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
HOSTGB=$1; shift; MIX=benchmark/mixed_results/mix4x4_h${HOSTGB}; MLOG=$MIX/mix.log
LANES=fact_check=4,BFCL_agent=4,coding_agent=4,doc_analysis=4
SESS=coding_agent=6,fact_check=0,BFCL_agent=0,doc_analysis=0
export CLIP=4096 MODEL=Qwen/Qwen3-14B-AWQ KV=31830
stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server|KVFlow_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
trap stop_server EXIT
trap on_signal INT TERM
mkdir -p $MIX
git rev-parse HEAD > $MIX/git_head.txt
git diff HEAD -- server model static_analysis benchmark/mixed_workload.py benchmark/applications > $MIX/working_tree.patch
grep -n "PROMOTE_MAX_USAGE = " server/kv_planner.py | head -1 > $MIX/promote_max_usage.txt
run_mix() {   # <arm name in the tag> <start_server arm> [server flags]
  local NAME=$1 ARM=$2; shift 2
  local TAG=mix4x4h${HOSTGB}_${NAME} OUT=$MIX/$NAME SLOG=$MIX/$NAME/server.log
  mkdir -p $OUT
  echo "$(date '+%m-%d %T') $TAG start lanes=$LANES sessions=$SESS ($MODEL xgrammar CLIP=$CLIP HOST=$HOSTGB KV=$KV, $ARM $*)" | tee -a $MLOG
  HOST=$HOSTGB LOG=$SLOG BENCH_REQUEST_LOG=$PWD/$OUT/requests.jsonl ./start_server.bash $ARM --engine-log info "$@" || { echo "$(date '+%m-%d %T') $TAG server failed" | tee -a $MLOG; stop_server; sleep 5; return; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --server-log $SLOG --out-dir $OUT > $OUT/driver.out 2>&1
  local RC=$?
  grep "^\[$TAG" $OUT/driver.out | tee -a $MLOG
  echo "$(date '+%m-%d %T') $TAG rc=$RC" | tee -a $MLOG
  stop_server; sleep 5
}
for A in "$@"; do case $A in ours_noprefetch) run_mix ours_noprefetch ours --no-promote;; *) run_mix $A $A;; esac; done
echo "$(date '+%m-%d %T') MIX4X4-DONE" | tee -a $MLOG
$P -m benchmark.mix_csv --prefix mix4x4h${HOSTGB} --dir $MIX --out $MIX/mix4x4_h${HOSTGB}.csv >> $MLOG 2>&1
