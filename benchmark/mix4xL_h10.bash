#!/bin/bash
# Concurrency sweep of the 4-program mix at HOST=10, one server per arm:
#   ./benchmark/mix4xL_h10.bash <LANES_PER_PROGRAM> <arm>...   e.g.  ./benchmark/mix4xL_h10.bash 2 ours relayout vanilla
# Same mix as mix4x4_hN.bash (fact_check, BFCL_agent, coding_agent, doc_analysis; CLIP=4096, KV=31830, HOST=10,
# Qwen3-14B-AWQ) with L lanes per program instead of 4. The coding program stays the fixed one and keeps the SAME
# total of 24 sessions (inputs 0..23) at every L: sessions per lane = ceil(24/L) = 24,12,8,6,5(->25),4 for
# L=1,2,3,4,5,6, so the steady window (mix_steady_csv --start-idx 12, coding input 12 -> first coding lane's end)
# covers the same inputs at every concurrency; the other three programs follow. L=4 is benchmark/mixed_results/
# mix4x4_h10/ (run 2026-09-27/28), not rerun here. Output benchmark/mixed_results/mix4x<L>_h10/<arm>/ plus
# mix.log, mix4x<L>_h10.csv and mix4x<L>_h10_steady.csv at the top; an arm whose summary.json already exists is
# skipped unless FORCE=1. Environment: HOSTGB (10), TIMEOUT (1800 s per session, as in the 4x4 run), FORCE.
# Queue levels with   nohup bash -c "for L in 1 2 3 5 6; do ./benchmark/mix4xL_h10.bash \$L ours relayout vanilla; done" &
# Cancel with kill -9 while it waits: its TERM trap kills every server.
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
L=$1; shift
HOSTGB=${HOSTGB:-10}; TIMEOUT=${TIMEOUT:-1800}; FORCE=${FORCE:-0}
CODING_TOTAL=24
SPL=$(( (CODING_TOTAL + L - 1) / L ))
MIX=benchmark/mixed_results/mix4x${L}_h${HOSTGB}; MLOG=$MIX/mix.log
LANES=fact_check=$L,BFCL_agent=$L,coding_agent=$L,doc_analysis=$L
SESS=coding_agent=$SPL,fact_check=0,BFCL_agent=0,doc_analysis=0
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
  local TAG=mix4x${L}h${HOSTGB}_${NAME} OUT=$MIX/$NAME SLOG=$MIX/$NAME/server.log
  if [[ -s $OUT/summary.json && $FORCE != 1 ]]; then
    echo "$(date '+%m-%d %T') $TAG exists, skipped (FORCE=1 to rerun)" | tee -a $MLOG; return
  fi
  rm -rf $OUT; mkdir -p $OUT
  echo "$(date '+%m-%d %T') $TAG start lanes=$LANES sessions=$SESS timeout=$TIMEOUT ($MODEL xgrammar CLIP=$CLIP HOST=$HOSTGB KV=$KV, $ARM $*)" | tee -a $MLOG
  HOST=$HOSTGB LOG=$SLOG BENCH_REQUEST_LOG=$PWD/$OUT/requests.jsonl ./start_server.bash $ARM --engine-log info "$@" || { echo "$(date '+%m-%d %T') $TAG server failed" | tee -a $MLOG; stop_server; sleep 5; return; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --timeout $TIMEOUT --server-log $SLOG --out-dir $OUT > $OUT/driver.out 2>&1
  local RC=$?
  grep "^\[$TAG" $OUT/driver.out | tee -a $MLOG
  echo "$(date '+%m-%d %T') $TAG rc=$RC" | tee -a $MLOG
  stop_server; sleep 5
}
for A in "$@"; do case $A in ours_noprefetch) run_mix ours_noprefetch ours --no-promote;; *) run_mix $A $A;; esac; done
echo "$(date '+%m-%d %T') MIX4X${L}-DONE" | tee -a $MLOG
$P -m benchmark.mix_csv --prefix mix4x${L}h${HOSTGB} --dir $MIX --out $MIX/mix4x${L}_h${HOSTGB}.csv >> $MLOG 2>&1
$P -m benchmark.mix_steady_csv --dir $MIX --start-idx 12 >> $MLOG 2>&1
