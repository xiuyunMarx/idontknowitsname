#!/bin/bash
# After the finance ten-arm sweep: 11-arm mix, four programs x 5 lanes (fact_check, BFCL_agent,
# coding_agent, doc_analysis), coding fixed 6 sessions per lane, the rest follow; HOST=10;
# tags mix4x5h10_<arm>, into mixed_results/mix/. Waits for the sweep driver first.
# Cancel with kill -9 while it waits: its TERM trap kills every server, including the sweep's.
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MIX=benchmark/mixed_results/mix; MLOG=$MIX/mix.log
LANES=fact_check=5,BFCL_agent=5,coding_agent=5,doc_analysis=5
SESS=coding_agent=6,fact_check=0,BFCL_agent=0,doc_analysis=0
export CLIP=4096 MODEL=Qwen/Qwen3-14B-AWQ KV=31830
stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server|KVFlow_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
while kill -0 2290682 2>/dev/null; do sleep 60; done
sleep 10
trap stop_server EXIT
trap on_signal INT TERM
run_mix() {   # <arm name in the tag> <start_server arm> [server flags]
  local NAME=$1 ARM=$2; shift 2
  local TAG=mix4x5h10_${NAME} SLOG=$MIX/mix4x5h10_${NAME}_server.log
  echo "$(date '+%m-%d %T') $TAG start lanes=$LANES sessions=$SESS ($MODEL xgrammar CLIP=$CLIP HOST=10 KV=$KV, $ARM $*)" | tee -a $MLOG
  HOST=10 LOG=$SLOG ./start_server.bash $ARM --engine-log info "$@" || { echo "$(date '+%m-%d %T') $TAG server failed" | tee -a $MLOG; stop_server; sleep 5; return; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --server-log $SLOG > $MIX/$TAG.out 2>&1
  local RC=$?
  grep "^\[$TAG" $MIX/$TAG.out | tee -a $MLOG
  echo "$(date '+%m-%d %T') $TAG rc=$RC" | tee -a $MLOG
  [[ -d benchmark/mixed_results/cache/$TAG ]] && mv benchmark/mixed_results/cache/$TAG $MIX/
  stop_server; sleep 5
}
run_mix ours ours
run_mix relayout relayout
run_mix vanilla vanilla
run_mix kvonly kvonly
run_mix ours_noprefetch ours --no-promote
run_mix continuum continuum
run_mix cachescout cachescout
run_mix kvflow kvflow
run_mix continuum_relayout continuum_relayout
run_mix cachescout_relayout cachescout_relayout
run_mix kvflow_relayout kvflow_relayout
echo "$(date '+%m-%d %T') MIX4X5-DONE" | tee -a $MLOG
$P -m benchmark.mix_csv --prefix mix4x5h10 --out $MIX/mix4x5_h10.csv >> $MLOG 2>&1
