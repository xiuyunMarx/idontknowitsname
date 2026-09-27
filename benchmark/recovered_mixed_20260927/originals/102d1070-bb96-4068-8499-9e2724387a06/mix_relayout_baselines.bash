#!/bin/bash
# HOST=10 mix: continuum_relayout then cachescout_relayout (tags mix20h10_<arm>), into mixed_results/mix/.
set -uo pipefail
cd /home/xiaoyu/idontknowitsname
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MIX=benchmark/mixed_results/mix; MLOG=$MIX/mix.log
LANES=fact_check=4,BFCL_agent=4,coding_agent=12
SESS=coding_agent=6,fact_check=0,BFCL_agent=0
export CLIP=4096 MODEL=Qwen/Qwen3-14B-AWQ KV=31830
stop_server() { pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server|KVFlow_server)|[s]glang::' 2>/dev/null || true; }
on_signal() { stop_server; exit 130; }
trap stop_server EXIT
trap on_signal INT TERM
for ARM in continuum_relayout cachescout_relayout; do
  TAG=mix20h10_${ARM}; SLOG=$MIX/${TAG}_server.log
  echo "$(date '+%m-%d %T') $TAG start lanes=$LANES sessions=$SESS ($MODEL xgrammar CLIP=$CLIP HOST=10 KV=$KV, $ARM)" | tee -a $MLOG
  HOST=10 LOG=$SLOG ./start_server.bash $ARM --engine-log info || { echo "$(date '+%m-%d %T') $TAG server failed" | tee -a $MLOG; stop_server; sleep 5; continue; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions "$SESS" --warmup 1 --server-log $SLOG > $MIX/$TAG.out 2>&1
  RC=$?
  grep "^\[$TAG" $MIX/$TAG.out | tee -a $MLOG
  echo "$(date '+%m-%d %T') $TAG rc=$RC" | tee -a $MLOG
  [[ -d benchmark/mixed_results/cache/$TAG ]] && mv benchmark/mixed_results/cache/$TAG $MIX/
  stop_server; sleep 5
done
echo "$(date '+%m-%d %T') RELAYOUT-BASELINES-DONE" | tee -a $MLOG
