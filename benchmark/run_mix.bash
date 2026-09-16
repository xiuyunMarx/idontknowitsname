#!/bin/bash
# Mixed workload, all four applications, one server per arm; throughput over fixed windows.
#   ./benchmark/run_mix.bash [lanes] [sessions per lane] [arms]
set -u
cd "$(dirname "$0")/.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
LANES=${1:-fact_check=2,coding_agent=2,doc_analysis=5,BFCL_agent=7}; S=${2:-4}; ARMS=${3:-"vanilla relayout ours"}
OUT=benchmark/mixed_results/mix; LOG=$OUT/mix.log
mkdir -p $OUT
N=$(echo "$LANES" | tr ',' '\n' | cut -d= -f2 | paste -sd+ | bc)
for ARM in $ARMS; do
  TAG=mix${N}_${ARM}; SLOG=$OUT/${TAG}_server.log
  echo "$(date +%T) $TAG start lanes=$LANES sessions=$S" | tee -a $LOG
  HOST=8 LOG=$SLOG ./start_server.bash $ARM --engine-log info || { echo "$ARM server failed" | tee -a $LOG; exit 1; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes "$LANES" --sessions $S --warmup 1 --server-log $SLOG > $OUT/$TAG.out 2>&1
  grep "^\[$TAG" $OUT/$TAG.out | tee -a $LOG
done
pkill -9 -f '[s]erver\.server|[s]glang::'
echo "$(date +%T) MIX-DONE" | tee -a $LOG
