#!/bin/bash
# Same load as the isolation run, with the admission reservation lowered: does the
# pool fill up (KV-bound) and do cache hits turn into TTFT?
#   ./benchmark/verify_clip.bash [clip] [c] [sessions per lane]
set -u
cd "$(dirname "$0")/.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
CLIP=${1:-512}; C=${2:-32}; S=${3:-2}; OUT=benchmark/mixed_results; LOG=$OUT/verify_clip.log
for ARM in lru ours; do
  TAG=clip${CLIP}_${ARM}_c$C; SLOG=$OUT/${TAG}_server.log
  echo "$(date +%T) $TAG start" | tee -a $LOG
  HOST=8 CLIP=$CLIP LOG=$SLOG ./start_server.bash $ARM --engine-log info || { echo "$ARM server failed" | tee -a $LOG; exit 1; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes coding_agent=$C --sessions $S --warmup 1 --server-log $SLOG > $OUT/$TAG.out 2>&1
  grep "^\[$TAG/" $OUT/$TAG.out | tee -a $LOG
done
bash benchmark/kill_bench.bash >/dev/null
echo "$(date +%T) VERIFY-CLIP-DONE" | tee -a $LOG
