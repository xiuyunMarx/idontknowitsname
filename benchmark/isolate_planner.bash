#!/bin/bash
# Which part of the planner costs TTFT on coding at one concurrency: four servers, same load.
#   ./benchmark/isolate_planner.bash [c] [sessions per lane]
# arms: relayout (re-layout only) | prio (relayout arm on the priority eviction policy, no planner)
#       | hold (planner steers eviction, no promotions) | ours (full)
set -u
cd "$(dirname "$0")/.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
C=${1:-32}; S=${2:-2}; OUT=benchmark/mixed_results; LOG=$OUT/isolate.log

arm() {  # $1 name, $2 start_server arm, $3 extra server flags
  local TAG=isolate_$1_c$C SLOG=$OUT/isolate_$1_server.log
  echo "$(date +%T) $TAG start" | tee -a $LOG
  HOST=8 LOG=$SLOG ./start_server.bash $2 $3 || { echo "$1 server failed" | tee -a $LOG; return 1; }
  $P -m benchmark.mixed_workload --tag $TAG --lanes coding_agent=$C --sessions $S --warmup 1 --server-log $SLOG > $OUT/$TAG.out 2>&1
  grep "^\[$TAG/" $OUT/$TAG.out | tee -a $LOG
}

arm relayout relayout "--engine-log info"
arm prio     relayout "--eviction priority --engine-log info"
arm hold     ours     "--no-promote --engine-log info"
arm ours     ours     "--engine-log info"
bash benchmark/kill_bench.bash >/dev/null
echo "$(date +%T) ISOLATE-DONE" | tee -a $LOG
