#!/bin/bash
# Coarse concurrency scan of every application, reorder-only vs ours, to find the
# load at which KV management starts to matter. Sequential on one GPU.
#   ./benchmark/find_regime.bash            (all four)
#   ./benchmark/find_regime.bash coding     (one program: bfcl | coding | finance | fact)
set -u
cd "$(dirname "$0")/.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
LOG=benchmark/mixed_results/find_regime.log
ONLY=${1:-all}

run() {  # $1 sweep module, $2 levels, $3 sessions per lane
  for ARM in lru ours; do
    echo "$(date +%T) $1 $ARM c=$2 sessions=$3 start" | tee -a $LOG
    $P -m benchmark.$1 $ARM --c "$2" --sessions "$3" >> $LOG 2>&1
    echo "$(date +%T) $1 $ARM exit $?" | tee -a $LOG
  done
}

[ "$ONLY" = all ] || [ "$ONLY" = bfcl ]    && run sweep_BFCL       20,40,60        2
[ "$ONLY" = all ] || [ "$ONLY" = coding ]  && run sweep_coding     16,32,48,64     2
[ "$ONLY" = all ] || [ "$ONLY" = finance ] && run sweep_finance    8,16,32         3
if [ "$ONLY" = all ] || [ "$ONLY" = fact ]; then
  # fill the Wikipedia cache at low concurrency first, then repair its failed entries
  echo "$(date +%T) fact cache fill start" | tee -a $LOG
  $P -m benchmark.sweep_fact_check lru --c 4 --sessions 30 >> $LOG 2>&1
  $P -m benchmark.applications.warm_web_cache --program fact_check >> $LOG 2>&1
  run sweep_fact_check 8,14,20 2
fi
echo "$(date +%T) FIND-REGIME-DONE" | tee -a $LOG
