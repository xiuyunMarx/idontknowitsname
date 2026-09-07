#!/bin/bash
# Qwen3-8B concurrency sweep, LRU vs ours, fresh server per (arm, level), level-major order.
# Benchmark (v2): 120 HoVer claims, rounds stop as soon as the verdict is decided
# (FC_MIN_ROUNDS=2), every web search costs >= FC_TOOL_DELAY_S seconds, fact_bench
# --sessions is PER LANE (total = concurrency * sessions) so lanes stay busy all run.
#   LEVELS="8 6 4" ARMS="lru ours" ./qwen8_sweep.bash
# Outputs: $OUT/{arm}_c{N}.{log,out} and $CSV (default qwen8_sweep.csv) in the repo root.
set -u
cd "$(dirname "$0")"
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MODEL=${MODEL:-Qwen/Qwen3-8B}; HOST=${HOST:-16}; WARMUP=2
ARMS=${ARMS:-"lru ours"}; LEVELS=${LEVELS:-"8 6 4"}
OURS_FLAGS=${OURS_FLAGS:-""}          # extra server flags for the ours arm only
OUT=${OUT:-results/qwen8_sweep_v2}; mkdir -p "$OUT"
CSV=${CSV:-qwen8_sweep.csv}
CLAIMS=${CLAIMS:-benchmark/fact_check/hover_claims_120.tsv}
export FC_CACHE_DIR=$PWD/benchmark/fact_check/wiki_cache
export FC_MIN_ROUNDS=${FC_MIN_ROUNDS:-2}        # stop when the verdict is decided (2..6 rounds)
export FC_TOOL_DELAY_S=${FC_TOOL_DELAY_S:-2}    # simulated search latency, cache hits included

per_lane() { case $1 in 1) echo 24;; 2) echo 12;; *) echo 6;; esac; }   # 118 usable claims cap c*N
log() { echo "$(date +%T) $*" | tee -a "$OUT/progress.log"; }
kill_srv() { pkill -9 -f "[s]glang::|[s]erver\.server|[f]act_bench|[j]ac run " 2>/dev/null; sleep 5; }

run_arm() {  # $1 tag, $2 server flags, $3 concurrency
  kill_srv
  nohup $P -m server.server "$MODEL" $2 --host "$HOST" > "$OUT/$1.log" 2>&1 &
  local sp=$!
  until grep -q "\[warmup\] engine ready" "$OUT/$1.log" 2>/dev/null; do
    if ! kill -0 $sp 2>/dev/null; then log "$1 server died"; tail -5 "$OUT/$1.log"; return 1; fi; sleep 3; done
  local n; n=$(per_lane "$3")
  log "$1 start sessions=$((n * $3)) ($n/lane) min_rounds=$FC_MIN_ROUNDS delay=${FC_TOOL_DELAY_S}s $(grep -o '\[ledger\] device=[0-9]* tokens host=[0-9]* tokens' "$OUT/$1.log")"
  $P -m benchmark.fact_check.fact_bench --sessions "$n" --concurrency "$3" --warmup $WARMUP --claims "$CLAIMS" \
     --tag "$1" --server-log "$OUT/$1.log" --logdir "$OUT/run_$1" > "$OUT/$1.out" 2>&1
  log "$1 bench exit $? $(grep -o 'JCT p50=.*accuracy=[0-9/]*' "$OUT/$1.out")"
}

for c in $LEVELS; do            # level-major: both arms of one level back to back
  for arm in $ARMS; do
    case $arm in
      lru)  run_arm "lru_c$c"  "--lru" "$c";;
      ours) run_arm "ours_c$c" "$OURS_FLAGS" "$c";;
    esac
  done
done
kill_srv
$P -m benchmark.fact_check.sweep_csv "$CSV" "q8b:$MODEL:$OUT" | tee -a "$OUT/progress.log"
log "sweep done"
