#!/bin/bash
# fact_check concurrency sweep, three arms per level, fresh server per (arm, level), level-major:
#   lruraw  --lru --no-relayout   raw SGLang HiCache LRU
#   lru     --lru                 LRU + IR-driven prompt re-layout (per-site header-last)
#   kvonly  --no-relayout         planner only (promotion, steering, retirement), no re-layout
#   ours                          re-layout + planner
#   cachescout                        server.cacheScout_server: opaque requests, online Markov agent model (no EXTRA flags)
# Same protocol as the Sep 3 qwen8 sweep: 120 HoVer claims, FC_TOOL_DELAY_S=2, rounds per fact_check.jac,
# --sessions per lane (6 above c=2); host 8 GB (54k tokens) to match coding_sweep.bash. Results in results/fact_sweep/{arm}_c{N}.{log,out}, fact_sweep.csv.
#   LEVELS="8 12 16" ARMS="lru ours" ./fact_sweep.bash
set -u
cd "$(dirname "$0")/.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MODEL=${MODEL:-Qwen/Qwen3-8B}; HOST=${HOST:-8}; KV=${KV:-}; WARMUP=2   # KV: device pool cap in tokens (--kv), needed for 14B-AWQ
ARMS=${ARMS:-"lruraw lru ours"}; LEVELS=${LEVELS:-"4 6 8 10 12 14 16 18"}
OUT=${OUT:-results/fact_sweep}; mkdir -p "$OUT"
CSV=${CSV:-fact_sweep.csv}
CLAIMS=${CLAIMS:-benchmark/fact_check/hover_claims_120.tsv}
PROGRAM=${PROGRAM:-}   # alternative .jac to drive (fact_bench --program), default fact_check.jac
EXTRA=${EXTRA:-}       # server flags added to every arm, e.g. --no-header-last
SCHED=${SCHED:-fcfs}   # engine queue order for every arm: fcfs (request priorities), lpm, or risk (cache-risk)
RISK_AGING=${RISK_AGING:-}   # --sched risk: aging seconds (server.server arms only)
export FC_CACHE_DIR=$PWD/benchmark/fact_check/wiki_cache
export FC_TOOL_DELAY_S=${FC_TOOL_DELAY_S:-2}

per_lane() { [ -n "${PER_LANE:-}" ] && { echo "$PER_LANE"; return; }; case $1 in 1) echo 24;; 2) echo 12;; *) echo 6;; esac; }   # 118 usable claims cap c*N
log() { echo "$(date +%T) $*" | tee -a "$OUT/progress.log"; }
kill_srv() { pkill -9 -f "[s]glang::|[s]erver\.server|[c]acheScout_server|[f]act_bench|[j]ac run " 2>/dev/null; sleep 5; }

run_arm() {  # $1 arm, $2 server flags, $3 concurrency
  kill_srv
  local mod=server.server extra="$EXTRA ${RISK_AGING:+--risk-aging-s $RISK_AGING}"; [ "$1" = cachescout ] && { mod=server.cacheScout_server; extra=; }
  nohup $P -m $mod "$MODEL" $2 $extra --sched "$SCHED" --host "$HOST" ${KV:+--kv $KV} > "$OUT/$1_c$3.log" 2>&1 &
  local sp=$!
  until grep -q "\[warmup\] engine ready" "$OUT/$1_c$3.log" 2>/dev/null; do
    if ! kill -0 $sp 2>/dev/null; then log "$1 c=$3 server died"; tail -5 "$OUT/$1_c$3.log"; return 1; fi; sleep 3; done
  local n; n=$(per_lane "$3")
  log "$1 c=$3 start sessions=$((n * $3)) ($n/lane) host=${HOST}GB $(grep -o '\[ledger\] device=[0-9]* tokens host=[0-9]* tokens' "$OUT/$1_c$3.log")"
  $P -m benchmark.fact_check.fact_bench ${PROGRAM:+--program $PROGRAM} --sessions "$n" --concurrency "$3" --warmup $WARMUP --claims "$CLAIMS" \
     --tag "$1" --server-log "$OUT/$1_c$3.log" --logdir "$OUT/run_$1_c$3" > "$OUT/$1_c$3.out" 2>&1
  log "$1 c=$3 bench exit $? $(grep -o 'JCT p50=.*accuracy=[0-9/]*' "$OUT/$1_c$3.out")"
}

for c in $LEVELS; do
  for arm in $ARMS; do
    case $arm in
      lruraw) run_arm lruraw "--lru --no-relayout" "$c";;
      lru)    run_arm lru "--lru" "$c";;
      kvonly) run_arm kvonly "--no-relayout" "$c";;
      ours)   run_arm ours "" "$c";;
      cachescout) run_arm cachescout "" "$c";;
    esac
  done
done
kill_srv
$P -m benchmark.fact_check.sweep_csv "$CSV" "q8b:$MODEL:$OUT" | tee -a "$OUT/progress.log"
log SWEEP-DONE
