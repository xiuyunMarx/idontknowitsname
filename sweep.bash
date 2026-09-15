#!/bin/bash
# Concurrency sweep of one workload, fresh server per (arm, level), level-major.
#
#   ./sweep.bash fact            HoVer fact_check (benchmark/fact_check/fact_check.jac)
#   ./sweep.bash coding          HumanEval coding agent (benchmark/coding/coding_agent.jac)
#
# Arms (ARMS="..."):
#   lruraw     --lru --no-relayout   raw SGLang HiCache LRU, prompts as the program lays them out
#   lru        --lru                 LRU + static-analysis re-layout
#   kvonly     --no-relayout         planner only (promotion, steering, retirement), no re-layout
#   ours                             re-layout + planner
#   cachescout                       server.cacheScout_server: opaque requests, online Markov agent model
#   continuum                        server.continuum_server: opaque requests, TTL pins, plain FCFS
#
# Every session is one `jac run` of the program (benchmark.fact_check.fact_bench drives both
# workloads); the driver registers the program's static analysis with the server before the
# sessions, so each request's call site resolves to its template. Header-last is a per-program
# choice made at registration: fact_check keeps its headers in front (NO_HEADER_LAST=1, the
# current model's output changes otherwise), coding_agent may go header-last.
#
# Environment overrides (defaults per workload below):
#   MODEL HOST KV        model, host KV tier GB (--host), device pool cap in tokens (--kv, 14B-AWQ needs it)
#   ARMS LEVELS          arms and concurrency levels; PER_LANE sessions per lane (default per level)
#   OUT CSV              results dir (results/{fact,coding}_sweep) and the CSV written at the end
#   EXTRA SCHED RISK_AGING  extra server.server flags, engine queue order (fcfs|lpm|risk), risk aging s
#   NO_HEADER_LAST       1 = register the program with headers in front (fact default 1, coding 0)
#   WARMUP               sequential warm-up sessions per arm, excluded from the numbers (2)
#
#   LEVELS="8 12 16" ARMS="lru ours" ./sweep.bash fact
#   LEVELS="32 40 48" ./sweep.bash coding
set -u
cd "$(dirname "$0")"
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
WORKLOAD=${1:-}
case "$WORKLOAD" in
  fact)
    PROGRAM=${PROGRAM:-benchmark/fact_check/fact_check.jac}
    INPUTS=${INPUTS:-benchmark/fact_check/hover_claims_120.tsv}; INPUT_ENV=FC_CLAIM
    LEVELS=${LEVELS:-"4 6 8 10 12 14 16 18"}
    OUT=${OUT:-results/fact_sweep}; NO_HEADER_LAST=${NO_HEADER_LAST:-1}
    export FC_CACHE_DIR=$PWD/benchmark/fact_check/wiki_cache
    export FC_TOOL_DELAY_S=${FC_TOOL_DELAY_S:-2}
    export FC_MIN_ROUNDS=${FC_MIN_ROUNDS:-2}
    per_lane() { [ -n "${PER_LANE:-}" ] && { echo "$PER_LANE"; return; }; case $1 in 1) echo 24;; 2) echo 12;; *) echo 6;; esac; }   # 118 usable claims cap c*N
    ;;
  coding)
    PROGRAM=${PROGRAM:-benchmark/coding/coding_agent.jac}
    INPUTS=${INPUTS:-benchmark/coding/tasks.txt}; INPUT_ENV=CA_TASKS
    LEVELS=${LEVELS:-"16 20 24 28 32 40 48"}
    OUT=${OUT:-results/coding_sweep}; NO_HEADER_LAST=${NO_HEADER_LAST:-0}
    export CA_TOOL_DELAY_S=${CA_TOOL_DELAY_S:-2}
    per_lane() { [ -n "${PER_LANE:-}" ] && { echo "$PER_LANE"; return; }; case $1 in 12) echo 2;; *) echo 1;; esac; }   # 64 bundles cap warmup + c*N
    ;;
  *) echo "usage: $0 fact|coding   (env: MODEL HOST KV ARMS LEVELS PER_LANE OUT CSV EXTRA SCHED RISK_AGING NO_HEADER_LAST WARMUP)" >&2; exit 2;;
esac
MODEL=${MODEL:-Qwen/Qwen3-8B}; HOST=${HOST:-16}; KV=${KV:-}; WARMUP=${WARMUP:-2}
ARMS=${ARMS:-"lruraw lru ours"}
EXTRA=${EXTRA:-}; SCHED=${SCHED:-fcfs}; RISK_AGING=${RISK_AGING:-}
mkdir -p "$OUT"; CSV=${CSV:-$OUT/sweep.csv}
N_INPUTS=$(grep -cv '^#' "$INPUTS")

log() { echo "$(date +%T) $*" | tee -a "$OUT/progress.log"; }
kill_srv() { pkill -9 -f "[s]glang::|[s]erver\.server|[c]acheScout_server|[c]ontinuum_server|[f]act_bench|[j]ac run " 2>/dev/null; sleep 5; }

run_arm() {  # $1 arm, $2 server flags, $3 concurrency
  kill_srv
  local mod=server.server extra="$EXTRA --sched $SCHED ${RISK_AGING:+--risk-aging-s $RISK_AGING}" reg=""
  [ "$1" = cachescout ] && { mod=server.cacheScout_server; extra=; reg="--no-register"; }
  [ "${1%%_nopin*}" = continuum ] && { mod=server.continuum_server; extra=; reg="--no-register"; }
  nohup $P -m $mod "$MODEL" $2 $extra --host "$HOST" ${KV:+--kv $KV} > "$OUT/$1_c$3.log" 2>&1 &
  local sp=$!
  until grep -q "\[warmup\] engine ready" "$OUT/$1_c$3.log" 2>/dev/null; do
    if ! kill -0 $sp 2>/dev/null; then log "$1 c=$3 server died"; tail -5 "$OUT/$1_c$3.log"; return 1; fi; sleep 3; done
  local n; n=$(per_lane "$3")
  if [ $((n * $3 + WARMUP)) -gt "$N_INPUTS" ]; then log "$1 c=$3 skipped: $((n * $3 + WARMUP)) sessions need more than the $N_INPUTS inputs in $INPUTS"; return 1; fi
  log "$1 c=$3 start sessions=$((n * $3)) ($n/lane) host=${HOST}GB $(grep -o '\[ledger\] device=[0-9]* tokens host=[0-9]* tokens' "$OUT/$1_c$3.log")"
  $P -m benchmark.fact_check.fact_bench --program "$PROGRAM" --claim-env "$INPUT_ENV" --claims "$INPUTS" \
     --sessions "$n" --concurrency "$3" --warmup "$WARMUP" $reg $([ "$NO_HEADER_LAST" = 1 ] && echo --no-header-last) \
     --tag "$1" --server-log "$OUT/$1_c$3.log" --logdir "$OUT/run_$1_c$3" > "$OUT/$1_c$3.out" 2>&1
  log "$1 c=$3 bench exit $? $(grep -o 'JCT p50=.*accuracy=[0-9/]*' "$OUT/$1_c$3.out")"
}

log "SWEEP $WORKLOAD model=$MODEL host=${HOST}GB kv=${KV:-default} arms=[$ARMS] levels=[$LEVELS] no_header_last=$NO_HEADER_LAST"
for c in $LEVELS; do
  for arm in $ARMS; do
    case $arm in
      lruraw) run_arm lruraw "--lru --no-relayout" "$c";;
      lru)    run_arm lru "--lru" "$c";;
      kvonly) run_arm kvonly "--no-relayout" "$c";;
      ours)   run_arm ours "" "$c";;
      cachescout) run_arm cachescout "" "$c";;
      continuum) run_arm continuum "" "$c";;
      continuum_nopin) run_arm continuum_nopin "--no-pin" "$c";;
      continuum_nopin_lru) run_arm continuum_nopin_lru "--no-pin --eviction lru" "$c";;
      *) log "unknown arm $arm";;
    esac
  done
done
kill_srv
$P -m benchmark.fact_check.sweep_csv "$CSV" "q8b:$MODEL:$OUT" | tee -a "$OUT/progress.log"
log SWEEP-DONE
