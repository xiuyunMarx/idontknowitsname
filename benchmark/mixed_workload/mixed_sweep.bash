#!/bin/bash
# Mixed-workload sweep: fact_check and coding_agent sessions interleaved on one server, fresh server per (arm, mix).
#   lruraw  --lru --no-relayout   raw SGLang HiCache LRU (recorded as `sglang`)
#   lru     --lru                 LRU + IR-driven prompt re-layout (recorded as `reorder-only`)
#   kvonly  --no-relayout         planner only (promotion, steering, retirement), no re-layout
#   ours                          re-layout + planner
#   cachescout                    server.cacheScout_server: opaque requests, online Markov agent model
#   continuum                     server.continuum_server: opaque requests, TTL pins, plain FCFS
# MIXES are fact_lanes+code_lanes pairs. Coding lanes run a fixed count (7 per lane, shrunk to fit the 64
# bundles); fact lanes "follow": they keep taking fresh claims until the coding lanes finish (see size_lanes,
# and run.py). Header-last is declared per program in the .jac (fact_check.jac: extra_body no_header_last),
# not by a server flag, so EXTRA stays empty for the mixed protocol. Host 8 GB (54k tokens) as in coding_sweep.bash.
# Results in results/mixed_sweep/{arm}_f{F}c{C}.{log,out}; consolidate with
#   python -m benchmark.mixed_workload.sweep_csv mixed_sweep.csv q8b:Qwen/Qwen3-8B:results/mixed_sweep
#   MIXES="4+12 8+8" ARMS="lruraw lru ours" ./benchmark/mixed_workload/mixed_sweep.bash   (any cwd)
set -u
cd "$(dirname "$0")/../.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MODEL=${MODEL:-Qwen/Qwen3-8B}; HOST=${HOST:-8}; KV=${KV:-}; WARMUP=${WARMUP:-2}
ARMS=${ARMS:-"lruraw lru ours"}; MIXES=${MIXES:-"4+12 8+8"}
FACT_SESSIONS=${FACT_SESSIONS:-}; CODE_SESSIONS=${CODE_SESSIONS:-}   # per lane; empty = auto-size per mix (see size_lanes)
N_CLAIMS=$(grep -cv '^#' benchmark/mixed_workload/hover_claims_120.tsv); N_BUNDLES=$(grep -cv '^#' benchmark/coding/tasks.txt)
OUT=${OUT:-results/mixed_sweep}; mkdir -p "$OUT"
EXTRA=${EXTRA:-}       # server flags added to every server.server arm, e.g. --no-header-last
SCHED=${SCHED:-fcfs}
export FC_CACHE_DIR=$PWD/benchmark/fact_check/wiki_cache
export FC_TOOL_DELAY_S=${FC_TOOL_DELAY_S:-2}
export FC_MIN_ROUNDS=${FC_MIN_ROUNDS:-2}
export CA_TOOL_DELAY_S=${CA_TOOL_DELAY_S:-2}

size_lanes() {  # $1 fact lanes, $2 code lanes -> FS CS: code 7/lane capped by the 64 bundles; fact follows (0)
  # unless FACT_SESSIONS is set, in which case it is capped by the 118 claims. Explicit CODE_SESSIONS wins.
  CS=${CODE_SESSIONS:-7}; [ $((CS * $2 + WARMUP)) -gt "$N_BUNDLES" ] && CS=$(( (N_BUNDLES - WARMUP) / $2 ))
  FS=${FACT_SESSIONS:-0}
  [ "$FS" -gt 0 ] && [ $((FS * $1 + WARMUP)) -gt "$N_CLAIMS" ] && FS=$(( (N_CLAIMS - WARMUP) / $1 ))
}
log() { echo "$(date +%T) $*" | tee -a "$OUT/progress.log"; }
kill_srv() { pkill -9 -f "[s]glang::|[s]erver\.server|[c]acheScout_server|[c]ontinuum_server|[m]ixed_workload\.run|[j]ac run " 2>/dev/null; sleep 5; }

run_arm() {  # $1 arm, $2 server flags, $3 fact lanes, $4 code lanes
  kill_srv
  local id="$1_f$3c$4" mod=server.server extra="$EXTRA --sched $SCHED" FS CS; size_lanes "$3" "$4"
  [ "$1" = cachescout ] && { mod=server.cacheScout_server; extra=; }
  [ "$1" = continuum ] && { mod=server.continuum_server; extra=; }
  nohup $P -m $mod "$MODEL" $2 $extra --host "$HOST" ${KV:+--kv $KV} > "$OUT/$id.log" 2>&1 &
  local sp=$!
  until grep -q "\[warmup\] engine ready" "$OUT/$id.log" 2>/dev/null; do
    if ! kill -0 $sp 2>/dev/null; then log "$id server died"; tail -5 "$OUT/$id.log"; return 1; fi; sleep 3; done
  log "$id start fact=$3x$FS code=$4x$CS host=${HOST}GB $(grep -o '\[ledger\] device=[0-9]* tokens host=[0-9]* tokens' "$OUT/$id.log")"
  $P -m benchmark.mixed_workload.run --fact-lanes "$3" --code-lanes "$4" \
     --fact-sessions "$FS" --code-sessions "$CS" --warmup "$WARMUP" \
     --tag "$1" --server-log "$OUT/$id.log" --logdir "$OUT/run_$id" > "$OUT/$id.out" 2>&1
  log "$id bench exit $? $(grep -o '^\[[^]]*/fact\] JCT p50= *[0-9.]*s' "$OUT/$id.out" | head -1) $(grep -o '^\[[^]]*/code\] JCT p50= *[0-9.]*s' "$OUT/$id.out" | head -1) $(grep -c 'no_header_last declared' "$OUT/$id.log") sites declared no_header_last"
}

for mix in $MIXES; do
  f=${mix%%+*}; c=${mix##*+}
  for arm in $ARMS; do
    case $arm in
      lruraw) run_arm lruraw "--lru --no-relayout" "$f" "$c";;
      lru)    run_arm lru "--lru" "$f" "$c";;
      kvonly) run_arm kvonly "--no-relayout" "$f" "$c";;
      ours)   run_arm ours "" "$f" "$c";;
      cachescout) run_arm cachescout "" "$f" "$c";;
      continuum) run_arm continuum "" "$f" "$c";;
    esac
  done
done
kill_srv
$P -m benchmark.mixed_workload.sweep_csv "${CSV:-mixed_sweep.csv}" "q8b:$MODEL:$OUT" | tee -a "$OUT/progress.log"
log SWEEP-DONE
