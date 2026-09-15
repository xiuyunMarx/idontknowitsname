#!/bin/bash
# Mixed workload: fact_check and coding_agent sessions interleaved on one server, one mix,
# every arm in turn (fresh server per arm), driven by benchmark.mixed_workload.run.
#
#   ./mixed.bash 8+8                      8 fact_check lanes + 8 coding_agent lanes, ARMS="lruraw lru ours"
#   ARMS="lru ours" ./mixed.bash 4+12
#
# Coding lanes run a fixed count per lane (CODE_SESSIONS, default 7, shrunk to fit the 64
# HumanEval bundles); fact lanes "follow": they keep taking fresh claims until the coding
# lanes finish, so both programs share the pool for the whole run (FACT_SESSIONS=N fixes them).
#
# Arms as in sweep.bash (lruraw, lru, kvonly, ours, cachescout, continuum). The driver registers
# both programs' static analyses before the warm-up; fact_check with its headers kept in front
# (the current model's output changes under header-last), coding_agent may go header-last
# (FACT_HEADER_LAST=1 lifts the restriction). Opaque baselines skip the registration.
#
# Environment overrides: MODEL HOST KV ARMS FACT_SESSIONS CODE_SESSIONS WARMUP OUT CSV EXTRA SCHED
#   FACT_HEADER_LAST FC_TOOL_DELAY_S FC_MIN_ROUNDS CA_TOOL_DELAY_S
# Results in results/mixed/{arm}_f{F}c{C}.{log,out}; the CSV at the end has one row per
# (arm, program): python -m benchmark.mixed_workload.sweep_csv.
set -u
cd "$(dirname "$0")"
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MIX=${1:-8+8}; F=${MIX%%+*}; C=${MIX##*+}
case "$F$C" in *[!0-9]*|"") echo "usage: $0 F+C   (F fact_check lanes, C coding_agent lanes; env ARMS MODEL HOST KV ...)" >&2; exit 2;; esac
MODEL=${MODEL:-Qwen/Qwen3-8B}; HOST=${HOST:-8}; KV=${KV:-}; WARMUP=${WARMUP:-2}
ARMS=${ARMS:-"lruraw lru ours"}
FACT_SESSIONS=${FACT_SESSIONS:-}; CODE_SESSIONS=${CODE_SESSIONS:-}   # per lane; empty = auto-size (see size_lanes)
FACT_HEADER_LAST=${FACT_HEADER_LAST:-0}
N_CLAIMS=$(grep -cv '^#' benchmark/fact_check/hover_claims_120.tsv); N_BUNDLES=$(grep -cv '^#' benchmark/coding/tasks.txt)
OUT=${OUT:-results/mixed}; mkdir -p "$OUT"; CSV=${CSV:-$OUT/mixed.csv}
EXTRA=${EXTRA:-}; SCHED=${SCHED:-fcfs}
export FC_CACHE_DIR=$PWD/benchmark/fact_check/wiki_cache
export FC_TOOL_DELAY_S=${FC_TOOL_DELAY_S:-2}
export FC_MIN_ROUNDS=${FC_MIN_ROUNDS:-2}
export CA_TOOL_DELAY_S=${CA_TOOL_DELAY_S:-2}

size_lanes() {  # -> FS CS: code 7/lane capped by the bundles; fact follows (0) unless FACT_SESSIONS is set
  CS=${CODE_SESSIONS:-7}; [ $((CS * C + WARMUP)) -gt "$N_BUNDLES" ] && CS=$(( (N_BUNDLES - WARMUP) / C ))
  FS=${FACT_SESSIONS:-0}
  [ "$FS" -gt 0 ] && [ $((FS * F + WARMUP)) -gt "$N_CLAIMS" ] && FS=$(( (N_CLAIMS - WARMUP) / F ))
}
log() { echo "$(date +%T) $*" | tee -a "$OUT/progress.log"; }
kill_srv() { pkill -9 -f "[s]glang::|[s]erver\.server|[c]acheScout_server|[c]ontinuum_server|[m]ixed_workload\.run|[j]ac run " 2>/dev/null; sleep 5; }

run_arm() {  # $1 arm, $2 server flags
  kill_srv
  local id="$1_f${F}c${C}" mod=server.server extra="$EXTRA --sched $SCHED" reg="" FS CS; size_lanes
  [ "$1" = cachescout ] && { mod=server.cacheScout_server; extra=; reg="--no-register"; }
  [ "$1" = continuum ] && { mod=server.continuum_server; extra=; reg="--no-register"; }
  nohup $P -m $mod "$MODEL" $2 $extra --host "$HOST" ${KV:+--kv $KV} > "$OUT/$id.log" 2>&1 &
  local sp=$!
  until grep -q "\[warmup\] engine ready" "$OUT/$id.log" 2>/dev/null; do
    if ! kill -0 $sp 2>/dev/null; then log "$id server died"; tail -5 "$OUT/$id.log"; return 1; fi; sleep 3; done
  log "$id start fact=${F}x$FS code=${C}x$CS host=${HOST}GB $(grep -o '\[ledger\] device=[0-9]* tokens host=[0-9]* tokens' "$OUT/$id.log")"
  $P -m benchmark.mixed_workload.run --fact-lanes "$F" --code-lanes "$C" \
     --fact-sessions "$FS" --code-sessions "$CS" --warmup "$WARMUP" $reg $([ "$FACT_HEADER_LAST" = 1 ] && echo --fact-header-last) \
     --tag "$1" --server-log "$OUT/$id.log" --logdir "$OUT/run_$id" > "$OUT/$id.out" 2>&1
  log "$id bench exit $? $(grep -o '^\[[^]]*/fact\] JCT p50= *[0-9.]*s' "$OUT/$id.out" | head -1) $(grep -o '^\[[^]]*/code\] JCT p50= *[0-9.]*s' "$OUT/$id.out" | head -1) $(grep -o 'programs registered=[0-9]* .*opaque_requests=[0-9]*' "$OUT/$id.out" | head -1)"
}

log "MIXED $MIX model=$MODEL host=${HOST}GB kv=${KV:-default} arms=[$ARMS] fact_header_last=$FACT_HEADER_LAST"
for arm in $ARMS; do
  case $arm in
    lruraw) run_arm lruraw "--lru --no-relayout";;
    lru)    run_arm lru "--lru";;
    kvonly) run_arm kvonly "--no-relayout";;
    ours)   run_arm ours "";;
    cachescout) run_arm cachescout "";;
    continuum) run_arm continuum "";;
    *) log "unknown arm $arm";;
  esac
done
kill_srv
$P -m benchmark.mixed_workload.sweep_csv "$CSV" "q8b:$MODEL:$OUT" | tee -a "$OUT/progress.log"
log MIXED-DONE
