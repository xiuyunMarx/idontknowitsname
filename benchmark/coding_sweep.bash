#!/bin/bash
# Concurrency sweep for the coding agent (benchmark/coding/coding_agent.jac), three arms per level:
#   lruraw  --lru --no-relayout   raw SGLang HiCache LRU, prompts as the program lays them out
#   lru     --lru                 LRU + the server's IR-driven prompt re-layout
#   kvonly  --no-relayout         planner only (promotion, steering, retirement), no re-layout
#   ours                          re-layout + planner
#   cachescout                        server.cacheScout_server: opaque requests, online Markov agent model
#   ARMS="cachescout" LEVELS="12 16" ./benchmark/coding_sweep.bash   (run from the repo root)
# One 5-function HumanEval bundle per session (benchmark/coding/tasks.txt), pytest floor CA_TOOL_DELAY_S.
# Results in results/coding_sweep/{arm}_c{N}.{out,log}; consolidate with
#   python -m benchmark.fact_check.sweep_csv coding_sweep.csv q8b:Qwen/Qwen3-8B:results/coding_sweep
set -u
cd "$(dirname "$0")/.."
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
MODEL=${MODEL:-Qwen/Qwen3-8B}; HOST=${HOST:-8}; WARMUP=2
LEVELS=${LEVELS:-"16 20 24 28"}
ARMS=${ARMS:-"lruraw lru ours"}
OUT=${OUT:-results/coding_sweep}; mkdir -p "$OUT"
export CA_TOOL_DELAY_S=${CA_TOOL_DELAY_S:-2}
per_lane() { case $1 in 12) echo 2;; *) echo 1;; esac; }   # 32 bundles cap warmup + c*N
log() { echo "$(date +%T) $*" | tee -a "$OUT/progress.log"; }
kill_srv() { pkill -9 -f "[s]glang::|[s]erver\.server|[c]acheScout_server|[f]act_bench|[j]ac run " 2>/dev/null; sleep 5; }

run_arm() {  # $1 arm, $2 server flags, $3 concurrency
  kill_srv
  local mod=server.server; [ "$1" = cachescout ] && mod=server.cacheScout_server
  nohup $P -m $mod "$MODEL" $2 --host "$HOST" > "$OUT/$1_c$3.log" 2>&1 &
  local sp=$!
  until grep -q "\[warmup\] engine ready" "$OUT/$1_c$3.log" 2>/dev/null; do
    if ! kill -0 $sp 2>/dev/null; then log "$1 c=$3 server died"; tail -5 "$OUT/$1_c$3.log"; return 1; fi; sleep 3; done
  local n; n=$(per_lane "$3")
  log "$1 c=$3 start sessions=$((n * $3)) host=${HOST}GB $(grep -o '\[ledger\] device=[0-9]* tokens host=[0-9]* tokens' "$OUT/$1_c$3.log")"
  $P -m benchmark.fact_check.fact_bench --program benchmark/coding/coding_agent.jac --claim-env CA_TASKS \
     --claims benchmark/coding/tasks.txt --sessions "$n" --concurrency "$3" --warmup $WARMUP \
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
log SWEEP-DONE
