#!/bin/bash
# Start the serving engine detached and wait until it is ready.
#   ./start_server.bash [arm] [extra server flags]
# arm: ours (default) | lru (relayout only) | lruraw (plain engine) | kvonly (planner only)
# Env: MODEL HOST (GB) KV (device tokens) SCHED LOG
set -u
cd "$(dirname "$0")"
P=/home/xiaoyu/miniconda3/envs/sglang/bin/python
ARM=${1:-ours}; [ $# -gt 0 ] && shift
MODEL=${MODEL:-Qwen/Qwen3-8B}; HOST=${HOST:-16}; KV=${KV:-}; SCHED=${SCHED:-fcfs}
LOG=${LOG:-benchmark/mixed_results/server_${ARM}.log}

case $ARM in
  ours)   FLAGS="";;
  lru)    FLAGS="--lru";;
  lruraw) FLAGS="--lru --no-relayout";;
  kvonly) FLAGS="--no-relayout";;
  *) echo "unknown arm: $ARM" >&2; exit 2;;
esac

if pgrep -f "[s]erver\.server" >/dev/null; then
  pkill -9 -f "[s]glang::|[s]erver\.server"
  sleep 5
fi
mkdir -p "$(dirname "$LOG")"
nohup $P -m server.server "$MODEL" $FLAGS --sched "$SCHED" --host "$HOST" ${KV:+--kv $KV} "$@" > "$LOG" 2>&1 &
PID=$!

until grep -q "\[warmup\] engine ready" "$LOG" 2>/dev/null; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server died, last lines of $LOG:" >&2
    tail -5 "$LOG" >&2
    exit 1
  fi
  sleep 3
done
echo "$ARM server ready: pid=$PID log=$LOG $(grep -o '\[ledger\] device=[0-9]* tokens host=[0-9]* tokens' "$LOG")"
