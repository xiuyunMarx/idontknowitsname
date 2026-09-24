#!/bin/bash
# Start one serving arm in the background and wait for model warmup.
#
#   ./start_server.bash [ours|kvonly|relayout|vanilla|continuum|cachescout|kvflow|kvflow_relayout|continuum_relayout|cachescout_relayout] [server flags]
#   relayout = re-layout only (no planner), vanilla = vanilla SGLang (no re-layout, no planner)
#
# Environment: MODEL, HOST, KV, SCHED, CLIP, LOG
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=/home/xiaoyu/miniconda3/envs/sglang/bin/python
export PATH="$(dirname "$PYTHON"):$PATH"

ARM=${1:-ours}
if (( $# > 0 )); then
  shift
fi

case "$ARM" in
  ours)             MODULE=server.server;            ARM_FLAGS=() ;;
  kvonly)           MODULE=server.server;            ARM_FLAGS=(--no-relayout) ;;
  relayout)         MODULE=server.server;            ARM_FLAGS=(--lru) ;;
  vanilla)          MODULE=server.server;            ARM_FLAGS=(--lru --no-relayout) ;;
  continuum)        MODULE=server.continuum_server;  ARM_FLAGS=() ;;
  kvflow)           MODULE=server.KVFlow_server;     ARM_FLAGS=() ;;
  kvflow_relayout)  MODULE=server.KVFlow_server;     ARM_FLAGS=(--relayout) ;;
  continuum_relayout)  MODULE=server.continuum_server;  ARM_FLAGS=(--relayout) ;;
  cachescout_relayout) MODULE=server.cacheScout_server; ARM_FLAGS=(--relayout) ;;
  cachescout)       MODULE=server.cacheScout_server; ARM_FLAGS=() ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

MODEL=${MODEL:-Qwen/Qwen3-14B-AWQ}
HOST=${HOST:-16}
KV=${KV:-}
SCHED=${SCHED:-fcfs}
CLIP=${CLIP:-}
LOG=${LOG:-benchmark/mixed_results/cache/server_log/server_${ARM}.log}

if [[ -n "$CLIP" ]]; then
  export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="$CLIP"
fi

SERVER_PATTERN='[s]erver\.(server|continuum_server|cacheScout_server|KVFlow_server)'
if pgrep -f "$SERVER_PATTERN" >/dev/null; then
  pkill -9 -f "$SERVER_PATTERN|[s]glang::"
  sleep 5
fi

mkdir -p "$(dirname "$LOG")"

COMMAND=("$PYTHON" -m "$MODULE" "$MODEL" "${ARM_FLAGS[@]}" --sched "$SCHED" --host "$HOST")
if [[ -n "$KV" ]]; then
  COMMAND+=(--kv "$KV")
fi
COMMAND+=("$@")

nohup "${COMMAND[@]}" >"$LOG" 2>&1 &
PID=$!

until grep -q '\[warmup\] engine ready' "$LOG" 2>/dev/null; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "$ARM server failed; last lines of $LOG:" >&2
    tail -n 5 "$LOG" >&2
    exit 1
  fi
  sleep 3
done

echo "$ARM server ready: pid=$PID log=$LOG"
