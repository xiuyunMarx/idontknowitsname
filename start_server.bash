#!/bin/bash
# Start one serving arm in the background and wait for model warmup.
#
#   ./start_server.bash [ours|kvonly|relayout|adapter|vanilla|continuum|cachescout|kvflow|kvflow_relayout|continuum_relayout|cachescout_relayout] [server flags]
#   relayout = re-layout only (no planner), adapter = framework-adapter layout only (header behind the values, native binding order, no planner), vanilla = vanilla SGLang (no re-layout, no planner)
#
# Environment: MODEL, HOST, KV, SCHED, CLIP, LOG
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/xiaoyu/miniconda3/envs/mortis/bin/python}
export PATH="$(dirname "$PYTHON"):$PATH"
# CUDA toolkit for sglang's JIT kernels (tvm-ffi resolves nvcc from CUDA_HOME, then PATH); the
# system /usr/bin/nvcc is CUDA 10.1 and cannot build them.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
# Host C++ compiler for nvcc and the JIT's .cpp units: the kernels need C++20 (<concepts>), which the
# system gcc 9 lacks; GCC 12 is installed in the serving env (conda-forge gcc=12 gxx=12).
export CC=${CC:-$(dirname "$PYTHON")/gcc}
export CXX=${CXX:-$(dirname "$PYTHON")/g++}
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin $CXX"

ARM=${1:-ours}
if (( $# > 0 )); then
  shift
fi

case "$ARM" in
  ours)             MODULE=server.server;            ARM_FLAGS=() ;;
  kvonly)           MODULE=server.server;            ARM_FLAGS=(--no-relayout) ;;
  relayout)         MODULE=server.server;            ARM_FLAGS=(--lru) ;;
  adapter)          MODULE=server.server;            ARM_FLAGS=(--lru --layout adapter) ;;
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
: >"$LOG"   # truncate before launching: a stale log from an earlier run already contains "engine ready"

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
