#!/bin/bash
# Start one serving arm in the background and wait for model warmup, or stop the one this script started.
#
#   ./start_server.bash <arm> [server flags]
#       arms: ours ours_noprefetch kvonly relayout adapter vanilla
#             continuum continuum_relayout cachescout cachescout_relayout kvflow kvflow_relayout
#   ./start_server.bash stop          stop the server recorded in $PIDFILE (its whole process group)
#   ./start_server.bash profile       measure the model's device profile on this GPU (prefill tps idle and
#                                     under load, host->device promote tps = H2D bandwidth in tokens/s) with
#                                     the same engine configuration as the arms, write model/<model>.profile.json
#                                     and exit; runs in the foreground (REPROFILE=1 re-measures an existing one)
#
# Defaults are the Great Lakes configuration (plan_on_greatlake.md): Qwen3-32B-AWQ on an A40, device pool
# 31830 tokens (--kv), host pool 20 GB (--host), CLIP 4096; everything else is sglang's default.
#
# Environment (all optional):
#   MODEL HOST KV CLIP SCHED   server configuration (defaults below)
#   LOG                        server stdout/stderr   (default benchmark/mixed_results/server_<arm>.log)
#   PIDFILE                    where the server pid is recorded (default benchmark/mixed_results/server.pid)
#   REQUEST_LOG                if set, the server appends per-request trace events there (BENCH_REQUEST_LOG)
#   READY_TIMEOUT              seconds to wait for "[warmup] engine ready" (default 2400: the first start of a
#                              model on a new GPU loads the weights, JIT-compiles kernels, runs the profiler)
#   PYTHON CUDA_HOME CC CXX    toolchain; the defaults are this box's, override them on the cluster
#   KILL_STRAY=1               also kill every other server/sglang process of this user before starting
#                              (single-user box only; never on a shared cluster node)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/xiaoyu/miniconda3/envs/mortis/bin/python}
export PATH="$(dirname "$PYTHON"):$PATH"
# nvcc for sglang's JIT kernels: CUDA_HOME if given, else the system CUDA 12.8, else whatever nvcc is on PATH.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -d /usr/local/cuda-12.8 ]]; then CUDA_HOME=/usr/local/cuda-12.8
  elif command -v nvcc >/dev/null 2>&1; then CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then export CUDA_HOME; export PATH="$CUDA_HOME/bin:$PATH"; fi
# The JIT needs a C++20 host compiler; prefer the serving env's conda-forge GCC 12 when it is there.
if [[ -z "${CXX:-}" && -x "$(dirname "$PYTHON")/g++" ]]; then
  export CC="$(dirname "$PYTHON")/gcc" CXX="$(dirname "$PYTHON")/g++"
fi
if [[ -n "${CXX:-}" ]]; then export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin $CXX"; fi

PIDFILE=${PIDFILE:-benchmark/mixed_results/server.pid}

stop_recorded() {   # stop the process group recorded in $PIDFILE if it is still alive
  [[ -f "$PIDFILE" ]] || return 0
  local pid; pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$PIDFILE"
}

ARM=${1:-ours}
if (( $# > 0 )); then shift; fi
if [[ "$ARM" == stop ]]; then stop_recorded; exit 0; fi
if [[ "$ARM" == profile ]]; then
  MODEL=${MODEL:-Qwen/Qwen3-32B-AWQ}; HOST=${HOST:-20}; KV=${KV:-31830}; CLIP=${CLIP:-4096}
  export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="$CLIP"
  FLAGS=(--profile-only --port "${GUARD_PORT:-8964}"); [[ "${REPROFILE:-0}" == 1 ]] && FLAGS+=(--reprofile)
  echo "[start_server] $(date '+%F %T') profile: $PYTHON -m server.server $MODEL --host $HOST --kv $KV ${FLAGS[*]} $*"
  exec "$PYTHON" -m server.server "$MODEL" --host "$HOST" --kv "$KV" "${FLAGS[@]}" "$@"
fi

case "$ARM" in
  ours)                MODULE=server.server;            ARM_FLAGS=() ;;
  ours_noprefetch)     MODULE=server.server;            ARM_FLAGS=(--no-promote) ;;
  kvonly)              MODULE=server.server;            ARM_FLAGS=(--no-relayout) ;;
  relayout)            MODULE=server.server;            ARM_FLAGS=(--lru) ;;
  adapter)             MODULE=server.server;            ARM_FLAGS=(--lru --layout adapter) ;;
  vanilla)             MODULE=server.server;            ARM_FLAGS=(--lru --no-relayout) ;;
  continuum)           MODULE=server.continuum_server;  ARM_FLAGS=() ;;
  continuum_relayout)  MODULE=server.continuum_server;  ARM_FLAGS=(--relayout) ;;
  cachescout)          MODULE=server.cacheScout_server; ARM_FLAGS=() ;;
  cachescout_relayout) MODULE=server.cacheScout_server; ARM_FLAGS=(--relayout) ;;
  kvflow)              MODULE=server.KVFlow_server;     ARM_FLAGS=() ;;
  kvflow_relayout)     MODULE=server.KVFlow_server;     ARM_FLAGS=(--relayout) ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

MODEL=${MODEL:-Qwen/Qwen3-32B-AWQ}
HOST=${HOST:-20}
KV=${KV:-31830}
CLIP=${CLIP:-4096}
SCHED=${SCHED:-fcfs}
LOG=${LOG:-benchmark/mixed_results/server_${ARM}.log}
READY_TIMEOUT=${READY_TIMEOUT:-2400}
export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="$CLIP"
if [[ -n "${REQUEST_LOG:-}" ]]; then mkdir -p "$(dirname "$REQUEST_LOG")"; export BENCH_REQUEST_LOG=$REQUEST_LOG; fi

stop_recorded
if [[ "${KILL_STRAY:-0}" == 1 ]]; then
  pkill -9 -f '[s]erver\.(server|continuum_server|cacheScout_server|KVFlow_server)|[s]glang::' 2>/dev/null || true
  sleep 3
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
: >"$LOG"   # a stale log already contains "engine ready"
# GUARD_PORT: the server's port (default 8964); run_arm.py and the jac agents read the same variable.
COMMAND=("$PYTHON" -m "$MODULE" "$MODEL" "${ARM_FLAGS[@]}" --sched "$SCHED" --host "$HOST" --kv "$KV" --port "${GUARD_PORT:-8964}" "$@")
echo "[start_server] $(date '+%F %T') $ARM: ${COMMAND[*]} (CLIP=$CLIP)" >>"$LOG"
# setsid: the server and every sglang subprocess share one process group, which `stop` kills as a unit.
setsid nohup "${COMMAND[@]}" >>"$LOG" 2>&1 </dev/null &
PID=$!
echo "$PID" >"$PIDFILE"

for (( t = 0; t < READY_TIMEOUT; t += 3 )); do
  if grep -q '\[warmup\] engine ready' "$LOG" 2>/dev/null; then
    echo "$ARM server ready: pid=$PID log=$LOG model=$MODEL kv=$KV host=${HOST}GB clip=$CLIP"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "$ARM server died; last lines of $LOG:" >&2; tail -n 8 "$LOG" >&2; rm -f "$PIDFILE"; exit 1
  fi
  sleep 3
done
echo "$ARM server not ready after ${READY_TIMEOUT}s, stopping it; last lines of $LOG:" >&2; tail -n 8 "$LOG" >&2
stop_recorded; exit 1
