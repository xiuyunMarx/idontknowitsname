#!/bin/sh
# Kill all vLLM-related processes (POSIX sh, works under dash/bash):
#   - VLLM::EngineCore workers (renamed via setproctitle, cmdline won't say "vllm")
#   - anything launched as `vllm serve` / `python -m vllm...`
#   - the parent process owning the EngineCore workers (e.g. start_server.py),
#     otherwise GPU memory stays allocated
#   - leftover GPU compute processes reported by nvidia-smi whose name mentions vllm
#
# Usage: ./kill_all_vllm.sh [--no-parent]     # --no-parent: keep the owning server alive
#        TERM_WAIT=10 ./kill_all_vllm.sh      # seconds to wait after SIGTERM (default 5)
set -u

KILL_PARENT=1
if [ "${1:-}" = "--no-parent" ]; then KILL_PARENT=0; fi
TERM_WAIT=${TERM_WAIT:-5}
ME=$$

collect() {
    {
        # 1. EngineCore workers: match on process *name* (comm), which setproctitle sets to
        #    "VLLM::EngineCore" (truncated to 15 chars by the kernel). Matching comm instead of
        #    the full cmdline avoids hitting shells/editors that merely mention "vllm".
        ps -eo pid=,comm= | awk '$2 ~ /^VLLM::/ {print $1}'
        # 2. Real vllm launch commands: `vllm serve ...`, `python -m vllm ...`, vllm.entrypoints
        pgrep -f '(^|/)vllm (serve|bench|run|chat|complete)( |$)' 2>/dev/null
        pgrep -f 'python[0-9.]* +-m +vllm(\.| |$)' 2>/dev/null
        pgrep -f 'vllm\.entrypoints' 2>/dev/null
        # 3. Parents of EngineCore workers (the process that created the LLM engine)
        if [ "$KILL_PARENT" -eq 1 ]; then
            for p in $(ps -eo pid=,comm= | awk '$2 ~ /^VLLM::/ {print $1}'); do
                pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
                if [ -n "$pp" ] && [ "$pp" != "1" ]; then echo "$pp"; fi
            done
        fi
        # 4. GPU compute processes whose name mentions vllm
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
                | awk -F', *' 'tolower($2) ~ /vllm/ {print $1}'
        fi
    } | grep -E '^[0-9]+$' | grep -vx "$ME" | grep -vx "$PPID" | sort -un
}

alive() {   # print the subset of given pids that are still running
    for p in "$@"; do
        if kill -0 "$p" 2>/dev/null; then echo "$p"; fi
    done
}

PIDS=$(collect | tr '\n' ' ')
if [ -z "$PIDS" ]; then
    echo "No vLLM processes found."
    exit 0
fi

echo "Found vLLM-related processes:"
ps -o pid,ppid,user,etime,comm,args -p "$(echo "$PIDS" | tr ' ' ',' | sed 's/,$//')" 2>/dev/null

echo "Sending SIGTERM..."
# shellcheck disable=SC2086
kill -TERM $PIDS 2>/dev/null

i=0
REMAIN=$PIDS
while [ "$i" -lt "$TERM_WAIT" ]; do
    sleep 1
    # shellcheck disable=SC2086
    REMAIN=$(alive $REMAIN | tr '\n' ' ')
    if [ -z "$REMAIN" ]; then break; fi
    i=$((i + 1))
done

if [ -n "$REMAIN" ]; then
    echo "Still alive after ${TERM_WAIT}s, sending SIGKILL: $REMAIN"
    # shellcheck disable=SC2086
    kill -KILL $REMAIN 2>/dev/null
    sleep 1
fi

# Second pass: catch anything spawned/renamed in the meantime
LEFT=$(collect | tr '\n' ' ')
if [ -n "$LEFT" ]; then
    echo "Second pass, SIGKILL: $LEFT"
    # shellcheck disable=SC2086
    kill -KILL $LEFT 2>/dev/null
    sleep 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Remaining GPU compute processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi
echo "Done."
