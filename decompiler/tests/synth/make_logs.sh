#!/usr/bin/env bash
# Regenerate decompiler/tests/logs/requests.jsonl: real byllm and DSPy request bodies,
# produced by running the benchmark programs against recorder.py (a mock
# chat-completions server that answers with schema-valid replies). No GPU.
#
#   DSPY_PYTHON=/path/to/venv/bin/python decompiler/tests/synth/make_logs.sh
#
# DSPY_PYTHON needs `pip install dspy` (3.3.x). The byllm programs run with the
# repo's `jac`. Port 8965 is used so a guard server on 8964 is left alone.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$REPO/decompiler/tests/synth
LOG=$REPO/decompiler/tests/logs/requests.jsonl
PORT=${PORT:-8965}
WORK=$(mktemp -d)
trap 'kill $REC 2>/dev/null || true; rm -rf "$WORK"' EXIT

mkdir -p "$(dirname "$LOG")" "$WORK/prog" "$WORK/state" "$WORK/wiki_cache"; rm -f "$LOG"
# copies of the programs with the interceptor pointed at the recorder's port
for p in fact_check/fact_check.jac coding/coding_agent.jac; do
  sed "s/InterceptorLLM(model_name=\"Qwen\/Qwen3-8B\")/InterceptorLLM(model_name=\"Qwen\/Qwen3-8B\", comm_port=$PORT)/" \
      "$REPO/benchmark/$p" > "$WORK/prog/$(basename "$p")"
done
python "$HERE/recorder.py" "$PORT" "$LOG" "$WORK/wiki_cache" & REC=$!
python - "$PORT" <<'PY'
import socket, sys, time
for _ in range(50):
    try: socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2).close(); break
    except OSError: time.sleep(0.1)
PY
i=0
while IFS=$'\t' read -r label hops claim; do
  [ "${label:0:1}" = "#" ] && continue
  i=$((i+1)); [ $i -gt 4 ] && break
  mkdir -p "$WORK/state/fc$i"
  FC_CLAIM="$claim" FC_CACHE_DIR="$WORK/wiki_cache" FC_MIN_ROUNDS=2 FC_ROUNDS=2 JAC_DATA_PATH="$WORK/state/fc$i" \
    jac run "$WORK/prog/fact_check.jac" > "$WORK/fc$i.out" 2>&1 && echo "fact_check $i ok"
done < "$REPO/benchmark/fact_check/hover_claims_120.tsv"
j=0
for T in "HumanEval/0,HumanEval/1" "HumanEval/2,HumanEval/3"; do
  j=$((j+1)); mkdir -p "$WORK/state/ca$j"
  CA_TASKS="$T" CA_DATA="$REPO/benchmark/coding/HumanEval.jsonl.gz" CA_MAX_ATTEMPTS=2 CA_TEST_TIMEOUT_S=30 JAC_DATA_PATH="$WORK/state/ca$j" \
    jac run "$WORK/prog/coding_agent.jac" > "$WORK/ca$j.out" 2>&1 && echo "coding $j ok"
done
"${DSPY_PYTHON:-python}" "$HERE/dspy_programs.py" "$PORT" \
  "Water boils at 100C at sea level." \
  "Ralek Gracie is the nephew of a retired mixed martial artist who was born in 1958." \
  "The Ford Fusion was introduced for model year 2006." \
  "Wendigo is a legend that gives its name to a medical term." \
  "Paris is the capital of France."
echo "$(wc -l < "$LOG") requests -> $LOG"
